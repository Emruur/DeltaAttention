"""
Final Optimized RULER benchmark for Llama-3.1-8B-Instruct.
Includes System Prompts, Noise-Resistant Filler, Explicit Task Instructions, 
Multi-row Table Generation, CSV export, and PNG Rendering.
"""

import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import json
import argparse
import random
import re
import gc
import time
import string
import uuid
import glob
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import globVR
from modeling_llama import LlamaForCausalLM, LlamaConfig
from deltaDecoding import HybridCompressedCache

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_CTX  = 131072

CTX_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
CTX_LABELS  = ["4k",  "8k",  "16k",  "32k",  "64k", "128k"]

RULER_TASKS = [
    "niah_single_1", "niah_single_2", "niah_single_3",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multivalue", "niah_multiquery", "vt", "cwe", "fwe",
    "qa_hotpot", "qa_squad",
]

EXPERIMENT_DEFINITIONS = {
    "baseline": {
        "injector": lambda args: {
            "delta_pf_key_on": 0, "delta_mlp": "Regular", "flash": True,
        }
    },
    "row_delta": {
        "injector": lambda args: {
            "delta_pf_key_on": 1, "delta_type": "row", "scale": args.scale,
            "delta_mlp": "Regular", "row_delta_threshold": args.row_thresh,
            "row_similarity_metric": args.row_sim, "chunk_size": args.chunk_size,
            "divide_to": 0, "flash": True, "delta_decode": False,
            "dense_window_size": args.dense_window_size,
        }
    },
}

# ==========================================
# JSON ENCODER & FILE I/O
# ==========================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, cls=NpEncoder)
    print(f"[IO] Saved {path}", flush=True)

# ==========================================
# TABLE EXPORT (CSV & PNG)
# ==========================================

def print_multi_table(all_results: dict[str, dict[int, float]], out_dir: str):
    header = f"{'Input Len':<18}|" + "".join(f"  {lbl:>5} |" for lbl in CTX_LABELS) + f"   Avg."
    sep    = "-" * len(header)
    
    print("\n" + "=" * len(header), flush=True)
    print("RULER Benchmark Results", flush=True)
    print("=" * len(header), flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    
    # Prepare data for Dataframe
    df_data = []
    
    for row_name, scores_by_ctx in all_results.items():
        vals  = [scores_by_ctx.get(c, float("nan")) for c in CTX_LENGTHS]
        valid = [v for v in vals if not np.isnan(v)]
        avg   = float(np.mean(valid)) if valid else float("nan")
        
        row_str = f"{row_name:<18}|" + "".join(
            f"  {v:5.2f} |" if not np.isnan(v) else f"  {'n/a':>5} |"
            for v in vals
        ) + f"  {avg:5.2f}"
        print(row_str, flush=True)
        
        # Build dict for Pandas
        row_dict = {"Input Len": row_name}
        for i, lbl in enumerate(CTX_LABELS):
            row_dict[lbl] = f"{vals[i]:.2f}" if not np.isnan(vals[i]) else "n/a"
        row_dict["Avg."] = f"{avg:.2f}" if not np.isnan(avg) else "n/a"
        df_data.append(row_dict)
        
    print("=" * len(header) + "\n", flush=True)
    
    # Generate Outputs
    df = pd.DataFrame(df_data)
    
    # 1. Save CSV
    csv_path = os.path.join(out_dir, "final_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"[IO] Final table saved to: {csv_path}", flush=True)
    
    # 2. Save PNG using Matplotlib
    png_path = os.path.join(out_dir, "final_table.png")
    fig, ax = plt.subplots(figsize=(10, len(df) * 0.5 + 1.5)) # Dynamically scale height
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Optional styling for header row
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f2f2f2')
            
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[IO] Final table image saved to: {png_path}", flush=True)

# ==========================================
# HAYSTACK CORPUS  (Paul Graham Essays — official RULER filler)
# ==========================================

_HAYSTACK_TOKENS: list[int] = []  # populated once in run_ruler()

_PG_ESSAYS_CACHE = Path("ruler_cache/PaulGrahamEssays.json")

# Exact URL list from hsiehjackson/RULER scripts/data/synthetic/json/PaulGrahamEssays_URLs.txt
_PG_ESSAYS_SOURCE_URLS = [
    "http://www.paulgraham.com/13sentences.html",
    "http://www.paulgraham.com/5founders.html",
    "http://www.paulgraham.com/6631327.html",
    "http://www.paulgraham.com/95.html",
    "http://www.paulgraham.com/ace.html",
    "http://www.paulgraham.com/airbnb.html",
    "http://www.paulgraham.com/airbnbs.html",
    "http://www.paulgraham.com/alien.html",
    "http://www.paulgraham.com/altair.html",
    "http://www.paulgraham.com/ambitious.html",
    "http://www.paulgraham.com/america.html",
    "http://www.paulgraham.com/angelinvesting.html",
    "http://www.paulgraham.com/artistsship.html",
    "http://www.paulgraham.com/badeconomy.html",
    "http://www.paulgraham.com/better.html",
    "http://www.paulgraham.com/bronze.html",
    "http://www.paulgraham.com/bubble.html",
    "http://www.paulgraham.com/charisma.html",
    "http://www.paulgraham.com/cities.html",
    "http://www.paulgraham.com/college.html",
    "http://www.paulgraham.com/colleges.html",
    "http://www.paulgraham.com/conformism.html",
    "http://www.paulgraham.com/control.html",
    "http://www.paulgraham.com/convergence.html",
    "http://www.paulgraham.com/convince.html",
    "http://www.paulgraham.com/cred.html",
    "http://www.paulgraham.com/credentials.html",
    "http://www.paulgraham.com/determination.html",
    "http://www.paulgraham.com/die.html",
    "http://www.paulgraham.com/disagree.html",
    "http://www.paulgraham.com/disc.html",
    "http://www.paulgraham.com/discover.html",
    "http://www.paulgraham.com/distraction.html",
    "http://www.paulgraham.com/divergence.html",
    "http://www.paulgraham.com/donate.html",
    "http://www.paulgraham.com/ds.html",
    "http://www.paulgraham.com/early.html",
    "http://www.paulgraham.com/earnest.html",
    "http://www.paulgraham.com/equity.html",
    "http://www.paulgraham.com/essay.html",
    "http://www.paulgraham.com/ffb.html",
    "http://www.paulgraham.com/fh.html",
    "http://www.paulgraham.com/fix.html",
    "http://www.paulgraham.com/fn.html",
    "http://www.paulgraham.com/foundersatwork.html",
    "http://www.paulgraham.com/fp.html",
    "http://www.paulgraham.com/fr.html",
    "http://www.paulgraham.com/fundraising.html",
    "http://www.paulgraham.com/future.html",
    "http://www.paulgraham.com/genius.html",
    "http://www.paulgraham.com/getideas.html",
    "http://www.paulgraham.com/good.html",
    "http://www.paulgraham.com/goodart.html",
    "http://www.paulgraham.com/googles.html",
    "http://www.paulgraham.com/greatwork.html",
    "http://www.paulgraham.com/growth.html",
    "http://www.paulgraham.com/guidetoinvestors.html",
    "http://www.paulgraham.com/hackernews.html",
    "http://www.paulgraham.com/head.html",
    "http://www.paulgraham.com/herd.html",
    "http://www.paulgraham.com/heresy.html",
    "http://www.paulgraham.com/heroes.html",
    "http://www.paulgraham.com/highres.html",
    "http://www.paulgraham.com/hiresfund.html",
    "http://www.paulgraham.com/hiring.html",
    "http://www.paulgraham.com/hp.html",
    "http://www.paulgraham.com/hs.html",
    "http://www.paulgraham.com/hundred.html",
    "http://www.paulgraham.com/hw.html",
    "http://www.paulgraham.com/hwh.html",
    "http://www.paulgraham.com/icad.html",
    "http://www.paulgraham.com/ideas.html",
    "http://www.paulgraham.com/identity.html",
    "http://www.paulgraham.com/ineq.html",
    "http://www.paulgraham.com/inequality.html",
    "http://www.paulgraham.com/investors.html",
    "http://www.paulgraham.com/invtrend.html",
    "http://www.paulgraham.com/javacover.html",
    "http://www.paulgraham.com/jessica.html",
    "http://www.paulgraham.com/judgement.html",
    "http://www.paulgraham.com/kate.html",
    "http://www.paulgraham.com/kids.html",
    "http://www.paulgraham.com/ladder.html",
    "http://www.paulgraham.com/lesson.html",
    "http://www.paulgraham.com/lies.html",
    "http://www.paulgraham.com/lwba.html",
    "http://www.paulgraham.com/mac.html",
    "http://www.paulgraham.com/makersschedule.html",
    "http://www.paulgraham.com/marginal.html",
    "http://www.paulgraham.com/maybe.html",
    "http://www.paulgraham.com/mean.html",
    "http://www.paulgraham.com/microsoft.html",
    "http://www.paulgraham.com/mit.html",
    "http://www.paulgraham.com/name.html",
    "http://www.paulgraham.com/nerds.html",
    "http://www.paulgraham.com/newthings.html",
    "http://www.paulgraham.com/noob.html",
    "http://www.paulgraham.com/noop.html",
    "http://www.paulgraham.com/notnot.html",
    "http://www.paulgraham.com/nov.html",
    "http://www.paulgraham.com/nthings.html",
    "http://www.paulgraham.com/opensource.html",
    "http://www.paulgraham.com/organic.html",
    "http://www.paulgraham.com/orth.html",
    "http://www.paulgraham.com/own.html",
    "http://www.paulgraham.com/patentpledge.html",
    "http://www.paulgraham.com/pgh.html",
    "http://www.paulgraham.com/pinch.html",
    "http://www.paulgraham.com/polls.html",
    "http://www.paulgraham.com/power.html",
    "http://www.paulgraham.com/prcmc.html",
    "http://www.paulgraham.com/procrastination.html",
    "http://www.paulgraham.com/progbot.html",
    "http://www.paulgraham.com/prop62.html",
    "http://www.paulgraham.com/property.html",
    "http://www.paulgraham.com/publishing.html",
    "http://www.paulgraham.com/pypar.html",
    "http://www.paulgraham.com/ramenprofitable.html",
    "http://www.paulgraham.com/randomness.html",
    "http://www.paulgraham.com/re.html",
    "http://www.paulgraham.com/read.html",
    "http://www.paulgraham.com/real.html",
    "http://www.paulgraham.com/really.html",
    "http://www.paulgraham.com/relres.html",
    "http://www.paulgraham.com/revolution.html",
    "http://www.paulgraham.com/richnow.html",
    "http://www.paulgraham.com/road.html",
    "http://www.paulgraham.com/ronco.html",
    "http://www.paulgraham.com/safe.html",
    "http://www.paulgraham.com/say.html",
    "http://www.paulgraham.com/schlep.html",
    "http://www.paulgraham.com/seesv.html",
    "http://www.paulgraham.com/segway.html",
    "http://www.paulgraham.com/selfindulgence.html",
    "http://www.paulgraham.com/sfp.html",
    "http://www.paulgraham.com/simply.html",
    "http://www.paulgraham.com/smart.html",
    "http://www.paulgraham.com/softwarepatents.html",
    "http://www.paulgraham.com/spam.html",
    "http://www.paulgraham.com/speak.html",
    "http://www.paulgraham.com/start.html",
    "http://www.paulgraham.com/startupfunding.html",
    "http://www.paulgraham.com/startuphubs.html",
    "http://www.paulgraham.com/startupideas.html",
    "http://www.paulgraham.com/startupmistakes.html",
    "http://www.paulgraham.com/stuff.html",
    "http://www.paulgraham.com/superlinear.html",
    "http://www.paulgraham.com/swan.html",
    "http://www.paulgraham.com/tablets.html",
    "http://www.paulgraham.com/talk.html",
    "http://www.paulgraham.com/taste.html",
    "http://www.paulgraham.com/think.html",
    "http://www.paulgraham.com/top.html",
    "http://www.paulgraham.com/trolls.html",
    "http://www.paulgraham.com/twitter.html",
    "http://www.paulgraham.com/usa.html",
    "http://www.paulgraham.com/users.html",
    "http://www.paulgraham.com/venturecapital.html",
    "http://www.paulgraham.com/wealth.html",
    "http://www.paulgraham.com/webstartups.html",
    "http://www.paulgraham.com/whyyc.html",
    "http://www.paulgraham.com/word.html",
    "http://www.paulgraham.com/words.html",
    "http://www.paulgraham.com/work.html",
    "http://www.paulgraham.com/writing44.html",
    "http://www.paulgraham.com/wtax.html",
    "http://www.paulgraham.com/yahoo.html",
    "http://www.paulgraham.com/ycombinator.html",
    "http://www.paulgraham.com/ycstart.html",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/addiction.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/aord.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/apple.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/avg.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/before.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/bias.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/boss.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/copy.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/corpdev.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/desres.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/diff.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/ecw.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/founders.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/foundervisa.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/gap.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/gba.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/gh.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/goodtaste.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/hubs.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/iflisp.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/island.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/know.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/langdes.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/laundry.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/love.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/mod.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/newideas.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/nft.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/philosophy.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/popular.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/pow.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/rootsoflisp.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/rss.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/siliconvalley.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/startuplessons.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/submarine.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/sun.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/superangels.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/todo.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/unions.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/useful.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/vb.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/vcsqueeze.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/vw.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/want.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/web20.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/weird.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/wisdom.txt",
    "https://github.com/gkamradt/LLMTest_NeedleInAHaystack/raw/main/needlehaystack/PaulGrahamEssays/worked.txt",
]


def _pg_html_to_text(content: bytes) -> str:
    """Extract essay text from a paulgraham.com HTML page.

    Mirrors the RULER download script: find the <font> tag and pull its text.
    Tries bs4 first; falls back to stdlib html.parser.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(content.decode("unicode_escape", errors="replace"), "html.parser")
        tag = soup.find("font")
        return tag.get_text(separator=" ") if tag else ""
    except ImportError:
        pass
    # stdlib fallback
    from html.parser import HTMLParser

    class _FontExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self._depth = 0
            self._parts = []
        def handle_starttag(self, tag, attrs):
            if tag == "font":
                self._depth += 1
        def handle_endtag(self, tag):
            if tag == "font" and self._depth:
                self._depth -= 1
        def handle_data(self, data):
            if self._depth:
                self._parts.append(data)
        def get_text(self):
            return "".join(self._parts)

    ex = _FontExtractor()
    ex.feed(content.decode("unicode_escape", errors="replace"))
    return ex.get_text()


def _load_paul_graham_essays() -> str:
    """Download (and cache) the official RULER Paul Graham essay corpus.

    Replicates the RULER download_paulgraham_essay.py script exactly:
    downloads from paulgraham.com (HTML) and gkamradt/LLMTest_NeedleInAHaystack (plain text),
    then caches the result as ruler_cache/PaulGrahamEssays.json.
    """
    if _PG_ESSAYS_CACHE.exists():
        with open(_PG_ESSAYS_CACHE) as f:
            data = json.load(f)
        if isinstance(data, list):
            return "\n\n".join(d["text"] if isinstance(d, dict) else d for d in data)
        return "\n\n".join(data.values())

    print("[RULER] Building Paul Graham essay corpus from original RULER sources...", flush=True)
    _PG_ESSAYS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    ok = fail = 0
    for url in _PG_ESSAYS_SOURCE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                raw = r.read()
            if url.endswith(".html"):
                text = _pg_html_to_text(raw)
            else:
                text = raw.decode("utf-8", errors="replace")
            if text.strip():
                parts.append(text)
            ok += 1
        except Exception as e:
            fail += 1
            print(f"[RULER]   skip {url.split('/')[-1]}: {e}", flush=True)

    print(f"[RULER] Downloaded {ok} essays ({fail} failed). Caching...", flush=True)
    combined = "\n\n".join(parts)
    with open(_PG_ESSAYS_CACHE, "w") as f:
        json.dump({"text": combined}, f)
    print("[RULER] Essays cached.", flush=True)
    return combined


def init_haystack(tokenizer):
    """Tokenize the Paul Graham essay corpus once; stored in _HAYSTACK_TOKENS."""
    global _HAYSTACK_TOKENS
    if _HAYSTACK_TOKENS:
        return
    text = _load_paul_graham_essays()
    _HAYSTACK_TOKENS = tokenizer.encode(text, add_special_tokens=False)
    print(f"[RULER] Haystack ready: {len(_HAYSTACK_TOKENS):,} tokens "
          f"(Paul Graham Essays)", flush=True)


def make_filler_tokens(n_tokens: int, tokenizer) -> list[int]:
    """Sample a contiguous slice from the essay corpus."""
    if len(_HAYSTACK_TOKENS) >= n_tokens:
        start = random.randint(0, len(_HAYSTACK_TOKENS) - n_tokens)
        return list(_HAYSTACK_TOKENS[start: start + n_tokens])
    # Corpus too short (shouldn't happen): tile and trim
    tiled = (_HAYSTACK_TOKENS * (n_tokens // len(_HAYSTACK_TOKENS) + 1))[:n_tokens]
    return list(tiled)


def tokens_to_text(token_ids: list[int], tokenizer) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=True)

# ==========================================
# NEEDLE HELPERS
# ==========================================

NAMES = [
    "Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi",
    "Ivan", "Judy", "Karl", "Laura", "Mallory", "Niaj", "Oscar", "Peggy",
    "Quinn", "Romeo", "Sybil", "Trent", "Ursula", "Victor", "Walter", "Zelda",
]

def _rand_int(lo=1000, hi=9999) -> int:
    return random.randint(lo, hi)

def _insert_needle_tokens(filler_tokens: list[int], needle_tokens: list[int],
                          depth: float, tokenizer) -> list[int]:
    idx = max(0, min(len(filler_tokens), int(len(filler_tokens) * depth)))
    return filler_tokens[:idx] + needle_tokens + filler_tokens[idx:]

def _build_prompt(filler_tokens, needle_tokens, question_tokens,
                  ctx_len, depth, tokenizer) -> tuple[str, int]:
    overhead = len(needle_tokens) + len(question_tokens) + 4
    filler_budget = max(10, ctx_len - overhead)
    filler = make_filler_tokens(filler_budget, tokenizer)
    combined = _insert_needle_tokens(filler, needle_tokens, depth, tokenizer)
    all_tokens = combined + question_tokens
    return tokens_to_text(all_tokens, tokenizer), len(all_tokens)

# ==========================================
# RULER TASK GENERATORS
# ==========================================

def make_niah_single(ctx_len, tokenizer, variant):
    depth = random.uniform(0.1, 0.9)
    if variant == 1:
        val = _rand_int()
        needle_txt = f" [NEEDLE] The secret passcode is {val}. [/NEEDLE] "
        question_txt = "\n\nQuestion: What is the secret passcode? Answer with the number only.\nAnswer:"
        answer = str(val)
    elif variant == 2:
        key = "".join(random.choices(string.ascii_uppercase, k=5))
        val = _rand_int()
        needle_txt = f" [NEEDLE] {key} equals {val}. [/NEEDLE] "
        question_txt = f"\n\nQuestion: What does {key} equal? Answer with the number only.\nAnswer:"
        answer = str(val)
    else:
        uid = str(uuid.uuid4())
        needle_txt = f" [NEEDLE] The magic token is {uid}. [/NEEDLE] "
        question_txt = "\n\nQuestion: What is the magic token? Answer exactly.\nAnswer:"
        answer = uid

    n_tok = tokenizer.encode(needle_txt, add_special_tokens=False)
    q_tok = tokenizer.encode(question_txt, add_special_tokens=False)
    prompt, _ = _build_prompt([], n_tok, q_tok, ctx_len, depth, tokenizer)
    return {"prompt": prompt, "answer": answer}

def make_niah_multikey(ctx_len, tokenizer, num_distractors):
    depth = random.uniform(0.1, 0.9)
    target_key = "KEY_" + "".join(random.choices(string.ascii_uppercase, k=4))
    target_val = _rand_int()
    needles = [f"[NEEDLE] {target_key} = {target_val} [/NEEDLE]"]
    for _ in range(num_distractors):
        dk = "KEY_" + "".join(random.choices(string.ascii_uppercase, k=4))
        needles.append(f"[NEEDLE] {dk} = {_rand_int()} [/NEEDLE]")
    random.shuffle(needles)
    needle_txt = " ".join(needles) + " "
    question_txt = f"\n\nQuestion: What is the value of {target_key}? Answer with the number only.\nAnswer:"

    n_tok = tokenizer.encode(needle_txt, add_special_tokens=False)
    q_tok = tokenizer.encode(question_txt, add_special_tokens=False)
    prompt, _ = _build_prompt([], n_tok, q_tok, ctx_len, depth, tokenizer)
    return {"prompt": prompt, "answer": str(target_val)}

def make_niah_multivalue(ctx_len, tokenizer, num_values=4):
    key = "MKEY_" + "".join(random.choices(string.ascii_uppercase, k=3))
    values = [str(_rand_int(10, 99)) for _ in range(num_values)]
    depths = sorted([random.uniform(0.05, 0.95) for _ in range(num_values)])

    q_tok = tokenizer.encode(
        f"\n\nQuestion: List all values of {key} in order of appearance, separated by commas.\nAnswer:",
        add_special_tokens=False)
    overhead = len(q_tok) + num_values * 15 + 4
    filler = make_filler_tokens(max(10, ctx_len - overhead), tokenizer)

    for i, (v, d) in enumerate(zip(values, depths)):
        n_tok = tokenizer.encode(f" [NEEDLE] {key} value: {v} [/NEEDLE] ", add_special_tokens=False)
        idx = int(len(filler) * d)
        filler = filler[:idx] + n_tok + filler[idx:]

    return {"prompt": tokens_to_text(filler + q_tok, tokenizer), "answer": ", ".join(values)}

def make_niah_multiquery(ctx_len, tokenizer, num_pairs=4):
    pairs = [(f"MQKEY_{i}", _rand_int()) for i in range(num_pairs)]
    depths = sorted([random.uniform(0.1, 0.9) for _ in range(num_pairs)])

    questions = "\n".join([f"{i+1}. What is the value of {k}?" for i, (k, v) in enumerate(pairs)])
    answers = ", ".join([str(v) for k, v in pairs])
    
    q_tok = tokenizer.encode(
        f"\n\nTask: Answer the following questions based on the [NEEDLE] tags above:\n{questions}\n"
        f"Answer with numbers only, separated by commas:", 
        add_special_tokens=False
    )
    
    filler = make_filler_tokens(max(10, ctx_len - (num_pairs*25) - len(q_tok) - 50), tokenizer)
    for (k, v), d in zip(pairs, depths):
        n_tok = tokenizer.encode(f" [NEEDLE] {k} is {v} [/NEEDLE] ", add_special_tokens=False)
        idx = int(len(filler) * d)
        filler = filler[:idx] + n_tok + filler[idx:]

    return {"prompt": tokenizer.decode(filler) + tokenizer.decode(q_tok), "answer": answers}

def make_vt(ctx_len, tokenizer, chain_len=4):
    names = random.sample(NAMES, chain_len + 1)
    chain_txts = [f" [NEEDLE] {names[i]} transfers to {names[i+1]} [/NEEDLE] " for i in range(chain_len)]
    depths = sorted([random.uniform(0.1, 0.9) for _ in range(chain_len)])

    q_tok = tokenizer.encode(
        f"\n\nQuestion: Following the transfer chain starting from {names[0]}, "
        f"who is the final person in the sequence? Answer with the name only.\nAnswer:", 
        add_special_tokens=False
    )
    filler = make_filler_tokens(max(10, ctx_len - (chain_len*25) - len(q_tok) - 50), tokenizer)
    for txt, d in zip(chain_txts, depths):
        n_tok = tokenizer.encode(txt, add_special_tokens=False)
        idx = int(len(filler) * d)
        filler = filler[:idx] + n_tok + filler[idx:]

    return {"prompt": tokenizer.decode(filler) + tokenizer.decode(q_tok), "answer": names[-1]}

_FREQ_WORD_POOL = ["alpha", "bravo", "delta", "echo", "foxtrot",
                   "gamma", "hotel", "india", "juliet", "kilo"]

def make_frequency_task(ctx_len, tokenizer, mode="cwe"):
    target = random.choice(_FREQ_WORD_POOL[:5])
    decoys = random.sample([w for w in _FREQ_WORD_POOL if w != target], 4)

    # target appears 10x, each decoy appears 2x — scattered throughout the full context
    target_count = 10
    decoy_count = 2
    words = [target] * target_count + [d for d in decoys for _ in range(decoy_count)]
    random.shuffle(words)

    word_set = ", ".join(sorted(set(_FREQ_WORD_POOL)))
    q_tok = tokenizer.encode(
        f"\n\nQuestion: Among the words [{word_set}], which one appears most frequently "
        f"in the document above? Answer with the single word only.\nAnswer:",
        add_special_tokens=False
    )

    # Build filler and scatter all word tokens throughout it
    filler = make_filler_tokens(max(100, ctx_len - len(q_tok) - len(words) * 4 - 50), tokenizer)
    positions = sorted(random.sample(range(len(filler)), min(len(words), len(filler))))
    for pos, word in zip(reversed(positions), reversed(words)):
        word_tok = tokenizer.encode(f" {word} ", add_special_tokens=False)
        filler = filler[:pos] + word_tok + filler[pos:]

    return {
        "prompt": tokens_to_text(filler, tokenizer) + tokens_to_text(q_tok, tokenizer),
        "answer": target,
    }

# ==========================================
# QA POOLS  (real HotpotQA + SQuAD — official RULER QA sources)
# ==========================================

_HOTPOT_SAMPLES: list[tuple[str, str, str]] = []
_SQUAD_SAMPLES:  list[tuple[str, str, str]] = []
_QA_POOL_SIZE = 500  # samples to cache from each dataset


def init_qa_pools():
    """Load QA samples from HotpotQA and SQuAD (done once in run_ruler)."""
    global _HOTPOT_SAMPLES, _SQUAD_SAMPLES

    if not _HOTPOT_SAMPLES:
        print("[RULER] Loading HotpotQA...", flush=True)
        ds = load_dataset("hotpot_qa", "distractor", split="validation",
                          trust_remote_code=True)
        for item in ds.shuffle(seed=42).select(range(min(_QA_POOL_SIZE, len(ds)))):
            # Concatenate ALL sentences from ALL supporting documents
            all_sents = [s for sent_list in item["context"]["sentences"] for s in sent_list]
            passage = " ".join(all_sents)
            if passage and item["question"] and item["answer"]:
                _HOTPOT_SAMPLES.append((passage, item["question"], item["answer"]))
        print(f"[RULER] HotpotQA pool: {len(_HOTPOT_SAMPLES)} samples", flush=True)

    if not _SQUAD_SAMPLES:
        print("[RULER] Loading SQuAD...", flush=True)
        ds = load_dataset("squad", split="validation",
                          trust_remote_code=True)
        for item in ds.shuffle(seed=42).select(range(min(_QA_POOL_SIZE, len(ds)))):
            answers = item["answers"]["text"]
            if item["context"] and item["question"] and answers:
                _SQUAD_SAMPLES.append((item["context"], item["question"], answers[0]))
        print(f"[RULER] SQuAD pool: {len(_SQUAD_SAMPLES)} samples", flush=True)


def make_qa(ctx_len, tokenizer, dataset="hotpot"):
    pool = _HOTPOT_SAMPLES if dataset == "hotpot" else _SQUAD_SAMPLES
    context, question, answer = random.choice(pool)
    depth = random.uniform(0.1, 0.9)

    passage_txt = f" [DOCUMENT] {context} [/DOCUMENT] "
    question_txt = f"\n\nQuestion: {question}\nAnswer:"
    p_tok = tokenizer.encode(passage_txt, add_special_tokens=False)
    q_tok = tokenizer.encode(question_txt, add_special_tokens=False)
    prompt, _ = _build_prompt([], p_tok, q_tok, ctx_len, depth, tokenizer)
    return {"prompt": prompt, "answer": answer}

def make_sample(task_name: str, ctx_len: int, tokenizer) -> dict:
    dispatch = {
        "niah_single_1":   lambda: make_niah_single(ctx_len, tokenizer, 1),
        "niah_single_2":   lambda: make_niah_single(ctx_len, tokenizer, 2),
        "niah_single_3":   lambda: make_niah_single(ctx_len, tokenizer, 3),
        "niah_multikey_1": lambda: make_niah_multikey(ctx_len, tokenizer, 1),
        "niah_multikey_2": lambda: make_niah_multikey(ctx_len, tokenizer, 2),
        "niah_multikey_3": lambda: make_niah_multikey(ctx_len, tokenizer, 3),
        "niah_multivalue": lambda: make_niah_multivalue(ctx_len, tokenizer),
        "niah_multiquery": lambda: make_niah_multiquery(ctx_len, tokenizer),
        "vt":              lambda: make_vt(ctx_len, tokenizer),
        "cwe":             lambda: make_frequency_task(ctx_len, tokenizer, "cwe"),
        "fwe":             lambda: make_frequency_task(ctx_len, tokenizer, "fwe"),
        "qa_hotpot":       lambda: make_qa(ctx_len, tokenizer, "hotpot"),
        "qa_squad":        lambda: make_qa(ctx_len, tokenizer, "squad"),
    }
    return dispatch[task_name]()

# ==========================================
# METRICS
# ==========================================

def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

def exact_match_robust(pred: str, gold: str) -> float:
    pred_n, gold_n = _norm(pred), _norm(gold)
    if pred_n == gold_n: return 1.0
    pattern = r'\b' + re.escape(gold_n) + r'\b'
    return 1.0 if re.search(pattern, pred_n) else 0.0

def token_f1(pred: str, gold: str) -> float:
    p_toks, g_toks = _norm(pred).split(), _norm(gold).split()
    if not p_toks or not g_toks: return 0.0
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0: return 0.0
    prec, rec = common / len(p_toks), common / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def existence_score(pred: str, gold: str) -> float:
    pred_n = _norm(pred)
    parts = [p.strip() for p in _norm(gold).split(",") if p.strip()]
    if not parts: return 0.0
    matches = sum(1 for p in parts if p in pred_n)
    return 1.0 if matches == len(parts) else 0.0

def score(pred: str, gold: str, task: str) -> float:
    if "multivalue" in task or "multiquery" in task: return existence_score(pred, gold)
    if "qa_" in task: return max(token_f1(pred, gold), exact_match_robust(pred, gold))
    return exact_match_robust(pred, gold)

# ==========================================
# GENERATION
# ==========================================

@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, device: str, max_new_tokens: int = 48) -> str:
    messages = [
        {
            "role": "system", 
            "content": "You are a precise information retrieval assistant. "
                       "You will be given a long context containing background noise and specific tagged information. "
                       "Ignore the random background words. Focus only on information inside [NEEDLE], [DOCUMENT], or [PASSAGE] tags."
        },
        {"role": "user", "content": prompt}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=MAX_CTX).to(device)
    
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        temperature=None, top_p=None, pad_token_id=tokenizer.eos_token_id,
    )
    
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()

# ==========================================
# MAIN EVALUATION
# ==========================================

def run_ruler(args, parent_dir):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    setattr(globVR, 'delta_pf_key_on', 0)
    setattr(globVR, 'delta_mlp', 'Regular')
    setattr(globVR, 'delta_decode', False)

    print(f"[RULER] Loading tokenizer: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load official RULER data sources once before evaluating any run
    init_haystack(tokenizer)
    init_qa_pools()

    print(f"[RULER] Loading model: {MODEL_ID}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    model.eval()

    exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
    glob_settings = exp_def["injector"](args)
    print(f"[RULER] Experiment: {args.experiment_type} | Settings: {glob_settings}", flush=True)
    for key, value in glob_settings.items():
        setattr(globVR, key, value)

    if getattr(globVR, 'delta_decode', False):
        print("[RULER] Patching generate() for HybridCompressedCache...", flush=True)
        _orig_generate = model.generate

        def _patched_generate(inputs=None, *a, **kw):
            src = inputs if inputs is not None else kw.get('input_ids')
            bsz = src.shape[0] if src is not None else 1
            kw['past_key_values'] = HybridCompressedCache(
                config=model.config, batch_size=bsz, dtype=model.dtype,
                exact_window_size=getattr(globVR, 'window_size', 0),
            )
            kw['use_cache'] = True
            return _orig_generate(inputs, *a, **kw)

        model.generate = _patched_generate

    # Group output files under the main run folder
    out_dir = os.path.join(parent_dir, args.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    ctx_subset = args.ctx_lens
    tasks      = args.tasks if args.tasks else RULER_TASKS
    all_scores: dict[int, dict[str, float]] = {c: {} for c in ctx_subset}

    for ctx_len in ctx_subset:
        lbl = CTX_LABELS[CTX_LENGTHS.index(ctx_len)] if ctx_len in CTX_LENGTHS else str(ctx_len)
        print(f"\n{'='*60}", flush=True)
        print(f"[RULER] Context length: {lbl} ({ctx_len} tokens)", flush=True)
        print(f"{'='*60}", flush=True)

        for task_name in tasks:
            print(f"  Task: {task_name}", flush=True)
            task_scores = []
            task_start  = time.time()

            for i in range(args.num_samples):
                sample = make_sample(task_name, ctx_len, tokenizer)
                pred   = generate_answer(model, tokenizer, sample["prompt"], device)
                s      = score(pred, sample["answer"], task_name)
                task_scores.append(s)

            acc = float(np.mean(task_scores)) * 100.0
            all_scores[ctx_len][task_name] = acc
            elapsed = time.time() - task_start
            print(f"  -> {task_name}: {acc:.2f}% ({args.num_samples} samples, {elapsed:.1f}s)", flush=True)

            save_json(
                os.path.join(out_dir, f"{task_name}_{lbl}.json"),
                {
                    "task": task_name, "ctx_len": ctx_len, "accuracy_pct": acc,
                    "num_samples": args.num_samples, "experiment_type": args.experiment_type,
                    "parameters": glob_settings, "timestamp": datetime.now().isoformat(),
                }
            )

            gc.collect()
            torch.cuda.empty_cache()

    ctx_avg: dict[int, float] = {}
    for ctx_len in ctx_subset:
        vals = list(all_scores[ctx_len].values())
        ctx_avg[ctx_len] = float(np.mean(vals)) if vals else float("nan")

    summary = {
        "row_name": args.row_name, "experiment_type": args.experiment_type,
        "parameters": glob_settings, "ctx_lengths": ctx_subset,
        "ctx_labels": [CTX_LABELS[CTX_LENGTHS.index(c)] if c in CTX_LENGTHS else str(c) for c in ctx_subset],
        "ctx_averages_pct": {str(c): ctx_avg[c] for c in ctx_subset},
        "overall_avg_pct": float(np.mean(list(ctx_avg.values()))),
        "task_scores_pct": {str(c): all_scores[c] for c in ctx_subset},
        "num_samples": args.num_samples, "timestamp": datetime.now().isoformat(),
    }
    save_json(os.path.join(out_dir, "_summary.json"), summary)

    return ctx_avg

# ==========================================
# DIRECTORY MANAGER
# ==========================================
def get_next_run_dir(base_path="."):
    existing_dirs = glob.glob(os.path.join(base_path, "ruler_results_*"))
    nums = []
    for d in existing_dirs:
        try:
            num = int(os.path.basename(d).split('_')[-1])
            nums.append(num)
        except ValueError:
            pass
    next_num = max(nums) + 1 if nums else 1
    return os.path.join(base_path, f"ruler_results_{next_num}")

def parse_ctx_lens(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",")]

# ==========================================
# ENTRY POINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RULER long-context benchmark for Llama-3.1-8B-Instruct (128k)")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--ctx_lens", type=parse_ctx_lens, default=CTX_LENGTHS)
    parser.add_argument("--tasks", type=lambda s: s.split(","), default=None)
    parser.add_argument('--scale', default=0.05, type=float)
    parser.add_argument('--thresh', default=0.6, type=float)
    parser.add_argument('--delta', default=1.0, type=float)
    parser.add_argument('--row_sim', default="cos", type=str)
    parser.add_argument('--chunk_size', default=512, type=int)
    parser.add_argument('--dense_window_size', default=0, type=int)
    parser.add_argument('--window_size', default=50, type=int)
    parser.add_argument('--row_thresh', default=0.5, type=float)

    args = parser.parse_args()

    # Define Output Architecture
    master_out_dir = get_next_run_dir()
    os.makedirs(master_out_dir, exist_ok=True)
    print(f"\n[IO] Initialized session folder: {master_out_dir}\n")

    runs = [
        {"row_name": "Full", "experiment_type": "baseline"},
        {"row_name": "Row Delta (t=15)", "experiment_type": "row_delta", "row_thresh": 15},
        {"row_name": "Row Delta (t=17)", "experiment_type": "row_delta", "row_thresh": 17},
    ]

    final_table_data = {}

    for run in runs:
        print(f"\n\n{'='*60}")
        print(f"*** STARTING EXPERIMENT: {run['row_name']} ***")
        print(f"{'='*60}\n")
        
        args.experiment_type = run["experiment_type"]
        args.row_name = run["row_name"]
        
        if "row_thresh" in run:
            args.row_thresh = run["row_thresh"]
            
        # Creates cleaner sub-folder names like "Row_Delta_t15" inside "ruler_results_1"
        safe_name = run["row_name"].replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
        args.exp_name = safe_name
        
        results = run_ruler(args, parent_dir=master_out_dir)
        final_table_data[run["row_name"]] = results

    # Hand off the parent directory to the table generator to save CSV and PNG
    print_multi_table(final_table_data, master_out_dir)