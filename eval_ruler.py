"""
Standalone RULER benchmark evaluator for Llama-3.1-8B-Instruct (128k context).

Runs all 13 RULER tasks at [4k, 8k, 16k, 32k, 64k, 128k] context lengths,
averages scores across tasks per length, and prints a summary table:

  Input Len |    4k |    8k |   16k |   32k |   64k |  128k |   Avg.
  baseline  | 96.74 | 94.03 | 92.02 | 84.17 | 81.32 | 76.89 | 87.52

Usage:
    python eval_ruler.py --experiment_type baseline --exp_name my_run
    python eval_ruler.py --experiment_type row_delta --num_samples 50
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
from collections import Counter
from datetime import datetime

import numpy as np
import torch
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
MAX_CTX  = 131072  # 128k — native context for Llama 3.1

CTX_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]
CTX_LABELS  = ["4k",  "8k",  "16k",  "32k",  "64k", "128k"]

RULER_TASKS = [
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_hotpot",
    "qa_squad",
]

EXPERIMENT_DEFINITIONS = {
    "baseline": {
        "injector": lambda args: {
            "delta_pf_key_on": 0,
            "delta_mlp": "Regular",
            "flash": True,
        }
    },
    "baseline_decoding": {
        "injector": lambda args: {
            "delta_decode": False,
            "delta_pf_key_on": False,
            "flash": True,
        }
    },
    "delta_decoding": {
        "injector": lambda args: {
            "delta_decode": True,
            "window_size": args.window_size,
            "row_delta_threshold": args.row_thresh,
            "row_similarity_metric": args.row_sim,
            "delta_pf_key_on": 1,
            "flash": True,
            "delta_type": "row",
            "divide_to": 0,
            "chunk_size": 512,
        }
    },
    "row_delta": {
        "injector": lambda args: {
            "delta_pf_key_on": 1,
            "delta_type": "row",
            "scale": args.scale,
            "delta_mlp": "Regular",
            "row_delta_threshold": args.delta,
            "row_similarity_metric": args.row_sim,
            "chunk_size": args.chunk_size,
            "divide_to": 0,
            "flash": True,
            "delta_decode": False,
            "dense_window_size": args.dense_window_size,
        }
    },
}

# ==========================================
# JSON ENCODER
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

# ==========================================
# FILLER GENERATION (token-precise)
# ==========================================

# A single sentence that tokenises predictably (~12 tokens)
_FILLER_SENTENCE = (
    "The sky is blue, the grass is green, and the sun rises in the east every morning. "
)

def make_filler_tokens(n_tokens: int, tokenizer) -> list[int]:
    """Return exactly n_tokens filler tokens (no special tokens)."""
    base = tokenizer.encode(_FILLER_SENTENCE, add_special_tokens=False)
    if not base:
        base = [tokenizer.eos_token_id]
    repeats = (n_tokens // len(base)) + 2
    return (base * repeats)[:n_tokens]

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
    """Insert needle_tokens at `depth` fraction through filler_tokens."""
    idx = max(0, min(len(filler_tokens), int(len(filler_tokens) * depth)))
    return filler_tokens[:idx] + needle_tokens + filler_tokens[idx:]

def _build_prompt(filler_tokens, needle_tokens, question_tokens,
                  ctx_len, depth, tokenizer) -> tuple[str, int]:
    """
    Build a prompt of exactly ctx_len tokens by sizing filler appropriately.
    Returns (prompt_text, actual_token_count).
    """
    overhead = len(needle_tokens) + len(question_tokens) + 4  # BOS + separators
    filler_budget = max(10, ctx_len - overhead)
    filler = make_filler_tokens(filler_budget, tokenizer)
    combined = _insert_needle_tokens(filler, needle_tokens, depth, tokenizer)
    all_tokens = combined + question_tokens
    # tokenize to get true count (decode → re-encode can drift slightly)
    prompt_text = tokens_to_text(all_tokens, tokenizer)
    return prompt_text, len(all_tokens)

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
    filler_budget = max(10, ctx_len - overhead)
    filler = make_filler_tokens(filler_budget, tokenizer)

    for i, (v, d) in enumerate(zip(values, depths)):
        n_tok = tokenizer.encode(f" [NEEDLE] {key} value: {v} [/NEEDLE] ", add_special_tokens=False)
        idx = int(len(filler) * d)
        filler = filler[:idx] + n_tok + filler[idx:]

    prompt = tokens_to_text(filler + q_tok, tokenizer)
    return {"prompt": prompt, "answer": ", ".join(values)}

def make_niah_multiquery(ctx_len, tokenizer, num_pairs=4):
    pairs = [(f"MQKEY_{i}", _rand_int()) for i in range(num_pairs)]
    depths = sorted([random.uniform(0.05, 0.95) for _ in range(num_pairs)])

    questions = " ".join([f"What is {k}?" for k, _ in pairs])
    answers   = ", ".join([f"{k}={v}" for k, v in pairs])
    q_tok = tokenizer.encode(f"\n\nQuestions: {questions}\nAnswers:", add_special_tokens=False)

    overhead = len(q_tok) + num_pairs * 20 + 4
    filler = make_filler_tokens(max(10, ctx_len - overhead), tokenizer)

    for (k, v), d in zip(pairs, depths):
        n_tok = tokenizer.encode(f" [NEEDLE] {k} = {v} [/NEEDLE] ", add_special_tokens=False)
        idx = int(len(filler) * d)
        filler = filler[:idx] + n_tok + filler[idx:]

    prompt = tokens_to_text(filler + q_tok, tokenizer)
    return {"prompt": prompt, "answer": answers}

def make_vt(ctx_len, tokenizer, chain_len=4):
    names = random.sample(NAMES, chain_len + 1)
    chain_txts = [f" [NEEDLE] {names[i]} = {names[i+1]}. [/NEEDLE] " for i in range(chain_len)]
    depths = sorted([random.uniform(0.05, 0.95) for _ in range(chain_len)])

    q_tok = tokenizer.encode(
        f"\n\nFollow assignments starting from {names[0]}. What is the final name?\nAnswer:",
        add_special_tokens=False)
    filler = make_filler_tokens(max(10, ctx_len - len(q_tok) - chain_len * 20 - 4), tokenizer)

    for txt, d in zip(chain_txts, depths):
        n_tok = tokenizer.encode(txt, add_special_tokens=False)
        idx = int(len(filler) * d)
        filler = filler[:idx] + n_tok + filler[idx:]

    prompt = tokens_to_text(filler + q_tok, tokenizer)
    return {"prompt": prompt, "answer": names[-1]}

def make_cwe(ctx_len, tokenizer):
    # One word repeated many times, others appear once
    target = random.choice(["alpha", "bravo", "delta", "echo", "foxtrot"])
    decoys = random.sample(["gamma", "hotel", "india", "juliet", "kilo", "lima"], 5)
    word_list = [target] * random.randint(5, 8) + decoys
    random.shuffle(word_list)
    passage = " ".join(word_list)

    p_tok = tokenizer.encode(f" [PASSAGE] {passage} [/PASSAGE] ", add_special_tokens=False)
    q_tok = tokenizer.encode(
        "\n\nWhich word in the passage above appears most frequently? Answer with that word only.\nAnswer:",
        add_special_tokens=False)
    filler = make_filler_tokens(max(10, ctx_len - len(p_tok) - len(q_tok) - 4), tokenizer)
    prompt = tokens_to_text(filler + p_tok + q_tok, tokenizer)
    return {"prompt": prompt, "answer": target}

def make_fwe(ctx_len, tokenizer):
    target = random.choice(["zeta", "theta", "sigma", "omega", "lambda"])
    count = random.randint(6, 10)
    noise = [random.choice(["one", "two", "three", "four", "five", "six"]) for _ in range(count * 2)]
    words = [target] * count + noise
    random.shuffle(words)
    passage = " ".join(words)

    p_tok = tokenizer.encode(f" [PASSAGE] {passage} [/PASSAGE] ", add_special_tokens=False)
    q_tok = tokenizer.encode(
        "\n\nWhat single word appears most frequently in the passage? Answer with that word only.\nAnswer:",
        add_special_tokens=False)
    filler = make_filler_tokens(max(10, ctx_len - len(p_tok) - len(q_tok) - 4), tokenizer)
    prompt = tokens_to_text(filler + p_tok + q_tok, tokenizer)
    return {"prompt": prompt, "answer": target}

_HOTPOT_SAMPLES = [
    ("Marie Curie was born in Warsaw in 1867 and later moved to Paris.", "In what city was Marie Curie born?", "Warsaw"),
    ("The Eiffel Tower was completed in 1889 and stands 330 metres tall.", "How tall is the Eiffel Tower?", "330 metres"),
    ("Einstein published special relativity in 1905 and general relativity in 1915.", "When did Einstein publish general relativity?", "1915"),
    ("Python was created by Guido van Rossum and first released in 1991.", "Who created Python?", "Guido van Rossum"),
    ("The Amazon River is the largest river by discharge and flows through Brazil.", "Through which country does the Amazon River flow?", "Brazil"),
    ("Shakespeare was born in Stratford-upon-Avon in 1564.", "Where was Shakespeare born?", "Stratford-upon-Avon"),
    ("The Great Wall of China was built over many centuries starting around 7th century BC.", "When did construction of the Great Wall begin?", "7th century BC"),
    ("Penicillin was discovered by Alexander Fleming in 1928.", "Who discovered penicillin?", "Alexander Fleming"),
]

_SQUAD_SAMPLES = [
    ("The Pacific Ocean is the largest and deepest of Earth's oceanic divisions.", "Which ocean is the largest?", "Pacific Ocean"),
    ("The speed of light in vacuum is approximately 299,792 kilometres per second.", "What is the speed of light?", "299,792 kilometres per second"),
    ("Mount Everest is Earth's highest mountain above sea level at 8,849 metres.", "What is the height of Mount Everest?", "8,849 metres"),
    ("The human body has 206 bones in adulthood.", "How many bones does the human body have?", "206"),
    ("Leonardo da Vinci painted the Mona Lisa between 1503 and 1519.", "Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("The chemical formula for water is H2O.", "What is the chemical formula for water?", "H2O"),
]

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

# ==========================================
# TASK DISPATCHER
# ==========================================

def make_sample(task_name: str, ctx_len: int, tokenizer) -> dict:
    dispatch = {
        "niah_single_1":  lambda: make_niah_single(ctx_len, tokenizer, 1),
        "niah_single_2":  lambda: make_niah_single(ctx_len, tokenizer, 2),
        "niah_single_3":  lambda: make_niah_single(ctx_len, tokenizer, 3),
        "niah_multikey_1": lambda: make_niah_multikey(ctx_len, tokenizer, 1),
        "niah_multikey_2": lambda: make_niah_multikey(ctx_len, tokenizer, 2),
        "niah_multikey_3": lambda: make_niah_multikey(ctx_len, tokenizer, 3),
        "niah_multivalue": lambda: make_niah_multivalue(ctx_len, tokenizer),
        "niah_multiquery": lambda: make_niah_multiquery(ctx_len, tokenizer),
        "vt":              lambda: make_vt(ctx_len, tokenizer),
        "cwe":             lambda: make_cwe(ctx_len, tokenizer),
        "fwe":             lambda: make_fwe(ctx_len, tokenizer),
        "qa_hotpot":       lambda: make_qa(ctx_len, tokenizer, "hotpot"),
        "qa_squad":        lambda: make_qa(ctx_len, tokenizer, "squad"),
    }
    return dispatch[task_name]()

# ==========================================
# METRICS
# ==========================================

def _norm(s: str) -> str:
    return " ".join(s.lower().strip().split())

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _norm(pred) == _norm(gold) else 0.0

def token_f1(pred: str, gold: str) -> float:
    p_toks = _norm(pred).split()
    g_toks = _norm(gold).split()
    if not p_toks or not g_toks:
        return 0.0
    common = sum((Counter(p_toks) & Counter(g_toks)).values())
    if common == 0:
        return 0.0
    prec = common / len(p_toks)
    rec  = common / len(g_toks)
    return 2 * prec * rec / (prec + rec)

def existence_score(pred: str, gold: str) -> float:
    """All comma-separated gold items must appear somewhere in pred."""
    pred_n = _norm(pred)
    for part in _norm(gold).split(","):
        part = part.strip()
        if part and part not in pred_n:
            return 0.0
    return 1.0

def score(pred: str, gold: str, task: str) -> float:
    if "multivalue" in task or "multiquery" in task:
        return existence_score(pred, gold)
    if task in ("qa_hotpot", "qa_squad"):
        return token_f1(pred, gold)
    return exact_match(pred, gold)

# ==========================================
# GENERATION
# ==========================================

@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, device: str,
                    max_new_tokens: int = 32) -> str:
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CTX,
    ).to(device)
    input_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()

# ==========================================
# RESULT I/O
# ==========================================

def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4, cls=NpEncoder)
    print(f"[IO] Saved {path}", flush=True)

# ==========================================
# TABLE PRINTER
# ==========================================

def print_table(row_name: str, scores_by_ctx: dict[int, float]):
    """
    scores_by_ctx: {ctx_len_int -> average_accuracy (0–100)}
    """
    vals  = [scores_by_ctx.get(c, float("nan")) for c in CTX_LENGTHS]
    valid = [v for v in vals if not np.isnan(v)]
    avg   = float(np.mean(valid)) if valid else float("nan")

    header = f"{'Input Len':<14}|" + "".join(f"  {lbl:>5} |" for lbl in CTX_LABELS) + f"   Avg."
    sep    = "-" * len(header)
    row    = f"{row_name:<14}|" + "".join(
        f"  {v:5.2f} |" if not np.isnan(v) else f"  {'n/a':>5} |"
        for v in vals
    ) + f"  {avg:5.2f}"

    print("\n" + "=" * len(header), flush=True)
    print("RULER Benchmark Results", flush=True)
    print("=" * len(header), flush=True)
    print(header, flush=True)
    print(sep, flush=True)
    print(row, flush=True)
    print("=" * len(header) + "\n", flush=True)

# ==========================================
# MAIN EVALUATION
# ==========================================

def run_ruler(args):
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Safe globVR defaults before model load
    setattr(globVR, 'delta_pf_key_on', 0)
    setattr(globVR, 'delta_mlp', 'Regular')
    setattr(globVR, 'delta_decode', False)

    print(f"[RULER] Loading tokenizer: {MODEL_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[RULER] Loading model: {MODEL_ID}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    # Inject experiment globals
    exp_def = EXPERIMENT_DEFINITIONS[args.experiment_type]
    glob_settings = exp_def["injector"](args)
    print(f"[RULER] Experiment: {args.experiment_type} | Settings: {glob_settings}", flush=True)
    for key, value in glob_settings.items():
        setattr(globVR, key, value)

    # HybridCompressedCache patch for delta_decode
    if getattr(globVR, 'delta_decode', False):
        print("[RULER] Patching generate() for HybridCompressedCache...", flush=True)
        _orig_generate = model.generate

        def _patched_generate(inputs=None, *a, **kw):
            src = inputs if inputs is not None else kw.get('input_ids')
            bsz = src.shape[0] if src is not None else 1
            kw['past_key_values'] = HybridCompressedCache(
                config=model.config,
                batch_size=bsz,
                dtype=model.dtype,
                exact_window_size=getattr(globVR, 'window_size', 50),
            )
            kw['use_cache'] = True
            return _orig_generate(inputs, *a, **kw)

        model.generate = _patched_generate

    # Output directory
    out_dir = os.path.join("snellius_experiments", "llama", args.exp_name, "ruler")
    os.makedirs(out_dir, exist_ok=True)

    ctx_subset = args.ctx_lens  # list of ints
    tasks      = args.tasks if args.tasks else RULER_TASKS

    # scores[ctx_len][task] = accuracy (0–100)
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

            acc = float(np.mean(task_scores)) * 100.0  # convert to %
            all_scores[ctx_len][task_name] = acc
            elapsed = time.time() - task_start
            print(f"  -> {task_name}: {acc:.2f}% ({args.num_samples} samples, {elapsed:.1f}s)",
                  flush=True)

            # Save per-task result
            save_json(
                os.path.join(out_dir, f"{task_name}_{lbl}.json"),
                {
                    "task": task_name,
                    "ctx_len": ctx_len,
                    "accuracy_pct": acc,
                    "num_samples": args.num_samples,
                    "experiment_type": args.experiment_type,
                    "parameters": glob_settings,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            gc.collect()
            torch.cuda.empty_cache()

    # Per-context-length averages (across tasks)
    ctx_avg: dict[int, float] = {}
    for ctx_len in ctx_subset:
        vals = list(all_scores[ctx_len].values())
        ctx_avg[ctx_len] = float(np.mean(vals)) if vals else float("nan")

    # Save summary
    row_name = args.row_name or args.experiment_type
    summary = {
        "row_name": row_name,
        "experiment_type": args.experiment_type,
        "parameters": glob_settings,
        "ctx_lengths": ctx_subset,
        "ctx_labels": [CTX_LABELS[CTX_LENGTHS.index(c)] if c in CTX_LENGTHS else str(c) for c in ctx_subset],
        "ctx_averages_pct": {str(c): ctx_avg[c] for c in ctx_subset},
        "overall_avg_pct": float(np.mean(list(ctx_avg.values()))),
        "task_scores_pct": {str(c): all_scores[c] for c in ctx_subset},
        "num_samples": args.num_samples,
        "timestamp": datetime.now().isoformat(),
    }
    save_json(os.path.join(out_dir, "_summary.json"), summary)

    # Print table row
    print_table(row_name, ctx_avg)

# ==========================================
# ENTRY POINT
# ==========================================

def parse_ctx_lens(s: str) -> list[int]:
    """Parse comma-separated context lengths, e.g. '4096,8192,131072'."""
    return [int(x.strip()) for x in s.split(",")]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RULER long-context benchmark for Llama-3.1-8B-Instruct (128k)"
    )

    # We can remove the --experiment_type argument from the parser 
    # since we will define the runs manually below.
    
    parser.add_argument(
        "--num_samples", type=int, default=50,
        help="Synthetic samples per (task, context_length) pair",
    )
    parser.add_argument(
        "--ctx_lens", type=parse_ctx_lens,
        default=CTX_LENGTHS,
        help="Comma-separated context lengths to evaluate "
             "(default: 4096,8192,16384,32768,65536,131072)",
    )
    parser.add_argument(
        "--tasks", type=lambda s: s.split(","), default=None,
        help="Comma-separated subset of RULER tasks (default: all 13)",
    )

    # Shared experiment args (mirrors eval_all.py)
    parser.add_argument('--scale',            default=0.05,  type=float)
    parser.add_argument('--thresh',           default=0.6,   type=float)
    parser.add_argument('--delta',            default=1.0,   type=float)
    parser.add_argument('--row_sim',          default="cos", type=str)
    parser.add_argument('--chunk_size',       default=512,   type=int)
    parser.add_argument('--dense_window_size',default=128,   type=int)
    parser.add_argument('--window_size',      default=50,    type=int)
    # Default row_thresh, but we will override it per run below
    parser.add_argument('--row_thresh',       default=0.5,   type=float)

    args = parser.parse_args()

    # ---------------------------------------------------------
    # DEFINE YOUR RUNS HERE
    # ---------------------------------------------------------
    # Each dictionary represents one row in your final table.
    runs = [
        {
            "row_name": "Baseline", 
            "experiment_type": "baseline"
        },
        {
            "row_name": "Row Delta (t=0.5)", 
            "experiment_type": "row_delta", 
            "row_thresh": 0.5
        },
        {
            "row_name": "Row Delta (t=0.7)", 
            "experiment_type": "row_delta", 
            "row_thresh": 0.7
        },
        {
            "row_name": "Row Delta (t=0.9)", 
            "experiment_type": "row_delta", 
            "row_thresh": 0.9
        }
    ]

    final_table_data = {}

    for run in runs:
        print(f"\n\n{'='*60}")
        print(f"*** STARTING EXPERIMENT: {run['row_name']} ***")
        print(f"{'='*60}\n")
        
        # Dynamically override the arguments for this specific run
        args.experiment_type = run["experiment_type"]
        args.row_name = run["row_name"]
        
        # If the run config specifies a threshold, overwrite the default arg
        if "row_thresh" in run:
            args.row_thresh = run["row_thresh"]
            
        # Create a unique output folder for this specific run's JSON files
        safe_name = run["row_name"].replace(" ", "_").replace("=", "").replace("(", "").replace(")", "")
        args.exp_name = f"ruler_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Execute the benchmark and store the averages
        results = run_ruler(args)
        final_table_data[run["row_name"]] = results

    # Print the combined multi-row table at the very end
    print_multi_table(final_table_data)