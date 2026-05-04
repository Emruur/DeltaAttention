import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import json
import argparse
import gc
import re
import subprocess
import time
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import globVR
import glob_set
from modeling_llama import LlamaForCausalLM, LlamaConfig

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SEQ_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536]

EXPERIMENT_SETTINGS = {
    "baseline": {
        "delta_pf_key_on": 0,
        "delta_type": "regular",
        "flash": True,
    },
    "row_delta": {
        "delta_pf_key_on": 1,
        "delta_type": "row",
        "flash": True,
        "row_delta_threshold": 15,
        "row_similarity_metric": "euclidean",
        "divide_to": 32,
    },
}


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        return super().default(obj)


def load_wikitext_tokens(tokenizer):
    """Tokenize wikitext into one flat CPU tensor. Falls back from 103 to 2."""
    for name in ("wikitext-103-raw-v1", "wikitext-2-raw-v1"):
        try:
            ds = load_dataset("wikitext", name, split="train")
            text = " ".join(t for t in ds["text"] if t.strip())
            ids = tokenizer(text, return_tensors="pt").input_ids[0]
            print(f"[Data] {ids.shape[0]:,} tokens from wikitext/{name}")
            return ids
        except Exception as e:
            print(f"[Data] Could not load {name}: {e}")
    raise RuntimeError("Could not load any wikitext split")


def random_slice(tokens, seq_len, rng):
    """Return a [1, seq_len] CPU tensor, repeating corpus if needed."""
    if tokens.shape[0] <= seq_len:
        repeats = (seq_len // tokens.shape[0]) + 2
        tokens = tokens.repeat(repeats)
    max_start = tokens.shape[0] - seq_len
    start = int(rng.integers(0, max_start + 1))
    return tokens[start : start + seq_len].unsqueeze(0)


def apply_settings(settings):
    for k, v in settings.items():
        setattr(globVR, k, v)
    globVR.time_internal = True


def reset_timing():
    globVR.latency_stats = {}
    globVR.latency_events = []
    globVR.spars = 0.0


def read_prefill_ms():
    """Sum of all-layer prefill time from the last resolved forward pass (ms)."""
    stats = getattr(globVR, "latency_stats", {})
    for key in ("time_prefill_forward_total", "time_forward_total"):
        entry = stats.get(key)
        if entry and entry["calls"] > 0:
            return entry["time_ms"]   # total across all layers, not per-call avg
    return 0.0


def read_sparsity():
    """Moving-average sparsity accumulated across layers during last forward pass."""
    v = getattr(globVR, "spars", 0.0)
    return float(v.item() if isinstance(v, torch.Tensor) else v)


def single_pass(model, input_ids):
    """One forward pass. Returns (prefill_internal_ms, wall_clock_ms, sparsity)."""
    reset_timing()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        model(input_ids)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    glob_set.resolve_latency_events()

    return read_prefill_ms(), (t1 - t0) * 1000.0, read_sparsity()


def benchmark_one_len(model, tokens, seq_len, n_samples, n_warmup, rng):
    """
    Benchmark a single sequence length.
    Returns stats dict or None on OOM.
    """
    # allocate a warmup input first to detect OOM early
    try:
        inp = random_slice(tokens, seq_len, rng).to("cuda")
        with torch.no_grad():
            for _ in range(n_warmup):
                model(inp)
                torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        gc.collect()
        print(f"   [OOM] N={seq_len:,} during warmup — skipping")
        return None

    prefill_samples, wall_samples, sparsity_samples = [], [], []

    for i in range(n_samples):
        inp = random_slice(tokens, seq_len, rng).to("cuda")
        try:
            pf, wl, spars = single_pass(model, inp)
            prefill_samples.append(pf)
            wall_samples.append(wl)
            sparsity_samples.append(spars)
            print(f"   sample {i+1}/{n_samples}: prefill={pf:.2f}ms  wall={wl:.2f}ms  sparsity={spars:.3f}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            print(f"   [OOM] N={seq_len:,} sample {i} — stopping")
            break

    if not prefill_samples:
        return None

    def stats(xs):
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "samples": [float(x) for x in xs]}

    return {
        "prefill_ms": stats(prefill_samples),
        "wall_ms": stats(wall_samples),
        "sparsity": stats(sparsity_samples),
        "n_valid": len(prefill_samples),
    }


def run_worker(args, output_dir):
    torch.set_grad_enabled(False)

    print(f"[Worker] Loading model: {MODEL_ID}")
    config = AutoConfig.from_pretrained(MODEL_ID)
    config.attn_implementation = "eager"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, config=config, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    tokens = load_wikitext_tokens(tokenizer)
    rng = np.random.default_rng(args.seed)

    settings = EXPERIMENT_SETTINGS[args.experiment_type]
    apply_settings(settings)
    print(f"[Worker] Experiment: {args.experiment_type}  settings: {settings}")

    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    results = {
        "experiment_type": args.experiment_type,
        "settings": settings,
        "timestamp": datetime.now().isoformat(),
        "n_samples": args.n_samples,
        "n_warmup": args.n_warmup,
        "data": {},
    }

    for N in seq_lengths:
        print(f"\n--- N={N:,} ---")
        stats = benchmark_one_len(model, tokens, N, args.n_samples, args.n_warmup, rng)
        results["data"][str(N)] = stats
        if stats:
            print(
                f"   => prefill {stats['prefill_ms']['mean']:.2f} ± {stats['prefill_ms']['std']:.2f} ms"
                f"   wall {stats['wall_ms']['mean']:.2f} ± {stats['wall_ms']['std']:.2f} ms"
                f"   sparsity {stats['sparsity']['mean']:.3f}"
                f"   ({stats['n_valid']} valid samples)"
            )
        gc.collect()
        torch.cuda.empty_cache()

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{args.experiment_type}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, cls=NpEncoder)
    print(f"\n[Worker] Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def _extract_series(data_dict, metric, seq_lengths):
    """Pull mean/std arrays aligned to seq_lengths; None entries become NaN."""
    ns, means, stds = [], [], []
    for N in seq_lengths:
        entry = data_dict.get(str(N))
        ns.append(N)
        if entry is None:
            means.append(float("nan"))
            stds.append(float("nan"))
        else:
            means.append(entry[metric]["mean"])
            stds.append(entry[metric]["std"])
    return np.array(ns), np.array(means), np.array(stds)


def _find_crossover(ns, bl_means, rd_means):
    """Return the approximate N where rd_means first drops below bl_means."""
    for i in range(len(ns)):
        if np.isnan(bl_means[i]) or np.isnan(rd_means[i]):
            continue
        if rd_means[i] < bl_means[i]:
            if i == 0:
                return float(ns[0])
            # linear interpolation in log-space for cleaner result
            diff_prev = bl_means[i - 1] - rd_means[i - 1]
            diff_curr = bl_means[i] - rd_means[i]
            if diff_curr == diff_prev:
                return float(ns[i])
            frac = diff_prev / (diff_prev - diff_curr)
            return ns[i - 1] + frac * (ns[i] - ns[i - 1])
    return None


def plot_results(baseline_path, row_delta_path, out_dir):
    with open(baseline_path) as f:
        bl = json.load(f)
    with open(row_delta_path) as f:
        rd = json.load(f)

    all_ns = sorted(set(int(k) for k in bl["data"]) | set(int(k) for k in rd["data"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Prefill Latency: Baseline vs Row-Delta", fontsize=13)

    panels = [
        ("prefill_ms", "Prefill Forward Time (ms)", "Internal prefill forward (all layers summed)"),
        ("wall_ms",    "Wall-Clock Time (ms)",       "End-to-end forward pass wall time"),
    ]

    for ax, (metric, ylabel, subtitle) in zip(axes, panels):
        bl_ns, bl_m, bl_s = _extract_series(bl["data"], metric, all_ns)
        rd_ns, rd_m, rd_s = _extract_series(rd["data"], metric, all_ns)

        ax.errorbar(bl_ns, bl_m, yerr=bl_s, marker="o", label="baseline",  capsize=4, linewidth=1.5)
        ax.errorbar(rd_ns, rd_m, yerr=rd_s, marker="s", label="row_delta", capsize=4, linewidth=1.5)

        cross = _find_crossover(all_ns, bl_m, rd_m)
        if cross is not None:
            ax.axvline(cross, color="gray", linestyle="--", alpha=0.75,
                       label=f"crossover ≈ {cross:,.0f} tok")

        ax.set_xscale("log")
        ax.set_xlabel("Sequence Length N (tokens, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(subtitle)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    out_path = os.path.join(plots_dir, "prefill_speedup.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved -> {out_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def get_next_run_id(base):
    if not os.path.exists(base):
        return 1
    pat = re.compile(r"^run_(\d+)$")
    ids = {int(m.group(1)) for d in os.listdir(base) if (m := pat.match(d))}
    i = 1
    while i in ids:
        i += 1
    return i


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefill latency benchmark: baseline vs row_delta")
    parser.add_argument("--mode", choices=["master", "worker", "plot"], default="master")

    # worker-only
    parser.add_argument("--experiment_type", choices=list(EXPERIMENT_SETTINGS.keys()), default="baseline")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--seq_lengths", default=",".join(str(x) for x in SEQ_LENGTHS),
                        help="Comma-separated sequence lengths (worker mode)")

    # shared
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Random wikitext sequences to average per N")
    parser.add_argument("--n_warmup",  type=int, default=5,
                        help="Warmup forward passes before timing (per N)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_dir", default=None,
                        help="Output directory (master creates; worker/plot reads)")

    # row_delta overrides
    parser.add_argument("--delta",    type=float, default=15.0)
    parser.add_argument("--row_sim",  type=str,   default="euclidean")
    parser.add_argument("--divide_to",type=int,   default=2)

    args = parser.parse_args()

    # Apply CLI overrides to row_delta settings
    EXPERIMENT_SETTINGS["row_delta"]["row_delta_threshold"]    = args.delta
    EXPERIMENT_SETTINGS["row_delta"]["row_similarity_metric"]  = args.row_sim
    EXPERIMENT_SETTINGS["row_delta"]["divide_to"]              = args.divide_to

    base_storage = "speedup_experiments"

    if args.mode == "master":
        run_id = get_next_run_id(base_storage)
        exp_dir = args.exp_dir or os.path.join(base_storage, f"run_{run_id}")
        os.makedirs(exp_dir, exist_ok=True)
        print(f"[Master] Output dir: {exp_dir}")
        print(f"[Master] Seq lengths: {SEQ_LENGTHS}")
        print(f"[Master] n_samples={args.n_samples}  n_warmup={args.n_warmup}")

        for exp_type in ("baseline", "row_delta"):
            print(f"\n{'='*60}")
            print(f"[Master] Launching worker: {exp_type}")
            print(f"{'='*60}")
            cmd = [
                sys.executable, sys.argv[0],
                "--mode", "worker",
                "--experiment_type", exp_type,
                "--output_dir", exp_dir,
                "--seq_lengths", args.seq_lengths,
                "--n_samples", str(args.n_samples),
                "--n_warmup",  str(args.n_warmup),
                "--seed",      str(args.seed),
                "--delta",     str(args.delta),
                "--row_sim",   args.row_sim,
                "--divide_to", str(args.divide_to),
            ]
            subprocess.run(cmd, check=True)

        # plot after both workers finish
        bl_path = os.path.join(exp_dir, "baseline_results.json")
        rd_path = os.path.join(exp_dir, "row_delta_results.json")
        if os.path.exists(bl_path) and os.path.exists(rd_path):
            plot_results(bl_path, rd_path, exp_dir)
        else:
            print("[Master] Missing result files — skipping plot")

    elif args.mode == "worker":
        if args.output_dir is None:
            print("[Worker] --output_dir required in worker mode")
            sys.exit(1)
        run_worker(args, args.output_dir)

    elif args.mode == "plot":
        exp_dir = args.exp_dir
        if exp_dir is None:
            print("[Plot] --exp_dir required in plot mode")
            sys.exit(1)
        bl_path = os.path.join(exp_dir, "baseline_results.json")
        rd_path = os.path.join(exp_dir, "row_delta_results.json")
        plot_results(bl_path, rd_path, exp_dir)
