import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import matplotlib
matplotlib.use("Agg")

import argparse
import gc
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

import globVR
import glob_set

from modeling_llama import LlamaForCausalLM, LlamaConfig

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SEQ_LENGTHS = [512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 32768, 64000, 96000, 128000]


# ==========================================
# GLOBVR HELPERS
# ==========================================

def reset_timing():
    globVR.latency_stats = {}
    globVR.latency_events = []
    globVR.spars = 0.0
    globVR.sequence_lengths = []
    if not hasattr(globVR, "kv_compression_samples"):
        globVR.kv_compression_samples = []
    globVR.kv_compression_samples = []


def set_baseline_flash():
    globVR.delta_pf_key_on = 0
    globVR.delta_mlp = "Regular"
    globVR.flash = True
    globVR.delta_decode = False
    globVR.time_internal = True


def set_row_delta():
    globVR.delta_pf_key_on = 1
    globVR.delta_type = "row"
    globVR.scale = 0.05
    globVR.delta_mlp = "Regular"
    globVR.row_delta_threshold = 15
    globVR.row_similarity_metric = "euclidian"
    globVR.chunk_size = 512
    globVR.divide_to = 0
    globVR.flash = True
    globVR.delta_decode = False
    globVR.time_internal = True


# ==========================================
# DATA
# ==========================================

def get_wikitext_tokens(tokenizer, max_tokens):
    from datasets import load_dataset
    print("Loading wikitext-103-raw-v1 ...", flush=True)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    # Concatenate until we have enough characters (~5 chars/token is conservative)
    char_budget = max_tokens * 6
    chunks, total = [], 0
    for t in dataset["text"]:
        if not t.strip():
            continue
        chunks.append(t)
        total += len(t)
        if total >= char_budget:
            break
    text = "\n".join(chunks)
    tokens = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    flat = tokens[0]
    if len(flat) < max_tokens:
        raise ValueError(f"Only got {len(flat)} tokens from wikitext, need {max_tokens}")
    print(f"Tokenized {len(flat)} tokens (using {len(chunks)} articles).", flush=True)
    return flat[:max_tokens]


# ==========================================
# BENCHMARK CORE
# ==========================================

@torch.no_grad()
def run_prefill(model, input_ids, device):
    """Single timed prefill forward. Returns total attention time in ms (summed across all layers)."""
    reset_timing()
    ids = input_ids.unsqueeze(0).to(device)
    mask = torch.ones_like(ids)
    torch.cuda.synchronize()
    model(input_ids=ids, attention_mask=mask, use_cache=False)
    torch.cuda.synchronize()
    glob_set.resolve_latency_events()
    stats = globVR.latency_stats.get("time_prefill_forward_total", {"time_ms": 0.0, "calls": 0})
    return stats["time_ms"]  # sum across all 32 attention layers


def benchmark_mode(model, tokens, seq_lengths, device, mode_name, setup_fn, n_warmup, n_runs):
    print(f"\n=== {mode_name} (warmup={n_warmup}, runs={n_runs}) ===", flush=True)

    # Initial warmup at smallest N to trigger Triton JIT
    print(f"  Warming up at N={seq_lengths[0]} ...", flush=True)
    for _ in range(n_warmup):
        setup_fn()
        try:
            run_prefill(model, tokens[: seq_lengths[0]], device)
        except torch.cuda.OutOfMemoryError:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    results = []
    for N in seq_lengths:
        if N > len(tokens):
            print(f"  N={N:>6}: skipped (not enough tokens)", flush=True)
            results.append(float("nan"))
            continue

        ids = tokens[:N]

        # Per-length warmup (new Triton tile sizes may recompile)
        for _ in range(n_warmup):
            setup_fn()
            try:
                run_prefill(model, ids, device)
            except torch.cuda.OutOfMemoryError:
                print(f"  N={N:>6}: OOM during warmup — skipping", flush=True)
                results.append(float("nan"))
                torch.cuda.empty_cache()
                gc.collect()
                break
        else:
            torch.cuda.empty_cache()
            gc.collect()

            # Timed runs
            run_times = []
            oom = False
            for _ in range(n_runs):
                setup_fn()
                try:
                    t = run_prefill(model, ids, device)
                    run_times.append(t)
                except torch.cuda.OutOfMemoryError:
                    print(f"  N={N:>6}: OOM during measurement", flush=True)
                    oom = True
                    break
                torch.cuda.empty_cache()
                gc.collect()

            if oom or not run_times:
                results.append(float("nan"))
            else:
                avg_t = float(np.mean(run_times))
                print(f"  N={N:>6}: {avg_t:9.2f} ms  (avg of {len(run_times)} runs)", flush=True)
                results.append(avg_t)

    return results


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_warmup", type=int, default=1, help="Warmup iterations per sequence length")
    parser.add_argument("--n_runs", type=int, default=3, help="Timed iterations per sequence length")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  warmup={args.n_warmup}  runs={args.n_runs}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokens = get_wikitext_tokens(tokenizer, max(SEQ_LENGTHS))

    # Disable timing/delta during model load to avoid spurious state
    globVR.delta_pf_key_on = 0
    globVR.delta_mlp = "Regular"
    globVR.delta_decode = False
    globVR.flash = False
    globVR.time_internal = False

    print(f"Loading {MODEL_ID} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    model.eval()
    torch.set_grad_enabled(False)
    print("Model ready.", flush=True)

    baseline_times = benchmark_mode(
        model, tokens, SEQ_LENGTHS, device,
        "Baseline Flash", set_baseline_flash,
        n_warmup=args.n_warmup, n_runs=args.n_runs,
    )
    row_delta_times = benchmark_mode(
        model, tokens, SEQ_LENGTHS, device,
        "Row Delta (prefill only)", set_row_delta,
        n_warmup=args.n_warmup, n_runs=args.n_runs,
    )

    # ------------------------------------------
    # Save raw results
    # ------------------------------------------
    os.makedirs("speedup_experiments", exist_ok=True)
    results = {
        "seq_lengths": SEQ_LENGTHS,
        "baseline_flash_total_ms": baseline_times,
        "row_delta_total_ms": row_delta_times,
    }
    json_path = "speedup_experiments/speedup_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}", flush=True)

    # ------------------------------------------
    # Plot
    # ------------------------------------------
    def valid(times):
        return [(N, t) for N, t in zip(SEQ_LENGTHS, times) if not np.isnan(t)]

    def paired_ratios(bl_list, rd_list):
        bl_d = {N: t for N, t in valid(bl_list)}
        return [(N, bl_d[N] / t) for N, t in valid(rd_list) if N in bl_d and t > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Row Delta vs Baseline Flash  |  LLaMA 3.1 8B-Instruct, Wikitext",
                 fontsize=14)

    # --- Left: raw latency ---
    for times, color, label in [
        (baseline_times,  "blue", "Baseline Flash"),
        (row_delta_times, "red",  "Row Delta (prefill)"),
    ]:
        pts = valid(times)
        if pts:
            ns, ts = zip(*pts)
            ax1.plot(ns, ts, "-o", color=color, label=label, linewidth=2, markersize=6)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Sequence Length N (tokens)", fontsize=12)
    ax1.set_ylabel("time_prefill_forward_total (ms, all layers)", fontsize=12)
    ax1.set_title("Absolute latency", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.35, which="both")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # --- Right: speedup ratio (baseline / row_delta) ---
    pr = paired_ratios(baseline_times, row_delta_times)
    if pr:
        ns_p, ratios = zip(*pr)
        ax2.plot(ns_p, ratios, "-o", color="red", linewidth=2, markersize=6,
                 label="Row Delta (prefill)")
        ax2.fill_between(ns_p, ratios, 1.0,
                         where=[r >= 1.0 for r in ratios],
                         alpha=0.15, color="green", label="Row delta faster")
        ax2.fill_between(ns_p, ratios, 1.0,
                         where=[r < 1.0 for r in ratios],
                         alpha=0.15, color="red", label="Row delta slower")
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Break-even")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence Length N (tokens)", fontsize=12)
    ax2.set_ylabel("Speedup  (baseline / row_delta)", fontsize=12)
    ax2.set_title("Speedup ratio", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.35, which="both")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = f"speedup_experiments/speedup_plot.{ext}"
        plt.savefig(path, dpi=150)
        print(f"Plot saved to {path}", flush=True)


if __name__ == "__main__":
    main()
