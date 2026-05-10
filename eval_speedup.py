import os
os.environ["HF_DATASETS_TRUST_REMOTE_CODE"] = "1"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import matplotlib
matplotlib.use("Agg")

import argparse
import gc
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM

import globVR
import glob_set

from modeling_llama import LlamaForCausalLM, LlamaConfig

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

MODEL_ID = "gradientai/Llama-3-8b-Instruct-Gradient-1048k"

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]


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


def set_row_delta():
    globVR.delta_pf_key_on = 1
    globVR.delta_type = "row"
    globVR.scale = 0.05
    globVR.delta_mlp = "Regular"
    globVR.row_delta_threshold = 17
    globVR.row_similarity_metric = "euclidian"
    globVR.chunk_size = 512
    globVR.divide_to = 0
    globVR.flash = True
    globVR.delta_decode = False
    globVR.time_internal = False
    globVR.skip_causal_mask = True  # safe: batch_size=1, no padding in speedup benchmark


# ==========================================
# DATA
# ==========================================

def get_wikitext_tokens(tokenizer, max_tokens):
    from datasets import load_dataset
    print("Loading wikitext-103-raw-v1 ...", flush=True)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
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
def run_prefill_e2e(model, input_ids, device, setup_fn=None):
    """Single timed prefill forward. Returns end-to-end latency in ms."""
    if setup_fn is not None:
        reset_timing()
        setup_fn()
    ids = input_ids.unsqueeze(0).to(device)
    mask = torch.ones_like(ids)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_evt.record()
    model(input_ids=ids, attention_mask=mask, use_cache=False)
    end_evt.record()
    torch.cuda.synchronize()
    return start_evt.elapsed_time(end_evt)


def is_cuda_error(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and
        any(x in str(e) for x in ["CUDA", "cuda", "Triton", "triton", "illegal"])
    )


def recover_cuda():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    torch.cuda.empty_cache()
    gc.collect()


def benchmark_mode(model, tokens, seq_lengths, device, mode_name, setup_fn, n_warmup, n_runs):
    print(f"\n=== {mode_name} (warmup={n_warmup}, runs={n_runs}) ===", flush=True)

    print(f"  Warming up at N={seq_lengths[0]} ...", flush=True)
    for _ in range(n_warmup):
        try:
            run_prefill_e2e(model, tokens[: seq_lengths[0]], device, setup_fn)
        except Exception as e:
            if is_cuda_error(e):
                recover_cuda()
            else:
                raise
        recover_cuda()

    results = []
    cuda_dead = False  # once GPU enters bad state, skip remaining lengths
    for N in seq_lengths:
        if cuda_dead:
            results.append(float("nan"))
            continue

        if N > len(tokens):
            print(f"  N={N:>7}: skipped (not enough tokens)", flush=True)
            results.append(float("nan"))
            continue

        ids = tokens[:N]
        error_msg = None

        for _ in range(n_warmup):
            try:
                run_prefill_e2e(model, ids, device, setup_fn)
            except Exception as e:
                if is_cuda_error(e):
                    error_msg = str(e)[:60]
                    recover_cuda()
                    break
                raise
        else:
            recover_cuda()

            run_times = []
            for _ in range(n_runs):
                try:
                    t = run_prefill_e2e(model, ids, device, setup_fn)
                    run_times.append(t)
                except Exception as e:
                    if is_cuda_error(e):
                        error_msg = str(e)[:60]
                        recover_cuda()
                        break
                    raise
                recover_cuda()

        if error_msg is not None:
            is_oom = "memory" in error_msg.lower() or "OutOfMemory" in error_msg
            label = "OOM" if is_oom else "CUDA error"
            print(f"  N={N:>7}: {label} — skipping ({error_msg})", flush=True)
            results.append(float("nan"))
            if not is_oom:
                # Illegal memory access puts GPU in unrecoverable state
                cuda_dead = True
        elif run_times:
            avg_t = float(np.mean(run_times))
            print(f"  N={N:>7}: {avg_t:9.2f} ms  (avg of {len(run_times)} runs)", flush=True)
            results.append(avg_t)
        else:
            results.append(float("nan"))

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

    # ------------------------------------------
    # Delta model: custom LlamaForCausalLM (row delta)
    # ------------------------------------------
    globVR.delta_pf_key_on = 0
    globVR.delta_mlp = "Regular"
    globVR.delta_decode = False
    globVR.flash = False
    globVR.time_internal = False

    print(f"\nLoading delta model: {MODEL_ID} ...", flush=True)
    delta_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    ).to(device)
    delta_model.eval()
    print("Delta model ready.", flush=True)

    row_delta_times = benchmark_mode(
        delta_model, tokens, SEQ_LENGTHS, device,
        "Row Delta (prefill only)", setup_fn=set_row_delta,
        n_warmup=args.n_warmup, n_runs=args.n_runs,
    )

    del delta_model
    torch.cuda.empty_cache()
    gc.collect()
    print("Delta model freed.", flush=True)

    # ------------------------------------------
    # Baseline: SDPA
    # Load standard HF class directly to bypass our custom registration
    # ------------------------------------------
    print(f"\nLoading baseline model (sdpa): {MODEL_ID} ...", flush=True)
    baseline_model = HFLlamaForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).to(device)
    baseline_model.eval()
    torch.set_grad_enabled(False)
    print("Baseline model ready.", flush=True)

    baseline_times = benchmark_mode(
        baseline_model, tokens, SEQ_LENGTHS, device,
        "Baseline SDPA", setup_fn=None,
        n_warmup=args.n_warmup, n_runs=args.n_runs,
    )

    del baseline_model
    torch.cuda.empty_cache()
    gc.collect()

    # ------------------------------------------
    # Save raw results
    # ------------------------------------------
    os.makedirs("speedup_experiments", exist_ok=True)
    results = {
        "model": MODEL_ID,
        "seq_lengths": SEQ_LENGTHS,
        "baseline_sdpa_e2e_ms": baseline_times,
        "row_delta_e2e_ms": row_delta_times,
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
    fig.suptitle("Row Delta vs SDPA  |  Llama-3-8B-1M (Gradient), Wikitext",
                 fontsize=14)

    for times, color, label in [
        (baseline_times,  "blue", "SDPA (baseline)"),
        (row_delta_times, "red",  "Row Delta (prefill)"),
    ]:
        pts = valid(times)
        if pts:
            ns, ts = zip(*pts)
            ax1.plot(ns, ts, "-o", color=color, label=label, linewidth=2, markersize=6)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Sequence Length N (tokens)", fontsize=12)
    ax1.set_ylabel("End-to-end prefill latency (ms)", fontsize=12)
    ax1.set_title("Absolute latency", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.35, which="both")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

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
    ax2.set_ylabel("Speedup  (SDPA / row_delta)", fontsize=12)
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
