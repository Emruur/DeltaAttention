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
from scipy.optimize import curve_fit
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM as HFLlamaForCausalLM

import globVR
import glob_set

from modeling_llama import LlamaForCausalLM, LlamaConfig

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]

OVERHEAD_KEYS = ["time_get_row_delta", "time_delta_mm_pattern", "time_prefill_forward_total"]


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


def set_row_delta(time_internal=False):
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
    globVR.time_internal = time_internal
    globVR.skip_causal_mask = True


def set_flash_baseline():
    globVR.delta_pf_key_on = 0
    globVR.delta_mlp = "Regular"
    globVR.delta_decode = False
    globVR.flash = True
    globVR.time_internal = False
    globVR.skip_causal_mask = True


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
    cuda_dead = False
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
                cuda_dead = True
        elif run_times:
            avg_t = float(np.mean(run_times))
            print(f"  N={N:>7}: {avg_t:9.2f} ms  (avg of {len(run_times)} runs)", flush=True)
            results.append(avg_t)
        else:
            results.append(float("nan"))

    return results


def collect_overhead_profile(model, tokens, seq_lengths, device):
    """One forward per N with time_internal=True to get per-operation breakdown."""
    print("\n=== Overhead profiling (row delta, time_internal=True) ===", flush=True)
    overhead_data = {key: [] for key in OVERHEAD_KEYS}

    for N in seq_lengths:
        if N > len(tokens):
            for key in OVERHEAD_KEYS:
                overhead_data[key].append(float("nan"))
            continue

        ids = tokens[:N]
        try:
            reset_timing()
            set_row_delta(time_internal=True)
            run_prefill_e2e(model, ids, device, setup_fn=None)
            stats = getattr(globVR, "latency_stats", {})
            for key in OVERHEAD_KEYS:
                val = stats.get(key, {}).get("time_ms", float("nan"))
                overhead_data[key].append(float(val))
            summary = "  ".join(
                f"{k.rsplit('_', 1)[-1]}={stats.get(k, {}).get('time_ms', 0):.1f}ms"
                for k in OVERHEAD_KEYS
            )
            print(f"  N={N:>7}: {summary}", flush=True)
        except Exception as e:
            if is_cuda_error(e):
                recover_cuda()
                for key in OVERHEAD_KEYS:
                    overhead_data[key].append(float("nan"))
                print(f"  N={N:>7}: CUDA error", flush=True)
            else:
                raise
        recover_cuda()

    return overhead_data


# ==========================================
# CURVE FITTING + OVERHEAD PLOT
# ==========================================

def _r2(y, y_hat):
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def plot_overhead(ax, seq_lengths, vals, title):
    ns = np.array(seq_lengths, dtype=float)
    vals = np.array(vals, dtype=float)
    mask = np.isfinite(vals) & np.isfinite(ns) & (ns > 0)
    ns_m, vals_m = ns[mask], vals[mask]

    ax.scatter(ns_m, vals_m, color="black", s=45, zorder=5, label="Measured")

    if len(ns_m) >= 3:
        ns_dense = np.linspace(ns_m.min(), ns_m.max(), 600)

        # Linear
        try:
            p = np.polyfit(ns_m, vals_m, 1)
            r2 = _r2(vals_m, np.polyval(p, ns_m))
            ax.plot(ns_dense, np.polyval(p, ns_dense), "--", color="royalblue",
                    linewidth=1.6, label=f"Linear (R²={r2:.3f})")
        except Exception:
            pass

        # N log N
        try:
            def nlogn(x, a, b):
                return a * x * np.log(x) + b
            p0 = [vals_m[-1] / (ns_m[-1] * np.log(ns_m[-1])), 0.0]
            popt, _ = curve_fit(nlogn, ns_m, vals_m, p0=p0, maxfev=10000)
            r2 = _r2(vals_m, nlogn(ns_m, *popt))
            ax.plot(ns_dense, nlogn(ns_dense, *popt), "--", color="forestgreen",
                    linewidth=1.6, label=f"N·log(N) (R²={r2:.3f})")
        except Exception:
            pass

        # Quadratic
        try:
            p = np.polyfit(ns_m, vals_m, 2)
            r2 = _r2(vals_m, np.polyval(p, ns_m))
            ax.plot(ns_dense, np.polyval(p, ns_dense), "--", color="crimson",
                    linewidth=1.6, label=f"Quadratic (R²={r2:.3f})")
        except Exception:
            pass

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length N (tokens)", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35, which="both")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))


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
    # Delta model: row delta + flash baseline + overhead profile
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
        "Row Delta (flash, prefill only)", setup_fn=set_row_delta,
        n_warmup=args.n_warmup, n_runs=args.n_runs,
    )

    flash_baseline_times = benchmark_mode(
        delta_model, tokens, SEQ_LENGTHS, device,
        "Flash Baseline (triton, no delta)", setup_fn=set_flash_baseline,
        n_warmup=args.n_warmup, n_runs=args.n_runs,
    )

    overhead_data = collect_overhead_profile(delta_model, tokens, SEQ_LENGTHS, device)

    del delta_model
    torch.cuda.empty_cache()
    gc.collect()
    print("Delta model freed.", flush=True)

    # ------------------------------------------
    # Baseline: SDPA (stock HF model, bypasses our custom registration)
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
        "flash_baseline_e2e_ms": flash_baseline_times,
        "row_delta_e2e_ms": row_delta_times,
        "overhead": {k: overhead_data[k] for k in OVERHEAD_KEYS},
    }
    json_path = "speedup_experiments/speedup_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {json_path}", flush=True)

    # ------------------------------------------
    # 4-subplot figure (2x2)
    # ------------------------------------------
    def valid(times):
        return [(N, t) for N, t in zip(SEQ_LENGTHS, times) if not np.isnan(t)]

    def paired_ratios(bl_list, rd_list):
        bl_d = {N: t for N, t in valid(bl_list)}
        return [(N, bl_d[N] / t) for N, t in valid(rd_list) if N in bl_d and t > 0]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        "Row Delta vs Baselines  |  Llama-3-8B-1M (Gradient), Wikitext",
        fontsize=14,
    )
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    # --- Plot 1: Absolute latency ---
    for times, color, label in [
        (baseline_times,       "steelblue",  "SDPA baseline"),
        (flash_baseline_times, "darkorange", "Flash baseline (triton)"),
        (row_delta_times,      "crimson",    "Row Delta (flash)"),
    ]:
        pts = valid(times)
        if pts:
            ns, ts = zip(*pts)
            ax1.plot(ns, ts, "-o", color=color, label=label, linewidth=2, markersize=6)

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Sequence Length N (tokens)", fontsize=11)
    ax1.set_ylabel("End-to-end prefill latency (ms)", fontsize=11)
    ax1.set_title("Absolute latency", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.35, which="both")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # --- Plot 2: Speedup ratios ---
    for bl_times, color, label in [
        (baseline_times,       "steelblue",  "SDPA / Row Delta"),
        (flash_baseline_times, "darkorange", "Flash / Row Delta"),
    ]:
        pr = paired_ratios(bl_times, row_delta_times)
        if pr:
            ns_p, ratios = zip(*pr)
            ax2.plot(ns_p, ratios, "-o", color=color, linewidth=2, markersize=6, label=label)

    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Break-even")
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence Length N (tokens)", fontsize=11)
    ax2.set_ylabel("Speedup  (baseline / row_delta)", fontsize=11)
    ax2.set_title("Speedup ratio", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.35, which="both")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # --- Plot 3: time_get_row_delta overhead ---
    plot_overhead(ax3, SEQ_LENGTHS, overhead_data["time_get_row_delta"],
                  "Overhead: time_get_row_delta")

    # --- Plot 4: time_delta_mm_pattern overhead ---
    plot_overhead(ax4, SEQ_LENGTHS, overhead_data["time_delta_mm_pattern"],
                  "Overhead: time_delta_mm_pattern")

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = f"speedup_experiments/speedup_plot.{ext}"
        plt.savefig(path, dpi=150)
        print(f"Plot saved to {path}", flush=True)


if __name__ == "__main__":
    main()
