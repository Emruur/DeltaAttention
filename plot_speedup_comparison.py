import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/emreuguremruur/Documents/Dev/Code/CS/Leiden/DeltaAttention"
OUT  = f"{BASE}/Paper/assets/speedup_comparison.png"

experiments = [
    (f"{BASE}/speedup_experiments_15/speedup_results.json", "thresh=15"),
    (f"{BASE}/speedup_experiments_17/speedup_results.json", "thresh=17"),
]

def valid_pairs(xs, ys):
    return [(x, y) for x, y in zip(xs, ys) if np.isfinite(x) and np.isfinite(y) and y > 0]

def paired_ratios(ns, bl, rd):
    rd_d = {n: t for n, t in valid_pairs(ns, rd)}
    return [(n, b / rd_d[n]) for n, b in valid_pairs(ns, bl) if n in rd_d]

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
fig.suptitle("Row Delta speedup vs baselines  |  Llama-3.1-8B, Wikitext", fontsize=13)

for ax, (path, label) in zip(axes, experiments):
    with open(path) as f:
        d = json.load(f)

    ns  = d["seq_lengths"]
    sdpa  = d["baseline_sdpa_e2e_ms"]
    flash = d["flash_baseline_e2e_ms"]
    rd    = d["row_delta_e2e_ms"]

    for bl, color, name in [
        (sdpa,  "steelblue",  "SDPA / Row Delta"),
        (flash, "darkorange", "Flash / Row Delta"),
    ]:
        pts = paired_ratios(ns, bl, rd)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-o", color=color, linewidth=2, markersize=6, label=name)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Break-even")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length N (tokens)", fontsize=11)
    ax.set_ylabel("Speedup  (baseline / row delta)", fontsize=11)
    ax.set_title(f"Speedup ratio  ({label})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.35, which="both")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
