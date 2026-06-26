import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/emreuguremruur/Documents/Dev/Code/CS/Leiden/DeltaAttention"
OUT  = f"{BASE}/Paper/assets/attention_dominance.png"

with open(f"{BASE}/speedup_experiments_15/speedup_results.json") as f:
    d = json.load(f)

ns      = np.array(d["seq_lengths"], dtype=float)
attn    = np.array(d["overhead"]["time_delta_mm_pattern"], dtype=float)
total   = np.array(d["overhead"]["time_prefill_forward_total"], dtype=float)
ratio   = attn / total

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(ns, ratio, "-o", color="#E6A817", linewidth=2, markersize=6)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.2)

ax.set_xscale("log", base=2)
ax.set_ylim(0, 1.05)
ax.set_xlabel("Sequence length (tokens)", fontsize=12)
ax.set_ylabel("Attention compute / Total prefill time", fontsize=12)
ax.set_title("Attention as fraction of total prefill time  |  thresh=15, LLaMA-3.1-8B", fontsize=12)
ax.grid(True, alpha=0.35, which="both")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
plt.savefig(OUT, dpi=150)
print(f"Saved to {OUT}")
