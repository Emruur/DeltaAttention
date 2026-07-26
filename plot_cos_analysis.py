"""
Regenerate assets/cos_analysis.png (paper Figure, Section 3.1) from raw data.

Reads key_similarity_data.npz and plots mean adjacent key cosine similarity
per layer, one bar per configuration. Five configurations.
"""

import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = "key_similarity_data.npz"
OUT_PATH  = "Paper/assets/cos_analysis.png"

d = np.load(DATA_PATH)
LAYERS = d["layers"].tolist()

# (array, label, color, hatch) — order matches the paper's config numbering
VARIANTS = [
    (d["pre_real"],          "pre-RoPE real",              "steelblue", ""),
    (d["pre_shuf"],          "pre-RoPE shuffled",          "steelblue", "//"),
    (d["post_real"],         "post-RoPE real",             "tomato",    ""),
    (d["post_content_shuf"], "post-RoPE content-shuffled", "tomato",    "//"),
    (d["post_shuf"],         "post-RoPE key-shuffled",     "tomato",    "xx"),
]

n_variants = len(VARIANTS)
n_layers   = len(LAYERS)

fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 2.5, 3.5), sharey=True,
                         gridspec_kw={"wspace": 0.10})

for c, (layer_idx, ax) in enumerate(zip(LAYERS, axes)):
    means  = [arr[c].mean() for arr, _, _, _ in VARIANTS]
    colors = [col           for _, _, col, _ in VARIANTS]
    bars   = ax.bar(range(n_variants), means, color=colors,
                    edgecolor="white", width=0.65)
    for bar, (_, _, _, hatch) in zip(bars, VARIANTS):
        bar.set_hatch(hatch)
    ax.set_xticks(range(n_variants))
    ax.set_xticklabels([str(i + 1) for i in range(n_variants)], fontsize=8)
    ax.set_title(f"Layer {layer_idx}", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    if c == 0:
        ax.set_ylabel("Mean adj. cos sim", fontsize=8)

legend_labels = [f"{i + 1}. {label}" for i, (_, label, _, _) in enumerate(VARIANTS)]
handles = [plt.Rectangle((0, 0), 1, 1, color=col, hatch=hatch, edgecolor="white")
           for _, _, col, hatch in VARIANTS]
fig.legend(handles, legend_labels, loc="lower center", ncol=4,
           fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.12))

fig.suptitle("Mean adjacent cosine similarity per layer (avg over all heads)", fontsize=9)
plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
plt.close()
print(f"Saved -> {OUT_PATH}")
