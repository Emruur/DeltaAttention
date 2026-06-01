"""
Suite of plots from key_similarity_data.npz.

Generated figures:
  1. ks_cdf_all.png        – CDF grid, all 4 variants per (layer, head)
  2. ks_cdf_pair_XY.png    – CDF grid for each of the 6 pairings
  3. ks_mean_by_layer.png  – Mean similarity per layer, grouped by variant (avg over heads)
  4. ks_gap_rope.png       – (post_real - post_shuf) vs (pre_real - pre_shuf): RoPE gap per layer/head
  5. ks_heatmap.png        – Mean similarity heatmap: variants × (layer, head)
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

DATA_PATH = "key_similarity_data.npz"
OUT_DIR   = "key_similarity_plots"

import os
os.makedirs(OUT_DIR, exist_ok=True)

d = np.load(DATA_PATH)

pre_real          = d["pre_real"]           # [L, H, T]
pre_shuf          = d["pre_shuf"]
post_real         = d["post_real"]
post_shuf         = d["post_shuf"]
post_content_shuf = d["post_content_shuf"]  # content shuffled, sequential positions → RoPE
LAYERS            = d["layers"].tolist()
SEQ_LEN           = int(d["seq_len"])

n_layers, n_heads, _ = pre_real.shape

VARIANTS = {
    "pre-RoPE real":              (pre_real,          "steelblue", "-",  1.5),
    "pre-RoPE shuffled":          (pre_shuf,          "steelblue", "--", 1.0),
    "post-RoPE real":             (post_real,         "tomato",    "-",  1.5),
    "post-RoPE content-shuffled": (post_content_shuf, "tomato",    ":",  1.2),
    "post-RoPE key-shuffled":     (post_shuf,         "tomato",    "--", 1.0),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def cdf_line(ax, vals, color, ls, lw, label):
    sv = np.sort(vals)
    ax.plot(sv, np.arange(1, len(sv)+1)/len(sv),
            color=color, ls=ls, lw=lw, label=label)


def cdf_grid(variant_subset, title, fname):
    """Grid: rows=layers, cols=heads. variant_subset: list of keys from VARIANTS."""
    fig, axes = plt.subplots(
        n_layers, n_heads,
        figsize=(n_heads * 2.0, n_layers * 1.8),
        sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.25, "wspace": 0.10},
    )
    for r, layer_idx in enumerate(LAYERS):
        for c in range(n_heads):
            ax = axes[r, c]
            for name in variant_subset:
                arr, color, ls, lw = VARIANTS[name]
                cdf_line(ax, arr[r, c], color, ls, lw, name)
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            ax.axvline(0, color="gray", lw=0.4, ls=":")
            ax.grid(True, alpha=0.2, lw=0.4)
            ax.tick_params(labelsize=6)
            if r == 0:
                ax.set_title(f"Head {c}", fontsize=7, pad=2)
            if c == 0:
                ax.set_ylabel(f"L{layer_idx}\nCDF", fontsize=7)
            if r == n_layers - 1:
                ax.set_xlabel("cos sim", fontsize=6)

    axes[0, 0].legend(fontsize=5.5, loc="upper left", framealpha=0.8)
    fig.suptitle(title, fontsize=9, y=1.01)
    plt.savefig(f"{OUT_DIR}/{fname}", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


# ── 1. All 4 variants ────────────────────────────────────────────────────────

cdf_grid(list(VARIANTS.keys()),
         "All variants · pre/post-RoPE × real/shuffled",
         "ks_cdf_all.png")


# ── 2. Every pairwise CDF ────────────────────────────────────────────────────

for a, b in combinations(VARIANTS.keys(), 2):
    safe = lambda s: s.replace(" ", "_").replace("-", "").replace("/", "")
    fname = f"ks_cdf_{safe(a)}_vs_{safe(b)}.png"
    cdf_grid([a, b], f"{a}  vs  {b}", fname)


# ── 3. Mean similarity per layer (averaged over heads) ───────────────────────

fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 2.5, 3), sharey=True,
                         gridspec_kw={"wspace": 0.12})

for c, (layer_idx, ax) in enumerate(zip(LAYERS, axes)):
    means = {name: VARIANTS[name][0][c].mean() for name in VARIANTS}
    colors = [VARIANTS[n][1] for n in means]
    hatches = ["", "//", "", "//"]
    bars = ax.bar(range(len(means)), list(means.values()),
                  color=colors, hatch=None, edgecolor="white", width=0.6)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels([n.replace(" ", "\n") for n in means], fontsize=5.5)
    ax.set_title(f"Layer {layer_idx}", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    if c == 0:
        ax.set_ylabel("Mean adj. cos sim", fontsize=8)

fig.suptitle("Mean adjacent cosine similarity per layer (avg over all heads)", fontsize=9)
plt.savefig(f"{OUT_DIR}/ks_mean_by_layer.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved ks_mean_by_layer.png")


# ── 4. RoPE gap: (post_real - post_shuf) vs (pre_real - pre_shuf) ────────────
# Per (layer, head): gap = mean(real) - mean(shuffled)

pre_gap  = pre_real.mean(axis=-1)  - pre_shuf.mean(axis=-1)   # [L, H]
post_gap = post_real.mean(axis=-1) - post_shuf.mean(axis=-1)  # [L, H]

fig, axes = plt.subplots(1, n_layers, figsize=(n_layers * 2.2, 2.8), sharey=True,
                         gridspec_kw={"wspace": 0.12})

x = np.arange(n_heads)
w = 0.35
for c, (layer_idx, ax) in enumerate(zip(LAYERS, axes)):
    ax.bar(x - w/2, pre_gap[c],  w, label="pre-RoPE gap",  color="steelblue", alpha=0.85)
    ax.bar(x + w/2, post_gap[c], w, label="post-RoPE gap", color="tomato",    alpha=0.85)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)], fontsize=6)
    ax.set_title(f"Layer {layer_idx}", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    if c == 0:
        ax.set_ylabel("mean(real) − mean(shuffled)", fontsize=7)

axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Real−shuffled similarity gap: pre-RoPE vs post-RoPE per head\n"
             "(larger post gap = RoPE is the driver)", fontsize=8, y=1.02)
plt.savefig(f"{OUT_DIR}/ks_gap_rope.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved ks_gap_rope.png")


# ── 5. Heatmap: mean similarity, variants × (layer×head) ─────────────────────

mat = np.stack([VARIANTS[n][0].mean(axis=-1) for n in VARIANTS])  # [4, L, H]
mat_flat = mat.reshape(4, -1)  # [4, L*H]

xlabels = [f"L{l}H{h}" for l in LAYERS for h in range(n_heads)]

fig, ax = plt.subplots(figsize=(n_layers * n_heads * 0.55, 2.5))
im = ax.imshow(mat_flat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
ax.set_yticks(range(4))
ax.set_yticklabels(list(VARIANTS.keys()), fontsize=8)
ax.set_xticks(range(len(xlabels)))
ax.set_xticklabels(xlabels, fontsize=6, rotation=45, ha="right")
fig.colorbar(im, ax=ax, label="mean adj. cos sim", fraction=0.03, pad=0.02)
fig.suptitle("Mean adjacent cosine similarity · variants × (layer, head)", fontsize=9)
plt.savefig(f"{OUT_DIR}/ks_heatmap.png", dpi=160, bbox_inches="tight")
plt.close()
print("Saved ks_heatmap.png")

print("\nDone. Files written:")
print("  ks_cdf_all.png")
print("  ks_cdf_<A>_vs_<B>.png  (6 pairwise CDFs)")
print("  ks_mean_by_layer.png")
print("  ks_gap_rope.png")
print("  ks_heatmap.png")
