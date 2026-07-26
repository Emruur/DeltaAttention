"""
Compare pre-RoPE vs post-RoPE adjacent query cosine similarity distributions.

Same structure as visualize_key_similarity.py but for Q instead of K.
Pre-RoPE Q: hooked from q_proj output.
Post-RoPE Q: captured by monkey-patching apply_rotary_pos_emb during forward.
"""

import os, sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import globVR
import modeling_llama as _ml
from modeling_llama import LlamaForCausalLM, LlamaConfig

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SEQ_LEN  = 512
LAYERS   = [0, 8, 16, 23]
HEADS    = [0, 2, 5, 7]
SEED     = 42
OUT_PATH = "Paper/assets/query_similarity.png"
# ─────────────────────────────────────────────

AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

globVR.delta_pf_key_on = 0
globVR.flash           = False
globVR.delta_decode    = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager"
).to(device)
model.eval()

num_heads    = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim     = model.config.hidden_size // num_heads

print("Loading wikitext…")
ds   = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
text = " ".join(row["text"] for row in ds if row["text"].strip())
ids  = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN).input_ids.to(device)

torch.manual_seed(SEED)
shuffled_ids = ids[:, torch.randperm(ids.shape[1])].to(device)

print(f"Sequence length: {ids.shape[1]} tokens")


def adj_cos(q):
    """q: [num_heads, seq, head_dim] → [num_heads, seq-1]"""
    q = F.normalize(q.float(), dim=-1)
    return (q[:, :-1, :] * q[:, 1:, :]).sum(dim=-1).cpu().numpy()


def get_sims(input_ids):
    pre_raw    = {}
    hooks      = []
    captured_qs = []   # one entry per layer, in order

    # pre-RoPE Q from q_proj hook
    for layer_idx in LAYERS:
        def _hook(module, inp, out, _idx=layer_idx):
            q = out[0].float().reshape(-1, num_heads, head_dim)
            pre_raw[_idx] = q.permute(1, 0, 2).clone()  # [h, seq, d]
        h = model.model.layers[layer_idx].self_attn.q_proj.register_forward_hook(_hook)
        hooks.append(h)

    # post-RoPE Q by patching apply_rotary_pos_emb in modeling_llama
    _orig_rope = _ml.apply_rotary_pos_emb

    def _capturing_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        q_rot, k_rot = _orig_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
        captured_qs.append(q_rot.detach().float())  # [1, heads, seq, d]
        return q_rot, k_rot

    _ml.apply_rotary_pos_emb = _capturing_rope

    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)

    _ml.apply_rotary_pos_emb = _orig_rope
    for h in hooks:
        h.remove()

    # captured_qs[i] = post-RoPE Q for layer i (all 32 layers in order)
    pre_real, pre_shuf, post_real, post_shuf = {}, {}, {}, {}
    for layer_idx in LAYERS:
        q_pre  = pre_raw[layer_idx]              # [h, seq, d]
        q_post = captured_qs[layer_idx][0]       # [h, seq, d]

        perm = torch.randperm(q_pre.shape[1])

        pre_real[layer_idx]  = adj_cos(q_pre)
        pre_shuf[layer_idx]  = adj_cos(q_pre[:, perm, :])
        post_real[layer_idx] = adj_cos(q_post)
        post_shuf[layer_idx] = adj_cos(q_post[:, perm, :])

    return pre_real, pre_shuf, post_real, post_shuf


torch.manual_seed(SEED)

print("Forward pass – real tokens…")
pre_real, pre_shuf, post_real, post_shuf = get_sims(ids)

print("Forward pass – content-shuffled tokens…")
content_shuf_data = get_sims(shuffled_ids)
# We only need the post-RoPE real from this run (content shuffled, positions sequential)
post_content_shuf = content_shuf_data[2]

print("Saving raw values to query_similarity_data.npz...")
np.savez(
    "query_similarity_data.npz",
    pre_real=np.stack([pre_real[l] for l in LAYERS]),
    pre_shuf=np.stack([pre_shuf[l] for l in LAYERS]),
    post_real=np.stack([post_real[l] for l in LAYERS]),
    post_content_shuf=np.stack([post_content_shuf[l] for l in LAYERS]),
    post_shuf=np.stack([post_shuf[l] for l in LAYERS]),
    layers=np.array(LAYERS),
    seq_len=np.array(ids.shape[1]),
)
print("Saved raw values → query_similarity_data.npz")

# ── Plot ──────────────────────────────────────────────────────────────────────
n_layers = len(LAYERS)
n_heads  = len(HEADS)

fig, axes = plt.subplots(
    n_heads, n_layers,
    figsize=(n_layers * 3, n_heads * 2.2),
    sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.30, "wspace": 0.15},
)

STYLES = [
    (pre_real,          "pre-RoPE real",              "steelblue", "-",  1.4),
    (pre_shuf,          "pre-RoPE Q-shuffled",        "steelblue", "--", 1.0),
    (post_real,         "post-RoPE real",             "tomato",    "-",  1.4),
    (post_content_shuf, "post-RoPE content-shuffled", "tomato",    "-.", 1.2),
    (post_shuf,         "post-RoPE Q-shuffled",       "tomato",    "--", 1.0),
]

for r, head_idx in enumerate(HEADS):
    for c, layer_idx in enumerate(LAYERS):
        ax = axes[r, c]

        for data, label, color, ls, lw in STYLES:
            vals = np.sort(data[layer_idx][head_idx])
            cdf  = np.arange(1, len(vals) + 1) / len(vals)
            ax.plot(vals, cdf, color=color, ls=ls, lw=lw, label=label)

        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.axvline(0, color="gray", lw=0.5, ls=":")
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)

        if r == 0:
            ax.set_title(f"Layer {layer_idx}", fontsize=9, pad=4)
        if c == 0:
            ax.set_ylabel(f"Head {head_idx}\nCDF", fontsize=8)
        if r == n_heads - 1:
            ax.set_xlabel("cos sim(Q[t], Q[t+1])", fontsize=7)

axes[0, 0].legend(fontsize=6.5, loc="upper left", framealpha=0.8)

fig.suptitle(
    "Adjacent query cosine similarity · pre/post-RoPE × real/Q-shuffled/content-shuffled\n"
    f"LLaMA 3.1-8B-Instruct  ·  {ids.shape[1]} tokens from WikiText-103",
    fontsize=10, y=1.01,
)

plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")

# ── Bar chart: mean adjacent cosine similarity per layer ──────────────────────
BAR_VARIANTS = [
    ("pre-RoPE real",              pre_real,          "#E6A817", ""),
    ("pre-RoPE shuffled",          pre_shuf,          "#E6A817", "//"),
    ("post-RoPE real",             post_real,         "#6B6B6B", ""),
    ("post-RoPE content-shuffled", post_content_shuf, "#6B6B6B", "//"),
    ("post-RoPE Q-shuffled",       post_shuf,         "#6B6B6B", "xx"),
]

fig2, axes2 = plt.subplots(1, n_layers, figsize=(n_layers * 2.5, 3.5), sharey=True,
                            gridspec_kw={"wspace": 0.10})

for c, (layer_idx, ax) in enumerate(zip(LAYERS, axes2)):
    means  = [v[layer_idx].mean() for _, v, _, _ in BAR_VARIANTS]
    colors = [col               for _, _, col, _ in BAR_VARIANTS]
    bars   = ax.bar(range(len(BAR_VARIANTS)), means, color=colors,
                    edgecolor="white", width=0.65)
    for bar, (_, _, _, h) in zip(bars, BAR_VARIANTS):
        bar.set_hatch(h)
    ax.set_xticks(range(len(BAR_VARIANTS)))
    ax.set_xticklabels([str(i+1) for i in range(len(BAR_VARIANTS))], fontsize=8)
    ax.set_title(f"Layer {layer_idx}", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    if c == 0:
        ax.set_ylabel("Mean adj. cos sim", fontsize=8)

legend_labels = [f"{i+1}. {name}" for i, (name, _, _, _) in enumerate(BAR_VARIANTS)]
handles = [plt.Rectangle((0,0), 1, 1, color=col, hatch=h, edgecolor="white")
           for _, _, col, h in BAR_VARIANTS]
fig2.legend(handles, legend_labels, loc="lower center", ncol=3,
            fontsize=7, framealpha=0.9, bbox_to_anchor=(0.5, -0.22))

fig2.suptitle("Mean adjacent query cosine similarity per layer (avg over all heads)", fontsize=9)
bar_path = "Paper/assets/query_similarity_mean_by_layer.png"
plt.savefig(bar_path, dpi=160, bbox_inches="tight")
print(f"Saved → {bar_path}")
