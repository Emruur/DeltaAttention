"""
Compare pre-RoPE vs post-RoPE adjacent query cosine similarity distributions.

Same structure as visualize_key_similarity.py but for Q instead of K.
Pre-RoPE Q is captured from the q_proj hook; post-RoPE Q is computed by
applying the layer's own rotary embedding to the captured tensor.
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
from modeling_llama import LlamaForCausalLM, LlamaConfig

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SEQ_LEN  = 512
LAYERS   = [0, 8, 16, 23]
HEADS    = [0, 2, 5, 7]
SEED     = 42
OUT_PATH = "query_similarity.png"
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


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, cos, sin):
    """q: [h, seq, d]; cos/sin: [1, seq, d] → [h, seq, d]"""
    return q * cos + rotate_half(q) * sin


def adj_cos(q):
    """q: [num_heads, seq, head_dim] → [num_heads, seq-1]"""
    q = F.normalize(q.float(), dim=-1)
    return (q[:, :-1, :] * q[:, 1:, :]).sum(dim=-1).cpu().numpy()


def get_post_rope_q(layer_idx, q_pre, seq_len):
    """Apply the layer's rotary embedding to pre-RoPE q_pre [h, seq, d]."""
    rotary_emb = model.model.layers[layer_idx].self_attn.rotary_emb
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq]

    with torch.no_grad():
        dummy = q_pre.unsqueeze(0)  # [1, h, seq, d] — used for dtype/device
        try:
            cos, sin = rotary_emb(dummy, position_ids)   # [1, seq, d]
        except TypeError:
            cos, sin = rotary_emb(seq_len=seq_len, device=device, dtype=q_pre.dtype)
            cos = cos[:seq_len]   # [seq, d]
            sin = sin[:seq_len]
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)

    cos = cos.float().squeeze(0)  # [seq, d]
    sin = sin.float().squeeze(0)
    return apply_rope(q_pre.float(), cos, sin)  # [h, seq, d]


def get_sims(input_ids):
    pre_raw = {}
    hooks   = []
    seq_len = input_ids.shape[1]

    for layer_idx in LAYERS:
        def _hook(module, inp, out, _idx=layer_idx):
            q = out[0].float().reshape(-1, num_heads, head_dim)
            pre_raw[_idx] = q.permute(1, 0, 2).clone()  # [h, seq, d]
        h = model.model.layers[layer_idx].self_attn.q_proj.register_forward_hook(_hook)
        hooks.append(h)

    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)

    for h in hooks:
        h.remove()

    pre_real, pre_shuf, post_real, post_shuf = {}, {}, {}, {}
    for layer_idx in LAYERS:
        q_pre  = pre_raw[layer_idx]                          # [h, seq, d]
        q_post = get_post_rope_q(layer_idx, q_pre, seq_len)  # [h, seq, d]

        perm = torch.randperm(q_pre.shape[1])

        pre_real[layer_idx]  = adj_cos(q_pre)
        pre_shuf[layer_idx]  = adj_cos(q_pre[:, perm, :])
        post_real[layer_idx] = adj_cos(q_post)
        post_shuf[layer_idx] = adj_cos(q_post[:, perm, :])

    return pre_real, pre_shuf, post_real, post_shuf


torch.manual_seed(SEED)

print("Forward pass…")
pre_real, pre_shuf, post_real, post_shuf = get_sims(ids)

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
    (pre_real,  "pre-RoPE real",          "steelblue", "-",  1.4),
    (pre_shuf,  "pre-RoPE Q-shuffled",    "steelblue", "--", 1.0),
    (post_real, "post-RoPE real",         "tomato",    "-",  1.4),
    (post_shuf, "post-RoPE Q-shuffled",   "tomato",    "--", 1.0),
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
    "Adjacent query cosine similarity · pre/post-RoPE × real/Q-shuffled\n"
    f"LLaMA 3.1-8B-Instruct  ·  {ids.shape[1]} tokens from WikiText-103",
    fontsize=10, y=1.01,
)

plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
