"""
Compare pre-RoPE vs post-RoPE adjacent key cosine similarity distributions.

Four CDFs per (layer, head) cell:
  pre-RoPE  real     – keys before positional encoding, natural order
  pre-RoPE  shuffled – keys before positional encoding, random order
  post-RoPE real     – keys after positional encoding, natural order
  post-RoPE shuffled – keys after positional encoding, random order

If RoPE is the driving factor for adjacent-token similarity structure,
pre-RoPE real vs shuffled should overlap, while post-RoPE real should
diverge (heavier low-similarity tail = more linguistic boundaries).
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
OUT_PATH = "key_similarity.png"
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

num_kv_heads = model.config.num_key_value_heads
head_dim     = model.config.hidden_size // model.config.num_attention_heads

print("Loading wikitext…")
ds   = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
text = " ".join(row["text"] for row in ds if row["text"].strip())
ids  = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN).input_ids.to(device)

torch.manual_seed(SEED)
shuffled_ids = ids[:, torch.randperm(ids.shape[1])].to(device)

print(f"Sequence length: {ids.shape[1]} tokens")


def extract_layer_keys(past_key_values, layer_idx):
    """Handle both DynamicCache and legacy tuple formats."""
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx]  # [1, num_kv_heads, seq, d]
    return past_key_values[layer_idx][0]


def adj_cos(k):
    """k: [num_kv_heads, seq, head_dim] → [num_kv_heads, seq-1]"""
    k = F.normalize(k.float(), dim=-1)
    return (k[:, :-1, :] * k[:, 1:, :]).sum(dim=-1).cpu().numpy()


def get_sims(input_ids):
    """
    Returns (pre, post) where each is dict layer_idx -> [num_kv_heads, seq-1].
    pre  = k_proj output (before RoPE)
    post = KV-cache keys (after RoPE)
    """
    pre_captured = {}
    hooks = []

    for layer_idx in LAYERS:
        def _hook(module, inp, out, _idx=layer_idx):
            # out: [batch, seq, num_kv_heads * head_dim]
            k = out[0].float().reshape(-1, num_kv_heads, head_dim)  # [seq, h, d]
            k = k.permute(1, 0, 2)                                   # [h, seq, d]
            pre_captured[_idx] = adj_cos(k)

        h = model.model.layers[layer_idx].self_attn.k_proj.register_forward_hook(_hook)
        hooks.append(h)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)

    for h in hooks:
        h.remove()

    post_captured = {}
    for layer_idx in LAYERS:
        k = extract_layer_keys(outputs.past_key_values, layer_idx)
        post_captured[layer_idx] = adj_cos(k[0])  # [num_kv_heads, seq-1]

    return pre_captured, post_captured


print("Forward pass – real tokens…")
pre_real, post_real = get_sims(ids)

print("Forward pass – shuffled tokens…")
pre_shuf, post_shuf = get_sims(shuffled_ids)

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
    (pre_real,  "pre  real",     "steelblue", "-",  1.4),
    (pre_shuf,  "pre  shuffled", "steelblue", "--", 1.0),
    (post_real, "post real",     "tomato",    "-",  1.4),
    (post_shuf, "post shuffled", "tomato",    "--", 1.0),
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
            ax.set_xlabel("cos sim(K[t], K[t+1])", fontsize=7)

axes[0, 0].legend(fontsize=6.5, loc="upper left", framealpha=0.8)

fig.suptitle(
    "Adjacent key cosine similarity · pre-RoPE vs post-RoPE · real vs shuffled\n"
    f"LLaMA 3.1-8B-Instruct  ·  {ids.shape[1]} tokens from WikiText-103",
    fontsize=10, y=1.01,
)

plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
