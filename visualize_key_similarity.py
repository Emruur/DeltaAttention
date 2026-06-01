"""
Visualise adjacent key-vector cosine similarity along the sequence dimension.

For each (layer, head) pair we show two thin vertical strips side-by-side:
  LEFT  – real wikitext token order
  RIGHT – same tokens but in a random permutation (shuffled)

If delta packing exploits genuine linguistic structure the real strips should
show bands of high similarity (adjacent tokens are semantically close),
while the shuffled strips should look flat / noisy.
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
# CONFIG  (edit here)
# ─────────────────────────────────────────────
MODEL_ID      = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SEQ_LEN       = 512
LAYERS        = [0, 8, 16, 23]   # transformer layers to visualise
HEADS         = [0, 2, 5, 7]     # KV-head indices to visualise
SEED          = 42
OUT_PATH      = "key_similarity.png"
# ─────────────────────────────────────────────

# ── Model registration ───────────────────────
AutoConfig.register("llama", LlamaConfig, exist_ok=True)
AutoModelForCausalLM.register(LlamaConfig, LlamaForCausalLM, exist_ok=True)

# Disable all delta machinery so we get clean key vectors
globVR.delta_pf_key_on = 0
globVR.flash           = False
globVR.delta_decode    = False

print("Loading model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model.eval()

num_kv_heads = model.config.num_key_value_heads
head_dim     = model.config.hidden_size // model.config.num_attention_heads

# ── Data ─────────────────────────────────────
print("Loading wikitext…")
ds   = load_dataset("wikitext", "wikitext-103-raw-v1", split="test", trust_remote_code=True)
text = " ".join(row["text"] for row in ds if row["text"].strip())
ids  = tokenizer(text, return_tensors="pt", truncation=True, max_length=SEQ_LEN).input_ids.cuda()

torch.manual_seed(SEED)
shuffled_ids = ids[:, torch.randperm(ids.shape[1])].cuda()

print(f"Sequence length: {ids.shape[1]} tokens")

# ── Forward-pass helper ───────────────────────
def get_adjacent_cosine_sims(input_ids):
    """Returns dict  layer_idx -> np.ndarray [seq-1, num_kv_heads]"""
    captured = {}
    hooks    = []

    for layer_idx in LAYERS:
        def _hook(module, inp, out, _idx=layer_idx):
            # out: [batch, seq, num_kv_heads * head_dim]
            k = out[0].float().reshape(-1, num_kv_heads, head_dim)  # [seq, h, d]
            k = F.normalize(k, dim=-1)
            # dot product of consecutive positions → [seq-1, h]
            captured[_idx] = (k[:-1] * k[1:]).sum(dim=-1).cpu().numpy()

        h = model.model.layers[layer_idx].self_attn.k_proj.register_forward_hook(_hook)
        hooks.append(h)

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    return captured


print("Forward pass – real tokens…")
real_sims = get_adjacent_cosine_sims(ids)

print("Forward pass – shuffled tokens…")
shuf_sims = get_adjacent_cosine_sims(shuffled_ids)

# ── Plot ──────────────────────────────────────
n_layers = len(LAYERS)
n_heads  = len(HEADS)

# Each head occupies 2 columns (real | shuffled), with a small gap between pairs
col_per_head = 2
total_cols   = n_heads * col_per_head

fig, axes = plt.subplots(
    n_layers, total_cols,
    figsize=(n_heads * 3, n_layers * 4),
    gridspec_kw={"wspace": 0.08, "hspace": 0.35},
)
if n_layers == 1:
    axes = axes[np.newaxis, :]

CMAP = "RdYlGn"
VMIN, VMAX = -1.0, 1.0

for r, layer_idx in enumerate(LAYERS):
    for c, head_idx in enumerate(HEADS):
        col_r = c * col_per_head
        col_s = c * col_per_head + 1

        real_vec = real_sims[layer_idx][:, head_idx, np.newaxis]   # [seq-1, 1]
        shuf_vec = shuf_sims[layer_idx][:, head_idx, np.newaxis]

        ax_r = axes[r, col_r]
        ax_s = axes[r, col_s]

        ax_r.imshow(real_vec, aspect="auto", cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
        ax_s.imshow(shuf_vec, aspect="auto", cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")

        for ax in (ax_r, ax_s):
            ax.set_xticks([])
            ax.set_yticks([])

        # column headers (top row only)
        if r == 0:
            ax_r.set_title(f"Head {head_idx}\nreal",    fontsize=8, pad=3)
            ax_s.set_title(f"Head {head_idx}\nshuffled", fontsize=8, pad=3)

        # row labels (first pair only)
        if c == 0:
            ax_r.set_ylabel(f"Layer {layer_idx}", fontsize=9)

        # subtle border to separate real/shuffled within a pair
        ax_s.spines["left"].set_linewidth(0.5)
        ax_s.spines["left"].set_color("gray")

# shared colour bar
sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(vmin=VMIN, vmax=VMAX))
sm.set_array([])
fig.colorbar(sm, ax=axes[:, -1], orientation="vertical",
             fraction=0.6, pad=0.04, label="cos sim with next token")

fig.suptitle(
    "Adjacent key-vector cosine similarity  ·  real vs shuffled token order\n"
    f"(LLaMA 3.1-8B-Instruct  ·  {ids.shape[1]} tokens from WikiText-103)",
    fontsize=11, y=1.02,
)

plt.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
print(f"Saved → {OUT_PATH}")
