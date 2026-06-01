"""
Compute adjacent key cosine similarities and save raw values to .npz.

Produces 4 variants per (layer, head):
  pre_real  – pre-RoPE keys, natural token order
  pre_shuf  – pre-RoPE keys, sequence shuffled
  post_real – post-RoPE keys, natural token order
  post_shuf – post-RoPE keys, sequence shuffled

Output: key_similarity_data.npz
  Arrays of shape [num_layers, num_kv_heads, seq-1]
  Plus metadata: LAYERS, HEADS, SEQ_LEN, MODEL_ID
"""

import os, sys
import torch
import torch.nn.functional as F
import numpy as np
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
SEED     = 42
OUT_PATH = "key_similarity_data.npz"
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
print(f"Sequence length: {ids.shape[1]} tokens")


def extract_layer_keys(past_key_values, layer_idx):
    if hasattr(past_key_values, "key_cache"):
        return past_key_values.key_cache[layer_idx]
    return past_key_values[layer_idx][0]


def adj_cos(k):
    """k: [num_kv_heads, seq, head_dim] → [num_kv_heads, seq-1]"""
    k = F.normalize(k.float(), dim=-1)
    return (k[:, :-1, :] * k[:, 1:, :]).sum(dim=-1).cpu().numpy()


def compute_all(input_ids, seed):
    pre_raw = {}
    hooks   = []

    for layer_idx in LAYERS:
        def _hook(module, inp, out, _idx=layer_idx):
            k = out[0].float().reshape(-1, num_kv_heads, head_dim)
            pre_raw[_idx] = k.permute(1, 0, 2).clone()  # [h, seq, d]
        h = model.model.layers[layer_idx].self_attn.k_proj.register_forward_hook(_hook)
        hooks.append(h)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)

    for h in hooks:
        h.remove()

    torch.manual_seed(seed)
    results = {"pre_real": [], "pre_shuf": [], "post_real": [], "post_shuf": []}

    for layer_idx in LAYERS:
        k_pre  = pre_raw[layer_idx]
        k_post = extract_layer_keys(outputs.past_key_values, layer_idx)[0].float()
        perm   = torch.randperm(k_pre.shape[1])

        results["pre_real"].append(adj_cos(k_pre))
        results["pre_shuf"].append(adj_cos(k_pre[:, perm, :]))
        results["post_real"].append(adj_cos(k_post))
        results["post_shuf"].append(adj_cos(k_post[:, perm, :]))

    # stack to [num_layers, num_kv_heads, seq-1]
    return {k: np.stack(v) for k, v in results.items()}


torch.manual_seed(SEED)
print("Forward pass – real tokens…")
data = compute_all(ids, seed=SEED)

# 5th variant: shuffle input tokens (random content), keep sequential positions
# → RoPE still applied at positions 0,1,2,... so positional adjacency is preserved
# → content adjacency is destroyed
torch.manual_seed(SEED)
shuffled_ids = ids[:, torch.randperm(ids.shape[1])]
print("Forward pass – content-shuffled tokens (pre-shuffle → RoPE)…")
content_shuf_data = compute_all(shuffled_ids, seed=SEED + 1)
# we only need the post-RoPE real from this run (content shuffled, positions sequential)
post_content_shuf = content_shuf_data["post_real"]

np.savez(
    OUT_PATH,
    pre_real          = data["pre_real"],
    pre_shuf          = data["pre_shuf"],
    post_real         = data["post_real"],
    post_shuf         = data["post_shuf"],
    post_content_shuf = post_content_shuf,   # content shuffled, RoPE on sequential positions
    layers            = np.array(LAYERS),
    seq_len           = np.array(ids.shape[1]),
)

print(f"Saved → {OUT_PATH}")
print(f"Array shape: {data['pre_real'].shape}  (layers, kv_heads, seq-1)")
