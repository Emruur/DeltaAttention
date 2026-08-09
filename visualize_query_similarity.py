"""
Compute pre-RoPE vs post-RoPE adjacent query cosine similarity distributions
and dump raw values to query_similarity_data.npz (no plotting).

Pre-RoPE Q: hooked from q_proj output.
Post-RoPE Q: captured by monkey-patching apply_rotary_pos_emb during forward.

Runs against stock HuggingFace transformers (venv_xattn) — no dependency on
DeltaLLM's modeling_llama / globVR. The similarity analysis does not use any
delta-attention machinery, so a plain LlamaForCausalLM is sufficient.
"""

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import modeling_llama as hf_llama

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SEQ_LEN  = 512
LAYERS   = [0, 8, 16, 23]
SEED     = 42
OUT_NPZ  = "query_similarity_data.npz"
# ─────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading model…")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="eager"
).to(device)
model.eval()

num_heads    = model.config.num_attention_heads
num_kv_heads = model.config.num_key_value_heads
head_dim     = getattr(model.config, "head_dim", None) or model.config.hidden_size // num_heads

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

    # post-RoPE Q by patching apply_rotary_pos_emb in transformers' llama module
    _orig_rope = hf_llama.apply_rotary_pos_emb

    def _capturing_rope(q, k, cos, sin, *args, **kwargs):
        q_rot, k_rot = _orig_rope(q, k, cos, sin, *args, **kwargs)
        captured_qs.append(q_rot.detach().float())  # [1, heads, seq, d]
        return q_rot, k_rot

    hf_llama.apply_rotary_pos_emb = _capturing_rope

    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)

    hf_llama.apply_rotary_pos_emb = _orig_rope
    for h in hooks:
        h.remove()

    # captured_qs[i] = post-RoPE Q for layer i (all layers in order)
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

print(f"Saving raw values to {OUT_NPZ}...")
np.savez(
    OUT_NPZ,
    pre_real=np.stack([pre_real[l] for l in LAYERS]),
    pre_shuf=np.stack([pre_shuf[l] for l in LAYERS]),
    post_real=np.stack([post_real[l] for l in LAYERS]),
    post_content_shuf=np.stack([post_content_shuf[l] for l in LAYERS]),
    post_shuf=np.stack([post_shuf[l] for l in LAYERS]),
    layers=np.array(LAYERS),
    seq_len=np.array(ids.shape[1]),
)
print(f"Saved raw values → {OUT_NPZ}")
