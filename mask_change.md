# Causal Mask Change — `_update_causal_mask` in `modeling_llama.py`

## What changed

Added an early-exit at the top of `LlamaModel._update_causal_mask` (line 943):

```python
def _update_causal_mask(self, attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
    # NEW: early exit added during eval_speedup session
    if getattr(globVR, 'skip_causal_mask', False) and input_tensor.shape[1] > 1:
        return None
    # ... rest of original code unchanged ...
```

The flag is set in `eval_speedup.py` inside `set_row_delta()`:

```python
def set_row_delta():
    ...
    globVR.skip_causal_mask = True  # <-- triggers the early exit
```

## Why it was added

At N=131K, the standard path in `_update_causal_mask` calls
`_prepare_4d_causal_attention_mask_with_cache_position`, which allocates a
`[batch, 1, N, N]` bf16 tensor. At 131K tokens:

```
1 × 1 × 131072 × 131072 × 2 bytes ≈ 34 GB
```

This alone OOMs an 80 GB H100. There was also a `.clone()` somewhere in the
attention path that doubled it to ~68 GB. Skipping the mask creation entirely
avoids both allocations.

## Why it is safe for `eval_speedup.py`

`eval_speedup.py` always runs with:
- `batch_size = 1`
- No padding (real WikiText tokens, no pad tokens)
- Full-sequence prefill (`use_cache=False`)

Under these conditions the causal mask is only needed for two things:
1. **Causality** — the attention implementation handles this internally when
   `attn_mask=None` is passed (eager uses `attn_bias=None` path, SDPA uses
   `is_causal=True`).
2. **Padding suppression** — not needed with batch_size=1 and no padding.

So `return None` is functionally correct here.

## Why it breaks accuracy in `eval_all.py`

`eval_all.py` runs `lm_eval`, which uses **batched, padded inputs**
(batch_size > 1). In that setting the 4D causal mask serves a second purpose:
**masking out pad tokens**. The standard mask generation encodes the pad
positions as `-inf` so attention scores from real tokens to pad positions become
zero after softmax.

When `skip_causal_mask=True` causes `_update_causal_mask` to return `None`,
the attention layers receive no mask at all. Real tokens then attend to pad
tokens, corrupting the output and lowering accuracy.

**`eval_all.py` never sets `globVR.skip_causal_mask`, so in theory this flag
should be `False` there and the early-exit should never trigger.**

## How to investigate the accuracy drop

If accuracy is still dropping in `eval_all.py` after this change, check:

1. **Is `skip_causal_mask` somehow `True` during eval_all?**
   Add a debug print at the top of `_update_causal_mask`:
   ```python
   if getattr(globVR, 'skip_causal_mask', False):
       print(f"[DEBUG] skip_causal_mask=True, seq={input_tensor.shape[1]}", flush=True)
   ```
   If you see this print during `eval_all.py`, something is setting the flag.

2. **Is the accuracy drop only for the row_delta condition, or also baseline?**
   - If baseline (no delta) also drops → the mask change itself is wrong even
     when the flag is `False`, which shouldn't happen.
   - If only row_delta drops → the flag IS being set somewhere during eval_all,
     OR the attention path with `attn_mask=None` behaves differently from
     before in a way unrelated to the flag.

3. **Check the attention forward path when `attn_mask=None`.**
   When `_update_causal_mask` returns `None`, the value reaches
   `LlamaAttention.forward` as `attention_mask=None`. Trace what happens:
   - In the eager path: does it default to causal? Check `scaled_dot_product_attention` call.
   - In the row-delta Triton path: does the kernel enforce causality? Look at
     the causal mask inside `hybrid_compressed_flash_kernel` — it applies
     `causal_mask = offs_m[:, None] >= offs_n[None, :]` only in the dense
     window phase, which IS causal. But if padding positions are in the dense
     window they will be attended to.

4. **Check `_forward_row_delta` when `causal_mask` is `None`.**
   The row-delta forward likely passes `causal_mask` to the standard eager
   attention for the "non-delta" tokens. If that path receives `None` and
   doesn't apply causality itself, you get incorrect attention for those tokens
   too.

## The cleanest fix

Instead of `return None` unconditionally, only skip the **allocation** but
still return a minimal causal mask for the batch_size>1 case:

```python
if getattr(globVR, 'skip_causal_mask', False) and input_tensor.shape[1] > 1:
    if input_tensor.shape[0] == 1:   # single sequence, no padding possible
        return None
    # fall through to normal path for batched inputs
```

This makes the flag safe even if it is accidentally set during eval_all.
