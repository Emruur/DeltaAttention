import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd_flat(
    Q, K, V, sm_scale, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    IS_CAUSAL: tl.constexpr
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    off_z = off_hz // H
    off_h = off_hz % H

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)

    # Calculate pointers using standard, safe arithmetic
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_kk
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vk + offs_k[None, :] * stride_vn
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_k[None, :] * stride_on

    # Initialize accumulators
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Load Q with safe boundary mask
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Determine inner loop limit based on causality
    if IS_CAUSAL:
        hi = tl.minimum(N_CTX, (start_m + 1) * BLOCK_M)
    else:
        hi = N_CTX

    # Inner loop over Key/Value blocks
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K and V with safe boundary masks
        k_mask = offs_n[None, :] + start_n < N_CTX
        k = tl.load(k_ptrs + start_n * stride_kn, mask=k_mask, other=0.0)
        
        v_mask = offs_n[:, None] + start_n < N_CTX
        v = tl.load(v_ptrs + start_n * stride_vk, mask=v_mask, other=0.0)

        # Compute QK dot product
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)
        qk *= sm_scale

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Flash Attention math
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])

        # Mask out-of-bounds keys so they don't affect the sum
        p = tl.where(offs_n[None, :] + start_n < N_CTX, p, 0.0)

        l_ij = tl.sum(p, 1)

        # Update accumulators
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # Dynamic cast to match V's dtype (handles float16 or bfloat16 automatically)
        p = p.to(v.dtype)
        acc += tl.dot(p, v)
        m_i = m_ij

    # Write back output
    acc = acc / l_i[:, None]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def triton_flash_attention(q, k, v, causal=True, sm_scale=None):
    """
    Clean inference wrapper. No padding required!
    """
    batch_size, num_heads, q_len, head_dim = q.shape

    # Triton kernels demand contiguous memory
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim ** 0.5)

    BLOCK_M = 128
    BLOCK_N = 128

    out = torch.empty_like(q)

    grid = (triton.cdiv(q_len, BLOCK_M), batch_size * num_heads, 1)

    _attn_fwd_flat[grid](
        q, k, v, sm_scale, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        batch_size, num_heads, q_len,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        num_warps=8,
        num_stages=3,
    )
    
    return out


