import torch
import triton
import triton.language as tl


@triton.jit
def hybrid_compressed_flash_kernel(
    Q, K_dense, V_dense,
    K_packed, V_packed,
    Packed_Counts,
    Packed_Block_Limits,   # [bsz*num_heads, num_q_blocks]  int32
    Dense_Starts,          # [bsz*num_heads, num_q_blocks]  int32
    Out,
    stride_qb,  stride_qh,  stride_qm,  stride_qd,
    stride_kdb, stride_kdh, stride_kdn, stride_kdd,
    stride_vdb, stride_vdh, stride_vdn, stride_vdd,
    stride_kpb, stride_kph, stride_kpn, stride_kpd,
    stride_vpb, stride_vph, stride_vpn, stride_vpd,
    stride_pcb, stride_pch, stride_pcn,
    num_q_blocks,
    stride_ob,  stride_oh,  stride_om,  stride_od,
    sm_scale,
    seq_len_q, seq_len_k, num_packed_keys, head_dim, num_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx  = pid_bh %  num_heads
    start_m   = pid_m * BLOCK_M

    # ----------------------------------------------------------------
    # Load precomputed per-CTA loop bounds
    # ----------------------------------------------------------------
    my_packed_limit = tl.load(Packed_Block_Limits + pid_bh * num_q_blocks + pid_m)
    my_dense_start  = tl.load(Dense_Starts        + pid_bh * num_q_blocks + pid_m)

    # ----------------------------------------------------------------
    # Query tile and running softmax state
    # ----------------------------------------------------------------
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (Q
              + batch_idx * stride_qb
              + head_idx  * stride_qh
              + offs_m[:, None] * stride_qm
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs,
                mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim),
                other=0.0)

    m_i  = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_pack_base = K_packed + batch_idx * stride_kpb + head_idx * stride_kph
    v_pack_base = V_packed + batch_idx * stride_vpb + head_idx * stride_vph
    pc_base     = Packed_Counts + batch_idx * stride_pcb + head_idx * stride_pch

    k_dense_base = K_dense + batch_idx * stride_kdb + head_idx * stride_kdh
    v_dense_base = V_dense + batch_idx * stride_vdb + head_idx * stride_vdh

    # ==================================================================
    # PHASE 1: Compressed history — static for loop, fully pipelineable.
    # my_packed_limit is a multiple of BLOCK_N, so every block is full;
    # no boundary mask or causal check needed here.
    # ==================================================================
    for start_n in range(0, my_packed_limit, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        k_ptrs = k_pack_base + offs_n[None, :] * stride_kpn + offs_d[:, None] * stride_kpd
        k_pack = tl.load(k_ptrs, mask=offs_d[:, None] < head_dim, other=0.0)

        qk = tl.dot(q, k_pack) * sm_scale

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p    = tl.math.exp(qk - m_ij[:, None])

        counts     = tl.load(pc_base + offs_n * stride_pcn)
        p_weighted = p * counts[None, :]

        l_ij  = tl.sum(p_weighted, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i   = l_i * alpha + l_ij
        acc   = acc * alpha[:, None]

        v_ptrs = v_pack_base + offs_n[:, None] * stride_vpn + offs_d[None, :] * stride_vpd
        v_pack = tl.load(v_ptrs, mask=offs_d[None, :] < head_dim, other=0.0)

        acc += tl.dot(p_weighted.to(v_pack.dtype), v_pack)
        m_i  = m_ij

    # ==================================================================
    # PHASE 2: Exact dense local window.
    # Starts where the last safe packed block ended (precomputed).
    # ==================================================================
    dense_hi = tl.minimum(seq_len_k, start_m + BLOCK_M)

    for start_n in range(my_dense_start, dense_hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < seq_len_k

        k_ptrs  = k_dense_base + offs_n[None, :] * stride_kdn + offs_d[:, None] * stride_kdd
        k_dense = tl.load(k_ptrs,
                          mask=k_mask[None, :] & (offs_d[:, None] < head_dim),
                          other=0.0)

        qk          = tl.dot(q, k_dense) * sm_scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk          = tl.where(causal_mask & k_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p    = tl.math.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp(m_i - m_ij)
        l_i   = l_i * alpha + l_ij
        acc   = acc * alpha[:, None]

        v_ptrs  = v_dense_base + offs_n[:, None] * stride_vdn + offs_d[None, :] * stride_vdd
        v_dense = tl.load(v_ptrs,
                          mask=k_mask[:, None] & (offs_d[None, :] < head_dim),
                          other=0.0)

        acc += tl.dot(p.to(v_dense.dtype), v_dense)
        m_i  = m_ij

    # ----------------------------------------------------------------
    # Finalise and write output
    # ----------------------------------------------------------------
    acc = acc / l_i[:, None]

    out_ptrs = (Out
                + batch_idx * stride_ob
                + head_idx  * stride_oh
                + offs_m[:, None] * stride_om
                + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))
