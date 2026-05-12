import torch
import triton
import triton.language as tl


@triton.jit
def hybrid_compressed_flash_kernel(
    Q, K_dense, V_dense,
    K_packed, V_packed,
    Packed_Counts,
    Chunk_Offsets,
    Chunk_Counts,
    Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kdb, stride_kdh, stride_kdn, stride_kdd,
    stride_vdb, stride_vdh, stride_vdn, stride_vdd,
    stride_kpb, stride_kph, stride_kpn, stride_kpd,
    stride_vpb, stride_vph, stride_vpn, stride_vpd,
    stride_pcb, stride_pch, stride_pcn,
    stride_cob, stride_coh, stride_con,
    stride_ccb, stride_cch, stride_ccn,
    stride_ob, stride_oh, stride_om, stride_od,
    sm_scale,
    seq_len_q, seq_len_k, num_chunks, head_dim, num_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # Each program handles one query block (pid_m) for one (batch, head) pair.
    # Inner loop iterates over dense chunk indices:
    #   chunks 0..pid_m-1  → compressed (load packed K/V for that chunk)
    #   chunk  pid_m       → exact diagonal (load dense K/V, apply causal mask)
    # chunk_size == BLOCK_N is required so each flash key block maps to exactly
    # one delta-compression chunk, preventing anchor bleed across blocks.
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = (Q + batch_idx * stride_qb + head_idx * stride_qh
              + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs,
                mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim),
                other=0.0)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    k_pack_base = K_packed + batch_idx * stride_kpb + head_idx * stride_kph
    v_pack_base = V_packed + batch_idx * stride_vpb + head_idx * stride_vph
    pc_base     = Packed_Counts + batch_idx * stride_pcb + head_idx * stride_pch
    co_base     = Chunk_Offsets + batch_idx * stride_cob + head_idx * stride_coh
    cc_base     = Chunk_Counts  + batch_idx * stride_ccb + head_idx * stride_cch
    k_dense_base = K_dense + batch_idx * stride_kdb + head_idx * stride_kdh
    v_dense_base = V_dense + batch_idx * stride_vdb + head_idx * stride_vdh

    # ===========================================================
    # PHASE 1: COMPRESSED HISTORY
    # All key chunks strictly before the diagonal (c < pid_m).
    # Every token in these chunks causally precedes every query in
    # this query block, so no causal masking is needed here.
    # ===========================================================
    for c in range(0, pid_m):
        chunk_offset = tl.load(co_base + c * stride_con)   # start in packed array
        chunk_count  = tl.load(cc_base + c * stride_ccn)   # #packed keys for chunk c

        offs_p = tl.arange(0, BLOCK_N)
        packed_valid = offs_p < chunk_count
        global_p = chunk_offset + offs_p

        # K packed: shape [BLOCK_D, BLOCK_N] for tl.dot(q, k_pack)
        k_ptrs = (k_pack_base + global_p[None, :] * stride_kpn
                  + offs_d[:, None] * stride_kpd)
        k_pack = tl.load(k_ptrs,
                         mask=packed_valid[None, :] & (offs_d[:, None] < head_dim),
                         other=0.0)

        qk = tl.dot(q, k_pack) * sm_scale
        qk = tl.where(packed_valid[None, :], qk, float("-inf"))

        # Per-entry count: how many original tokens this packed slot represents.
        counts = tl.load(pc_base + global_p * stride_pcn,
                         mask=packed_valid, other=0.0)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])
        p_weighted = p * counts[None, :]

        l_ij = tl.sum(p_weighted, 1)
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        # V packed: shape [BLOCK_N, BLOCK_D]
        v_ptrs = (v_pack_base + global_p[:, None] * stride_vpn
                  + offs_d[None, :] * stride_vpd)
        v_pack = tl.load(v_ptrs,
                         mask=packed_valid[:, None] & (offs_d[None, :] < head_dim),
                         other=0.0)

        acc += tl.dot(p_weighted.to(v_pack.dtype), v_pack)
        m_i = m_ij

    # ===========================================================
    # PHASE 2: EXACT DIAGONAL
    # The key chunk aligned with this query block (c == pid_m).
    # Uses full dense K/V and a standard causal mask.
    # ===========================================================
    start_n = pid_m * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    k_mask = offs_n < seq_len_k

    k_ptrs = (k_dense_base + offs_n[None, :] * stride_kdn
              + offs_d[:, None] * stride_kdd)
    k_dense = tl.load(k_ptrs,
                      mask=k_mask[None, :] & (offs_d[:, None] < head_dim),
                      other=0.0)

    qk = tl.dot(q, k_dense) * sm_scale
    causal_mask = offs_m[:, None] >= offs_n[None, :]
    qk = tl.where(causal_mask & k_mask[None, :], qk, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(qk, 1))
    p = tl.math.exp(qk - m_ij[:, None])
    l_ij = tl.sum(p, 1)
    alpha = tl.math.exp(m_i - m_ij)
    l_i = l_i * alpha + l_ij
    acc = acc * alpha[:, None]

    v_ptrs = (v_dense_base + offs_n[:, None] * stride_vdn
              + offs_d[None, :] * stride_vdd)
    v_dense_block = tl.load(v_ptrs,
                            mask=k_mask[:, None] & (offs_d[None, :] < head_dim),
                            other=0.0)

    acc += tl.dot(p.to(v_dense_block.dtype), v_dense_block)

    # Finalize
    acc = acc / l_i[:, None]
    out_ptrs = (Out + batch_idx * stride_ob + head_idx * stride_oh
                + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty),
             mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))
