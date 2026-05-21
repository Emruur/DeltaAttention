import torch
import triton
import triton.language as tl

import torch
import triton
import triton.language as tl

@triton.jit
def hybrid_compressed_flash_kernel(
    Q, K_dense, V_dense,
    K_packed, V_packed,
    Packed_Timestamps, Packed_Counts,
    Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kdb, stride_kdh, stride_kdn, stride_kdd,
    stride_vdb, stride_vdh, stride_vdn, stride_vdd,
    stride_kpb, stride_kph, stride_kpn, stride_kpd,
    stride_vpb, stride_vph, stride_vpn, stride_vpd,
    stride_ptb, stride_pth, stride_ptn,
    stride_pcb, stride_pch, stride_pcn,
    stride_ob, stride_oh, stride_om, stride_od,
    sm_scale,
    seq_len_q, seq_len_k, num_packed_keys, head_dim, num_heads,
    denseWindowSize,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    # -----------------------------------------------------------
    # 1. Setup Query Pointers
    # -----------------------------------------------------------
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)

    # Running state for Softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Base pointers for packed and dense data
    k_pack_base = K_packed + batch_idx * stride_kpb + head_idx * stride_kph
    v_pack_base = V_packed + batch_idx * stride_vpb + head_idx * stride_vph
    pt_base = Packed_Timestamps + batch_idx * stride_ptb + head_idx * stride_pth
    pc_base = Packed_Counts + batch_idx * stride_pcb + head_idx * stride_pch

    k_dense_base = K_dense + batch_idx * stride_kdb + head_idx * stride_kdh
    v_dense_base = V_dense + batch_idx * stride_vdb + head_idx * stride_vdh

    # The boundary between Phase 1 (compressed) and Phase 2 (exact dense).
    # Phase 2 covers exactly [target_boundary, query_pos].
    target_boundary = start_m - denseWindowSize

    # ===========================================================
    # PHASE 1: COMPRESSED HISTORY LOOP
    # ===========================================================
    # For each packed anchor, we clip its count to only the portion of the
    # original sequence that falls before target_boundary.
    # Since sum(packed_counts) == seq_len, target_boundary - timestamp gives
    # exactly how many original tokens this anchor covers before the boundary.
    # This prevents Phase 2 from ballooning when compression is aggressive
    # and packed timestamps are sparse.
    start_n = 0

    while start_n < num_packed_keys:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        packed_mask = offs_n < num_packed_keys

        timestamps = tl.load(pt_base + offs_n * stride_ptn, mask=packed_mask, other=seq_len_k + 1)
        counts = tl.load(pc_base + offs_n * stride_pcn, mask=packed_mask, other=0.0)

        # Clip each anchor's count to only the tokens it represents before the boundary.
        # Anchors at or past target_boundary get effective_count = 0.
        effective_counts = tl.minimum(counts, tl.maximum(0.0, target_boundary - timestamps.to(tl.float32)))
        effective_counts = tl.where(packed_mask, effective_counts, 0.0)

        # Packed timestamps are sorted; once a block has no contribution, neither
        # will any subsequent block.
        if tl.sum(effective_counts) <= 0.0:
            start_n = num_packed_keys
        else:
            # --- SAFE PACKED MATH ---
            k_ptrs = k_pack_base + offs_n[None, :] * stride_kpn + offs_d[:, None] * stride_kpd
            k_pack = tl.load(k_ptrs, mask=(packed_mask[None, :]) & (offs_d[:, None] < head_dim), other=0.0)

            qk = tl.dot(q, k_pack) * sm_scale
            # Exclude zero-effective-count anchors from softmax numerics.
            qk = tl.where((effective_counts[None, :] > 0.0) & packed_mask[None, :], qk, float("-inf"))

            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp(qk - m_ij[:, None])

            p_weighted = p * effective_counts[None, :]

            l_ij = tl.sum(p_weighted, 1)
            alpha = tl.math.exp(m_i - m_ij)

            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]

            v_ptrs = v_pack_base + offs_n[:, None] * stride_vpn + offs_d[None, :] * stride_vpd
            v_pack = tl.load(v_ptrs, mask=(packed_mask[:, None]) & (offs_d[None, :] < head_dim), other=0.0)

            acc += tl.dot(p_weighted.to(v_pack.dtype), v_pack)
            m_i = m_ij

            start_n += BLOCK_N

    # ===========================================================
    # PHASE 2: EXACT DENSE LOCAL WINDOW
    # ===========================================================
    # Always starts at exactly target_boundary in the original sequence,
    # covering exactly denseWindowSize tokens regardless of packing sparsity.
    dense_start_n = tl.maximum(0, start_m - denseWindowSize)
    dense_hi = tl.minimum(seq_len_k, start_m + BLOCK_M)

    for start_n in range(dense_start_n, dense_hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < seq_len_k

        k_ptrs = k_dense_base + offs_n[None, :] * stride_kdn + offs_d[:, None] * stride_kdd
        k_dense = tl.load(k_ptrs, mask=(k_mask[None, :]) & (offs_d[:, None] < head_dim), other=0.0)

        qk = tl.dot(q, k_dense) * sm_scale

        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & k_mask[None, :], qk, float("-inf"))

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])

        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp(m_i - m_ij)

        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]

        v_ptrs = v_dense_base + offs_n[:, None] * stride_vdn + offs_d[None, :] * stride_vdd
        v_dense = tl.load(v_ptrs, mask=(k_mask[:, None]) & (offs_d[None, :] < head_dim), other=0.0)

        acc += tl.dot(p.to(v_dense.dtype), v_dense)
        m_i = m_ij

    # -----------------------------------------------------------
    # 3. Finalize and Write Output
    # -----------------------------------------------------------
    acc = acc / l_i[:, None]

    out_ptrs = Out + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))
