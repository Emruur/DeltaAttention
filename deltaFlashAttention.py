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
    Packed_Timestamps, Packed_Counts, # The crucial metadata
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

    # Track where we are in physical time to ensure seamless handoff
    last_safe_physical_timestamp = -1

    # ===========================================================
    # PHASE 1: COMPRESSED HISTORY LOOP
    # ===========================================================
    last_safe_physical_timestamp = -1
    start_n = 0
    
    while start_n < num_packed_keys:
        offs_n = start_n + tl.arange(0, BLOCK_N)
        packed_mask = offs_n < num_packed_keys
        
        timestamps = tl.load(pt_base + offs_n * stride_ptn, mask=packed_mask, other=seq_len_k + 1)
        max_timestamp_in_block = tl.max(timestamps)
        
        if max_timestamp_in_block >= start_m:
            # We hit the critical diagonal boundary!
            # To "break" in Triton, we just force the loop counter past the boundary.
            start_n = num_packed_keys 
        else:
            # --- SAFE PACKED MATH ---
            k_ptrs = k_pack_base + offs_n[None, :] * stride_kpn + offs_d[:, None] * stride_kpd
            k_pack = tl.load(k_ptrs, mask=(packed_mask[None, :]) & (offs_d[:, None] < head_dim), other=0.0)
            
            qk = tl.dot(q, k_pack) * sm_scale
            qk = tl.where(packed_mask[None, :], qk, float("-inf"))
            
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp(qk - m_ij[:, None])
            
            counts = tl.load(pc_base + offs_n * stride_pcn, mask=packed_mask, other=0.0)
            p_weighted = p * counts[None, :]
            
            l_ij = tl.sum(p_weighted, 1)
            alpha = tl.math.exp(m_i - m_ij)
            
            l_i = l_i * alpha + l_ij
            acc = acc * alpha[:, None]
            
            v_ptrs = v_pack_base + offs_n[:, None] * stride_vpn + offs_d[None, :] * stride_vpd
            v_pack = tl.load(v_ptrs, mask=(packed_mask[:, None]) & (offs_d[None, :] < head_dim), other=0.0)
            
            acc += tl.dot(p_weighted.to(v_pack.dtype), v_pack)
            m_i = m_ij
            
            last_safe_physical_timestamp = max_timestamp_in_block
            
            # Manually increment the loop counter
            start_n += BLOCK_N
    # ===========================================================
    # PHASE 2: EXACT DENSE LOCAL WINDOW
    # ===========================================================
    # Start dense loop precisely where the packed loop left off.
    # Align to nearest block for optimal memory reads.
    dense_start_n = ((last_safe_physical_timestamp + 1) // BLOCK_N) * BLOCK_N
    dense_start_n = tl.maximum(0, dense_start_n)
    
    # We only need to go up to the end of the current query block
    dense_hi = tl.minimum(seq_len_k, start_m + BLOCK_M)
    
    for start_n in range(dense_start_n, dense_hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < seq_len_k
        
        k_ptrs = k_dense_base + offs_n[None, :] * stride_kdn + offs_d[:, None] * stride_kdd
        k_dense = tl.load(k_ptrs, mask=(k_mask[None, :]) & (offs_d[:, None] < head_dim), other=0.0)
        
        qk = tl.dot(q, k_dense) * sm_scale
        
        # Standard Physical Causal Masking
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & k_mask[None, :], qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])
        
        # NO COUNTS HERE. Exact 1-to-1 mapping.
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

@triton.jit
def absolute_flash_kernel(
    Q, K_dense, K_packed, V_dense, Index_Map, Keep_Mask, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kdb, stride_kdh, stride_kdn, stride_kdd,
    stride_kpb, stride_kph, stride_kpn, stride_kpd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_imb, stride_imh, stride_ims,
    stride_kb, stride_kh, stride_kn,
    stride_ob, stride_oh, stride_om, stride_od,
    sm_scale, blk_size, seq_len_q, seq_len_k, head_dim, num_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)
    
    out_ptrs = Out + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    
    k_dense_base = K_dense + batch_idx * stride_kdb + head_idx * stride_kdh
    k_packed_base = K_packed + batch_idx * stride_kpb + head_idx * stride_kph
    v_dense_base = V_dense + batch_idx * stride_vb + head_idx * stride_vh
    im_base = Index_Map + batch_idx * stride_imb + head_idx * stride_imh
    km_base = Keep_Mask + batch_idx * stride_kb + head_idx * stride_kh
    
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    # ==========================================
    # --- THE MAGIC FP32 STATE VARIABLES ---
    # ==========================================
    running_score = tl.zeros([BLOCK_M], dtype=tl.float32)
    prev_anchor_idx = -1
    
    q_start_bin = start_m // blk_size
    hi = tl.minimum(seq_len_k, (pid_m + 1) * BLOCK_M)
    
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < seq_len_k
        
        local_im = tl.load(im_base + offs_n * stride_ims, mask=k_mask, other=-1)
        current_anchor_idx = tl.max(local_im)
        
        k_end_bin = (start_n + BLOCK_N - 1) // blk_size
        is_diagonal = (k_end_bin >= q_start_bin)
        
        if is_diagonal:
            # --- PATH A: EXACT DIAGONAL ---
            k_dense_ptrs = k_dense_base + offs_n[None, :] * stride_kdn + offs_d[:, None] * stride_kdd
            k_dense = tl.load(k_dense_ptrs, mask=(offs_n[None, :] < seq_len_k) & (offs_d[:, None] < head_dim), other=0.0)
            qk = tl.dot(q, k_dense)
            
            # Extract exact baseline to carry into the sparse blocks
            local_keep = tl.load(km_base + offs_n * stride_kn, mask=k_mask, other=0)
            active_indices = tl.where(local_keep, offs_n, -1)
            last_active_idx = tl.max(active_indices)
            
            extract_mask = (offs_n == last_active_idx)
            extracted_baseline = tl.sum(tl.where(extract_mask[None, :], qk, 0.0), axis=1)
            running_score = tl.where(last_active_idx >= 0, extracted_baseline, running_score)
            prev_anchor_idx = tl.maximum(prev_anchor_idx, current_anchor_idx)
            
        else:
            # --- PATH B: STATEFUL ABSOLUTE COMPRESSED ---
            pack_start = prev_anchor_idx + 1
            pack_end = tl.maximum(current_anchor_idx, prev_anchor_idx)
            num_new_active = pack_end - pack_start + 1
            
            offs_k_active = pack_start + tl.arange(0, BLOCK_N)
            k_active_mask = tl.arange(0, BLOCK_N) < num_new_active
            k_packed_ptrs = k_packed_base + offs_k_active[None, :] * stride_kpn + offs_d[:, None] * stride_kpd
            
            # Load only the NEW absolute keys
            k_active = tl.load(k_packed_ptrs, mask=k_active_mask[None, :] & (offs_d[:, None] < head_dim), other=0.0)
            
            # Direct Dot Product (Exact FP32)
            qk_active = tl.dot(q, k_active)
            
            # Tensor Core Routing for NEW keys
            offs_i = tl.arange(0, BLOCK_N)
            routing_idx = local_im - pack_start
            routing_matrix = (offs_i[:, None] == routing_idx[None, :])
            new_scores = tl.dot(qk_active.to(q.dtype), routing_matrix.to(q.dtype))
            
            # INJECT STATE: Route `running_score` directly to dropped boundary tokens!
            is_old_anchor = local_im <= prev_anchor_idx
            qk = tl.where(is_old_anchor[None, :], running_score[:, None], new_scores)
            
            # UPDATE STATE: Carry the very last computed active score forward
            extract_carry_mask = (tl.arange(0, BLOCK_N) == (num_new_active - 1))
            new_running_score = tl.sum(tl.where(extract_carry_mask[None, :], qk_active, 0.0), axis=1)
            running_score = tl.where(num_new_active > 0, new_running_score, running_score)
            prev_anchor_idx = tl.maximum(prev_anchor_idx, current_anchor_idx)

        # --- UNIFIED MATH ---
        qk = qk * sm_scale
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & k_mask[None, :], qk, float("-inf"))
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        
        v_dense_ptrs = v_dense_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v_dense = tl.load(v_dense_ptrs, mask=(offs_n[:, None] < seq_len_k) & (offs_d[None, :] < head_dim), other=0.0)
        
        p = p.to(v_dense.dtype)
        acc += tl.dot(p, v_dense)
        m_i = m_ij

    acc = acc / l_i[:, None]
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))
