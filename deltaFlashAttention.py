import torch
import triton
import triton.language as tl

@triton.jit
def fused_delta_flash_kernel(
    Q, K_dense, K_packed, V_dense, Cumsum_Mask, Keep_Mask, Out, Scratch,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kdb, stride_kdh, stride_kdn, stride_kdd,
    stride_kpb, stride_kph, stride_kpn, stride_kpd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_cb, stride_ch, stride_cn,
    stride_kb, stride_kh, stride_kn,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_sb, stride_sh, stride_sm, stride_sn,
    sm_scale, blk_size, seq_len_q, seq_len_k, head_dim, num_heads,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    
    # -----------------------------------------------------------
    # 1. Base Pointers & Offsets
    # -----------------------------------------------------------
    start_m = pid_m * BLOCK_M
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    
    q_ptrs = Q + batch_idx * stride_qb + head_idx * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim), other=0.0)
    
    out_ptrs = Out + batch_idx * stride_ob + head_idx * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    
    k_dense_base = K_dense + batch_idx * stride_kdb + head_idx * stride_kdh
    k_packed_base = K_packed + batch_idx * stride_kpb + head_idx * stride_kph
    v_dense_base = V_dense + batch_idx * stride_vb + head_idx * stride_vh
    cm_base = Cumsum_Mask + batch_idx * stride_cb + head_idx * stride_ch
    km_base = Keep_Mask + batch_idx * stride_kb + head_idx * stride_kh
    scratch_base = Scratch + batch_idx * stride_sb + head_idx * stride_sh + offs_m[:, None] * stride_sm
    
    # -----------------------------------------------------------
    # 2. Flash Attention Accumulators & State
    # -----------------------------------------------------------
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    
    running_score = tl.zeros([BLOCK_M], dtype=tl.float32)
    q_start_bin = start_m // blk_size
    
    # -----------------------------------------------------------
    # 3. Inner Loop over K-Blocks
    # -----------------------------------------------------------
    hi = tl.minimum(seq_len_k, (pid_m + 1) * BLOCK_M)
    
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = offs_n < seq_len_k
        
        k_end_bin = (start_n + BLOCK_N - 1) // blk_size
        is_diagonal = (k_end_bin >= q_start_bin)
        
        if is_diagonal:
            # ==========================================
            # PATH A: EXACT ATTENTION (Diagonal)
            # ==========================================
            k_dense_ptrs = k_dense_base + offs_n[None, :] * stride_kdn + offs_d[:, None] * stride_kdd
            k_dense = tl.load(k_dense_ptrs, mask=(offs_n[None, :] < seq_len_k) & (offs_d[:, None] < head_dim), other=0.0)
            
            qk = tl.dot(q, k_dense)
            
            local_keep_mask = tl.load(km_base + offs_n * stride_kn, mask=k_mask, other=0)
            active_indices = tl.where(local_keep_mask, offs_n, -1)
            last_active_idx = tl.max(active_indices)
            
            extract_mask = (offs_n == last_active_idx)
            extracted_baseline = tl.sum(tl.where(extract_mask[None, :], qk, 0.0), axis=1)
            running_score = tl.where(last_active_idx >= 0, extracted_baseline, running_score)
            
        else:
            # ==========================================
            # PATH B: COMPRESSED DELTA ATTENTION
            # ==========================================
            pack_start_raw = tl.load(cm_base + start_n * stride_cn)
            is_start_kept = tl.load(km_base + start_n * stride_kn)
            
            # [FIX 1]: Identify the true start to prevent double-counting keys from the previous block
            true_pack_start = pack_start_raw + (1 - is_start_kept) 
            
            pack_end_idx = tl.minimum(start_n + BLOCK_N - 1, seq_len_k - 1)
            pack_end = tl.load(cm_base + pack_end_idx * stride_cn)
            num_active = pack_end - pack_start_raw + 1
            
            offs_k_active = pack_start_raw + tl.arange(0, BLOCK_N)
            k_active_mask = tl.arange(0, BLOCK_N) < num_active
            k_packed_ptrs = k_packed_base + offs_k_active[None, :] * stride_kpn + offs_d[:, None] * stride_kpd
            
            k_active = tl.load(k_packed_ptrs, mask=k_active_mask[None, :] & (offs_d[:, None] < head_dim), other=0.0)
            
            qk_delta = tl.dot(q, k_active)
            
            # [FIX 1 APPLIED]: Zero out the score of any key we already processed in the last block
            valid_delta_mask = offs_k_active >= true_pack_start
            qk_delta = tl.where(valid_delta_mask[None, :], qk_delta, 0.0)
            
            qk_cumsum = tl.cumsum(qk_delta, axis=1) + running_score[:, None]
            
            # [FIX 2]: Only update running_score if this block actually contained active keys
            extract_carry_mask = (tl.arange(0, BLOCK_N) == (num_active - 1))
            new_running_score = tl.sum(tl.where(extract_carry_mask[None, :], qk_cumsum, 0.0), axis=1)
            running_score = tl.where(num_active > 0, new_running_score, running_score)
            
            # The SRAM Bounce
            scratch_ptrs = scratch_base + tl.arange(0, BLOCK_N)[None, :] * stride_sn
            tl.store(scratch_ptrs, qk_cumsum, mask=k_active_mask[None, :])
            
            local_routing_mask = tl.load(cm_base + offs_n * stride_cn, mask=k_mask, other=0)
            routing_idx = local_routing_mask - pack_start_raw
            
            read_ptrs = scratch_base + routing_idx[None, :] * stride_sn
            qk = tl.load(read_ptrs, mask=k_mask[None, :], other=0.0)

        # ==========================================
        # UNIFIED FLASH ATTENTION MATH
        # ==========================================
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

    # -----------------------------------------------------------
    # 4. Final Write Back
    # -----------------------------------------------------------
    acc = acc / l_i[:, None]
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=(offs_m[:, None] < seq_len_q) & (offs_d[None, :] < head_dim))


def fused_delta_flash_attention(
    q, k_dense, k_packed, v_dense, cumsum_mask, keep_mask, 
    sm_scale, blk_size
):
    batch_size, num_heads, q_len, head_dim = q.shape
    k_len = k_dense.shape[2]
    
    # Assert contiguity
    q = q.contiguous()
    k_dense = k_dense.contiguous()
    k_packed = k_packed.contiguous()
    v_dense = v_dense.contiguous()
    cumsum_mask = cumsum_mask.contiguous()
    keep_mask = keep_mask.contiguous().to(torch.int32) # Ensure boolean is cast for triton
    
    out = torch.empty_like(q)
    
    # Tuning parameters
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = triton.next_power_of_2(head_dim)
    
    # HBM Scratchpad for the register expansion bounce
    # Size: [Batch, Heads, Q_Len, BLOCK_N]
    scratch = torch.empty((batch_size, num_heads, q_len, BLOCK_N), dtype=torch.float32, device=q.device)
    
    grid = (triton.cdiv(q_len, BLOCK_M), batch_size * num_heads, 1)
    
    fused_delta_flash_kernel[grid](
        q, k_dense, k_packed, v_dense, cumsum_mask, keep_mask, out, scratch,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_dense.stride(0), k_dense.stride(1), k_dense.stride(2), k_dense.stride(3),
        k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
        v_dense.stride(0), v_dense.stride(1), v_dense.stride(2), v_dense.stride(3),
        cumsum_mask.stride(0), cumsum_mask.stride(1), cumsum_mask.stride(2),
        keep_mask.stride(0), keep_mask.stride(1), keep_mask.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        scratch.stride(0), scratch.stride(1), scratch.stride(2), scratch.stride(3),
        sm_scale, blk_size, q_len, k_len, head_dim, num_heads,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D
    )
    
    return out