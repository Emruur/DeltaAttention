import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def chunked_eval_kernel(
    in_ptr, keep_mask_ptr, chunk_counts_ptr,
    threshold_sq,
    stride_in_b, stride_in_h, stride_in_s, stride_in_d,
    stride_km_b, stride_km_h, stride_km_s,
    stride_cc_b, stride_cc_h, stride_cc_c,
    seq_len, head_dim, chunk_size, 
    num_heads,
    BLOCK_D: tl.constexpr
):
    pid_bh = tl.program_id(0) # Batch * Head
    pid_c = tl.program_id(1)  # Chunk Index
    
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    
    in_seq_ptr = in_ptr + batch_idx * stride_in_b + head_idx * stride_in_h
    km_seq_ptr = keep_mask_ptr + batch_idx * stride_km_b + head_idx * stride_km_h
    
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    
    start_idx = pid_c * chunk_size
    end_idx = tl.minimum(start_idx + chunk_size, seq_len)
    
    if start_idx >= seq_len:
        return
        
    # --- Token 0 of this chunk is ALWAYS an anchor locally ---
    ptrs_0 = in_seq_ptr + start_idx * stride_in_s + offs_d * stride_in_d
    ref_state = tl.load(ptrs_0, mask=mask_d, other=0.0)
    
    tl.store(km_seq_ptr + start_idx * stride_km_s, 1) # Keep = True
    local_count = 1
    
    for i in range(start_idx + 1, end_idx):
        curr_ptrs = in_seq_ptr + i * stride_in_s + offs_d * stride_in_d
        curr_state = tl.load(curr_ptrs, mask=mask_d, other=0.0)
        
        diff = curr_state - ref_state
        sq_dist = tl.sum(diff * diff, axis=0)
        
        if sq_dist > threshold_sq:
            tl.store(km_seq_ptr + i * stride_km_s, 1)
            ref_state = curr_state
            local_count += 1
        else:
            tl.store(km_seq_ptr + i * stride_km_s, 0)
            
    # Store the count for this specific chunk
    cc_ptr = chunk_counts_ptr + batch_idx * stride_cc_b + head_idx * stride_cc_h + pid_c * stride_cc_c
    tl.store(cc_ptr, local_count)


@triton.jit
def parallel_scatter_pack_kv_kernel(
    k_in_ptr, v_in_ptr, 
    k_packed_ptr, v_packed_ptr, pt_ptr, # pt = packed_timestamps
    keep_mask_ptr, index_map_ptr,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_kpb, stride_kph, stride_kpa, stride_kpd,
    stride_vpb, stride_vph, stride_vpa, stride_vpd,
    stride_ptb, stride_pth, stride_pta,
    stride_km_b, stride_km_h, stride_km_s,
    stride_im_b, stride_im_h, stride_im_s,
    seq_len, head_dim, num_heads,
    BLOCK_S: tl.constexpr, BLOCK_D: tl.constexpr
):
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1) # Sequence block
    
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    
    start_s = pid_s * BLOCK_S
    offs_s = start_s + tl.arange(0, BLOCK_S)
    mask_s = offs_s < seq_len
    
    km_ptrs = keep_mask_ptr + batch_idx * stride_km_b + head_idx * stride_km_h + offs_s * stride_km_s
    keep_mask = tl.load(km_ptrs, mask=mask_s, other=0)
    
    # If nothing in this block is kept, exit early to save bandwidth
    if tl.sum(keep_mask) == 0:
        return
        
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    
    # ----------------------------------------------------
    # LOAD K, V, and Target Destinations
    # ----------------------------------------------------
    im_ptrs = index_map_ptr + batch_idx * stride_im_b + head_idx * stride_im_h + offs_s * stride_im_s
    dest_indices = tl.load(im_ptrs, mask=mask_s, other=0)
    
    k_in_ptrs = k_in_ptr + batch_idx * stride_kb + head_idx * stride_kh + offs_s[:, None] * stride_ks + offs_d[None, :] * stride_kd
    v_in_ptrs = v_in_ptr + batch_idx * stride_vb + head_idx * stride_vh + offs_s[:, None] * stride_vs + offs_d[None, :] * stride_vd
    
    keys = tl.load(k_in_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
    vals = tl.load(v_in_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)
    
    # ----------------------------------------------------
    # SCATTER WRITE K, V, and Timestamps
    # ----------------------------------------------------
    write_mask = (keep_mask[:, None] == 1) & mask_s[:, None] & mask_d[None, :]
    
    k_packed_ptrs = k_packed_ptr + batch_idx * stride_kpb + head_idx * stride_kph + dest_indices[:, None] * stride_kpa + offs_d[None, :] * stride_kpd
    tl.store(k_packed_ptrs, keys, mask=write_mask)
    
    v_packed_ptrs = v_packed_ptr + batch_idx * stride_vpb + head_idx * stride_vph + dest_indices[:, None] * stride_vpa + offs_d[None, :] * stride_vpd
    tl.store(v_packed_ptrs, vals, mask=write_mask)
    
    # Timestamps only need 1D masking
    pt_ptrs = pt_ptr + batch_idx * stride_ptb + head_idx * stride_pth + dest_indices * stride_pta
    tl.store(pt_ptrs, offs_s, mask=(keep_mask == 1) & mask_s)

