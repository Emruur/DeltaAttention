import torch
from torch import nn
import triton
import triton.language as tl

@triton.jit
def row_delta_euclidean_kernel(
    in_ptr, delta_ptr, mask_ptr,          # Pointers to memory
    threshold_sq,                         # Pre-squared threshold for speed
    stride_in_b, stride_in_h, stride_in_s, stride_in_d,  # Input/Delta strides
    stride_m_b, stride_m_h, stride_m_s,                  # Mask strides
    seq_len, head_dim,                    # Dimensions
    BLOCK_D: tl.constexpr                 # Must be a power of 2 (e.g., 64, 128)
):
    # 1. Identify which Batch and Head this specific program is processing
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # 2. Calculate the starting memory address for this specific sequence
    in_seq_ptr = in_ptr + pid_b * stride_in_b + pid_h * stride_in_h
    delta_seq_ptr = delta_ptr + pid_b * stride_in_b + pid_h * stride_in_h
    mask_seq_ptr = mask_ptr + pid_b * stride_m_b + pid_h * stride_m_h
    
    # 3. Create memory offsets for the head dimension (e.g., [0, 1, ..., 63])
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    
    # --- PROCESS TOKEN 0 ---
    # Load the first token to act as our initial reference state
    ptrs_0 = in_seq_ptr + 0 * stride_in_s + offs_d * stride_in_d
    ref_state = tl.load(ptrs_0, mask=mask_d, other=0.0)
    
    # Store delta[0] (which is just the input itself) and keep_mask[0] = True (1)
    delta_ptrs_0 = delta_seq_ptr + 0 * stride_in_s + offs_d * stride_in_d
    tl.store(delta_ptrs_0, ref_state, mask=mask_d)
    tl.store(mask_seq_ptr + 0 * stride_m_s, 1, mask=None) 
    
    # --- PROCESS TOKENS 1 TO N ---
    # We iterate sequentially on the GPU, avoiding Python overhead
    for i in range(1, seq_len):
        # Load current token
        curr_in_ptrs = in_seq_ptr + i * stride_in_s + offs_d * stride_in_d
        curr_state = tl.load(curr_in_ptrs, mask=mask_d, other=0.0)
        
        # Compute difference and Euclidean distance squared
        diff = curr_state - ref_state
        sq_dist = tl.sum(diff * diff, axis=0)
        
        # Evaluate if distance exceeds threshold
        should_keep = sq_dist > threshold_sq
        
        # Store the difference to the delta matrix
        curr_delta_ptrs = delta_seq_ptr + i * stride_in_s + offs_d * stride_in_d
        tl.store(curr_delta_ptrs, diff, mask=mask_d)
        
        # Store the boolean mask (cast automatically to uint8 by Triton)
        tl.store(mask_seq_ptr + i * stride_m_s, should_keep, mask=None)
        
        # Dynamically update the reference state if we kept this token
        if should_keep:
            ref_state = curr_state


@triton.jit
def sparse_delta_mm_scatter_kernel(
    Q_ptr, K_ptr, Out_ptr, Indices_ptr, Counts_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kd, stride_kn,
    stride_ob, stride_oh, stride_om, stride_on,
    stride_ib, stride_ih, stride_in,
    stride_cb, stride_ch,
    num_heads, num_kv_groups, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    # 1D Grid over (batch * num_heads)
    pid_bh = tl.program_id(0)
    # 1D Grid over query sequence blocks
    pid_m = tl.program_id(1)
    
    # Decode batch and head indices
    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads
    kv_head_idx = head_idx // num_kv_groups
    
    # Calculate Query (M) offsets
    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask = m_offsets < seq_len
    
    # Load the number of active keys for this specific batch/kv_head
    count_ptr = Counts_ptr + batch_idx * stride_cb + kv_head_idx * stride_ch
    num_active = tl.load(count_ptr)
    
    # Setup base pointers
    q_head_ptr = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_head_ptr = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    out_head_ptr = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    idx_head_ptr = Indices_ptr + batch_idx * stride_ib + kv_head_idx * stride_ih
    
    d_offsets = tl.arange(0, BLOCK_D)
    
    # Load Q block (Invariant to the inner N loop)
    # Shape: (BLOCK_M, BLOCK_D)
    q_ptrs = q_head_ptr + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_mask = m_mask[:, None] & (d_offsets[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # Iterate over active keys
    for n_start in range(0, num_active, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offsets < num_active
        
        # Load the *original* sequence indices of the active keys
        idx_ptrs = idx_head_ptr + n_offsets * stride_in
        active_cols = tl.load(idx_ptrs, mask=n_mask, other=0)
        
        # Indirect Loading: Load K block using the active columns
        # Shape: (BLOCK_D, BLOCK_N)
        k_ptrs = k_head_ptr + d_offsets[:, None] * stride_kd + active_cols[None, :] * stride_kn
        k_mask = (d_offsets[:, None] < head_dim) & n_mask[None, :]
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # Compute Q @ K (Standard FP16/BF16 -> FP32 accumulation is handled by Triton natively)
        acc = tl.dot(q, k)
        
        # Indirect Storing: Scatter results back into the correct sequence positions
        out_ptrs = out_head_ptr + m_offsets[:, None] * stride_om + active_cols[None, :] * stride_on
        out_mask = m_mask[:, None] & n_mask[None, :]
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)
