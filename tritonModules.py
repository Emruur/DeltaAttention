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
    k_head_ptr = K_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
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

@triton.jit
def fused_nm_delta_kernel(
    states_ptr,       # [B * H, seq_len, head_dim]
    out_ptr,          # [B * H, seq_len, head_dim]
    stride_bh, stride_seq, stride_dim,
    seq_len,
    thresh,           # Temporal threshold
    BLOCK_DIM: tl.constexpr = 4
):
    # 1. Map programs
    bh_idx = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    
    dim_offsets = chunk_idx * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    
    base_state_ptr = states_ptr + bh_idx * stride_bh + dim_offsets * stride_dim
    base_out_ptr   = out_ptr + bh_idx * stride_bh + dim_offsets * stride_dim
    
    block_indices = tl.arange(0, BLOCK_DIM)

    # --- 2. THE ATTENTION SINK FIX (t = 0) ---
    # Load the dense state to use as our reference for the rest of the sequence
    ref_states = tl.load(base_state_ptr + 0 * stride_seq)
    
    # Apply 2:4 sparsity to the first token so we don't destroy the attention sink
    abs_ref = tl.abs(ref_states)
    idx_1_t0 = tl.argmax(abs_ref, axis=0)
    is_max_1_t0 = block_indices == idx_1_t0
    
    abs_ref_masked = tl.where(is_max_1_t0, -1.0, abs_ref)
    idx_2_t0 = tl.argmax(abs_ref_masked, axis=0)
    
    mask_t0 = is_max_1_t0 | (block_indices == idx_2_t0)
    sparse_t0 = tl.where(mask_t0, ref_states, 0.0)
    
    # Store the 2:4 sparsified first token
    tl.store(base_out_ptr + 0 * stride_seq, sparse_t0)


    # --- 3. Iterate down the sequence ---
    for t in range(1, seq_len):
        curr_states = tl.load(base_state_ptr + t * stride_seq)
        
        # PHASE 1: TEMPORAL DELTA
        sub = curr_states - ref_states
        abs_sub = tl.abs(sub)
        
        exceeds_thresh = abs_sub > thresh
        
        delta_row = tl.where(exceeds_thresh, sub, 0.0)
        abs_delta_row = tl.abs(delta_row)
        
        # Update dense reference state unconditionally based on threshold
        ref_states = tl.where(exceeds_thresh, curr_states, ref_states)
        
        # PHASE 2: 2:4 STRUCTURED SPARSITY
        idx_1 = tl.argmax(abs_delta_row, axis=0)
        is_max_1 = block_indices == idx_1
        
        abs_delta_masked = tl.where(is_max_1, -1.0, abs_delta_row)
        idx_2 = tl.argmax(abs_delta_masked, axis=0)
        
        mask_24 = is_max_1 | (block_indices == idx_2)
        
        final_delta = tl.where(mask_24, delta_row, 0.0)
        
        tl.store(base_out_ptr + t * stride_seq, final_delta)

@triton.jit
def row_delta_euclidean_partitioned_kernel(
    in_ptr, delta_ptr, mask_ptr,          
    threshold_sq,                         
    stride_in_b, stride_in_h, stride_in_s, stride_in_d,  
    stride_m_b, stride_m_h, stride_m_s,                  
    seq_len, head_dim, num_heads, chunk_size,            
    BLOCK_D: tl.constexpr                 
):
    pid_bh = tl.program_id(0)
    pid_p = tl.program_id(1)
    
    pid_b = pid_bh // num_heads
    pid_h = pid_bh % num_heads
    
    start_seq = pid_p * chunk_size
    end_seq = tl.minimum(start_seq + chunk_size, seq_len)
    
    if start_seq >= seq_len:
        return

    in_seq_ptr = in_ptr + pid_b * stride_in_b + pid_h * stride_in_h
    delta_seq_ptr = delta_ptr + pid_b * stride_in_b + pid_h * stride_in_h
    mask_seq_ptr = mask_ptr + pid_b * stride_m_b + pid_h * stride_m_h
    
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    
    # --- ANCHOR TOKEN ---
    ptrs_start = in_seq_ptr + start_seq * stride_in_s + offs_d * stride_in_d
    ref_state = tl.load(ptrs_start, mask=mask_d, other=0.0)
    
    delta_ptrs_start = delta_seq_ptr + start_seq * stride_in_s + offs_d * stride_in_d
    tl.store(delta_ptrs_start, ref_state, mask=mask_d)
    tl.store(mask_seq_ptr + start_seq * stride_m_s, 1, mask=None) 
    
    # --- REMAINING TOKENS (Exact match to your original math) ---
    for i in range(start_seq + 1, end_seq):
        curr_in_ptrs = in_seq_ptr + i * stride_in_s + offs_d * stride_in_d
        curr_state = tl.load(curr_in_ptrs, mask=mask_d, other=0.0)
        
        diff = curr_state - ref_state
        sq_dist = tl.sum(diff * diff, axis=0)
        should_keep = sq_dist > threshold_sq
        
        # Store the raw diff unconditionally, just like your baseline
        curr_delta_ptrs = delta_seq_ptr + i * stride_in_s + offs_d * stride_in_d
        tl.store(curr_delta_ptrs, diff, mask=mask_d)
        tl.store(mask_seq_ptr + i * stride_m_s, should_keep, mask=None)
        
        # Original block-uniform reference update
        if should_keep:
            ref_state = curr_state




@triton.jit
def _triton_gather_expand(
    delta_y, active_indices, k_packed_q,
    stride_dy_b, stride_dy_h, stride_dy_d, stride_dy_l,
    stride_idx_b, stride_idx_h, stride_idx_a,
    stride_out_b, stride_out_q, stride_out_d, stride_out_a,
    num_q_heads, num_groups, seq_len, head_dim,
    BLOCK_D: tl.constexpr
):
    """
    Gathers active keys and expands them for GQA/MQA in a single fused pass.
    """
    pid_bq = tl.program_id(0)
    pid_a = tl.program_id(1)

    b = pid_bq // num_q_heads
    q = pid_bq % num_q_heads
    kv_h = q // num_groups

    # Locate the original sequence index for this active element
    idx_offset = b * stride_idx_b + kv_h * stride_idx_h + pid_a * stride_idx_a
    orig_idx = tl.load(active_indices + idx_offset)

    offsets_d = tl.arange(0, BLOCK_D)
    mask_d = offsets_d < head_dim

    # Calculate input offset. Mask out invalid indices (padded values)
    dy_offset = b * stride_dy_b + kv_h * stride_dy_h + offsets_d * stride_dy_d + orig_idx * stride_dy_l
    load_mask = mask_d & (orig_idx < seq_len)
    
    # Load from delta_y (writes 0.0 if it was padded/inactive)
    vals = tl.load(delta_y + dy_offset, mask=load_mask, other=0.0)

    # Store into densely packed tensor
    out_offset = b * stride_out_b + q * stride_out_q + offsets_d * stride_out_d + pid_a * stride_out_a
    tl.store(k_packed_q + out_offset, vals, mask=mask_d)


@triton.jit
def _triton_scatter(
    scores_packed, active_indices, delta_out,
    stride_sp_b, stride_sp_q, stride_sp_l, stride_sp_a,
    stride_idx_b, stride_idx_h, stride_idx_a,
    stride_out_b, stride_out_q, stride_out_l, stride_out_s,
    num_q_heads, num_groups, seq_len, max_active,
    BLOCK_L: tl.constexpr
):
    """
    Scatters the tightly packed cuBLAS results back into the sparse L x L attention matrix.
    """
    pid_bq = tl.program_id(0)
    pid_a = tl.program_id(1)
    pid_l_chunk = tl.program_id(2)

    b = pid_bq // num_q_heads
    q = pid_bq % num_q_heads
    kv_h = q // num_groups

    # Locate the original sequence index
    idx_offset = b * stride_idx_b + kv_h * stride_idx_h + pid_a * stride_idx_a
    orig_idx = tl.load(active_indices + idx_offset)

    # Only process if this is a valid token (not a pad token)
    if orig_idx < seq_len:
        offsets_l = pid_l_chunk * BLOCK_L + tl.arange(0, BLOCK_L)
        mask_l = offsets_l < seq_len

        # Load the computed scores
        sp_offset = b * stride_sp_b + q * stride_sp_q + offsets_l * stride_sp_l + pid_a * stride_sp_a
        vals = tl.load(scores_packed + sp_offset, mask=mask_l, other=0.0)

        # Scatter back into the sparse attention map
        out_offset = b * stride_out_b + q * stride_out_q + offsets_l * stride_out_l + orig_idx * stride_out_s
        tl.store(delta_out + out_offset, vals, mask=mask_l)