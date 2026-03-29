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
def _triton_expand_cumsum(
    packed_cumsum, cumsum_mask, delta_out,
    stride_pc_b, stride_pc_h, stride_pc_q, stride_pc_a,
    stride_cm_b, stride_cm_h, stride_cm_k,
    stride_out_b, stride_out_h, stride_out_q, stride_out_k,
    num_heads, num_groups, seq_len_q, seq_len_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Broadcasts the pre-cumsummed packed scores back to a dense matrix
    using the cumsum_mask as an O(1) routing index. ZERO scattered writes.
    """
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    b = pid_bh // num_heads
    h = pid_bh % num_heads
    kv_h = h // num_groups

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_q = offs_q < seq_len_q
    mask_k = offs_k < seq_len_k

    # 1. Load the routing index from the cumsum_mask
    cm_ptrs = cumsum_mask + b * stride_cm_b + kv_h * stride_cm_h + offs_k * stride_cm_k
    cm_vals = tl.load(cm_ptrs, mask=mask_k, other=0)

    # 2. Map dense coordinates directly to the packed coordinates
    pc_ptrs = packed_cumsum + b * stride_pc_b + h * stride_pc_h + offs_q[:, None] * stride_pc_q + cm_vals[None, :] * stride_pc_a
    
    mask_load = mask_q[:, None] & mask_k[None, :]
    vals = tl.load(pc_ptrs, mask=mask_load, other=0.0)

    # 3. Contiguous, coalesced store (No scattered writes!)
    out_ptrs = delta_out + b * stride_out_b + h * stride_out_h + offs_q[:, None] * stride_out_q + offs_k[None, :] * stride_out_k
    tl.store(out_ptrs, vals, mask=mask_load)



import triton
import triton.language as tl

@triton.jit
def opt_triton_expand_cumsum(
    packed_cumsum_ptr, cumsum_mask_ptr, delta_out_ptr,
    stride_pc_b, stride_pc_h, stride_pc_q, stride_pc_k,
    stride_cm_b, stride_cm_h, stride_cm_k,
    stride_out_b, stride_out_h, stride_out_q, stride_out_k,
    num_heads, num_kv_groups, seq_len_q, seq_len_k,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr
):
    # 1. Identify our position in the 3D Grid
    pid_bh = tl.program_id(0)
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)

    # 2. Decode Batch and Head IDs
    batch_id = pid_bh // num_heads
    head_id = pid_bh % num_heads

    # 3. Compute block offsets for Q and K dimensions
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    # 4. Create bounds masks to prevent reading/writing out of bounds
    mask_q = offs_q < seq_len_q
    mask_k = offs_k < seq_len_k

    # ==========================================
    # STEP A: LOAD THE ROUTING INDICES (1D)
    # ==========================================
    # The cumsum_mask tells us how many active tokens exist up to position K.
    # Shape of cumsum_mask is (bsz, num_heads, seq_len_k)
    cm_ptrs = cumsum_mask_ptr + (
        batch_id * stride_cm_b + 
        head_id * stride_cm_h + 
        offs_k * stride_cm_k
    )
    
    # Load 1D row of routing indices. If out of bounds, return 0.
    routing_idx = tl.load(cm_ptrs, mask=mask_k, other=0)

    # ==========================================
    # STEP B: PREPARE 2D POINTERS FOR UNPADDED FETCH
    # ==========================================
    # We want to map this to a 2D block of shape (BLOCK_Q, BLOCK_K)
    # routing_idx > 0 means it's a valid mapped token. 0 means it was skipped/padded.
    
    # Broadcast Q to column vector (BLOCK_Q, 1) and K to row vector (1, BLOCK_K)
    offs_q_2d = offs_q[:, None]
    
    # Shift index back by 1 because the unpadded tensor is 0-indexed
    fetch_idx_2d = (routing_idx - 1)[None, :]
    
    # 2D Mask: Valid only if within Q/K sequence bounds AND the token is actually active
    is_active_token = (routing_idx > 0)[None, :]
    fetch_mask_2d = mask_q[:, None] & mask_k[None, :] & is_active_token

    # Compute the 2D memory pointers for the unpadded packed_cumsum tensor
    pc_ptrs = packed_cumsum_ptr + (
        batch_id * stride_pc_b + 
        head_id * stride_pc_h + 
        offs_q_2d * stride_pc_q + 
        fetch_idx_2d * stride_pc_k
    )

    # ==========================================
    # STEP C: PREDICATED LOAD & STORE
    # ==========================================
    # Fetch from the unpadded tensor. 
    # If fetch_mask_2d is False, it skips memory and instantly loads 0.0!
    vals = tl.load(pc_ptrs, mask=fetch_mask_2d, other=0.0)

    # Compute final 2D pointers for the output tensor
    out_ptrs = delta_out_ptr + (
        batch_id * stride_out_b + 
        head_id * stride_out_h + 
        offs_q_2d * stride_out_q + 
        offs_k[None, :] * stride_out_k
    )

    # Store the results. (We only need standard bounds checking here)
    store_mask_2d = mask_q[:, None] & mask_k[None, :]
    tl.store(out_ptrs, vals, mask=store_mask_2d)