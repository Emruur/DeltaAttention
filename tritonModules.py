import torch
from torch import nn
import triton
import triton.language as tl

@triton.jit
def fused_row_delta_pack_kernel(
    in_ptr, packed_delta_ptr, cumsum_mask_ptr, counts_ptr, # Pointers
    threshold_sq,                                          # Pre-squared threshold
    stride_in_b, stride_in_h, stride_in_s, stride_in_d,    # Input strides
    stride_pd_b, stride_pd_h, stride_pd_a, stride_pd_d,    # Packed Delta strides
    stride_cm_b, stride_cm_h, stride_cm_s,                 # Cumsum Mask strides
    stride_c_b, stride_c_h,                                # Counts strides
    seq_len, head_dim,                                     
    BLOCK_D: tl.constexpr                 
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    # Base pointers for this specific Batch/Head
    in_seq_ptr = in_ptr + pid_b * stride_in_b + pid_h * stride_in_h
    packed_seq_ptr = packed_delta_ptr + pid_b * stride_pd_b + pid_h * stride_pd_h
    cm_seq_ptr = cumsum_mask_ptr + pid_b * stride_cm_b + pid_h * stride_cm_h
    
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    
    # --- PROCESS TOKEN 0 ---
    ptrs_0 = in_seq_ptr + 0 * stride_in_s + offs_d * stride_in_d
    ref_state = tl.load(ptrs_0, mask=mask_d, other=0.0)
    
    # Write Token 0 to the first slot (index 0)
    packed_ptrs_0 = packed_seq_ptr + 0 * stride_pd_a + offs_d * stride_pd_d
    tl.store(packed_ptrs_0, ref_state, mask=mask_d)
    
    # Token 0 is at index 0, so its routing mask is 0.
    active_idx = 0
    tl.store(cm_seq_ptr + 0 * stride_cm_s, active_idx) 
    
    # Now increment because the NEXT kept token belongs in slot 1
    active_idx += 1 
    
    # --- TEMPORAL LOOP ---
    for i in range(1, seq_len):
        curr_in_ptrs = in_seq_ptr + i * stride_in_s + offs_d * stride_in_d
        curr_state = tl.load(curr_in_ptrs, mask=mask_d, other=0.0)
        
        diff = curr_state - ref_state
        sq_dist = tl.sum(diff * diff, axis=0)
        
        should_keep = sq_dist > threshold_sq
        
        if should_keep:
            # active_idx points to the next available slot
            curr_packed_ptrs = packed_seq_ptr + active_idx * stride_pd_a + offs_d * stride_pd_d
            tl.store(curr_packed_ptrs, diff, mask=mask_d)
            ref_state = curr_state
            
            # Store the index we JUST wrote to, then increment
            tl.store(cm_seq_ptr + i * stride_cm_s, active_idx)
            active_idx += 1
        else:
            # If we skip, route this token's answer to the PREVIOUS kept token
            tl.store(cm_seq_ptr + i * stride_cm_s, active_idx - 1)
            
    # The total count of active tokens is still exactly active_idx
    count_ptr = counts_ptr + pid_b * stride_c_b + pid_h * stride_c_h
    tl.store(count_ptr, active_idx)

@triton.jit
def row_delta_euclidean_kernel(
    in_ptr, delta_ptr, mask_ptr,          # Pointers to memory
    threshold_sq,                         # Pre-squared threshold for speed
    stride_in_b, stride_in_h, stride_in_s, stride_in_d,  # Input/Delta strides
    stride_m_b, stride_m_h, stride_m_s,                  # Mask strides
    seq_len, head_dim,                    # Dimensions
    BLOCK_D: tl.constexpr                 # Must be a power of 2 (e.g., 64, 128)
):
    # Identify which Batch and Head this program is processing
    pid_b = tl.program_id(0) #batch
    pid_h = tl.program_id(1) #head
    
    # Calculate the starting memory address for this sequence
    in_seq_ptr = in_ptr + pid_b * stride_in_b + pid_h * stride_in_h

    ## Output pointers to write the delta
    delta_seq_ptr = delta_ptr + pid_b * stride_in_b + pid_h * stride_in_h
    mask_seq_ptr = mask_ptr + pid_b * stride_m_b + pid_h * stride_m_h
    
    # Head Dimension (the vector of numbers that represents one token)
    # Create memory offsets for the head dimension (e.g., [0, 1, ..., 63])
    # We create 
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    ##offs_d = tl.arange(0, 64) tells the worker: "Spread your 64 hands across the 64 columns."
    #mask_d tells the worker: "Actually, we only have 50 columns of real data. Hands 50 through 63, 
    # grab zeros instead so you don't break anything."
    
    # --- PROCESS TOKEN 0 ---
    # Load the first token to act as our initial reference state
    # TODO stride_in_d should be 1 for coalasced reads
    ptrs_0 = in_seq_ptr + 0 * stride_in_s + offs_d * stride_in_d
    ref_state = tl.load(ptrs_0, mask=mask_d, other=0.0)
    
    # Store delta[0] (which is just the input itself) and keep_mask[0] = True (1)
    delta_ptrs_0 = delta_seq_ptr + 0 * stride_in_s + offs_d * stride_in_d
    tl.store(delta_ptrs_0, ref_state, mask=mask_d)
    tl.store(mask_seq_ptr + 0 * stride_m_s, 1, mask=None) 
    
    # PROCESS TOKENS 1 TO N
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
    in_ptr, delta_ptr, mask_ptr, chunk_counts_ptr,       # [NEW] Added chunk_counts_ptr
    threshold_sq,                         
    stride_in_b, stride_in_h, stride_in_s, stride_in_d,  
    stride_m_b, stride_m_h, stride_m_s,    
    stride_cc_b, stride_cc_h, stride_cc_c,               # [NEW] Added strides for the counts tensor              
    seq_len, head_dim, num_heads, chunk_size,            
    BLOCK_D: tl.constexpr                 
):
    # Identify which Batch and Head this program is processing
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
    
    # [NEW] Initialize the counter. 
    # It starts at 1 because the anchor token is unconditionally kept!
    active_count = 1 
    
    # --- ANCHOR TOKEN ---
    ptrs_start = in_seq_ptr + start_seq * stride_in_s + offs_d * stride_in_d
    ref_state = tl.load(ptrs_start, mask=mask_d, other=0.0)
    
    delta_ptrs_start = delta_seq_ptr + start_seq * stride_in_s + offs_d * stride_in_d
    tl.store(delta_ptrs_start, ref_state, mask=mask_d)
    tl.store(mask_seq_ptr + start_seq * stride_m_s, 1, mask=None) 
    
    # --- REMAINING TOKENS ---
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
            active_count += 1  # [NEW] Increment our local register

    # [NEW] Write the final count to HBM right before the block exits
    # pid_p is our exact chunk index
    count_ptr = chunk_counts_ptr + pid_b * stride_cc_b + pid_h * stride_cc_h + pid_p * stride_cc_c
    tl.store(count_ptr, active_count)


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

@triton.jit
def _triton_segmented_cumsum_kernel(
    scores_ptr, bounds_ptr,
    stride_s_b, stride_s_h, stride_s_l, stride_s_a,
    stride_b_b, stride_b_h, stride_b_c,
    max_active, num_chunks,
    BLOCK_A: tl.constexpr,
    MAX_CHUNKS: tl.constexpr
):
    # 3D Grid: [Batch, Heads, Seq_Len]
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_l = tl.program_id(2)

    # Base pointers for this specific row and boundary array
    scores_row_ptr = scores_ptr + pid_b * stride_s_b + pid_h * stride_s_h + pid_l * stride_s_l
    bounds_ptr_base = bounds_ptr + pid_b * stride_b_b + pid_h * stride_b_h

    offs = tl.arange(0, BLOCK_A)
    mask = offs < max_active

    # 1. Load the raw matrix row into ultra-fast SRAM
    row = tl.load(scores_row_ptr + offs * stride_s_a, mask=mask, other=0.0)

    # 2. Run the blind cumsum directly in registers
    cumsum_row = tl.cumsum(row, axis=0)

    # 3. Calculate exact subtractions in registers
    subtractions = tl.zeros((BLOCK_A,), dtype=tl.float32)

    # Unroll the loop over the chunks
    for i in tl.static_range(MAX_CHUNKS):
        if i < num_chunks - 1:
            # Load the boundary index
            boundary = tl.load(bounds_ptr_base + i * stride_b_c)
            
            # Find the "explosion" value exactly one step before the boundary
            explosion_mask = offs == (boundary - 1)
            
            # tl.sum collapses the masked register into a single scalar value
            explosion_val = tl.sum(tl.where(explosion_mask, cumsum_row, 0.0), axis=0)
            
            # Overwrite the subtraction offset for all elements AFTER this boundary.
            # Because the loop runs sequentially, later boundaries overwrite earlier ones 
            # for the elements furthest to the right, which is mathematically perfect.
            apply_mask = offs >= boundary
            subtractions = tl.where(apply_mask, explosion_val, subtractions)

    # 4. Apply correction and write back directly! (In-place modification)
    corrected_row = cumsum_row - subtractions
    tl.store(scores_row_ptr + offs * stride_s_a, corrected_row, mask=mask)
