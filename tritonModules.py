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
def block_delta_mm_cumsum_kernel(
    q_ptr, k_ptr, mask_ptr, out_ptr,          
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_m_b, stride_m_h, stride_m_s,
    stride_o_b, stride_o_h, stride_o_s1, stride_o_s2,
    seq_len, head_dim, num_kv_groups,         
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)
    
    kv_head_idx = head_idx // num_kv_groups

    # 1. Query Offsets
    offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < seq_len
    mask_d = offs_d < head_dim
    
    # 2. Load the block of Queries (Shape: [BLOCK_M, BLOCK_D])
    q_ptrs = q_ptr + batch_idx * stride_q_b + head_idx * stride_q_h + \
             offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # 3. Setup the Carry-Over Register [BLOCK_M]
    carry_over = tl.zeros([BLOCK_M], dtype=tl.float32)

    # 4. Generate the "CumSum Matrix" (Upper Triangular matrix of 1s)
    # Shape: [BLOCK_N, BLOCK_N]
    row_i = tl.arange(0, BLOCK_N)[:, None]
    col_j = tl.arange(0, BLOCK_N)[None, :]
    cumsum_matrix = tl.where(row_i <= col_j, 1.0, 0.0).to(q.dtype)

    # 5. Iterate over Keys in BLOCKS (Not one by one!)
    num_k_blocks = tl.cdiv(seq_len, BLOCK_N)
    for k_idx in range(num_k_blocks):
        offs_n = k_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        # Load Keep Mask for this block [BLOCK_N]
        mask_ptrs = mask_ptr + batch_idx * stride_m_b + kv_head_idx * stride_m_h + offs_n * stride_m_s
        keep_mask = tl.load(mask_ptrs, mask=mask_n, other=0)
        
        # Load block of Keys [BLOCK_N, BLOCK_D]
        k_ptrs = k_ptr + batch_idx * stride_k_b + kv_head_idx * stride_k_h + \
                 offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        
        # Apply the delta mask: If a key wasn't kept, we force it to 0.0
        k = tl.where(keep_mask[:, None], k, 0.0)
        
        # --- THE MAGIC HAPPENS HERE ---
        # 1. Block MatMul: Q [M, D] @ K^T [D, N] -> [M, N]
        # This triggers the hardware Tensor Cores!
        scores = tl.dot(q, tl.trans(k))
        scores = scores.to(q.dtype)
        
        # 2. Local Block CumSum: scores [M, N] @ cumsum_matrix [N, N] -> [M, N]
        # This triggers the Tensor Cores again to instantly sum the columns!
        local_cumsum = tl.dot(scores, cumsum_matrix)
        
        # 3. Add the carry-over from the previous block
        out_block = local_cumsum + carry_over[:, None]

        last_col_mask = (tl.arange(0, BLOCK_N) == (BLOCK_N - 1)).to(tl.float32)
        
        # Multiply and sum to legally extract the last column in Triton
        carry_over = tl.sum(out_block * last_col_mask[None, :], axis=1)
        
        
        # 5. Store the chunk back to High Bandwidth Memory
        out_ptrs = out_ptr + batch_idx * stride_o_b + head_idx * stride_o_h + \
                   offs_m[:, None] * stride_o_s1 + offs_n[None, :] * stride_o_s2
                   
        tl.store(out_ptrs, out_block.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def fused_delta_mm_cumsum_kernel(
    q_ptr, k_delta_ptr, k_exact_ptr, mask_ptr, out_ptr,          
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,  # exact and delta keys share strides
    stride_m_b, stride_m_h, stride_m_s,
    stride_o_b, stride_o_h, stride_o_s1, stride_o_s2,
    seq_len, head_dim, num_kv_groups,
    sink_size, blk_size,                              # NEW: Patching parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr
):
    batch_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    q_block_idx = tl.program_id(2)
    kv_head_idx = head_idx // num_kv_groups

    # Query Setup
    offs_m = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    mask_m = offs_m < seq_len
    mask_d = offs_d < head_dim
    
    q_ptrs = q_ptr + batch_idx * stride_q_b + head_idx * stride_q_h + \
             offs_m[:, None] * stride_q_s + offs_d[None, :] * stride_q_d
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    carry_over = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    row_i = tl.arange(0, BLOCK_N)[:, None]
    col_j = tl.arange(0, BLOCK_N)[None, :]
    cumsum_matrix = tl.where(row_i <= col_j, 1.0, 0.0).to(q.dtype)

    # Pre-calculate Query Block Boundaries for the Local Block check
    m_start = q_block_idx * BLOCK_M
    m_end = m_start + BLOCK_M - 1

    num_k_blocks = tl.cdiv(seq_len, BLOCK_N)
    for k_idx in range(num_k_blocks):
        offs_n = k_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        n_start = k_idx * BLOCK_N
        n_end = n_start + BLOCK_N - 1
        
        # --- 1. COMPUTE DELTA APPROXIMATION ---
        mask_ptrs = mask_ptr + batch_idx * stride_m_b + kv_head_idx * stride_m_h + offs_n * stride_m_s
        keep_mask = tl.load(mask_ptrs, mask=mask_n, other=0)
        
        k_delta_ptrs = k_delta_ptr + batch_idx * stride_k_b + kv_head_idx * stride_k_h + \
                       offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
        k_delta = tl.load(k_delta_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        k_delta = tl.where(keep_mask[:, None], k_delta, 0.0)
        
        scores = tl.dot(q, tl.trans(k_delta))
        scores = scores.to(q.dtype)
        
        local_cumsum = tl.dot(scores, cumsum_matrix)
        out_block = local_cumsum + carry_over[:, None]
        
        # Update carry-over strictly using the delta scores (Preserves PyTorch math exactly!)
        last_col_mask = (tl.arange(0, BLOCK_N) == (BLOCK_N - 1)).to(tl.float32)
        carry_over = tl.sum(out_block * last_col_mask[None, :], axis=1)

        # --- 2. FUSED SINK & LOCAL PATCHING ---
        is_sink = n_start < sink_size
        
        # Avoid division by zero if blk_size is 0
        if blk_size > 0:
            is_local = (m_start // blk_size <= n_end // blk_size) and (n_start // blk_size <= m_end // blk_size)
        else:
            is_local = False
            
        # Only do the heavy exact math if this block intersects with the Sink or Diagonal
        if is_sink or is_local:
            # Load exact keys
            k_exact_ptrs = k_exact_ptr + batch_idx * stride_k_b + kv_head_idx * stride_k_h + \
                           offs_n[:, None] * stride_k_s + offs_d[None, :] * stride_k_d
            k_exact = tl.load(k_exact_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
            
            # Compute exact attention scores (fp32)
            exact_scores = tl.dot(q, tl.trans(k_exact))
            
            # Create precise element-wise masks
            sink_mask = offs_n[None, :] < sink_size
            
            if blk_size > 0:
                local_mask = (offs_m[:, None] // blk_size) == (offs_n[None, :] // blk_size)
            else:
                local_mask = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.int1) # False
                
            patch_mask = sink_mask | local_mask
            
            # Overwrite the approximated scores with exact scores where required
            out_block = tl.where(patch_mask, exact_scores, out_block)

        # --- 3. STORE RESULT ---
        out_ptrs = out_ptr + batch_idx * stride_o_b + head_idx * stride_o_h + \
                   offs_m[:, None] * stride_o_s1 + offs_n[None, :] * stride_o_s2
                   
        tl.store(out_ptrs, out_block.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])