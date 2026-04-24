import torch
from transformers import Cache
import triton
import triton.language as tl
import globVR 


from tritonModules import (
    chunked_eval_kernel,
    parallel_scatter_pack_kv_kernel
)
import torch
import triton
from transformers.cache_utils import Cache
import globVR


@triton.jit
def fused_hybrid_decode_update_kernel(
    new_k_ptr, new_v_ptr,
    k_exact_ptr, v_exact_ptr,
    k_packed_ptr, v_packed_ptr,
    packed_counts_ptr, packed_timestamps_ptr, packed_lengths_ptr,
    exact_seq_len, total_seq_len, threshold_sq, exact_window_size,
    stride_new_b, stride_new_h, stride_new_s, stride_new_d,
    stride_exact_b, stride_exact_h, stride_exact_s, stride_exact_d,
    stride_pack_b, stride_pack_h, stride_pack_s, stride_pack_d,
    stride_pc_b, stride_pc_h, stride_pc_s,
    num_heads, head_dim: tl.constexpr, BLOCK_D: tl.constexpr,
    USE_COSINE: tl.constexpr,
):
    # 1 Program ID = 1 Batch + 1 Head
    pid = tl.program_id(0)
    pid_b = pid // num_heads
    pid_h = pid % num_heads

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    # Pointers to the brand new generated token
    new_k_ptrs = new_k_ptr + pid_b * stride_new_b + pid_h * stride_new_h + offs_d * stride_new_d
    new_v_ptrs = new_v_ptr + pid_b * stride_new_b + pid_h * stride_new_h + offs_d * stride_new_d

    write_idx = exact_seq_len % exact_window_size

    # Base pointers for the Exact Ring Buffer (sliced per layer)
    exact_k_base = k_exact_ptr + pid_b * stride_exact_b + pid_h * stride_exact_h
    exact_v_base = v_exact_ptr + pid_b * stride_exact_b + pid_h * stride_exact_h

    # Base pointers for the Packed Cache (sliced per layer)
    pack_k_base = k_packed_ptr + pid_b * stride_pack_b + pid_h * stride_pack_h
    pack_v_base = v_packed_ptr + pid_b * stride_pack_b + pid_h * stride_pack_h
    count_base = packed_counts_ptr + pid_b * stride_pc_b + pid_h * stride_pc_h
    ts_base = packed_timestamps_ptr + pid_b * stride_pc_b + pid_h * stride_pc_h

    # Pointer to the current length tracker for this head
    len_ptr = packed_lengths_ptr + pid
    packed_len = tl.load(len_ptr)

    # ----------------------------------------------------
    # EVICTION & MERGE LOGIC (Only if Ring Buffer is full)
    # ----------------------------------------------------
    if exact_seq_len >= exact_window_size:
        # Load the token that is about to be overwritten into SRAM
        evict_k_ptrs = exact_k_base + write_idx * stride_exact_s + offs_d * stride_exact_d
        evict_v_ptrs = exact_v_base + write_idx * stride_exact_s + offs_d * stride_exact_d
        evict_k = tl.load(evict_k_ptrs, mask=mask_d)
        evict_v = tl.load(evict_v_ptrs, mask=mask_d)

        if packed_len > 0:
            last_idx = packed_len - 1
            last_k_ptrs = pack_k_base + last_idx * stride_pack_s + offs_d * stride_pack_d
            last_k = tl.load(last_k_ptrs, mask=mask_d)

            # Distance/similarity calculation in SRAM (upcast to float32 for safety)
            if USE_COSINE:
                dot = tl.sum(evict_k.to(tl.float32) * last_k.to(tl.float32), axis=0)
                norm_e = tl.sqrt(tl.sum(evict_k.to(tl.float32) * evict_k.to(tl.float32), axis=0) + 1e-8)
                norm_l = tl.sqrt(tl.sum(last_k.to(tl.float32)  * last_k.to(tl.float32),  axis=0) + 1e-8)
                cos_sim = dot / (norm_e * norm_l)
                should_merge = cos_sim >= threshold_sq  # threshold_sq holds cosine threshold here
            else:
                diff = evict_k.to(tl.float32) - last_k.to(tl.float32)
                sq_dist = tl.sum(diff * diff, axis=0)
                should_merge = sq_dist <= threshold_sq

            if should_merge:
                # MERGE: Increment the count of the last packed token
                count_ptr = count_base + last_idx * stride_pc_s
                old_count = tl.load(count_ptr)
                tl.store(count_ptr, old_count + 1)
            else:
                # APPEND: Write the evicted token to the end of the packed array
                tl.store(pack_k_base + packed_len * stride_pack_s + offs_d * stride_pack_d, evict_k, mask=mask_d)
                tl.store(pack_v_base + packed_len * stride_pack_s + offs_d * stride_pack_d, evict_v, mask=mask_d)
                tl.store(count_base + packed_len * stride_pc_s, 1)
                tl.store(ts_base + packed_len * stride_pc_s, total_seq_len - exact_window_size)
                tl.store(len_ptr, packed_len + 1)
        else:
            # FIRST APPEND: The packed array is completely empty
            tl.store(pack_k_base + 0 * stride_pack_s + offs_d * stride_pack_d, evict_k, mask=mask_d)
            tl.store(pack_v_base + 0 * stride_pack_s + offs_d * stride_pack_d, evict_v, mask=mask_d)
            tl.store(count_base + 0 * stride_pc_s, 1)
            tl.store(ts_base + 0 * stride_pc_s, total_seq_len - exact_window_size)
            tl.store(len_ptr, 1)

    # ----------------------------------------------------
    # INSERTION: Put the new token into the Ring Buffer
    # ----------------------------------------------------
    new_k = tl.load(new_k_ptrs, mask=mask_d)
    new_v = tl.load(new_v_ptrs, mask=mask_d)
    tl.store(exact_k_base + write_idx * stride_exact_s + offs_d * stride_exact_d, new_k, mask=mask_d)
    tl.store(exact_v_base + write_idx * stride_exact_s + offs_d * stride_exact_d, new_v, mask=mask_d)

import torch
import triton
from transformers.cache_utils import Cache
import globVR

class HybridCompressedCache(Cache):
    def __init__(self, config, batch_size, dtype=torch.float16, exact_window_size=50, initial_capacity=1024):
        super().__init__()
        self.exact_window_size = exact_window_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.bsz = batch_size
        self.dtype = dtype
        self.device = torch.device("cuda")
        self.num_layers = config.num_hidden_layers
        
        # --- Memory Allocations ---
        self.capacity = initial_capacity
        self.k_packed = torch.zeros((self.num_layers, self.bsz, self.n_kv_heads, self.capacity, self.head_dim), device=self.device, dtype=self.dtype)
        self.v_packed = torch.zeros_like(self.k_packed)
        self.packed_counts = torch.zeros((self.num_layers, self.bsz, self.n_kv_heads, self.capacity), device=self.device, dtype=torch.int32)
        self.packed_timestamps = torch.zeros((self.num_layers, self.bsz, self.n_kv_heads, self.capacity), device=self.device, dtype=torch.int32)
        self.packed_lengths = torch.zeros((self.num_layers, self.bsz * self.n_kv_heads), device=self.device, dtype=torch.int32)
        
        self.k_exact = torch.zeros((self.num_layers, self.bsz, self.n_kv_heads, self.exact_window_size, self.head_dim), device=self.device, dtype=self.dtype)
        self.v_exact = torch.zeros_like(self.k_exact)
        
        # --- Pure Python Trackers (Zero GPU Sync) ---
        self.exact_seq_lens = [0 for _ in range(self.num_layers)]
        self.max_packed_len = [0 for _ in range(self.num_layers)]
        self.total_seq_lens = [0 for _ in range(self.num_layers)]

    def _expand_capacity(self, target_capacity):
        new_capacity = max(self.capacity * 2, target_capacity)
        
        new_k = torch.zeros((self.num_layers, self.bsz, self.n_kv_heads, new_capacity, self.head_dim), device=self.device, dtype=self.dtype)
        new_v = torch.zeros_like(new_k)
        new_counts = torch.zeros((self.num_layers, self.bsz, self.n_kv_heads, new_capacity), device=self.device, dtype=torch.int32)
        new_timestamps = torch.zeros((self.num_layers, self.bsz, self.n_kv_heads, new_capacity), device=self.device, dtype=torch.int32)
        
        old_cap = self.k_packed.shape[3]
        if old_cap > 0:
            new_k[:, :, :, :old_cap, :] = self.k_packed
            new_v[:, :, :, :old_cap, :] = self.v_packed
            new_counts[:, :, :, :old_cap] = self.packed_counts
            new_timestamps[:, :, :, :old_cap] = self.packed_timestamps
            
        self.k_packed = new_k
        self.v_packed = new_v
        self.packed_counts = new_counts
        self.packed_timestamps = new_timestamps
        self.capacity = new_capacity

    def initialize_from_prefill(self, layer_idx, k_packed, v_packed, counts, timestamps, k_exact, v_exact, original_q_len):
        p_len = k_packed.shape[2]
        
        # Safely track capacity on the CPU
        self.max_packed_len[layer_idx] = p_len
        if p_len > self.capacity:
            self._expand_capacity(p_len + 1024)
            
        self.k_packed[layer_idx, :, :, :p_len, :] = k_packed
        self.v_packed[layer_idx, :, :, :p_len, :] = v_packed
        self.packed_counts[layer_idx, :, :, :p_len] = counts
        self.packed_timestamps[layer_idx, :, :, :p_len] = timestamps
        self.packed_lengths[layer_idx, :] = p_len 
        
        e_len = k_exact.shape[2]
        self.k_exact[layer_idx, :, :, :e_len, :] = k_exact
        self.v_exact[layer_idx, :, :, :e_len, :] = v_exact
        
        # Initialize the Python trackers
        self.exact_seq_lens[layer_idx] = e_len
        self.total_seq_lens[layer_idx] = original_q_len

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape[2] > 1:
            return key_states, value_states

        # 1. Ensure GPU memory is large enough BEFORE launching the kernel.
        if self.max_packed_len[layer_idx] + 1 >= self.capacity:
            self._expand_capacity(self.capacity + 1024)

        # 2. Launch the Fused Triton Kernel
        BLOCK_D = triton.next_power_of_2(self.head_dim)
        grid = (self.bsz * self.n_kv_heads, )
        similarity_metric = getattr(globVR, 'row_similarity_metric', 'euclidean')
        use_cosine = (similarity_metric == 'cosine')
        raw_threshold = getattr(globVR, 'row_delta_threshold', 0.0)
        threshold_val = raw_threshold if use_cosine else raw_threshold ** 2

        fused_hybrid_decode_update_kernel[grid](
            key_states, value_states,
            self.k_exact[layer_idx], self.v_exact[layer_idx],
            self.k_packed[layer_idx], self.v_packed[layer_idx],
            self.packed_counts[layer_idx], self.packed_timestamps[layer_idx], self.packed_lengths[layer_idx],
            self.exact_seq_lens[layer_idx], self.total_seq_lens[layer_idx],
            threshold_val, self.exact_window_size,
            key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
            self.k_exact.stride(1), self.k_exact.stride(2), self.k_exact.stride(3), self.k_exact.stride(4),
            self.k_packed.stride(1), self.k_packed.stride(2), self.k_packed.stride(3), self.k_packed.stride(4),
            self.packed_counts.stride(1), self.packed_counts.stride(2), self.packed_counts.stride(3),
            self.n_kv_heads, self.head_dim,
            BLOCK_D=BLOCK_D,
            USE_COSINE=use_cosine,
        )

        # 3. Update capacity tracker (sync-free in the common case)
        # Packed cache only grows when the ring buffer is full AND a token gets appended
        # (vs merged). Worst case per step: +1. Only bump while eviction is active.
        if self.exact_seq_lens[layer_idx] >= self.exact_window_size:
            self.max_packed_len[layer_idx] += 1

            # Periodically reconcile with GPU truth to recover from merge-savings
            if self.total_seq_lens[layer_idx] % 256 == 0:
                self.max_packed_len[layer_idx] = self.packed_lengths[layer_idx].max().item()

        # 4. Increment the exact trackers (These are uncompressed)
        self.exact_seq_lens[layer_idx] += 1
        self.total_seq_lens[layer_idx] += 1

        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.total_seq_lens[layer_idx]

    def get_max_length(self) -> int: return None
    def get_max_cache_shape(self): return None


@triton.jit
def fused_hybrid_decode_kernel(
    Q,
    K_packed, V_packed,
    Packed_Counts,
    Packed_Lengths,
    K_exact, V_exact,
    Out,
    stride_qb, stride_qh, stride_qd,
    stride_kpb, stride_kph, stride_kpn, stride_kpd,
    stride_vpb, stride_vph, stride_vpn, stride_vpd,
    stride_cb, stride_ch, stride_cn,
    stride_lb, stride_lh,
    stride_keb, stride_keh, stride_ken, stride_ked,
    stride_veb, stride_veh, stride_ven, stride_ved,
    stride_ob, stride_oh, stride_od,
    sm_scale,
    exact_len,
    num_kv_groups,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,   # Padding dimension for tensor cores (16)
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    kv_head = pid_h // num_kv_groups

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim

    # Mask for "real" row (only row 0 is valid; others are zero padding)
    row_mask = offs_m == 0  # [BLOCK_M], True only at row 0

    # ---- Load query into row 0 of a [BLOCK_M, BLOCK_D] tile ----
    q_ptrs = Q + pid_b * stride_qb + pid_h * stride_qh + offs_d * stride_qd
    q_vec = tl.load(q_ptrs, mask=mask_d, other=0.0)   # [BLOCK_D]

    # Broadcast q into row 0, zeros elsewhere: [BLOCK_M, BLOCK_D]
    q_tile = tl.where(row_mask[:, None], q_vec[None, :], 0.0)

    # Load per-head packed length (stays on GPU)
    my_packed_len = tl.load(Packed_Lengths + pid_b * stride_lb + kv_head * stride_lh)

    # Running softmax state (only row 0 matters, but keep tile-shaped)
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Base pointers
    kp_base = K_packed + pid_b * stride_kpb + kv_head * stride_kph
    vp_base = V_packed + pid_b * stride_vpb + kv_head * stride_vph
    c_base  = Packed_Counts + pid_b * stride_cb + kv_head * stride_ch
    ke_base = K_exact + pid_b * stride_keb + kv_head * stride_keh
    ve_base = V_exact + pid_b * stride_veb + kv_head * stride_veh

    # =========================================================
    # PHASE 1: Packed cache with log-count weighting
    # =========================================================
    for start_n in range(0, my_packed_len, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < my_packed_len

        # Load K block: K is stored as [N, D]; we want K^T for the dot → [D, N]
        k_ptrs = kp_base + cur_n[None, :] * stride_kpn + offs_d[:, None] * stride_kpd
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)  # [BLOCK_D, BLOCK_N]

        # Tensor-core dot: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] → [BLOCK_M, BLOCK_N]
        qk = tl.dot(q_tile, k) * sm_scale

        # Load counts, apply log-weight (broadcasts across BLOCK_M rows)
        counts = tl.load(c_base + cur_n * stride_cn, mask=mask_n, other=1.0)
        qk = qk + tl.log(counts.to(tl.float32))[None, :]
        qk = tl.where(mask_n[None, :], qk, -float("inf"))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        # Load V block: [BLOCK_N, BLOCK_D]
        v_ptrs = vp_base + cur_n[:, None] * stride_vpn + offs_d[None, :] * stride_vpd
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Accumulate: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] → [BLOCK_M, BLOCK_D]
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # =========================================================
    # PHASE 2: Exact ring buffer
    # =========================================================
    for start_n in range(0, exact_len, BLOCK_N):
        cur_n = start_n + offs_n
        mask_n = cur_n < exact_len

        k_ptrs = ke_base + cur_n[None, :] * stride_ken + offs_d[:, None] * stride_ked
        k = tl.load(k_ptrs, mask=mask_n[None, :] & mask_d[:, None], other=0.0)

        qk = tl.dot(q_tile, k) * sm_scale
        qk = tl.where(mask_n[None, :], qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        v_ptrs = ve_base + cur_n[:, None] * stride_ven + offs_d[None, :] * stride_ved
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # =========================================================
    # Finalize — extract row 0 (the real result)
    # =========================================================
    acc = acc / l_i[:, None]

    # Store only row 0 (index 0 along BLOCK_M)
    out_ptrs = Out + pid_b * stride_ob + pid_h * stride_oh + offs_d * stride_od

    # Extract row 0 by masked sum (rows 1..15 are zero because q was zero there,
    # so their softmax accumulates uniformly — we just pick row 0 directly).
    acc_row0 = tl.sum(tl.where(row_mask[:, None], acc, 0.0), axis=0)
    tl.store(out_ptrs, acc_row0.to(Out.dtype.element_ty), mask=mask_d)