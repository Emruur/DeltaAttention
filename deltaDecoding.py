import torch
from transformers import Cache
import triton
import triton.language as tl




@triton.jit
def compressed_cache_update_kernel(
    evicted_k_ptr, evicted_v_ptr,
    k_packed_ptr, v_packed_ptr, 
    packed_counts_ptr, packed_lengths_ptr, # tracks the current length per head
    threshold_sq,
    stride_ek_b, stride_ek_h, stride_ek_d, # evicted strides (seq_len is 1)
    stride_pk_b, stride_pk_h, stride_pk_n, stride_pk_d, # packed strides
    head_dim, max_packed_len,
    BLOCK_D: tl.constexpr
):
    # One program per batch * head
    pid_bh = tl.program_id(0)
    
    # 1. Load the current packed length for this specific head
    current_len = tl.load(packed_lengths_ptr + pid_bh)
    last_idx = current_len - 1
    
    # Setup pointers for the head dimension
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < head_dim
    
    # 2. Load the evicted token's K and V
    ek_ptrs = evicted_k_ptr + pid_bh * stride_ek_h + offs_d * stride_ek_d
    evicted_k = tl.load(ek_ptrs, mask=mask_d, other=0.0)
    
    ev_ptrs = evicted_v_ptr + pid_bh * stride_ek_h + offs_d * stride_ek_d
    evicted_v = tl.load(ev_ptrs, mask=mask_d, other=0.0)
    
    # 3. Load the LAST anchor from the packed history
    pk_last_ptrs = k_packed_ptr + pid_bh * stride_pk_h + last_idx * stride_pk_n + offs_d * stride_pk_d
    last_anchor_k = tl.load(pk_last_ptrs, mask=mask_d, other=0.0)
    
    # 4. Compute Euclidean Distance
    diff = evicted_k - last_anchor_k
    sq_dist = tl.sum(diff * diff, axis=0)
    
    # 5. Branch: Merge or Append
    if sq_dist <= threshold_sq:
        # MERGE: Just increment the count of the last anchor
        count_ptr = packed_counts_ptr + pid_bh * max_packed_len + last_idx
        old_count = tl.load(count_ptr)
        tl.store(count_ptr, old_count + 1)
    else:
        # APPEND: Create a new anchor (if we haven't hit memory limits)
        if current_len < max_packed_len:
            new_idx = current_len
            
            # Store new K
            pk_new_ptrs = k_packed_ptr + pid_bh * stride_pk_h + new_idx * stride_pk_n + offs_d * stride_pk_d
            tl.store(pk_new_ptrs, evicted_k, mask=mask_d)
            
            # Store new V
            pv_new_ptrs = v_packed_ptr + pid_bh * stride_pk_h + new_idx * stride_pk_n + offs_d * stride_pk_d
            tl.store(pv_new_ptrs, evicted_v, mask=mask_d)
            
            # Set count to 1
            count_ptr = packed_counts_ptr + pid_bh * max_packed_len + new_idx
            tl.store(count_ptr, 1)
            
            # Update the length tracker
            tl.store(packed_lengths_ptr + pid_bh, current_len + 1)



class HybridCompressedCache(Cache):
    def __init__(self, config, batch_size, dtype=torch.float16, exact_window_size=50, initial_capacity=1024):
        super().__init__()
        self.exact_window_size = exact_window_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.bsz = batch_size
        self.dtype = dtype
        self.device = torch.device("cuda")
        
        # DYNAMIC CAPACITY TRACKING
        self.capacity = initial_capacity
        
        # PRE-ALLOCATED (BUT EXPANDABLE) COMPRESSED HISTORY
        self.k_packed = torch.zeros((self.bsz, self.n_kv_heads, self.capacity, self.head_dim), device=self.device, dtype=self.dtype)
        self.v_packed = torch.zeros_like(self.k_packed)
        self.packed_counts = torch.zeros((self.bsz, self.n_kv_heads, self.capacity), device=self.device, dtype=torch.int32)
        
        # Tracker for how many packed tokens exist PER HEAD
        self.packed_lengths = torch.zeros((self.bsz * self.n_kv_heads,), device=self.device, dtype=torch.int32)
        
        # EXACT WINDOW BUFFER
        self.k_exact = torch.zeros((self.bsz, self.n_kv_heads, 0, self.head_dim), device=self.device, dtype=self.dtype)
        self.v_exact = torch.zeros_like(self.k_exact)

    def _expand_capacity_if_needed(self):
        """Doubles the capacity of the packed cache if any head hits the limit."""
        max_current_len = self.packed_lengths.max().item()
        
        if max_current_len >= self.capacity:
            new_capacity = self.capacity * 2  
            
            # Allocate new larger tensors
            new_k = torch.zeros((self.bsz, self.n_kv_heads, new_capacity, self.head_dim), device=self.device, dtype=self.dtype)
            new_v = torch.zeros_like(new_k)
            new_counts = torch.zeros((self.bsz, self.n_kv_heads, new_capacity), device=self.device, dtype=torch.int32)
            
            # Copy over existing data
            new_k[:, :, :self.capacity, :] = self.k_packed
            new_v[:, :, :self.capacity, :] = self.v_packed
            new_counts[:, :, :self.capacity] = self.packed_counts
            
            # Overwrite references
            self.k_packed = new_k
            self.v_packed = new_v
            self.packed_counts = new_counts
            self.capacity = new_capacity

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        seq_len = key_states.shape[2]
        
        # 1. PREFILL PHASE
        if seq_len > 1:
            tail_len = min(seq_len, self.exact_window_size)
            self.k_exact = key_states[:, :, -tail_len:, :]
            self.v_exact = value_states[:, :, -tail_len:, :]
            return self.k_exact, self.v_exact
            
        # 2. DECODING PHASE
        current_exact_len = self.k_exact.shape[2]
        
        if current_exact_len == self.exact_window_size:
            self._expand_capacity_if_needed()
            
            evicted_k = self.k_exact[:, :, 0:1, :]
            evicted_v = self.v_exact[:, :, 0:1, :]
            
            self.k_exact = self.k_exact[:, :, 1:, :]
            self.v_exact = self.v_exact[:, :, 1:, :]
            
            BLOCK_D = triton.next_power_of_2(self.head_dim)
            grid = (self.bsz * self.n_kv_heads,)
            
            import globVR 
            threshold_sq = getattr(globVR, 'row_delta_threshold', 0.0) ** 2
            
            # Assumes compressed_cache_update_kernel is defined above this in modeling_llama.py
            compressed_cache_update_kernel[grid](
                evicted_k, evicted_v,
                self.k_packed, self.v_packed, 
                self.packed_counts, self.packed_lengths,
                threshold_sq,
                evicted_k.stride(0), evicted_k.stride(1), evicted_k.stride(3),
                self.k_packed.stride(0), self.k_packed.stride(1), self.k_packed.stride(2), self.k_packed.stride(3),
                self.head_dim, self.capacity,
                BLOCK_D=BLOCK_D
            )
            
        self.k_exact = torch.cat([self.k_exact, key_states], dim=2)
        self.v_exact = torch.cat([self.v_exact, value_states], dim=2)
        
        return self.k_exact, self.v_exact
    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Returns the total logical sequence length.
        The physical length is the sum of all represented tokens (packed counts) 
        plus the current length of the exact window buffer.
        """
        if self.k_exact is None or self.k_exact.shape[2] == 0:
            return 0
            
        # Since all heads process the same prompt, the sum of counts for any 
        # single head gives the total compressed history length.
        compressed_len = int(self.packed_counts[0, 0].sum().item())
        exact_len = self.k_exact.shape[2]
        
        return compressed_len + exact_len

    def get_max_length(self) -> int:
        """
        Returns the maximum sequence length. 
        Since our cache dynamically rolls and compresses to save memory, 
        it conceptually has no strict physical upper limit.
        """
        return None
        
    def get_max_cache_shape(self):
        # Newer versions of transformers sometimes look for this specific method
        return None