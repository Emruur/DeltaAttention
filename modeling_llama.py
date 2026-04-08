from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# --- MODIFIED IMPORTS START ---
from flashAttention import triton_flash_attention
from flashAttention import triton_flash_attention
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

try:
    from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
except ImportError:
    class FlashAttentionKwargs(TypedDict, total=False):
        padding_mask: torch.Tensor

try:
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:
    GradientCheckpointingLayer = nn.Module

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack

try:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
except ImportError:
    ALL_LAYERNORM_LAYERS = []

from transformers.utils import auto_docstring, can_return_tuple, logging

try:
    from transformers.utils import LossKwargs
except ImportError:
    class LossKwargs: pass

try:
    from transformers.utils import is_torch_flex_attn_available
except ImportError:
    def is_torch_flex_attn_available(): return False

# Import config directly from the main library
from transformers import LlamaConfig

if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask
    try:
        from transformers.integrations.flex_attention import make_flex_block_causal_mask
    except ImportError:
        pass

try:
    from transformers.integrations import use_kernel_forward_from_hub
except ImportError:
    def dummy_decorator(*args, **kwargs):
        def wrapper(obj): return obj
        return wrapper
    use_kernel_forward_from_hub = dummy_decorator

# --- MODIFIED IMPORTS END ---

# --- INJECTED GLOBAL TRACKERS ---
import globVR
import glob_set


## IMPORT THE KERNELS
import triton
import triton.language as tl
from tritonModules import (
    fused_row_delta_pack_kernel, 
    row_delta_euclidean_partitioned_kernel,
    _triton_gather_expand,
    _triton_expand_cumsum,
    _triton_segmented_cumsum_kernel
)

## FLASH ATTENTION IMPORTS
from flashAttention import triton_flash_attention
from deltaFlashAttention import fused_delta_flash_attention
from nmDeltaFlashAttention import fused_online_24_compress_kernel, run_24_sparse_flash

logger = logging.get_logger(__name__)
@use_kernel_forward_from_hub("RMSNorm")
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    
    # --- GUARDED TIMING START ---
    # In eager mode, query shape is [bsz, num_heads, seq_len, head_dim]
    is_prefill = query.shape[2] > 1
    do_time = getattr(globVR, 'time_internal', False) and is_prefill

    if do_time:
        start_evt_reg = torch.cuda.Event(enable_timing=True)
        end_evt_reg = torch.cuda.Event(enable_timing=True)
        start_evt_reg.record()

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    
    if do_time:
        end_evt_reg.record()
        # Only queue the event if we are in the prefill stage
        glob_set.queue_event_pair('time_regular_matmul', start_evt_reg, end_evt_reg)
    # --- GUARDED TIMING END ---


    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    ### DELTA ATTENTION METHODS
    def get_row_delta_mat_triton(self, input_states, threshold, similarity_metric="euclidean", divide_to=4):
        """Partitioned Triton-accelerated structured delta matrix computation."""
        bsz, n_head, seq_len, head_dim = input_states.shape
        device = input_states.device
        
        if not input_states.is_contiguous():
            input_states = input_states.contiguous()

        # SAFE ALLOCATION
        delta_all = torch.empty_like(input_states)
        keep_mask = torch.zeros((bsz, n_head, seq_len), dtype=torch.bool, device=device)
        
        BLOCK_D = triton.next_power_of_2(head_dim)
        chunk_size = (seq_len + divide_to - 1) // divide_to
        
        # SAFE ALLOCATION
        chunk_counts = torch.empty((bsz, n_head, divide_to), dtype=torch.int32, device=device)
        
        if similarity_metric == "euclidean":
            threshold_sq = threshold ** 2
            grid = (bsz * n_head, divide_to)
            
            row_delta_euclidean_partitioned_kernel[grid](
                input_states, delta_all, keep_mask, chunk_counts,
                threshold_sq,
                input_states.stride(0), input_states.stride(1), input_states.stride(2), input_states.stride(3),
                keep_mask.stride(0), keep_mask.stride(1), keep_mask.stride(2),
                chunk_counts.stride(0), chunk_counts.stride(1), chunk_counts.stride(2),
                seq_len, head_dim, n_head, chunk_size,
                BLOCK_D=BLOCK_D
            )
        else:
            raise NotImplementedError(f"Partitioned Triton kernel for '{similarity_metric}' is not yet implemented.")
            
        return delta_all, keep_mask, chunk_counts
    def _patch_hybrid_attention(self, out, regular_x, regular_y, bsz, seq_len, blk_size):
        """Efficiently computes exact attention ONLY for the Sink and Local Diagonal Blocks."""
        sink_size = getattr(globVR, 'sink_size', 0)
        num_heads = self.config.num_attention_heads
        
        if sink_size > 0:
            k_sink = regular_y[..., :sink_size]
            sink_scores = torch.matmul(regular_x, k_sink)
            out[..., :sink_size] = sink_scores

        if blk_size > 0:
            n_blocks = seq_len // blk_size
            trunc_len = n_blocks * blk_size
            
            if n_blocks > 0:
                q_blocked = regular_x[..., :trunc_len, :].view(bsz, num_heads, n_blocks, blk_size, -1)
                k_blocked = regular_y[..., :trunc_len].view(bsz, num_heads, -1, n_blocks, blk_size).permute(0, 1, 3, 2, 4)
                block_scores = torch.matmul(q_blocked, k_blocked)
                out_view = out[..., :trunc_len, :trunc_len].view(bsz, num_heads, n_blocks, blk_size, n_blocks, blk_size)
                
                for i in range(n_blocks):
                    out_view[:, :, i, :, i, :] = block_scores[:, :, i, :, :]

            if seq_len > trunc_len:
                q_tail = regular_x[..., trunc_len:, :]
                k_tail = regular_y[..., trunc_len:]
                tail_scores = torch.matmul(q_tail, k_tail)
                out[..., trunc_len:, trunc_len:] = tail_scores

        return out

    def opt_delta_mm_pattern_dn(self, delta_y, regular_x, regular_y, bsz, seq_len, dim_out, blk_size, keep_mask=None, divide_to=None, chunk_counts=None):
        num_heads = self.config.num_attention_heads
        
        if keep_mask is not None:
            counts = keep_mask.sum(dim=-1)
            max_active = counts.max().item()

            if max_active == 0:
                # SAFE ALLOCATION
                delta_out = torch.zeros(bsz, num_heads, seq_len, seq_len, dtype=regular_x.dtype, device=regular_x.device)
            else:
                if getattr(globVR, 'time_internal', False):
                    start_evt_sm = torch.cuda.Event(enable_timing=True)
                    end_evt_sm = torch.cuda.Event(enable_timing=True)
                    start_evt_sm.record()

                seq_idx = torch.arange(seq_len, dtype=torch.int32, device=keep_mask.device)
                sort_keys = torch.where(keep_mask, seq_idx, seq_len)
                active_indices = sort_keys.sort(dim=-1)[0][:, :, :max_active]

                # VRAM MGMT + CLAMP
                del sort_keys
                del seq_idx
                active_indices = torch.clamp(active_indices, max=seq_len - 1).to(torch.int32)

                if getattr(globVR, 'time_internal', False):
                    start_evt_gather = torch.cuda.Event(enable_timing=True)
                    end_evt_gather = torch.cuda.Event(enable_timing=True)
                    start_evt_gather.record()

                head_dim = delta_y.shape[2]
                
                # SAFE ALLOCATION
                k_packed_view = torch.empty((bsz, num_heads, head_dim, max_active), dtype=delta_y.dtype, device=delta_y.device)

                BLOCK_D = triton.next_power_of_2(head_dim)
                grid_gather = (bsz * num_heads, max_active)
                
                _triton_gather_expand[grid_gather](
                    delta_y, active_indices, k_packed_view,
                    delta_y.stride(0), delta_y.stride(1), delta_y.stride(2), delta_y.stride(3),
                    active_indices.stride(0), active_indices.stride(1), active_indices.stride(2),
                    k_packed_view.stride(0), k_packed_view.stride(1), k_packed_view.stride(2), k_packed_view.stride(3),
                    num_heads, self.num_key_value_groups, seq_len, head_dim,
                    BLOCK_D=BLOCK_D
                )

                del active_indices

                if getattr(globVR, 'time_internal', False):
                    end_evt_gather.record()
                    glob_set.queue_event_pair('time_gather_expand', start_evt_gather, end_evt_gather)

                if getattr(globVR, 'time_internal', False):
                    start_evt_matmul = torch.cuda.Event(enable_timing=True)
                    end_evt_matmul = torch.cuda.Event(enable_timing=True)
                    start_evt_matmul.record()

                packed_cumsum = torch.matmul(regular_x, k_packed_view)
                del k_packed_view

                if getattr(globVR, 'time_internal', False):
                    end_evt_matmul.record()
                    glob_set.queue_event_pair('time_dense_matmul', start_evt_matmul, end_evt_matmul)

                if getattr(globVR, 'time_internal', False):
                    start_evt_cumsum = torch.cuda.Event(enable_timing=True)
                    end_evt_cumsum = torch.cuda.Event(enable_timing=True)
                    start_evt_cumsum.record()

                boundaries = chunk_counts.cumsum(dim=-1).to(torch.int32)
                boundaries = torch.repeat_interleave(boundaries, self.num_key_value_groups, dim=1)
                
                BLOCK_A = triton.next_power_of_2(max_active)
                MAX_CHUNKS = triton.next_power_of_2(divide_to)

                grid_cumsum = (bsz, num_heads, seq_len)
                _triton_segmented_cumsum_kernel[grid_cumsum](
                    packed_cumsum, boundaries,
                    packed_cumsum.stride(0), packed_cumsum.stride(1), packed_cumsum.stride(2), packed_cumsum.stride(3),
                    boundaries.stride(0), boundaries.stride(1), boundaries.stride(2),
                    max_active, divide_to,
                    BLOCK_A=BLOCK_A,
                    MAX_CHUNKS=MAX_CHUNKS
                )
                del boundaries

                cumsum_mask = torch.cumsum(keep_mask.to(torch.int32), dim=-1) -1

                if getattr(globVR, 'time_internal', False):
                    end_evt_cumsum.record()
                    glob_set.queue_event_pair('time_small_cumsum', start_evt_cumsum, end_evt_cumsum)

                if getattr(globVR, 'time_internal', False):
                    start_evt_expand = torch.cuda.Event(enable_timing=True)
                    end_evt_expand = torch.cuda.Event(enable_timing=True)
                    start_evt_expand.record()

                # SAFE ALLOCATION
                delta_out = torch.empty(bsz, num_heads, seq_len, seq_len, dtype=regular_x.dtype, device=regular_x.device)
                
                BLOCK_Q = 64
                BLOCK_K = 64
                grid_expand = (bsz * num_heads, triton.cdiv(seq_len, BLOCK_Q), triton.cdiv(seq_len, BLOCK_K))
                
                _triton_expand_cumsum[grid_expand](
                    packed_cumsum, cumsum_mask, delta_out,
                    packed_cumsum.stride(0), packed_cumsum.stride(1), packed_cumsum.stride(2), packed_cumsum.stride(3),
                    cumsum_mask.stride(0), cumsum_mask.stride(1), cumsum_mask.stride(2),
                    delta_out.stride(0), delta_out.stride(1), delta_out.stride(2), delta_out.stride(3),
                    num_heads, self.num_key_value_groups, seq_len, seq_len,
                    BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
                )

                del packed_cumsum
                del cumsum_mask

                if getattr(globVR, 'time_internal', False):
                    end_evt_expand.record()
                    glob_set.queue_event_pair('time_triton_expand', start_evt_expand, end_evt_expand)

                if getattr(globVR, 'time_internal', False):
                    end_evt_sm.record()
                    glob_set.queue_event_pair('time_sparse_matmul_total', start_evt_sm, end_evt_sm)
                                    
        else:
            delta_out = torch.cumsum(torch.matmul(regular_x, delta_y), dim=-1)

        if getattr(globVR, 'time_internal', False):
            start_evt_patch = torch.cuda.Event(enable_timing=True)
            end_evt_patch = torch.cuda.Event(enable_timing=True)
            start_evt_patch.record()

        output = self._patch_hybrid_attention(delta_out, regular_x, regular_y, bsz, seq_len, blk_size)
        
        if getattr(globVR, 'time_internal', False):
            end_evt_patch.record()
            glob_set.queue_event_pair('time_patch_hybrid_attention', start_evt_patch, end_evt_patch)

        return output

    ## UNPARTITIONED DELTA

    def get_row_delta_mat_triton_unpartitioned(self, input_states, threshold, similarity_metric="euclidean"):
        """
        Fused temporal delta and memory packing kernel.
        Only valid when processing the entire sequence sequentially (divide_to=1).
        """
        bsz, n_head, seq_len, head_dim = input_states.shape
        device = input_states.device
        
        if not input_states.is_contiguous():
            input_states = input_states.contiguous()

        # 1. Pre-allocate outputs
        # Packed delta is the same size, but all valid data will be shoved to the left (dim 2)
        packed_delta_all = torch.zeros_like(input_states) 
        
        # Cumsum mask replaces the boolean keep_mask. It tracks the running active_idx
        cumsum_mask = torch.zeros((bsz, n_head, seq_len), dtype=torch.int32, device=device)
        
        # We need to know exactly how many tokens were kept per batch/head
        counts = torch.zeros((bsz, n_head), dtype=torch.int32, device=device)
        
        # 2. Setup Triton launch parameters
        BLOCK_D = triton.next_power_of_2(head_dim)
        
        if similarity_metric == "euclidean":
            threshold_sq = threshold ** 2
            
            # Grid: 1D grid over Batch * Heads (No sequence partitioning!)
            grid = (bsz, n_head)
            
            fused_row_delta_pack_kernel[grid](
                input_states, packed_delta_all, cumsum_mask, counts,
                threshold_sq,
                # Input strides
                input_states.stride(0), input_states.stride(1), input_states.stride(2), input_states.stride(3),
                # Packed Delta strides
                packed_delta_all.stride(0), packed_delta_all.stride(1), packed_delta_all.stride(2), packed_delta_all.stride(3),
                # Cumsum strides
                cumsum_mask.stride(0), cumsum_mask.stride(1), cumsum_mask.stride(2),
                # Counts strides
                counts.stride(0), counts.stride(1),
                seq_len, head_dim,
                BLOCK_D=BLOCK_D
            )
        else:
            raise NotImplementedError(f"Fused kernel for '{similarity_metric}' is not yet implemented.")
            
        return packed_delta_all, cumsum_mask, counts

    def opt_packed_mm_pattern_dn(
        self, k_packed, cumsum_mask, active_counts, regular_x, regular_y, 
        bsz, seq_len, dim_out, blk_size
    ):
        """
        Streamlined Matrix Multiplication for pre-packed key matrices.
        Dynamically slices padding based on active_counts.
        """
        num_heads = self.config.num_attention_heads

        if getattr(globVR, 'time_internal', False):
            start_evt_sm = torch.cuda.Event(enable_timing=True)
            end_evt_sm = torch.cuda.Event(enable_timing=True)
            start_evt_sm.record()

        # ==========================================
        # --- 0. THE DYNAMIC SLICE ---
        # ==========================================
        max_active = active_counts.max().item()

        if max_active == 0:
            delta_out = torch.zeros(bsz, num_heads, seq_len, seq_len, dtype=regular_x.dtype, device=regular_x.device)
            return self._patch_hybrid_attention(delta_out, regular_x, regular_y, bsz, seq_len, blk_size)

        k_packed_sliced = k_packed[:, :, :max_active, :]

        # ==========================================
        # --- 1. DENSE MATMUL [cuBLAS] ---
        # ==========================================
        if getattr(globVR, 'time_internal', False):
            start_evt_matmul = torch.cuda.Event(enable_timing=True)
            end_evt_matmul = torch.cuda.Event(enable_timing=True)
            start_evt_matmul.record()

        k_packed_repeated = repeat_kv(k_packed_sliced, self.num_key_value_groups)
        packed_scores = torch.matmul(regular_x, k_packed_repeated.transpose(-1, -2))

        if getattr(globVR, 'time_internal', False):
            end_evt_matmul.record()
            glob_set.queue_event_pair('time_dense_matmul', start_evt_matmul, end_evt_matmul)

        # ==========================================
        # --- 2. THE (ACTUALLY) SMALL CUMSUM ---
        # ==========================================
        if getattr(globVR, 'time_internal', False):
            start_evt_cumsum = torch.cuda.Event(enable_timing=True)
            end_evt_cumsum = torch.cuda.Event(enable_timing=True)
            start_evt_cumsum.record()

        packed_scores.cumsum_(dim=-1)

        if getattr(globVR, 'time_internal', False):
            end_evt_cumsum.record()
            glob_set.queue_event_pair('time_small_cumsum', start_evt_cumsum, end_evt_cumsum)

        # ==========================================
        # --- 3. O(1) TRITON EXPAND ---
        # ==========================================
        if getattr(globVR, 'time_internal', False):
            start_evt_expand = torch.cuda.Event(enable_timing=True)
            end_evt_expand = torch.cuda.Event(enable_timing=True)
            start_evt_expand.record()

        delta_out = torch.empty(bsz, num_heads, seq_len, seq_len, dtype=regular_x.dtype, device=regular_x.device)
        
        BLOCK_Q = 64
        BLOCK_K = 64
        grid_expand = (bsz * num_heads, triton.cdiv(seq_len, BLOCK_Q), triton.cdiv(seq_len, BLOCK_K))
        
        _triton_expand_cumsum[grid_expand](
            packed_scores, cumsum_mask, delta_out,
            packed_scores.stride(0), packed_scores.stride(1), packed_scores.stride(2), packed_scores.stride(3),
            cumsum_mask.stride(0), cumsum_mask.stride(1), cumsum_mask.stride(2),
            delta_out.stride(0), delta_out.stride(1), delta_out.stride(2), delta_out.stride(3),
            num_heads, self.num_key_value_groups, seq_len, seq_len,
            BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K
        )

        if getattr(globVR, 'time_internal', False):
            end_evt_expand.record()
            glob_set.queue_event_pair('time_triton_expand', start_evt_expand, end_evt_expand)

        if getattr(globVR, 'time_internal', False):
            end_evt_sm.record()
            glob_set.queue_event_pair('time_sparse_matmul_total', start_evt_sm, end_evt_sm)

        # ==========================================
        # --- 4. PATCH HYBRID ATTENTION ---
        # ==========================================
        output = self._patch_hybrid_attention(delta_out, regular_x, regular_y, bsz, seq_len, blk_size)
        
        return output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        is_prefill = hidden_states.shape[1] > 1
        do_time = getattr(globVR, 'time_internal', False) and is_prefill

        if do_time:
            start_evt_attn = torch.cuda.Event(enable_timing=True)
            end_evt_attn = torch.cuda.Event(enable_timing=True)
            start_evt_attn.record()

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        bsz, q_len, _ = hidden_states.size()

        # =========================================================================
        # --- PATH 1: DELTA ATTENTION (PREFILL ONLY) ---
        # =========================================================================
        if is_prefill and getattr(globVR, 'delta_pf_key_on', 0) == 1 and globVR.delta_type == "row":
            divide_to = getattr(globVR, 'divide_to', 0)
            active_counts = None
            
            if do_time:
                start_evt_rd = torch.cuda.Event(enable_timing=True)
                end_evt_rd = torch.cuda.Event(enable_timing=True)
                start_evt_rd.record()

            # --- DYNAMIC BRANCH: Unpartitioned vs Partitioned ---
            if divide_to == 0:
                k_packed, cumsum_mask, active_counts = self.get_row_delta_mat_triton_unpartitioned(
                    key_states, 
                    getattr(globVR, 'row_delta_threshold', 0.0), 
                    getattr(globVR, 'row_similarity_metric', 'euclidean')
                )
                key_delta_all = k_packed
                
                # Derive keep_mask: A token is kept if its cumsum index increased, OR if it's index 0.
                shifted_cumsum = torch.cat([torch.zeros_like(cumsum_mask[:, :, :1]), cumsum_mask[:, :, :-1]], dim=-1)
                keep_mask = (cumsum_mask > shifted_cumsum)
                keep_mask[:, :, 0] = True # Anchor token is always kept
                
                chunk_counts = None
            else:
                key_delta_all, keep_mask, chunk_counts = self.get_row_delta_mat_triton(
                    key_states, 
                    getattr(globVR, 'row_delta_threshold', 0.0), 
                    getattr(globVR, 'row_similarity_metric', 'euclidean'), 
                    divide_to
                )

            if do_time:
                end_evt_rd.record()
                glob_set.queue_event_pair('time_get_row_delta', start_evt_rd, end_evt_rd)

            # Metadata and Sparsity Tracking
            glob_set.store_delta(getattr(globVR, 'delta_key', ''), self.layer_idx, key_delta_all, getattr(globVR, 'collect_delta_pf_key', 0))
            blk_size = round(q_len * getattr(globVR, 'scale', 0.0))
            new_scale = blk_size / q_len
            glob_set.compute_sparsity_scale(key_delta_all, new_scale, keep_mask=keep_mask, active_counts=active_counts)

            # Expand K and V for Grouped Query Attention (GQA)
            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

            if do_time:
                start_evt_mm = torch.cuda.Event(enable_timing=True)
                end_evt_mm = torch.cuda.Event(enable_timing=True)
                start_evt_mm.record()

            # --- DYNAMIC BRANCH: Unpartitioned FUSED MM vs Partitioned MM ---
            if divide_to == 0:
                # Fused Triton Path: Handles Exact Diagonal, Delta Math, Cumsum, Expand, Causal Mask, Softmax, and V MatMul
                
                # Expand the tracking masks for GQA to match the expanded Q/K/V heads
                cumsum_mask_expanded = repeat_kv(cumsum_mask.unsqueeze(-1), self.num_key_value_groups).squeeze(-1)
                keep_mask_expanded = repeat_kv(keep_mask.unsqueeze(-1), self.num_key_value_groups).squeeze(-1).to(torch.int32)
                k_packed_expanded = repeat_kv(k_packed, self.num_key_value_groups)
                
                attn_output = fused_delta_flash_attention(
                    query_states, 
                    key_states_expanded,          # Dense Keys for the exact diagonal
                    k_packed_expanded,            # Packed delta keys
                    value_states_expanded,        # Dense Values
                    cumsum_mask_expanded,         # Routing mask
                    keep_mask_expanded,           # Baseline extraction mask
                    self.scaling, 
                    int(blk_size)
                )
                
                # We do not materialize the NxN attention matrix in the fused path
                attn_weights = None 
                
            else:
                # Fallback Partitioned Path
                attn_weights = self.opt_delta_mm_pattern_dn(
                    key_delta_all.transpose(2,3), 
                    query_states, 
                    key_states_expanded.transpose(2,3), 
                    bsz, q_len, q_len, int(blk_size), 
                    keep_mask=keep_mask, 
                    divide_to=divide_to, 
                    chunk_counts=chunk_counts
                )
                
                attn_weights.mul_(self.scaling)

                if attention_mask is not None:
                    causal_mask = attention_mask[:, :, :, : key_states_expanded.shape[-2]]
                    attn_weights.add_(causal_mask)

                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = nn.functional.dropout(attn_weights, p=0.0 if not self.training else self.attention_dropout, training=self.training)

                attn_output = torch.matmul(attn_weights, value_states_expanded)

            if do_time:
                end_evt_mm.record()
                glob_set.queue_event_pair('time_delta_mm_pattern', start_evt_mm, end_evt_mm)

            # Final Reshape and Projection
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)


        # =========================================================================
        # --- PATH 2: TRITON FLASH ATTENTION BASELINE ---
        # =========================================================================
        elif getattr(globVR, 'flash', False) and is_prefill:
            if do_time:
                start_evt_flash = torch.cuda.Event(enable_timing=True)
                end_evt_flash = torch.cuda.Event(enable_timing=True)
                start_evt_flash.record()

            # Expand K and V for Grouped Query Attention (GQA)
            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)
            
            # If attention_mask is passed during prefill, it's typically causal
            is_causal = True  # Decoder prefill is ALWAYS causal!
            ##TODo delta path??

            # Execute our imported custom Triton kernel
            attn_output = triton_flash_attention(
                query_states, 
                key_states_expanded, 
                value_states_expanded, 
                causal=is_causal, 
                sm_scale=self.scaling
            )
            
            # Flash Attention does not instantiate the N x N attention matrix
            attn_weights = None
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            if do_time:
                end_evt_flash.record()
                glob_set.queue_event_pair('time_triton_flash_attention', start_evt_flash, end_evt_flash)

        # =========================================================================
        # --- PATH 3: STANDARD LLAMA PATH (DECODING / EAGER FALLBACK) ---
        # =========================================================================
        else:
            attention_interface: Callable = eager_attention_forward

            if self.config._attn_implementation != "eager":
                if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                    logger.warning_once(
                        "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                        'eager attention.'
                    )
                else:
                    attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

            attn_output, attn_weights = attention_interface(
                self, query_states, key_states, value_states, attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling, **kwargs,
            )

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

        if do_time:
            end_evt_attn.record()
            glob_set.queue_event_pair('time_forward_total', start_evt_attn, end_evt_attn)

        return attn_output, attn_weights

class LlamaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


@auto_docstring
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LlamaRMSNorm):
            module.weight.data.fill_(1.0)


@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:


        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        is_prefill = (input_ids.shape[1] > 1) if input_ids is not None else (inputs_embeds.shape[1] > 1)
        if hasattr(globVR, "sequence_lengths") and is_prefill:
            globVR.sequence_lengths.append(inputs_embeds.shape[1])

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_compilable_cache = past_key_values.is_compileable if past_key_values is not None else False

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_compilable_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype = input_tensor.dtype
        sequence_length = input_tensor.shape[1]
        if using_compilable_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=cache_position.device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring
class LlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring(
    custom_intro="""
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`LlamaForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """
)
class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@auto_docstring
class LlamaForQuestionAnswering(LlamaPreTrainedModel):
    base_model_prefix = "transformer"

    # Copied from transformers.models.bloom.modeling_bloom.BloomForQuestionAnswering.__init__ with Bloom->Llama
    def __init__(self, config):
        super().__init__(config)
        self.transformer = LlamaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class LlamaForTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "LlamaForCausalLM",
    "LlamaModel",
    "LlamaPreTrainedModel",
    "LlamaForSequenceClassification",
    "LlamaForQuestionAnswering",
    "LlamaForTokenClassification",
]
