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
    chunked_eval_kernel,
    parallel_scatter_pack_kv_kernel
)

## FLASH ATTENTION IMPORTS
from flashAttention import triton_flash_attention
from deltaFlashAttention import hybrid_compressed_flash_kernel, hybrid_compressed_flash_kernel_sanity
from deltaDecoding import fused_hybrid_decode_kernel

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

    def _forward_hybrid_flash(
        self, q, k_dense, v_dense, k_packed, v_packed,
        packed_timestamps, packed_counts, dense_window_size
    ):
        """Python wrapper for the Two-Phase Hybrid Flash Kernel."""
        batch_size, num_heads, q_len, head_dim = q.shape
        k_len = k_dense.shape[2]
        num_packed = k_packed.shape[2]

        q = q.contiguous()
        k_dense = k_dense.contiguous()
        v_dense = v_dense.contiguous()
        k_packed = k_packed.contiguous()
        v_packed = v_packed.contiguous()
        packed_timestamps = packed_timestamps.contiguous().to(torch.int32)
        packed_counts = packed_counts.contiguous().to(torch.float32)

        if getattr(globVR, 'no_count', False):
            packed_counts = torch.ones_like(packed_counts)

        out = torch.empty_like(q)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_D = triton.next_power_of_2(head_dim)

        grid = (triton.cdiv(q_len, BLOCK_M), batch_size * num_heads, 1)

        hybrid_compressed_flash_kernel[grid](
            q, k_dense, v_dense,
            k_packed, v_packed,
            packed_timestamps, packed_counts,
            out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_dense.stride(0), k_dense.stride(1), k_dense.stride(2), k_dense.stride(3),
            v_dense.stride(0), v_dense.stride(1), v_dense.stride(2), v_dense.stride(3),
            k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
            v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
            packed_timestamps.stride(0), packed_timestamps.stride(1), packed_timestamps.stride(2),
            packed_counts.stride(0), packed_counts.stride(1), packed_counts.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            self.scaling,
            q_len, k_len, num_packed, head_dim, num_heads,
            dense_window_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=8,
            num_stages=1,
        )
        return out

    def _forward_hybrid_flash_sanity(
        self, q, k_dense, v_dense, k_packed, v_packed,
        packed_timestamps, packed_counts, dense_window_size
    ):
        """Wrapper for the sanity-check kernel: Phase 1 iterates packed K without contributing."""
        batch_size, num_heads, q_len, head_dim = q.shape
        k_len = k_dense.shape[2]
        num_packed = k_packed.shape[2]

        q = q.contiguous()
        k_dense = k_dense.contiguous()
        v_dense = v_dense.contiguous()
        k_packed = k_packed.contiguous()
        v_packed = v_packed.contiguous()
        packed_timestamps = packed_timestamps.contiguous().to(torch.int32)
        packed_counts = packed_counts.contiguous().to(torch.float32)

        out = torch.empty_like(q)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_D = triton.next_power_of_2(head_dim)

        grid = (triton.cdiv(q_len, BLOCK_M), batch_size * num_heads, 1)

        hybrid_compressed_flash_kernel_sanity[grid](
            q, k_dense, v_dense,
            k_packed, v_packed,
            packed_timestamps, packed_counts,
            out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_dense.stride(0), k_dense.stride(1), k_dense.stride(2), k_dense.stride(3),
            v_dense.stride(0), v_dense.stride(1), v_dense.stride(2), v_dense.stride(3),
            k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
            v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
            packed_timestamps.stride(0), packed_timestamps.stride(1), packed_timestamps.stride(2),
            packed_counts.stride(0), packed_counts.stride(1), packed_counts.stride(2),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            self.scaling,
            q_len, k_len, num_packed, head_dim, num_heads,
            dense_window_size,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
            num_warps=8,
            num_stages=1,
        )
        return out

    def _forward_fused_hybrid_decode(self, q, past_key_value):
        """
        q: [bsz, num_heads, 1, head_dim]  (decode, q_len=1)
        Returns: [bsz, num_heads, 1, head_dim]
        """
        bsz, num_heads, _, head_dim = q.shape
        layer_idx = self.layer_idx

        # Squeeze q_len=1 for the kernel
        q = q.squeeze(2).contiguous()  # [bsz, num_heads, head_dim]

        # Fetch this layer's cache slices (views, no copies)
        k_packed = past_key_value.k_packed[layer_idx]          # [bsz, n_kv, cap, d]
        v_packed = past_key_value.v_packed[layer_idx]
        counts   = past_key_value.packed_counts[layer_idx]     # [bsz, n_kv, cap]
        lengths  = past_key_value.packed_lengths[layer_idx]    # [bsz * n_kv] flat
        k_exact  = past_key_value.k_exact[layer_idx]           # [bsz, n_kv, window, d]
        v_exact  = past_key_value.v_exact[layer_idx]

        # Reshape flat lengths to [bsz, n_kv] for 2D strides
        n_kv = self.config.num_key_value_heads
        lengths = lengths.view(bsz, n_kv)

        exact_len = min(
            past_key_value.exact_seq_lens[layer_idx],
            past_key_value.exact_window_size,
        )

        out = torch.empty_like(q)

        BLOCK_M = 16      # tensor-core minimum
        BLOCK_N = 128     # bigger blocks, fewer iters
        BLOCK_D = triton.next_power_of_2(head_dim)
        grid = (bsz, num_heads)

        #TODO doTime
        if True:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        fused_hybrid_decode_kernel[grid](
            q, k_packed, v_packed, counts, lengths,
            k_exact, v_exact, out,
            q.stride(0), q.stride(1), q.stride(2),
            k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
            v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
            counts.stride(0), counts.stride(1), counts.stride(2),
            lengths.stride(0), lengths.stride(1),
            k_exact.stride(0), k_exact.stride(1), k_exact.stride(2), k_exact.stride(3),
            v_exact.stride(0), v_exact.stride(1), v_exact.stride(2), v_exact.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            self.scaling,
            exact_len,
            self.num_key_value_groups,
            head_dim=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
        )

        if True:
            end.record()
            glob_set.queue_event_pair('time_fused_kernel_only', start, end)

        return out.unsqueeze(2)  # back to [bsz, num_heads, 1, head_dim]

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
        do_time = getattr(globVR, 'time_internal', False)

        # Mode Identification
        is_delta_decode = not is_prefill and getattr(globVR, 'delta_decode', False)
        is_baseline_decode = not is_prefill and not getattr(globVR, 'delta_decode', False)

        # =========================================================================
        # --- GLOBAL TIMING INIT ---
        # =========================================================================
        if do_time:
            start_evt_forward_total = torch.cuda.Event(enable_timing=True)
            end_evt_forward_total = torch.cuda.Event(enable_timing=True)
            start_evt_forward_total.record()

        # =========================================================================
        # --- STANDARD PROJECTIONS & ROPE ---
        # =========================================================================
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # =========================================================================
        # --- CACHE UPDATE (Handles both Triton Kernel & Baseline) ---
        # =========================================================================
        if do_time and not is_prefill:
            start_evt_update = torch.cuda.Event(enable_timing=True)
            end_evt_update = torch.cuda.Event(enable_timing=True)
            start_evt_update.record()

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if do_time and not is_prefill:
            end_evt_update.record()
            if is_delta_decode:
                glob_set.queue_event_pair('time_delta_decode_cache_update', start_evt_update, end_evt_update)
            else:
                glob_set.queue_event_pair('time_baseline_decode_cache_update', start_evt_update, end_evt_update)

        bsz, q_len, _ = hidden_states.size()

        # =========================================================================
        # --- KV CACHE SIZE TRACKER (Runs only on Layer 0 to prevent spam) ---
        # =========================================================================
        if not is_prefill and self.layer_idx == 0 and hasattr(past_key_value, 'exact_seq_lens'):
            total_len = past_key_value.total_seq_lens[0]
            log_interval = getattr(globVR, 'kv_log_interval', 10)
            
            # Print at specific intervals (e.g., every 10 or 100 steps)
            if total_len > 0 and total_len % log_interval == 0:
                exact_len = min(past_key_value.exact_seq_lens[0], past_key_value.exact_window_size)
                # Average packed tokens per head (max_packed_len is the worst-case head and understates compression)
                avg_packed_len = past_key_value.packed_lengths[0].float().mean().item()

                savings = (1 - (exact_len + avg_packed_len) / total_len) * 100 if total_len > 0 else 0

                if not hasattr(globVR, 'kv_compression_samples'):
                    globVR.kv_compression_samples = []
                globVR.kv_compression_samples.append(savings)

        # =========================================================================
        # --- PATH 1: SMART HYBRID ATTENTION (PREFILL ONLY) ---
        # =========================================================================
        if is_prefill and getattr(globVR, 'delta_pf_key_on', 0) == 1 and globVR.delta_type == "row":
            
            fixed_chunk = getattr(globVR, 'chunk_size', 0)
            if fixed_chunk > 0:
                chunk_size = min(fixed_chunk, q_len)
                num_chunks = max(1, (q_len + chunk_size - 1) // chunk_size)
            else:
                divide_to = getattr(globVR, 'divide_to', 0)
                actual_divide_to = 1 if divide_to == 0 else divide_to
                chunk_size = (q_len + actual_divide_to - 1) // actual_divide_to
                num_chunks = actual_divide_to

            if do_time:
                start_evt_rd = torch.cuda.Event(enable_timing=True)
                end_evt_rd = torch.cuda.Event(enable_timing=True)
                start_evt_rd.record()

            # --- 1. EVALUATE DELTAS ---
            BLOCK_D_EVAL = triton.next_power_of_2(self.head_dim)

            keep_mask = torch.zeros((bsz, self.config.num_key_value_heads, q_len), dtype=torch.int32, device=key_states.device)
            chunk_counts = torch.zeros((bsz, self.config.num_key_value_heads, num_chunks), dtype=torch.int32, device=key_states.device)

            grid_eval = (bsz * self.config.num_key_value_heads, num_chunks)
            similarity_metric = getattr(globVR, 'row_similarity_metric', 'euclidean')
            use_cosine = (similarity_metric == 'cosine')
            raw_threshold = getattr(globVR, 'row_delta_threshold', 0.0)
            threshold_val = raw_threshold if use_cosine else raw_threshold ** 2

            chunked_eval_kernel[grid_eval](
                key_states, keep_mask, chunk_counts,
                threshold_val,
                key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
                keep_mask.stride(0), keep_mask.stride(1), keep_mask.stride(2),
                chunk_counts.stride(0), chunk_counts.stride(1), chunk_counts.stride(2),
                q_len, self.head_dim, chunk_size, num_heads=self.config.num_key_value_heads,
                BLOCK_D=BLOCK_D_EVAL,
                USE_COSINE=use_cosine,
            )
            
            index_map = (torch.cumsum(keep_mask, dim=-1) - 1).to(torch.int32)
            active_counts = chunk_counts.sum(dim=-1)
            max_packed_len_gpu = active_counts.max().item() # Safe here: only runs once per sequence
            
            k_packed = torch.zeros((bsz, self.config.num_key_value_heads, max_packed_len_gpu, self.head_dim), device=key_states.device, dtype=key_states.dtype)
            v_packed = torch.zeros_like(k_packed)
            packed_timestamps = torch.zeros((bsz, self.config.num_key_value_heads, max_packed_len_gpu), device=key_states.device, dtype=torch.int32)
            
            # --- 2. PACK K, V, AND TIMESTAMPS ---
            BLOCK_S = 64
            BLOCK_D_SCATTER = triton.next_power_of_2(self.head_dim)
            grid_scatter = (bsz * self.config.num_key_value_heads, triton.cdiv(q_len, BLOCK_S))
            
            parallel_scatter_pack_kv_kernel[grid_scatter](
                key_states, value_states,
                k_packed, v_packed, packed_timestamps,
                keep_mask, index_map,
                key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
                value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
                k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
                v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
                packed_timestamps.stride(0), packed_timestamps.stride(1), packed_timestamps.stride(2),
                keep_mask.stride(0), keep_mask.stride(1), keep_mask.stride(2),
                index_map.stride(0), index_map.stride(1), index_map.stride(2),
                q_len, self.head_dim, self.config.num_key_value_heads,
                BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D_SCATTER
            )

            # --- 3. CALCULATE PACKED COUNTS ---
            packed_counts = torch.zeros((bsz, self.config.num_key_value_heads, max_packed_len_gpu), device=key_states.device, dtype=torch.int32)
            ones = torch.ones_like(index_map, dtype=torch.int32)
            packed_counts.scatter_add_(2, index_map.long(), ones)

            if do_time:
                end_evt_rd.record()
                glob_set.queue_event_pair('time_get_row_delta', start_evt_rd, end_evt_rd)

            # --- METADATA & SPARSITY TRACKING ---
            key_delta_all = k_packed
            glob_set.store_delta(getattr(globVR, 'delta_key', ''), self.layer_idx, key_delta_all, getattr(globVR, 'collect_delta_pf_key', 0))
            # Sparsity = fraction of tokens dropped, computed purely from keep_mask.
            kept_tokens  = active_counts.sum().item()
            total_tokens = keep_mask.shape[0] * keep_mask.shape[1] * keep_mask.shape[2]
            layer_spars  = 1.0 - kept_tokens / total_tokens if total_tokens > 0 else 0.0
            globVR.spars = layer_spars if globVR.spars == 0.0 else (globVR.spars + layer_spars) / 2

            # --- 4. GQA EXPANSION ---
            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)

            k_packed_expanded = repeat_kv(k_packed, self.num_key_value_groups)
            v_packed_expanded = repeat_kv(v_packed, self.num_key_value_groups)

            timestamps_expanded = repeat_kv(packed_timestamps.unsqueeze(-1), self.num_key_value_groups).squeeze(-1)
            counts_expanded = repeat_kv(packed_counts.unsqueeze(-1), self.num_key_value_groups).squeeze(-1)

            # --- 5. HYBRID FLASH ATTENTION ---
            if do_time:
                start_evt_mm = torch.cuda.Event(enable_timing=True)
                end_evt_mm = torch.cuda.Event(enable_timing=True)
                start_evt_mm.record()

            _use_sanity = getattr(globVR, 'sanity', False)
            if _use_sanity:
                attn_output = self._forward_hybrid_flash_sanity(
                    query_states,
                    key_states_expanded,
                    value_states_expanded,
                    k_packed_expanded,
                    v_packed_expanded,
                    timestamps_expanded,
                    counts_expanded,
                    getattr(globVR, 'dense_window_size', 128),
                )
            else:
                attn_output = self._forward_hybrid_flash(
                    query_states,
                    key_states_expanded,
                    value_states_expanded,
                    k_packed_expanded,
                    v_packed_expanded,
                    timestamps_expanded,
                    counts_expanded,
                    getattr(globVR, 'dense_window_size', 128),
                )
            attn_weights = None

            if do_time:
                end_evt_mm.record()
                glob_set.queue_event_pair('time_delta_mm_pattern', start_evt_mm, end_evt_mm)

            # --- 6. OUTPUT PROJECTION ---
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            
            # ---> 7. PREFILL CACHE HAND-OFF <---
            if past_key_value is not None and hasattr(past_key_value, 'initialize_from_prefill'):
                tail_len = min(q_len, past_key_value.exact_window_size)
                past_key_value.initialize_from_prefill(
                    layer_idx=self.layer_idx,
                    k_packed=k_packed,
                    v_packed=v_packed,
                    counts=packed_counts,
                    timestamps=packed_timestamps,
                    k_exact=key_states[:, :, -tail_len:, :],
                    v_exact=value_states[:, :, -tail_len:, :],
                    original_q_len=q_len  # <--- Pass the true length here!
                )

        # =========================================================================
        # --- PATH 2: TRITON FLASH ATTENTION BASELINE ---
        # =========================================================================
        elif getattr(globVR, 'flash', False) and is_prefill:
            if do_time:
                start_evt_flash = torch.cuda.Event(enable_timing=True)
                end_evt_flash = torch.cuda.Event(enable_timing=True)
                start_evt_flash.record()

            key_states_expanded = repeat_kv(key_states, self.num_key_value_groups)
            value_states_expanded = repeat_kv(value_states, self.num_key_value_groups)
            is_causal = True

            attn_output = triton_flash_attention(
                query_states, 
                key_states_expanded, 
                value_states_expanded, 
                causal=is_causal, 
                sm_scale=self.scaling
            )
            
            attn_weights = None
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

            if do_time:
                end_evt_flash.record()
                glob_set.queue_event_pair('time_triton_flash_attention', start_evt_flash, end_evt_flash)

            # decode_only_delta path: regular prefill attention, but pack K/V into HybridCompressedCache
            # so that the fused hybrid decode kernel can use it during generation.
            if getattr(globVR, 'delta_decode', False) and past_key_value is not None and hasattr(past_key_value, 'initialize_from_prefill'):
                _fixed_chunk = getattr(globVR, 'chunk_size', 0)
                if _fixed_chunk > 0:
                    _chunk_size = min(_fixed_chunk, q_len)
                    _num_chunks = max(1, (q_len + _chunk_size - 1) // _chunk_size)
                else:
                    _divide_to = getattr(globVR, 'divide_to', 0)
                    _actual_divide_to = 1 if _divide_to == 0 else _divide_to
                    _chunk_size = (q_len + _actual_divide_to - 1) // _actual_divide_to
                    _num_chunks = _actual_divide_to

                _BLOCK_D_EVAL = triton.next_power_of_2(self.head_dim)
                _keep_mask = torch.zeros((bsz, self.config.num_key_value_heads, q_len), dtype=torch.int32, device=key_states.device)
                _chunk_counts = torch.zeros((bsz, self.config.num_key_value_heads, _num_chunks), dtype=torch.int32, device=key_states.device)
                _grid_eval = (bsz * self.config.num_key_value_heads, _num_chunks)
                _similarity_metric = getattr(globVR, 'row_similarity_metric', 'euclidean')
                _use_cosine = (_similarity_metric == 'cosine')
                _raw_threshold = getattr(globVR, 'row_delta_threshold', 0.0)
                _threshold_val = _raw_threshold if _use_cosine else _raw_threshold ** 2

                chunked_eval_kernel[_grid_eval](
                    key_states, _keep_mask, _chunk_counts,
                    _threshold_val,
                    key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
                    _keep_mask.stride(0), _keep_mask.stride(1), _keep_mask.stride(2),
                    _chunk_counts.stride(0), _chunk_counts.stride(1), _chunk_counts.stride(2),
                    q_len, self.head_dim, _chunk_size, num_heads=self.config.num_key_value_heads,
                    BLOCK_D=_BLOCK_D_EVAL,
                    USE_COSINE=_use_cosine,
                )

                _index_map = (torch.cumsum(_keep_mask, dim=-1) - 1).to(torch.int32)
                _active_counts = _chunk_counts.sum(dim=-1)
                _max_packed_len = _active_counts.max().item()

                _k_packed = torch.zeros((bsz, self.config.num_key_value_heads, _max_packed_len, self.head_dim), device=key_states.device, dtype=key_states.dtype)
                _v_packed = torch.zeros_like(_k_packed)
                _packed_timestamps = torch.zeros((bsz, self.config.num_key_value_heads, _max_packed_len), device=key_states.device, dtype=torch.int32)

                _BLOCK_S = 64
                _BLOCK_D_SCATTER = triton.next_power_of_2(self.head_dim)
                _grid_scatter = (bsz * self.config.num_key_value_heads, triton.cdiv(q_len, _BLOCK_S))

                parallel_scatter_pack_kv_kernel[_grid_scatter](
                    key_states, value_states,
                    _k_packed, _v_packed, _packed_timestamps,
                    _keep_mask, _index_map,
                    key_states.stride(0), key_states.stride(1), key_states.stride(2), key_states.stride(3),
                    value_states.stride(0), value_states.stride(1), value_states.stride(2), value_states.stride(3),
                    _k_packed.stride(0), _k_packed.stride(1), _k_packed.stride(2), _k_packed.stride(3),
                    _v_packed.stride(0), _v_packed.stride(1), _v_packed.stride(2), _v_packed.stride(3),
                    _packed_timestamps.stride(0), _packed_timestamps.stride(1), _packed_timestamps.stride(2),
                    _keep_mask.stride(0), _keep_mask.stride(1), _keep_mask.stride(2),
                    _index_map.stride(0), _index_map.stride(1), _index_map.stride(2),
                    q_len, self.head_dim, self.config.num_key_value_heads,
                    BLOCK_S=_BLOCK_S, BLOCK_D=_BLOCK_D_SCATTER
                )

                _packed_counts = torch.zeros((bsz, self.config.num_key_value_heads, _max_packed_len), device=key_states.device, dtype=torch.int32)
                _ones = torch.ones_like(_index_map, dtype=torch.int32)
                _packed_counts.scatter_add_(2, _index_map.long(), _ones)

                _tail_len = min(q_len, past_key_value.exact_window_size)
                past_key_value.initialize_from_prefill(
                    layer_idx=self.layer_idx,
                    k_packed=_k_packed,
                    v_packed=_v_packed,
                    counts=_packed_counts,
                    timestamps=_packed_timestamps,
                    k_exact=key_states[:, :, -_tail_len:, :],
                    v_exact=value_states[:, :, -_tail_len:, :],
                    original_q_len=q_len
                )


        # =========================================================================
        # --- PATH 3: STANDARD DECODING / HYBRID DUAL-MATMUL ---
        # =========================================================================
        else:
            # ---> HYBRID COMPRESSED DECODING PATH (Dual-Matmul) <---
            if is_delta_decode and hasattr(past_key_value, 'k_packed'):

                if do_time:
                    start_evt_dec_total = torch.cuda.Event(enable_timing=True)
                    end_evt_dec_total   = torch.cuda.Event(enable_timing=True)
                    start_evt_dec_total.record()

                    start_evt_attn = torch.cuda.Event(enable_timing=True)
                    end_evt_attn   = torch.cuda.Event(enable_timing=True)
                    start_evt_attn.record()

                attn_output = self._forward_fused_hybrid_decode(query_states, past_key_value)
                attn_weights = None

                if do_time:
                    end_evt_attn.record()
                    glob_set.queue_event_pair('time_delta_decode_attn_calc',
                                            start_evt_attn, end_evt_attn)
                    end_evt_dec_total.record()
                    glob_set.queue_event_pair('time_delta_decode_inner_total',
                                            start_evt_dec_total, end_evt_dec_total)
            # ---> STANDARD EAGER/SDPA FALLBACK <---
            else:
                if do_time and not is_prefill:
                    start_evt_base_total = torch.cuda.Event(enable_timing=True)
                    end_evt_base_total = torch.cuda.Event(enable_timing=True)
                    start_evt_base_total.record()
                    
                    start_evt_base_attn = torch.cuda.Event(enable_timing=True)
                    end_evt_base_attn = torch.cuda.Event(enable_timing=True)
                    start_evt_base_attn.record()

                attention_interface: Callable = eager_attention_forward

                if self.config._attn_implementation != "eager":
                    if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                        pass # Handle SDPA fallback warning
                    else:
                        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

                attn_output, attn_weights = attention_interface(
                    self, query_states, key_states, value_states, attention_mask,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling, **kwargs,
                )

                if do_time and not is_prefill:
                    end_evt_base_attn.record()
                    glob_set.queue_event_pair('time_baseline_decode_attn_calc', start_evt_base_attn, end_evt_base_attn)

                    end_evt_base_total.record()
                    glob_set.queue_event_pair('time_baseline_decode_inner_total', start_evt_base_total, end_evt_base_total)

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)

        # =========================================================================
        # --- GLOBAL TIMING RESOLVE ---
        # =========================================================================
        if do_time:
            end_evt_forward_total.record()
            
            if is_prefill:
                glob_set.queue_event_pair('time_prefill_forward_total', start_evt_forward_total, end_evt_forward_total)
            else:
                glob_set.queue_event_pair('time_decode_forward_total', start_evt_forward_total, end_evt_forward_total)

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
        # Skip O(N²) mask only when explicitly requested (e.g. single-sequence speedup benchmarks
        # with no padding). Never set this in eval_all.py — lm_eval uses batched+padded inputs
        # and relies on the mask for padding suppression.
        if getattr(globVR, 'skip_causal_mask', False) and input_tensor.shape[1] > 1:
            return None

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
