# SPDX-License-Identifier: Apache-2.0
"""
Qwen3 MoE BF16 Implementation
======================================

Annotated implementation of Qwen3 MoE for the Neuron backend.

Supported parallelism: TP, SP, EP.

ANNOTATION GUIDE:
  # >>> PARALLELISM: ... <<<   Reusable parallelism code. Keep when porting.
  # <-- MODEL-SPECIFIC: ...    Qwen3 MoE specific. Change when porting.
"""

import logging
import math

import nki.language as nl
import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    last_dim_padding_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig

from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    RouterActFnType,
)
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation

from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_degree,
    get_neuron_ep_rank,
    get_neuron_ep_tp_group,
)

from .config import Qwen3MoeConfig
from .weight_loaders_bf16 import (
    expert_gate_up_weight_loader,
    expert_down_weight_loader,
)
from vllm_neuron.functional.moe.router import RouterComputationOrder

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# <-- MODEL-SPECIFIC: Qwen3 uses standard RMSNorm (no hidden dim padding)
# =============================================================================


class Qwen3MoeRMSNorm(nn.Module):
    """RMS Normalization.

    <-- MODEL-SPECIFIC: Qwen3 uses standard RMSNorm over the full hidden dim.
    Unlike GPT-OSS, there is no hidden dim padding, so variance is computed
    over all dimensions and no zeroing of padded positions is needed.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        # <-- MODEL-SPECIFIC: Variance over full hidden dim (no padding)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding
# <-- MODEL-SPECIFIC: Qwen3 uses standard RoPE with interleaved rotation
# =============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """<-- MODEL-SPECIFIC: Interleaved RoPE rotation (rotate_half style)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors.

    <-- MODEL-SPECIFIC: Qwen3 uses interleaved RoPE (rotate_half).
    GPT-OSS uses non-interleaved (split in half). The rotation style must
    match the checkpoint — using the wrong one produces silent accuracy bugs.
    """
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0)  # [1, T, Dh]
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0)  # [1, T, Dh]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3MoeRotaryEmbedding(nn.Module):
    """Rotary Position Embedding with standard RoPE.

    <-- MODEL-SPECIFIC: Qwen3 uses standard RoPE (no YaRN scaling, no
    concentration factor). Only rope_theta and head_dim are needed.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

        # <-- MODEL-SPECIFIC: Standard inverse frequency (no YaRN interpolation)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float, device="cpu")
                / self.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :].float()  # [1, Dh/2]
        position_ids_expanded = position_ids[:, None].float()  # [T, 1]
        freqs = position_ids_expanded @ inv_freq_expanded  # [T, Dh/2]
        # <-- MODEL-SPECIFIC: No concentration scaling (unlike GPT-OSS YaRN)
        cos = freqs.cos()
        sin = freqs.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


# =============================================================================
# Section 3: Attention
# Mixed: PARALLELISM (TP head sharding, SP, collectives) +
#        MODEL-SPECIFIC (QK-norm, no sinks, no sliding window, no biases)
# =============================================================================


class Qwen3MoeAttention(nn.Module):
    """Multi-head attention with TP head sharding.

    >>> PARALLELISM: TP <<<
    - Q/K/V heads are sharded across TP ranks
    - KV heads are replicated when fewer than TP size (GQA)
    - Prefill: all-gather input → QKV proj → attention → O proj → reduce-scatter
    - Decode: fused megakernel with TP all-reduce

    <-- MODEL-SPECIFIC:
    - GQA with separate Q and KV head counts
    - Per-head QK RMSNorm before RoPE
    - No attention sinks (unlike GPT-OSS)
    - No sliding window (unlike GPT-OSS)
    - No biases on QKV/O projections (unlike GPT-OSS)
    - Interleaved RoPE (externally computed)
    """

    def __init__(self, config: Qwen3MoeConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: Head sharding calculation <<<
        # Q heads divided evenly across TP ranks
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
        # KV heads: replicated if fewer than world_size, else divided
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1

        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        # >>> PARALLELISM: QKV weight shapes for TP <<<
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_proj_in_features = (
            self.num_attention_heads * self.head_dim
        ) // self.world_size

        # <-- MODEL-SPECIFIC: No biases on QKV/O projections (unlike GPT-OSS)
        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        # <-- MODEL-SPECIFIC: Per-head QK RMSNorm before RoPE (Qwen3 specific)
        self.q_layernorm = Qwen3MoeRMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )
        self.k_layernorm = Qwen3MoeRMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach weight loaders for checkpoint → parameter transformation.

        >>> PARALLELISM: Weight loaders handle TP sharding of checkpoint tensors.
        <-- MODEL-SPECIFIC: Uses generic weight loaders (not GPT-OSS MXFP4 loaders).
        No bias loaders needed. is_storage_transposed=True for HF checkpoint format.
        """
        # >>> PARALLELISM: Fused QKV with TP head sharding <<<
        set_weight_loader(
            self.qkv_proj_weight,
            fused_qkv_weight_loader(
                q_size=self.q_size,
                kv_size=self.kv_size,
                shard_dim=1,
                num_shards=self.world_size,
                is_storage_transposed=True,
                num_kv_replicas=self.num_kv_replicas,
            ),
        )
        # >>> PARALLELISM: O-proj sharded on input dim <<<
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=(self.num_attention_heads * self.head_dim)
                // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    # ── Forward dispatch ─────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        """Dispatch to prefill or decode path based on metadata.

        >>> PARALLELISM: Dispatch logic <<<
        - Prefill: all-gather for SP before attention, reduce-scatter after
        - Decode: fused megakernel handles TP internally
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
            )
        else:
            # >>> PARALLELISM: All-gather from SP before attention <<<
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_prefill(
                hidden_states,
                positions,
                position_embeddings,
                attn_metadata,
            )

    # ── Prefill path ─────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Prefill: full-sequence attention with flash attention.

        Pipeline:
        1. QKV projection          >>> PARALLELISM: TP sharded heads <<<
        2. QK norm + RoPE          <-- MODEL-SPECIFIC: per-head QK RMSNorm, interleaved RoPE
        3. KV cache update         >>> PARALLELISM: per-rank cache <<<
        4. Flash attention          <-- MODEL-SPECIFIC: no sinks, no sliding window
        5. Output projection       >>> PARALLELISM: reduce-scatter after O proj <<<
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # ── Step 1: QKV Projection ───────────────────────────────────────
        # >>> PARALLELISM: Each rank computes only its sharded Q/K/V heads <<<
        # <-- MODEL-SPECIFIC: No bias (unlike GPT-OSS)
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(
            0, 1
        )

        # ── Step 2: QK Norm + RoPE ──────────────────────────────────────
        # <-- MODEL-SPECIFIC: Per-head QK RMSNorm before RoPE (Qwen3 specific)
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        # <-- MODEL-SPECIFIC: Interleaved RoPE (rotate_half)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── Step 3: Update KV Cache ─────────────────────────────────────
        # >>> PARALLELISM: Cache is per-rank (TP sharded KV heads) <<<
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        k_flat = k.reshape(-1, self.head_dim)
        v_flat = v.reshape(-1, self.head_dim)

        head_indices_for_put = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_indices_for_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        position_indices_for_put = position_indices.repeat(
            self.num_key_value_heads_per_rank
        )

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_flat,
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_flat,
        )

        # ── Step 4: Attention ─────────────────────────────────────────────
        # <-- MODEL-SPECIFIC: No sinks, no sliding window (unlike GPT-OSS)
        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.repeat_interleave(self.num_key_value_groups, dim=0)

        q_flash = q.transpose(1, 2)  # [Nh, Dh, T]
        k_flash = k.transpose(1, 2)  # [Nh, Dh, T]
        v_flash = v  # [Nh, T, Dh]

        attn_output = NF.flash_attention(
            q_flash,
            k_flash,
            v_flash,
            scale=self.scaling,
            tp_q=False,
            tp_out=True,
        )

        # ── Step 5: Output Projection ────────────────────────────────────
        attn_output = attn_output.unsqueeze(0)
        # <-- MODEL-SPECIFIC: No bias on O proj (unlike GPT-OSS)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight)
        attn_output = attn_output.squeeze(0)

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode path ──────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode: fused megakernel for single-token generation.

        >>> PARALLELISM: The megakernel handles TP internally. <<<
        The kernel performs QKV proj, RoPE, attention, and O proj in one fused call.
        TP all-reduce is done after the kernel.

        <-- MODEL-SPECIFIC: Qwen3 specific kernel arguments:
        - QK RMSNorm before RoPE (rmsnorm_QK_pre_rope_enabled=True)
        - Interleaved RoPE layout (rope_contiguous_layout=True)
        - No sinks, no sliding window, no biases
        """
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        assert tokens == B * S_decode

        hidden_states = hidden_states.to(self.dtype)
        S_ctx = max_blocks_per_seq * block_size
        nkh = self.num_key_value_heads_per_rank

        X = hidden_states.view(B, S_decode, hidden)

        cos, sin = position_embeddings
        half_d = self.head_dim // 2
        cos_kernel = (
            cos[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )
        sin_kernel = (
            sin[:, :half_d]
            .view(B, S_decode, half_d)
            .permute(2, 0, 1)
            .contiguous()
            .to(self.dtype)
        )

        # <-- MODEL-SPECIFIC: No sliding window, no sinks — full causal mask
        pos_ids = positions.view(1, B * S_decode)
        attention_mask = NF.gen_attention_decode_mask(
            pos_ids=pos_ids.to(torch.float32),
            bs=B,
            q_head=self.num_attention_heads_per_rank,
            s_active=S_decode,
            s_prior=S_ctx,
            start_pos=None,
            block_len=block_size,
        )

        k_cache = (
            self.k_cache.squeeze(1) if self.k_cache.dim() == 4 and nkh else self.k_cache
        )
        v_cache = (
            self.v_cache.squeeze(1) if self.v_cache.dim() == 4 and nkh else self.v_cache
        )

        active_blocks_table = block_table

        # <-- MODEL-SPECIFIC: QK norm weights for megakernel, interleaved RoPE
        W_q_norm = self.q_layernorm.weight.view(self.head_dim, 1)
        W_k_norm = self.k_layernorm.weight.view(self.head_dim, 1)

        # >>> PARALLELISM: Fused megakernel with TP-sharded weights <<<
        output, K_new, V_new = NF.attention_decode(
            X=X,
            W_qkv=self.qkv_proj_weight,
            rmsnorm_X_enabled=False,
            # <-- MODEL-SPECIFIC: QK RMSNorm before RoPE
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_W_Q=W_q_norm,
            rmsnorm_QK_pre_rope_W_K=W_k_norm,
            cos=cos_kernel,
            sin=sin_kernel,
            # <-- MODEL-SPECIFIC: Interleaved RoPE (rotate_half)
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            # <-- MODEL-SPECIFIC: No sinks (unlike GPT-OSS)
            update_cache=False,
            W_out=self.o_proj_weight,
            # <-- MODEL-SPECIFIC: No bias on O proj
            transposed_out=False,
            out_in_sb=False,
        )

        # ── Manual KV cache update ──────────────────────────────────────
        # >>> PARALLELISM: Cache is per-rank <<<
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]

        k_new = (
            K_new.permute(1, 2, 0)
            .reshape(B, nkh, S_decode, self.head_dim)
            .transpose(0, 1)
            .reshape(nkh, B * S_decode, self.head_dim)
        )
        k_new_flat = k_new.reshape(-1, self.head_dim)
        v_new_flat = V_new.transpose(0, 1).reshape(-1, self.head_dim)

        head_indices_for_put = torch.arange(
            nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_indices_for_put = block_indices.repeat(nkh)
        position_indices_for_put = position_indices.repeat(nkh)

        self.k_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            k_new_flat.to(self.k_cache.dtype),
        )
        self.v_cache.index_put_(
            (block_indices_for_put, head_indices_for_put, position_indices_for_put),
            v_new_flat.to(self.v_cache.dtype),
        )

        # >>> PARALLELISM: TP all-reduce after megakernel <<<
        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: MoE Experts
# Mixed: PARALLELISM (TP sharding of expert weights, EP partitioning, SP collectives) +
#        MODEL-SPECIFIC (SiLU activation, softmax-before-topk routing, no clamping)
# =============================================================================


class Qwen3MoeExperts(nn.Module):
    """Expert feed-forward layers with TP, optional EP support.

    >>> PARALLELISM: TP + EP <<<
    - EP disabled (ep_degree=1): all experts on all ranks, intermediate sharded by TP
    - EP enabled: experts partitioned across EP ranks (linear placement),
      intermediate sharded across tp_degree = world_size / ep_degree

    <-- MODEL-SPECIFIC:
    - Softmax before topk with L1 renormalization (norm_topk_prob)
    - SiLU activation, no clamping (unlike GPT-OSS SwiGLU with clamping)
    - No biases on expert weights (unlike GPT-OSS)
    - Individual expert weights in checkpoint (not MXFP4 blocks/scales)
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()

        # >>> PARALLELISM: TP + EP configuration <<<
        self.tp_group = get_tp_group()

        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        self.ep_enabled = vllm_config.parallel_config.enable_expert_parallel

        self.ep_degree = get_neuron_ep_degree() if self.ep_enabled else 1
        self.ep_rank = get_neuron_ep_rank() if self.ep_enabled else 0
        self.tp_degree = self.tp_group.world_size // self.ep_degree
        self.ep_tp_group = (
            get_neuron_ep_tp_group() if self.ep_enabled else self.tp_group
        )
        self.moe_group = self.tp_group

        self.total_num_experts = config.num_experts
        # >>> PARALLELISM: Linear placement — EP rank k owns experts [k*L, (k+1)*L) <<<
        self.num_local_experts = config.num_experts // self.ep_degree
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        # <-- MODEL-SPECIFIC: L1 renormalization after topk
        self.norm_topk_prob = config.norm_topk_prob

        # >>> PARALLELISM: Intermediate dim sharded across TP degree <<<
        self.intermediate_size_per_rank = config.moe_intermediate_size // self.tp_degree
        self.block_size = 256

        # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
        self.post_attention_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )

        # >>> PARALLELISM: Router weights replicated on all ranks (NOT EP-sharded) <<<
        self.router_weight = nn.Parameter(
            torch.empty(self.total_num_experts, self.hidden_size, dtype=torch.bfloat16)
        )

        # >>> PARALLELISM: Expert weights sharded on intermediate dim <<<
        # <-- MODEL-SPECIFIC: No biases on expert weights (unlike GPT-OSS)
        # gate_up_proj: [E_local, H, I_per_rank*2] (gate and up interleaved)
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size_per_rank * 2,
                dtype=config.torch_dtype,
            )
        )
        # down_proj: [E_local, I_per_rank, H]
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.intermediate_size_per_rank,
                self.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Set up weight loaders for TP + EP sharding.

        >>> PARALLELISM: TP sharding of intermediate dim, EP filtering of experts <<<
        <-- MODEL-SPECIFIC: Uses Qwen3-specific weight loaders for individual
        expert weights (not MXFP4 blocks/scales like GPT-OSS). No bias loaders.
        """
        from vllm_neuron.utils.weight_loader import (
            expert_parallel_weight_loader,
        )

        # Router: pad hidden dim (always full, not EP-sharded)
        set_weight_loader(
            self.router_weight, last_dim_padding_weight_loader(self.hidden_size)
        )
        set_weight_loader(
            self.post_attention_layernorm.weight,
            last_dim_padding_weight_loader(self.hidden_size),
        )

        # >>> PARALLELISM: Linear EP placement — rank k owns experts [k*L, (k+1)*L) <<<
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )

        def _maybe_ep_wrap(loader):
            """Wrap loader with EP expert filtering if ep_degree > 1."""
            if self.ep_degree > 1:
                return expert_parallel_weight_loader(local_expert_indices, loader)
            return loader

        # <-- MODEL-SPECIFIC: Individual expert BF16 weights + TP sharding (+ EP filtering)
        set_weight_loader(
            self.gate_up_proj_weight,
            _maybe_ep_wrap(
                expert_gate_up_weight_loader(
                    num_experts=self.total_num_experts,
                    shard_size=self.intermediate_size_per_rank * 2,
                    num_shards=self.tp_degree,
                )
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            _maybe_ep_wrap(
                expert_down_weight_loader(
                    num_experts=self.total_num_experts,
                    shard_size=self.intermediate_size_per_rank,
                    num_shards=self.tp_degree,
                )
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        if is_decode:
            return self.forward_decode(hidden_states, rank)
        else:
            return self.forward_prefill(hidden_states, positions, rank)

    def forward_decode(self, hidden_states: torch.Tensor, rank: torch.Tensor):
        """Decode: fused MoE kernel.

        >>> PARALLELISM: TKG kernel with EP rank_id, all-reduce after <<<
        <-- MODEL-SPECIFIC: Softmax routing with pre-norm, SiLU activation,
        norm_topk_prob, no clamping, no biases
        """
        # >>> PARALLELISM: Derive ep_rank from runtime rank tensor <<<
        # Avoids baking rank as a constant in the FX graph (per-rank compilations).
        if self.ep_degree > 1:
            ep_rank = (rank % (self.ep_degree * self.tp_degree)) // self.tp_degree
        else:
            ep_rank = torch.zeros_like(rank)
        rank_id = ep_rank.view(1, 1).to(torch.int32)

        output = NF.moe_block_tkg(
            inp=hidden_states.unsqueeze(0),
            gamma=self.post_attention_layernorm.weight.unsqueeze(0).to(torch.float32),
            router_weights=self.router_weight.T,
            expert_gate_up_weights=self.gate_up_proj_weight.reshape(
                self.num_local_experts,
                self.hidden_size,
                2,
                self.intermediate_size_per_rank,
            ),
            expert_down_weights=self.down_proj_weight,
            rank_id=rank_id,
            top_k=self.num_experts_per_token,
            eps=self.rms_norm_eps,
            # <-- MODEL-SPECIFIC: Softmax before topk (router_pre_norm=True)
            router_act_fn=RouterActFnType.SOFTMAX,
            router_pre_norm=True,
            norm_topk_prob=self.norm_topk_prob,
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            # <-- MODEL-SPECIFIC: SiLU activation, no clamping
            hidden_act_fn=ActFnType.SiLU,
            router_mm_dtype=nl.bfloat16,
            is_all_expert=True,
            skip_router_logits=True,
        )

        # >>> PARALLELISM: All-reduce across MoE group <<<
        if self.moe_group.world_size > 1:
            output = self.moe_group.all_reduce(output)

        return output

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        rank: torch.Tensor,
    ):
        """Prefill: blockwise MoE with CTE kernel.

        >>> PARALLELISM: All-gather from SP → MoE → reduce-scatter back to SP <<<
        <-- MODEL-SPECIFIC: Softmax-before-topk routing with L1 renormalization,
        SiLU activation, no clamping, no biases
        """
        # <-- MODEL-SPECIFIC: Pre-MLP RMSNorm
        hidden_states = self.post_attention_layernorm(hidden_states)

        # <-- MODEL-SPECIFIC: Softmax before topk, then L1 renormalize
        expert_affinities = NF.router(
            hidden_states=hidden_states,
            router_weights=self.router_weight.T,
            top_k=self.num_experts_per_token,
            activation="softmax",
            computation_dtype=torch.float32,
            router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        )

        # >>> PARALLELISM: All-gather from SP for full sequence (within TP group) <<<
        if self.tp_group.world_size > 1:
            expert_affinities = self.tp_group.all_gather(expert_affinities, dim=0)
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        # >>> PARALLELISM: With EP, map global affinities to local experts <<<
        if self.ep_degree > 1:
            # Derive ep_rank from the runtime rank tensor to avoid baking rank
            # as a constant in the FX graph.
            ep_rank = (rank % (self.ep_degree * self.tp_degree)) // self.tp_degree
            local_expert_indices = (
                torch.arange(
                    self.num_local_experts,
                    device=hidden_states.device,
                    dtype=torch.int32,
                )
                + ep_rank.to(torch.int32) * self.num_local_experts
            )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_expert_indices
            )

        # >>> PARALLELISM: Build blockwise mapping for efficient MoE dispatch <<<
        (
            expert_affinities_masked,
            token_position_to_id,
            block_to_expert,
            conditions,
        ) = NF.build_blockwise_mapping(
            expert_affinities=expert_affinities,
            num_local_experts=self.num_local_experts,
            num_experts_per_token=self.num_experts_per_token,
            block_size=self.block_size,
            moe_group=self.ep_tp_group,
            tp_degree=self.tp_degree,
            padding_mask=None,
        )

        num_tokens = hidden_states.shape[0]
        num_static_block = math.ceil(
            num_tokens * self.num_experts_per_token / self.ep_degree / self.block_size
        )

        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj_weight.reshape(
                self.num_local_experts,
                self.hidden_size,
                2,
                self.intermediate_size_per_rank,
            ),
            down_proj_weight=self.down_proj_weight,
            # <-- MODEL-SPECIFIC: SiLU activation, no clamping, no biases
            activation_function=ActFnType.SiLU,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
            num_static_block=num_static_block,
        )

        # >>> PARALLELISM: Reduce-scatter to return to SP layout <<<
        if self.moe_group.world_size > 1:
            output = self.moe_group.reduce_scatter(output, dim=0)

        return output


# =============================================================================
# Section 5: MLP Wrapper
# <-- MODEL-SPECIFIC: Thin wrapper around MoE experts
# =============================================================================


class Qwen3MoeMLP(nn.Module):
    """MLP layer with Mixture of Experts.

    <-- MODEL-SPECIFIC: This wrapper exists because Qwen3 uses MoE.
    A dense model would have a simple MLP here instead.
    """

    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
        self.dtype = config.torch_dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(self.dtype)
        return self.experts(
            hidden_states, positions=positions, is_decode=is_decode, rank=rank
        )


# =============================================================================
# Section 6: Decoder Layer
# Mixed: PARALLELISM (is_decode dispatch pattern) +
#        MODEL-SPECIFIC (pre-attention norm, residual connections)
# =============================================================================


class Qwen3MoeDecoderLayer(nn.Module):
    """Single transformer decoder layer.

    Architecture (MODEL-SPECIFIC):
        hidden_states → RMSNorm → Attention → residual → MoE → residual

    Parallelism (TP + SP):
        - Input arrives in SP layout (T/world_size tokens per rank) during prefill
        - Attention all-gathers before and reduce-scatters after
        - MoE all-gathers before and reduce-scatters after
        - Residual connection operates on SP-local slice
    """

    def __init__(self, config: Qwen3MoeConfig, batch_size: int, layer_idx: int):
        super().__init__()
        # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
        self.input_layernorm = Qwen3MoeRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Qwen3MoeAttention(config, layer_idx=layer_idx)
        self.mlp = Qwen3MoeMLP(config)
        self.layer_idx = layer_idx

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        # ── Self Attention ───────────────────────────────────────────────
        residual = hidden_states
        # <-- MODEL-SPECIFIC: Pre-attention RMSNorm
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # ── MoE Feed-Forward ─────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.mlp(
            hidden_states,
            positions=positions,
            is_decode=is_decode,
            rank=rank,
        )
        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# Mixed: PARALLELISM (SP chunking / gathering) +
#        MODEL-SPECIFIC (embedding, layer stack, final norm)
# =============================================================================


class Qwen3MoeModel(nn.Module):
    """Qwen3 MoE transformer backbone.

    >>> PARALLELISM: SP (Sequence Parallelism) <<<
    During prefill:
    - After embedding: chunk sequence across TP ranks (each gets T/world_size)
    - After all layers: all-gather to reconstruct full sequence
    During decode:
    - No SP; all ranks process all tokens
    """

    def __init__(self, config: Qwen3MoeConfig, batch_size: int):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        # Each rank stores vocab_size/tp_size entries, combined via reduce_scatter (prefill)
        # or all_reduce (decode).
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        # <-- MODEL-SPECIFIC: Stack of decoder layers
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(config, batch_size, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        # <-- MODEL-SPECIFIC: Final RMSNorm
        self.norm = Qwen3MoeRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config)

        # >>> PARALLELISM: Shard embedding on vocab dim <<<
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.hidden_size,
            ),
        )
        set_weight_loader(
            self.norm.weight, last_dim_padding_weight_loader(config.hidden_size)
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        # scatter_tokens=True (prefill): reduce_scatter → [T/world_size, H]
        # scatter_tokens=False (decode): all_reduce → [T, H]
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                rank=rank,
            )

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — all-gather to reconstruct full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# =============================================================================
# Section 8: Language Model Head
# Mixed: PARALLELISM (column-parallel LM head) +
#        MODEL-SPECIFIC (vocabulary projection, sampling, weight mappings)
# =============================================================================


class Qwen3MoeForCausalLM(nn.Module):
    """Qwen3 MoE model with language modeling head.

    >>> PARALLELISM: Column-parallel linear for LM head <<<
    The vocabulary projection is sharded across TP ranks. Each rank computes
    a portion of the logits, then either:
    - Gathered for full logits (when not using on-device sampling)
    - Kept sharded for on-device sampling (sampler handles TP internally)
    """

    def __init__(self, config: Qwen3MoeConfig, batch_size: int):
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config, batch_size)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        # >>> PARALLELISM: Column-parallel LM head <<<
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

        # >>> PARALLELISM: Shard lm_head on vocab dim (dim 0) <<<
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.hidden_size,
            ),
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError("Input Embedding as Inputs is Not Supported Yet.")

        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        # >>> PARALLELISM: SP length validation <<<
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        hidden_states_for_logits = hidden_states_for_logits.to(self.config.torch_dtype)

        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
        gathered_logits = None
        if self._gather_logits:
            if self.tp_group is not None:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            rejection_sampled_tokens = rejection_sampler(
                spec_decode_metadata, sampled_tokens
            )
            return rejection_sampled_tokens

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Qwen3MoeConfig.from_configs(hf_config, neuron_config)
        return cls(config, batch_size=1)

    # ── KV Cache Management ──────────────────────────────────────────────
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<
    # <-- MODEL-SPECIFIC: No sliding window on any layer

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=layer.self_attn.num_key_value_heads_per_rank,
                    head_size=layer.self_attn.head_dim,
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from a checkpoint with pipelined data movement.

        >>> PARALLELISM: Weight loaders (attached to each parameter) handle TP sharding <<<
        <-- MODEL-SPECIFIC: The mapping between HF names and our parameter names
        """
        tp_rank = self.tp_group.rank_in_group
        tp_size = self.world_size

        mappings = dict()

        for layer_id in range(len(self.model.layers)):
            layer_prefix = f"model.layers.{layer_id}"

            # <-- MODEL-SPECIFIC: Attention weight mappings
            # HF stores separate q_proj, k_proj, v_proj → we fuse into qkv_proj
            mappings[f"{layer_prefix}.self_attn.qkv_proj_weight"] = [
                f"{layer_prefix}.self_attn.q_proj.weight",
                f"{layer_prefix}.self_attn.k_proj.weight",
                f"{layer_prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{layer_prefix}.input_layernorm.weight"] = (
                f"{layer_prefix}.input_layernorm.weight"
            )
            mappings[f"{layer_prefix}.self_attn.o_proj_weight"] = (
                f"{layer_prefix}.self_attn.o_proj.weight"
            )

            # <-- MODEL-SPECIFIC: Qwen3 QK LayerNorm (HF uses q_norm/k_norm)
            mappings[f"{layer_prefix}.self_attn.q_layernorm.weight"] = (
                f"{layer_prefix}.self_attn.q_norm.weight"
            )
            mappings[f"{layer_prefix}.self_attn.k_layernorm.weight"] = (
                f"{layer_prefix}.self_attn.k_norm.weight"
            )

            # <-- MODEL-SPECIFIC: MoE weight mappings
            mappings[f"{layer_prefix}.mlp.experts.post_attention_layernorm.weight"] = (
                f"{layer_prefix}.post_attention_layernorm.weight"
            )
            mappings[f"{layer_prefix}.mlp.experts.router_weight"] = (
                f"{layer_prefix}.mlp.gate.weight"
            )

            # <-- MODEL-SPECIFIC: Individual expert weights (not MXFP4 blocks/scales)
            num_experts = self.config.num_experts
            gate_proj_keys = [
                f"{layer_prefix}.mlp.experts.{i}.gate_proj.weight"
                for i in range(num_experts)
            ]
            up_proj_keys = [
                f"{layer_prefix}.mlp.experts.{i}.up_proj.weight"
                for i in range(num_experts)
            ]
            down_proj_keys = [
                f"{layer_prefix}.mlp.experts.{i}.down_proj.weight"
                for i in range(num_experts)
            ]

            mappings[f"{layer_prefix}.mlp.experts.gate_up_proj_weight"] = (
                gate_proj_keys + up_proj_keys
            )
            mappings[f"{layer_prefix}.mlp.experts.down_proj_weight"] = down_proj_keys

        # Embedding and final norm
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["model.norm.weight"] = "model.norm.weight"
        mappings["lm_head.weight"] = "lm_head.weight"

        # >>> PARALLELISM: load_sharded_pipelined handles TP sharding via weight loaders <<<
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded_checkpoint = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        ).state_dict

        self.load_state_dict(rank_sharded_checkpoint, strict=False, assign=True)
