# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-VL Text Decoder Model
============================

Text decoder for Qwen3-VL multimodal model. Dense architecture (no MoE)
with GQA, QK-norm, SwiGLU MLP, and M-RoPE (multimodal rotary position
embedding with interleaved 3D position IDs).

Ported from HuggingFace Qwen3VLTextModel / Qwen3VLForConditionalGeneration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_tp_group
from transformers import PretrainedConfig

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig, VisionNeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from .config import Qwen3VLConfig, Qwen3VLTextConfig


# ---------------------------------------------------------------------------
# Section 1: RMSNorm
# ---------------------------------------------------------------------------
# Standard RMSNorm without padding. hidden_size=5120 for the 32B variant
# is already hardware-aligned so no unpadded_hidden_size handling is needed.


class Qwen3VLTextRMSNorm(nn.Module):
    """RMS Normalization for Qwen3-VL text decoder."""

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# ---------------------------------------------------------------------------
# Section 2: M-RoPE (Multimodal Rotary Position Embedding)
# ---------------------------------------------------------------------------
# Qwen3-VL uses M-RoPE: 3D position IDs (temporal, height, width) with
# interleaved frequency sections controlled by mrope_section=[24,20,20].
# Full head_dim rotation (no partial_rotary_factor). Text-only inputs use
# identical values across all three dimensions; multimodal inputs encode
# spatial grid positions in the height/width dimensions.


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key tensors.

    Full head_dim rotation using the rotate_half pattern.
    cos/sin are [T, Dh/2] from the RoPE module; they are doubled and
    broadcast over the head dimension.
    """
    cos = torch.cat((cos, cos), dim=-1).unsqueeze(0)  # [1, T, Dh]
    sin = torch.cat((sin, sin), dim=-1).unsqueeze(0)  # [1, T, Dh]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3VLTextRotaryEmbedding(nn.Module):
    """M-RoPE for Qwen3-VL.

    Computes rotary embeddings from 3D position IDs [temporal, height, width].
    Frequencies for each dimension are interleaved according to mrope_section
    so that nearby spatial positions produce similar embeddings. For the 32B
    model: head_dim=128, inv_freq has 64 entries, mrope_section=[24,20,20]
    (sums to 64), rope_theta=5e6.

    The interleaving uses torch.where for XLA compatibility (no in-place ops).
    """

    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        self.config = config

        dim = config.head_dim
        base = config.rope_theta

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float, device="cpu") / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.attention_scaling = 1.0
        self.mrope_section = config.rope_scaling.get("mrope_section", [24, 20, 20])

    def forward(
        self,
        position_ids: torch.Tensor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin embeddings from position IDs.

        Matches the Llama-style call convention: (position_ids, device, dtype).

        Args:
            position_ids: Position IDs.
                - Shape [T] for text-only (1D, expanded to 3D internally)
                - Shape [3, T] or [3, bs, T] for M-RoPE (temporal, height, width)
            device: Target device (unused, kept for API compatibility).
            dtype: Output dtype.

        Returns:
            (cos, sin) each of shape [T, head_dim/2]. The apply_rotary_pos_emb
            function handles doubling to full head_dim via cat((cos, cos)).
        """
        # Expand 1D [T] → [3, 1, T] for M-RoPE
        if position_ids.ndim == 1:
            position_ids = position_ids[None, None, :].expand(3, 1, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(
            2, 3
        )

        # freqs: [bs, seq_len, head_dim/2] after interleaving
        freqs = self.compute_interleaved_mrope(freqs, self.mrope_section)

        cos = freqs.cos() * self.attention_scaling
        sin = freqs.sin() * self.attention_scaling

        # Squeeze batch dim if input was 1D
        if cos.shape[0] == 1:
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)

        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    @staticmethod
    def compute_interleaved_mrope(
        freqs: torch.Tensor, mrope_section: list[int]
    ) -> torch.Tensor:
        """Interleave T/H/W frequencies for M-RoPE.

        Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
        interleaved [THWTHWTHW...TT] so that each group of 3 consecutive
        frequency slots encodes one temporal, one height, and one width
        component. Remaining slots (when T section > H or W) stay as T.

        Uses torch.where instead of in-place slice assignment for XLA
        compatibility.

        Args:
            freqs: [3, bs, seq_len, head_dim/2] — per-dimension frequencies
            mrope_section: freq pairs per T/H/W dimension (e.g. [24, 20, 20])

        Returns:
            Interleaved frequencies [bs, seq_len, head_dim/2]
        """
        last_dim = freqs.shape[-1]
        indices = torch.arange(last_dim, device=freqs.device, dtype=torch.int64)

        freqs_t = freqs[0].clone()

        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            mask = (indices % 3 == offset) & (indices < length)
            freqs_t = torch.where(mask, freqs[dim], freqs_t)

        return freqs_t


# ---------------------------------------------------------------------------
# Section 3: Attention
# ---------------------------------------------------------------------------
# GQA (Grouped Query Attention) with per-head QK RMSNorm before RoPE.
# 64 Q heads, 8 KV heads for the 32B model. No attention bias, no sinks,
# no sliding window. TP shards Q/K/V heads across ranks. Prefill uses
# flash attention with SP all-gather/reduce-scatter. Decode uses the
# fused megakernel with built-in QK-norm and RoPE support.


class Qwen3VLTextAttention(nn.Module):
    """Multi-head attention with GQA and per-head QK RMSNorm.

    Architecture:
    - Fused QKV projection (no bias)
    - Per-head RMSNorm on Q and K before RoPE
    - M-RoPE applied to Q and K
    - Flash attention (prefill) or fused megakernel (decode)
    - Output projection (no bias)

    Parallelism (keep as-is when porting):
    - Q/K/V heads sharded across TP ranks
    - KV heads replicated when fewer than TP size
    - Prefill: SP all-gather → compute → reduce-scatter
    - Decode: fused megakernel with TP all-reduce
    """

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
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
        self.num_attention_heads_per_rank = self.num_attention_heads // self.world_size
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

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in_features, self.hidden_size, dtype=self.dtype)
        )

        # Per-head QK RMSNorm before RoPE
        self.q_layernorm = Qwen3VLTextRMSNorm(
            self.head_dim, config.rms_norm_eps, self.dtype
        )
        self.k_layernorm = Qwen3VLTextRMSNorm(
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
        No bias. is_storage_transposed=True for HF checkpoint format.
        """
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
        """Dispatch to prefill or decode path based on metadata."""
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
        """Prefill: QKV proj → QK norm → RoPE → flash attention → O proj."""
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # Step 1: QKV Projection
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

        # Step 2: Per-head QK RMSNorm + M-RoPE
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        cos, sin = position_embeddings
        # TODO: fuse MRope with QKV kernel
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Step 3: Update KV Cache
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

        # Step 4: Flash Attention
        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.repeat_interleave(self.num_key_value_groups, dim=0)

        q_flash = q.transpose(1, 2)
        k_flash = k.transpose(1, 2)
        v_flash = v

        attn_output = NF.flash_attention(
            q_flash,
            k_flash,
            v_flash,
            scale=self.scaling,
            tp_q=False,
            tp_out=True,
        )

        # Step 5: Output Projection
        attn_output = attn_output.unsqueeze(0)
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
        """Decode: fused megakernel with QK-norm and RoPE."""
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

        W_q_norm = self.q_layernorm.weight.view(self.head_dim, 1)
        W_k_norm = self.k_layernorm.weight.view(self.head_dim, 1)

        output, K_new, V_new = NF.attention_decode(
            X=X,
            W_qkv=self.qkv_proj_weight,
            rmsnorm_X_enabled=False,
            rmsnorm_QK_pre_rope_enabled=True,
            rmsnorm_QK_pre_rope_W_Q=W_q_norm,
            rmsnorm_QK_pre_rope_W_K=W_k_norm,
            cos=cos_kernel,
            sin=sin_kernel,
            rope_contiguous_layout=True,
            K_cache_transposed=False,
            active_blocks_table=active_blocks_table,
            K_cache=k_cache,
            V_cache=v_cache,
            attention_mask=attention_mask,
            softmax_scale=self.scaling,
            update_cache=False,
            W_out=self.o_proj_weight,
            transposed_out=False,
            out_in_sb=False,
        )

        # Manual KV cache update
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


# ---------------------------------------------------------------------------
# Section 5: MLP
# ---------------------------------------------------------------------------
# SwiGLU MLP: gate_proj and up_proj map hidden → intermediate, silu
# activation on gate, element-wise multiply, then down_proj maps back.
# No bias on any projection. intermediate_size=25600 for the 32B model.
# TP shards the intermediate dimension across ranks. SP all-gather before
# and reduce-scatter after during prefill; all-reduce during decode.


class Qwen3VLTextMLP(nn.Module):
    """Dense SwiGLU MLP with TP intermediate sharding.

    Architecture: down_proj(silu(gate_proj(x)) * up_proj(x))
    No bias on any projection.

    Parallelism (keep as-is when porting):
    - gate/up/down projections sharded on intermediate dim across TP
    - Prefill: SP all-gather → compute → reduce-scatter
    - Decode: compute → all-reduce
    """

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()

        # >>> PARALLELISM: TP group setup <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        # >>> PARALLELISM: Intermediate dim sharded across TP <<<
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                config.hidden_size,
                self.intermediate_size_per_rank,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_size_per_rank,
                config.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """>>> PARALLELISM: TP sharding of MLP weights. <<<"""
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )

        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(self, hidden_states: torch.Tensor, is_prefill: bool) -> torch.Tensor:
        """SwiGLU forward with SP/TP collectives."""
        # >>> PARALLELISM: All-gather from SP for full sequence <<<
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        # >>> PARALLELISM: Combine TP shards <<<
        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            if self.world_size > 1:
                self.tp_group.all_reduce(output)

        return output


# ---------------------------------------------------------------------------
# Section 6: Decoder Layer
# ---------------------------------------------------------------------------
# Pre-norm residual architecture:
#   hidden → RMSNorm → Attention → residual → RMSNorm → MLP → residual
# Prefill and decode dispatch handled by the attention and MLP modules
# internally (SP all-gather/reduce-scatter for prefill, all-reduce for decode).


class Qwen3VLTextDecoderLayer(nn.Module):
    """Single transformer decoder layer for Qwen3-VL text model."""

    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.input_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = Qwen3VLTextAttention(config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.layer_idx = layer_idx

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
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # MLP Feed-Forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, is_prefill=not is_decode)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Section 7: Model Backbone
# ---------------------------------------------------------------------------
# Vocab-sharded embedding → decoder layers → final RMSNorm.
# SP: embedding reduce-scatters during prefill, all-reduces during decode.
# After all layers, all-gather reconstructs the full sequence for prefill.


class Qwen3VLTextModel(nn.Module):
    """Qwen3-VL text transformer backbone.

    Supports optional vision inputs for multimodal inference:
    - vision_embeddings: scattered into token embeddings at vision_mask positions
    - deepstack_vision_embeds: added to hidden states after specific decoder layers
      (DeepStack: visual features from intermediate vision encoder layers are
      injected into early text decoder layers)

    """

    def __init__(self, config: Qwen3VLTextConfig):
        super().__init__()
        self.config = config

        # >>> PARALLELISM: TP group for SP <<<
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: Vocab-sharded embedding <<<
        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                Qwen3VLTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = Qwen3VLTextRMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config)

        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.embed_tokens.tp_size,
                is_storage_transposed=False,
            ),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        # <-- MODEL-SPECIFIC: Deepstack features
        deepstack_vision_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional vision inputs.

        Args:
            vision_embeddings: Vision encoder output, full sequence length.
                Scattered into token embeddings at vision_mask positions.
            deepstack_vision_embeds: List of tensors from intermediate vision
                encoder layers, full sequence length each. Added to hidden
                states after decoder layers 0, 1, 2 (for deepstack_visual_indexes
                [8, 16, 24]).
        """
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        # >>> PARALLELISM: VocabDimShardedEmbedding handles SP internally <<<
        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        # >>> PARALLELISM: SP — slice prompt embeds to local chunk <<<
        if is_prefill and inputs_embeds is not None and is_token_ids is not None:
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
            )

            # <-- MODEL-SPECIFIC: Deepstack: add vision features after first few decoder layers
            # TODO: Re-enable once vision encoder is integrated and vision_mask
            # is passed through the forward signature.

        hidden_states = self.norm(hidden_states)

        # >>> PARALLELISM: SP — all-gather to reconstruct full sequence <<<
        if is_prefill:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states


# ---------------------------------------------------------------------------
# Section 8: LM Head + Weight Loading
# ---------------------------------------------------------------------------
# Top-level multimodal model. Contains the text backbone (language_model)
# and will later contain the vision encoder (visual). For now, text-only
# flow with dummy vision inputs passed through.

# HF checkpoint prefix for the text model weights
HF_TEXT_PREFIX = "model.language_model"


class Qwen3VLForConditionalGeneration(nn.Module):
    """Qwen3-VL multimodal model with language modeling head.

    Currently implements text-only flow. Vision encoder will be added later.
    The forward signature accepts vision_embeddings, vision_mask, and
    deepstack_vision_embeds which are dummy (empty) for text-only inputs.
    """

    def __init__(self, config: Qwen3VLConfig):
        super().__init__()
        self.config = config
        self.text_config = config.text_config

        self.language_model = Qwen3VLTextModel(config.text_config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        nc = config.text_config.neuron_config
        self.on_device_sampling_config = nc.on_device_sampling_config if nc else None
        debug_logits_enabled = nc is not None and nc.debug_logits_dir is not None
        self._gather_logits = (
            nc is not None and nc.max_logprobs != 0
        ) or debug_logits_enabled

        # >>> PARALLELISM: Column-parallel LM head <<<
        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            dtype=config.text_config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )

        # >>> PARALLELISM: Shard lm_head on vocab dim (dim 0) <<<
        set_weight_loader(
            self.lm_head.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.text_config.vocab_size // self.world_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=self.text_config.hidden_size,
                unpadded_size=self.text_config.hidden_size,
            ),
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        # TODO: read deepstack_vision_embed from buffer after integrating vision encoder
        deepstack_vision_embeds = None

        hidden_states = self.language_model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
            deepstack_vision_embeds=deepstack_vision_embeds,
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        logits = self.lm_head(hidden_states_for_logits)

        # >>> PARALLELISM: Gather sharded logits for logprobs computation <<<
        gathered_logits = None
        if self._gather_logits:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        text_neuron_config: NeuronConfig,
        vision_neuron_config: VisionNeuronConfig,
    ):
        """Build model from HF config and NeuronConfig.

        For now, uses the same NeuronConfig for text. Vision NeuronConfig
        will be added when the vision encoder is implemented.
        """
        config = Qwen3VLConfig.from_configs(
            hf_config,
            text_neuron_config=text_neuron_config,
            vision_neuron_config=vision_neuron_config,
        )
        return cls(config)

    # ── KV Cache Management ──────────────────────────────────────────────
    # >>> PARALLELISM: KV spec uses per-rank head counts (TP-sharded) <<<

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.language_model.layers):
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
        for i, layer in enumerate(self.language_model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        """Load weights from checkpoint.

        HF checkpoint key prefix: model.language_model.layers.{i}.*
        >>> PARALLELISM: Weight loaders handle TP sharding <<<
        <-- MODEL-SPECIFIC: HF checkpoint key → model parameter mappings
        """
        tp_rank = self.rank
        tp_size = self.world_size

        mappings = dict()
        for layer_id in range(len(self.language_model.layers)):
            hf_prefix = f"{HF_TEXT_PREFIX}.layers.{layer_id}"
            our_prefix = f"language_model.layers.{layer_id}"

            # Attention: separate Q, K, V → fused QKV
            mappings[f"{our_prefix}.self_attn.qkv_proj_weight"] = [
                f"{hf_prefix}.self_attn.q_proj.weight",
                f"{hf_prefix}.self_attn.k_proj.weight",
                f"{hf_prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{our_prefix}.self_attn.o_proj_weight"] = (
                f"{hf_prefix}.self_attn.o_proj.weight"
            )

            # QK LayerNorm
            mappings[f"{our_prefix}.self_attn.q_layernorm.weight"] = (
                f"{hf_prefix}.self_attn.q_norm.weight"
            )
            mappings[f"{our_prefix}.self_attn.k_layernorm.weight"] = (
                f"{hf_prefix}.self_attn.k_norm.weight"
            )

            # Decoder layer norms
            mappings[f"{our_prefix}.input_layernorm.weight"] = (
                f"{hf_prefix}.input_layernorm.weight"
            )
            mappings[f"{our_prefix}.post_attention_layernorm.weight"] = (
                f"{hf_prefix}.post_attention_layernorm.weight"
            )

            # Dense MLP
            mappings[f"{our_prefix}.mlp.gate_proj_weight"] = (
                f"{hf_prefix}.mlp.gate_proj.weight"
            )
            mappings[f"{our_prefix}.mlp.up_proj_weight"] = (
                f"{hf_prefix}.mlp.up_proj.weight"
            )
            mappings[f"{our_prefix}.mlp.down_proj_weight"] = (
                f"{hf_prefix}.mlp.down_proj.weight"
            )

        # Embedding, final norm, LM head
        mappings["language_model.embed_tokens.weight"] = (
            f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        )
        mappings["language_model.norm.weight"] = f"{HF_TEXT_PREFIX}.norm.weight"

        if self.text_config.tie_word_embeddings:
            mappings["lm_head.weight"] = f"{HF_TEXT_PREFIX}.embed_tokens.weight"
        else:
            mappings["lm_head.weight"] = "lm_head.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        ).state_dict

        target_dtype = self.text_config.torch_dtype
        for name, tensor in rank_sharded.items():
            if tensor.dtype != target_dtype:
                rank_sharded[name] = tensor.to(target_dtype)

        self.load_state_dict(rank_sharded, strict=False, assign=True)
