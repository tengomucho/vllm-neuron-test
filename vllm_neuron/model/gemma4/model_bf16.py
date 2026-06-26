# SPDX-License-Identifier: Apache-2.0
"""
Gemma4 (text decoder) BF16 implementation for the Neuron backend.

Correctness-first port. Notable Gemma4 traits handled here:
  - Every layer runs a DENSE MLP and a 128-expert MoE in parallel and sums them.
  - 7 RMSNorms per layer (Gemma4 plain `* weight` form, fp32 accumulation).
  - Hybrid attention: sliding-window layers (head_dim 256, 8 KV heads, default
    RoPE) and full-attention layers (head_dim 512, 2 KV heads, proportional /
    partial RoPE, K reused as V via attention_k_eq_v).
  - q_norm / k_norm before RoPE, parameterless v_norm on values, softmax scale 1.0.
  - Embedding scaled by sqrt(hidden), final logit softcapping, tied LM head.

Parallelism (v1): pure tensor parallel, NO sequence parallelism. Hidden states
are replicated across ranks; sharded matmuls are followed by all_reduce.
Attention uses plain torch ops (head_dim > 128 rules out the flash kernel).

Env-var overrides (bisect / debug):
  GEMMA4_STUB_ATTN=1   Replace attention computation with a zero tensor (no QK^T
                        or softmax); use to confirm whether attention is the
                        neuronx-cc NCC_IGCA024 trigger before the full fix lands.
"""

import logging
import math
import os as _os

import nki.language as nl
import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
import vllm_neuron.nn as neuron_nn

from transformers import PretrainedConfig

from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
)
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation
from vllm_neuron.functional.moe.router import RouterComputationOrder

from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.nn.sampler import Sampler
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    set_weight_loader,
    sharding_weight_loader_with_padding,
)

from .config import Gemma4Config
from . import weight_loaders as WL

logger = logging.getLogger(__name__)

# Set GEMMA4_STUB_ATTN=1 to replace attention with zeros for bisect testing.
_STUB_ATTN = _os.environ.get("GEMMA4_STUB_ATTN", "0") == "1"


# =============================================================================
# Norms
# =============================================================================


def gemma_rms_norm(
    x: torch.Tensor, weight: torch.Tensor | None, eps: float
) -> torch.Tensor:
    """Gemma4 RMSNorm over the last dim: (x / rms(x)) * weight, fp32.

    NOTE: Gemma4 uses a plain `* weight` (weights centered around 1), NOT the
    Gemma1/2/3 `* (1 + weight)` convention. weight=None gives the parameterless
    variant (used for v_norm)."""
    input_dtype = x.dtype
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(-1, keepdim=True)
    xn = xf * torch.pow(var + eps, -0.5)
    if weight is not None:
        xn = xn * weight.to(torch.float32)
    return xn.to(input_dtype)


class Gemma4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gemma_rms_norm(x, self.weight, self.eps)


# =============================================================================
# Rotary embeddings (per layer type)
# =============================================================================


_GELU_C = 0.7978845608028654  # sqrt(2/pi)


def _gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """gelu_pytorch_tanh, written with elementary ops (traceable by dynamo)."""
    xf = x.to(torch.float32)
    out = 0.5 * xf * (1.0 + torch.tanh(_GELU_C * (xf + 0.044715 * xf * xf * xf)))
    return out.to(x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, nheads, head_dim]; cos/sin: [T, head_dim]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class Gemma4RotaryEmbedding(nn.Module):
    """Computes (cos, sin) for both sliding (default) and full (proportional)
    attention layer types."""

    def __init__(self, config: Gemma4Config):
        super().__init__()
        # Sliding: default RoPE over head_dim.
        s_rope = config.layer_rope_for_type("sliding_attention")
        s_dim = config.head_dim
        s_theta = s_rope.get("rope_theta", 10000.0)
        s_inv = 1.0 / (
            s_theta
            ** (torch.arange(0, s_dim, 2, dtype=torch.float, device="cpu") / s_dim)
        )
        self.register_buffer("sliding_inv_freq", s_inv, persistent=False)

        # Full: proportional RoPE with partial rotary over global_head_dim.
        f_rope = config.layer_rope_for_type("full_attention")
        f_dim = config.global_head_dim
        f_theta = f_rope.get("rope_theta", 1000000.0)
        proportion = f_rope.get("partial_rotary_factor", 0.25)
        rope_angles = int(proportion * f_dim // 2)
        inv_rotated = 1.0 / (
            f_theta
            ** (
                torch.arange(0, 2 * rope_angles, 2, dtype=torch.float, device="cpu")
                / f_dim
            )
        )
        nope = f_dim // 2 - rope_angles
        f_inv = torch.cat(
            [inv_rotated, torch.zeros(nope, dtype=torch.float, device="cpu")]
        )
        self.register_buffer("full_inv_freq", f_inv, persistent=False)

    def _cos_sin(self, inv_freq, positions, dtype):
        freqs = positions.float().unsqueeze(1) * inv_freq.unsqueeze(0)  # [T, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, dim]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, positions: torch.Tensor, dtype: torch.dtype):
        cs_s = self._cos_sin(self.sliding_inv_freq, positions, dtype)
        cs_f = self._cos_sin(self.full_inv_freq, positions, dtype)
        return {"sliding_attention": cs_s, "full_attention": cs_f}


# =============================================================================
# Attention (manual; plain torch ops to support head_dim > 128)
# =============================================================================


class Gemma4Attention(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.is_sliding = config.is_sliding(layer_idx)
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.head_dim = config.layer_head_dim(layer_idx)
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.layer_num_kv_heads(layer_idx)
        # attention_k_eq_v: full-attention layers have no v_proj; reuse K.
        self.k_eq_v = (not self.is_sliding)
        self.scaling = 1.0  # Gemma uses no 1/sqrt(d); q/k norm handle scaling.

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        nh = self.num_attention_heads
        nkv = self.num_key_value_heads
        assert nh % self.world_size == 0, "num_heads must be divisible by TP"
        self.q_heads_per_rank = nh // self.world_size
        if self.world_size <= nkv:
            self.kv_heads_per_rank = nkv // self.world_size
            self.num_kv_replicas = 1
        else:
            self.kv_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // nkv
        self.num_kv_groups = self.q_heads_per_rank // self.kv_heads_per_rank
        self.heads_per_kv = nh // nkv  # q heads served by one kv head

        hd = self.head_dim
        # Weights stored as [hidden, out_per_rank] for matmul (h @ W).
        self.q_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.q_heads_per_rank * hd, dtype=self.dtype)
        )
        self.k_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.kv_heads_per_rank * hd, dtype=self.dtype)
        )
        if not self.k_eq_v:
            self.v_proj_weight = nn.Parameter(
                torch.empty(
                    self.hidden_size, self.kv_heads_per_rank * hd, dtype=self.dtype
                )
            )
        # o_proj stored as [q_heads_per_rank*hd, hidden].
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.q_heads_per_rank * hd, self.hidden_size, dtype=self.dtype)
        )
        self.q_norm = nn.Parameter(torch.zeros(hd, dtype=self.dtype))
        self.k_norm = nn.Parameter(torch.zeros(hd, dtype=self.dtype))

        self.k_cache = None
        self.v_cache = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        hd = self.head_dim
        q_start = lambda r: r * self.q_heads_per_rank  # noqa: E731
        kv_start = lambda r: (r * self.q_heads_per_rank) // self.heads_per_kv  # noqa: E731
        set_weight_loader(
            self.q_proj_weight,
            WL.attn_in_proj_loader(hd, self.q_heads_per_rank, q_start),
        )
        set_weight_loader(
            self.k_proj_weight,
            WL.attn_in_proj_loader(hd, self.kv_heads_per_rank, kv_start),
        )
        if not self.k_eq_v:
            set_weight_loader(
                self.v_proj_weight,
                WL.attn_in_proj_loader(hd, self.kv_heads_per_rank, kv_start),
            )
        set_weight_loader(
            self.o_proj_weight,
            WL.attn_o_proj_loader(hd, self.q_heads_per_rank, q_start),
        )
        set_weight_loader(self.q_norm, WL.passthrough_loader())
        set_weight_loader(self.k_norm, WL.passthrough_loader())

    def forward(self, hidden_states, positions, cos_sin, attn_metadata, is_prefill):
        T = hidden_states.shape[0]
        hd = self.head_dim
        nqh = self.q_heads_per_rank
        nkv = self.kv_heads_per_rank
        cos, sin = cos_sin
        hidden_states = hidden_states.to(self.dtype)

        q = torch.matmul(hidden_states, self.q_proj_weight).view(T, nqh, hd)
        q = gemma_rms_norm(q, self.q_norm, self.eps)
        q = apply_rotary(q, cos, sin)

        k_raw = torch.matmul(hidden_states, self.k_proj_weight).view(T, nkv, hd)
        if self.k_eq_v:
            v_raw = k_raw
        else:
            v_raw = torch.matmul(hidden_states, self.v_proj_weight).view(T, nkv, hd)
        k = apply_rotary(gemma_rms_norm(k_raw, self.k_norm, self.eps), cos, sin)
        v = gemma_rms_norm(v_raw, None, self.eps)  # parameterless v_norm

        self._write_cache(k, v, attn_metadata)

        if is_prefill:
            attn = self._attn_prefill(q, k, v, positions)
        else:
            attn = self._attn_decode(q, positions, attn_metadata)

        out = attn.reshape(T, nqh * hd)
        out = torch.matmul(out, self.o_proj_weight)  # [T, hidden]
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out

    def _layer_name(self):
        return f"layers.{self.layer_idx}.self_attn"

    def _write_cache(self, k, v, attn_metadata):
        meta = attn_metadata[self._layer_name()]
        slot_mapping = meta["slot_mapping"]
        block_size = meta["block_size"]
        nkv = self.kv_heads_per_rank
        hd = self.head_dim
        block_idx = slot_mapping // block_size
        pos_idx = slot_mapping % block_size
        ntok = slot_mapping.shape[0]
        # k, v: [T, nkv, hd] -> [nkv, T, hd]
        k_t = k.transpose(0, 1).reshape(-1, hd).to(self.k_cache.dtype)
        v_t = v.transpose(0, 1).reshape(-1, hd).to(self.v_cache.dtype)
        head_idx = torch.arange(
            nkv, dtype=torch.long, device=k.device
        ).repeat_interleave(ntok)
        b_idx = block_idx.repeat(nkv)
        p_idx = pos_idx.repeat(nkv)
        self.k_cache.index_put_((b_idx, head_idx, p_idx), k_t)
        self.v_cache.index_put_((b_idx, head_idx, p_idx), v_t)

    def _attn_prefill(self, q, k, v, positions):
        # q: [T, nqh, hd]; k, v: [T, nkv, hd]
        if _STUB_ATTN:
            return torch.zeros_like(q)

        T = q.shape[0]
        g = self.num_kv_groups
        kg = k.repeat_interleave(g, dim=1)  # [T, nqh, hd]
        vg = v.repeat_interleave(g, dim=1)
        q_ = q.permute(1, 0, 2)  # [nqh, T, hd]
        k_ = kg.permute(1, 0, 2)
        v_ = vg.permute(1, 0, 2)
        scores = torch.matmul(q_, k_.transpose(1, 2)) * self.scaling  # [nqh, T, T]
        # Index-based causal mask: a full prefill is a single contiguous sequence
        # (token index == position). Using indices (not the `positions` tensor)
        # ensures bucket-padding tokens (appended at higher indices) are never
        # attended to by real tokens, regardless of their padded position ids.
        ar = torch.arange(T, device=q.device)
        qi = ar.view(T, 1)
        kj = ar.view(1, T)
        allowed = kj <= qi  # causal
        if self.sliding_window is not None:
            allowed = allowed & ((qi - kj) < self.sliding_window)
        # Additive mask: avoids scalar-broadcast torch.where which triggers
        # neuronx-cc NCC_IGCA024 at large head_dim (256/512).
        mask = allowed.logical_not().to(torch.float32) * -1e30  # [T, T]
        scores = scores.to(torch.float32) + mask.unsqueeze(0)
        attn = torch.softmax(scores, dim=-1).to(v_.dtype)
        out = torch.matmul(attn, v_)  # [nqh, T, hd]
        return out.permute(1, 0, 2)  # [T, nqh, hd]

    def _attn_decode(self, q, positions, attn_metadata):
        meta = attn_metadata[self._layer_name()]
        # kc/vc are pre-gathered by the runner (outside torch.compile) before this
        # model call, then passed here via attn_metadata.  This avoids
        # self.k_cache[block_table] inside the compiled graph: that gather forces
        # neuronx-cc to create a full-pool intermediate (~629 MB per layer) which
        # drives the decode NEFF past the 24 GB NCC_EOOM002 HBM limit.
        kc = meta["kc"]  # [B, max_blocks, nkv, block_size, hd] — pre-gathered
        vc = meta["vc"]
        B, max_blocks, nkv, block_size, _ = kc.shape
        S_ctx = max_blocks * block_size
        T = q.shape[0]
        S_dec = T // B
        hd = self.head_dim
        nqh = self.q_heads_per_rank
        g = self.num_kv_groups

        if _STUB_ATTN:
            return q.view(B, S_dec, nqh, hd).view(T, nqh, hd)

        kc = kc.permute(0, 2, 1, 3, 4).reshape(B, nkv, S_ctx, hd)
        vc = vc.permute(0, 2, 1, 3, 4).reshape(B, nkv, S_ctx, hd)
        kc = kc.repeat_interleave(g, dim=1)  # [B, nqh, S_ctx, hd]
        vc = vc.repeat_interleave(g, dim=1)

        q_ = q.view(B, S_dec, nqh, hd).permute(0, 2, 1, 3)  # [B, nqh, S_dec, hd]
        scores = torch.matmul(q_, kc.transpose(2, 3)) * self.scaling

        qpos = positions.view(B, S_dec).to(torch.float32)
        kidx = torch.arange(S_ctx, device=q.device, dtype=torch.float32).view(1, 1, S_ctx)
        qp = qpos.view(B, S_dec, 1)
        allowed = kidx <= qp
        if self.sliding_window is not None:
            allowed = allowed & ((qp - kidx) < self.sliding_window)
        # Additive mask: avoids scalar-broadcast torch.where which triggers
        # neuronx-cc NCC_IGCA024 at large head_dim (256/512).
        mask = allowed.logical_not().to(torch.float32) * -1e30
        scores = scores.to(torch.float32) + mask.unsqueeze(1)
        attn = torch.softmax(scores, dim=-1).to(vc.dtype)
        out = torch.matmul(attn, vc)
        out = out.permute(0, 2, 1, 3).reshape(T, nqh, hd)
        return out


# =============================================================================
# Dense MLP
# =============================================================================


class Gemma4DenseMLP(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.inter_per_rank = config.intermediate_size // self.world_size
        self.gate_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.inter_per_rank, dtype=self.dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(config.hidden_size, self.inter_per_rank, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.inter_per_rank, config.hidden_size, dtype=self.dtype)
        )
        set_weight_loader(
            self.gate_proj_weight, WL.mlp_gate_up_loader(self.inter_per_rank)
        )
        set_weight_loader(
            self.up_proj_weight, WL.mlp_gate_up_loader(self.inter_per_rank)
        )
        set_weight_loader(
            self.down_proj_weight, WL.mlp_down_loader(self.inter_per_rank)
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.to(self.dtype)
        gate = torch.matmul(x, self.gate_proj_weight)  # [T, inter_per_rank]
        up = torch.matmul(x, self.up_proj_weight)
        gate = _gelu_tanh(gate)
        out = torch.matmul(gate * up, self.down_proj_weight)  # [T, hidden]
        if self.world_size > 1:
            out = self.tp_group.all_reduce(out)
        return out


# =============================================================================
# MoE experts
# =============================================================================


class Gemma4Experts(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.rms_norm_eps = config.rms_norm_eps
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.top_k_experts
        self.norm_topk_prob = config.norm_topk_prob

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.block_size = 256

        # >>> PARALLELISM: TP + EP. EP shards the expert COUNT across ep ranks;
        # TP shards each expert's intermediate dim across tp ranks (tp = world/ep).
        from vllm.config import get_current_vllm_config

        self.ep_enabled = (
            get_current_vllm_config().parallel_config.enable_expert_parallel
        )
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_ep_degree,
            get_neuron_ep_rank,
            get_neuron_ep_tp_group,
        )

        self.ep_degree = get_neuron_ep_degree() if self.ep_enabled else 1
        self.ep_rank = get_neuron_ep_rank() if self.ep_enabled else 0
        self.tp_degree = self.world_size // self.ep_degree
        self.ep_tp_group = (
            get_neuron_ep_tp_group() if self.ep_enabled else self.tp_group
        )
        self.num_local_experts = self.num_experts // self.ep_degree
        self.expert_offset = self.ep_rank * self.num_local_experts
        self.inter_per_rank = config.moe_intermediate_size // self.tp_degree
        # moe_cte NKI kernel tiles the hidden dim and needs it a multiple of 512.
        # Pad the expert hidden dim with zeros (numerical no-op); unpad output.
        self.hidden_pad = ((self.hidden_size + 511) // 512) * 512

        self.router_weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=torch.bfloat16)
        )
        self.router_gamma = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.float32)
        )
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_pad,
                self.inter_per_rank * 2,
                dtype=self.dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.inter_per_rank,
                self.hidden_pad,
                dtype=self.dtype,
            )
        )

        set_weight_loader(self.router_weight, WL.passthrough_loader())
        set_weight_loader(self.router_gamma, WL.router_gamma_loader(self.hidden_size))
        set_weight_loader(
            self.gate_up_proj_weight,
            WL.expert_gate_up_loader(
                self.num_local_experts,
                config.moe_intermediate_size,
                self.inter_per_rank,
                self.tp_degree,
                self.hidden_pad,
            ),
        )
        set_weight_loader(
            self.down_proj_weight,
            WL.expert_down_loader(
                self.num_local_experts,
                self.inter_per_rank,
                self.tp_degree,
                self.hidden_pad,
            ),
        )

    def forward(
        self,
        router_input: torch.Tensor,
        expert_input: torch.Tensor,
        is_prefill: bool,
        rank: torch.Tensor | None = None,
    ):
        """router_input: raw post-attn residual [T, H]; expert_input:
        pre_feedforward_layernorm_2(residual) [T, H]."""
        expert_affinities = NF.router(
            hidden_states=router_input,
            router_weights=self.router_weight.T,
            top_k=self.num_experts_per_token,
            activation="softmax",
            gamma=self.router_gamma,
            eps=self.rms_norm_eps,
            computation_dtype=torch.float32,
            router_computation_order=RouterComputationOrder.PRENORM_LINEAR_ACT_TOPK_RENORM_SCATTER,
        )

        # >>> EP: keep only this rank's local experts' affinities [T, E_local] <<<
        if self.ep_degree > 1:
            ep_rank = (rank % (self.ep_degree * self.tp_degree)) // self.tp_degree
            local_idx = (
                torch.arange(
                    self.num_local_experts,
                    device=expert_input.device,
                    dtype=torch.int32,
                )
                + ep_rank.to(torch.int32) * self.num_local_experts
            )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_idx
            )

        # Pad hidden dim to the kernel-friendly size (zeros).
        h_in = expert_input.to(self.dtype)
        if self.hidden_pad != self.hidden_size:
            h_in = torch.nn.functional.pad(
                h_in, (0, self.hidden_pad - self.hidden_size)
            )

        num_tokens = expert_input.shape[0]

        # Use the vectorized einsum path for decode OR for small prefill buckets.
        # The moe_cte NKI kernel tiles the token dimension in 128-element NKI
        # partitions — it asserts num_tokens >= 128. For decode (always few tokens)
        # and for small prefill buckets (e.g. the 64-token test bucket), fall back
        # to the einsum which has no such constraint. In production, prefill buckets
        # are ≥ 128 so moe_cte is used there.
        _NKI_PARTITION = 128
        if not is_prefill or num_tokens < _NKI_PARTITION:
            # Decode / small-prefill: vectorized local-expert einsum, POST_SCALE.
            gu = self.gate_up_proj_weight.reshape(
                self.num_local_experts, self.hidden_pad, 2, self.inter_per_rank
            )
            gate = torch.einsum("th,ehi->eti", h_in, gu[:, :, 0, :])
            up = torch.einsum("th,ehi->eti", h_in, gu[:, :, 1, :])
            inter = _gelu_tanh(gate) * up  # [E_local, T, I]
            outs = torch.einsum("eti,eih->eth", inter, self.down_proj_weight)
            aff = expert_affinities.to(outs.dtype).T.unsqueeze(-1)  # [E_local, T, 1]
            output = (outs * aff).sum(0)  # [T, hidden_pad]
            if self.hidden_pad != self.hidden_size:
                output = output[:, : self.hidden_size]
            if self.world_size > 1:
                output = self.tp_group.all_reduce(output)
            return output
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
        num_static_block = math.ceil(
            num_tokens * self.num_experts_per_token / self.ep_degree / self.block_size
        )
        output = NF.moe_cte(
            implementation=MoECTEImplementation.shard_on_block,
            conditions=conditions,
            hidden_states=h_in,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=self.gate_up_proj_weight.reshape(
                self.num_local_experts, self.hidden_pad, 2, self.inter_per_rank
            ),
            down_proj_weight=self.down_proj_weight,
            activation_function=ActFnType.GELU_Tanh_Approx,
            block_size=self.block_size,
            token_position_to_id=token_position_to_id.to(dtype=torch.int32),
            block_to_expert=block_to_expert.to(dtype=torch.int32),
            expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
            skip_token=True,
            is_tensor_update_accumulating=True,
            compute_dtype=nl.bfloat16,
            num_static_block=num_static_block,
        )
        # Unpad hidden dim.
        if self.hidden_pad != self.hidden_size:
            output = output[:, : self.hidden_size]
        if self.world_size > 1:
            output = self.tp_group.all_reduce(output)
        return output


# =============================================================================
# Decoder layer
# =============================================================================


class Gemma4DecoderLayer(nn.Module):
    def __init__(self, config: Gemma4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        dt = config.torch_dtype
        eps = config.rms_norm_eps
        H = config.hidden_size

        self.self_attn = Gemma4Attention(config, layer_idx)
        self.mlp = Gemma4DenseMLP(config)
        self.experts = Gemma4Experts(config)

        self.input_layernorm = Gemma4RMSNorm(H, eps, dt)
        self.post_attention_layernorm = Gemma4RMSNorm(H, eps, dt)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(H, eps, dt)
        self.post_feedforward_layernorm = Gemma4RMSNorm(H, eps, dt)
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(H, eps, dt)
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(H, eps, dt)
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(H, eps, dt)
        # Per-layer learned output scalar (Gemma4: hidden_states *= layer_scalar).
        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=dt))
        set_weight_loader(self.layer_scalar, WL.passthrough_loader())

    def forward(
        self, hidden_states, positions, cos_sin, attn_metadata, is_prefill, rank=None
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, cos_sin, attn_metadata, is_prefill
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        # Dense MLP branch
        mlp_in = self.pre_feedforward_layernorm(hidden_states)
        mlp_out = self.mlp(mlp_in)
        h1 = self.post_feedforward_layernorm_1(mlp_out)
        # MoE branch (router reads raw residual; experts read their own norm)
        expert_in = self.pre_feedforward_layernorm_2(residual)
        moe_out = self.experts(residual, expert_in, is_prefill, rank)
        h2 = self.post_feedforward_layernorm_2(moe_out)

        hidden_states = h1 + h2
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = hidden_states * self.layer_scalar
        return hidden_states


# =============================================================================
# Backbone
# =============================================================================


class Gemma4Model(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.dtype = config.torch_dtype
        self.embed_scale = float(config.hidden_size) ** 0.5

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )
        self.layers = nn.ModuleList(
            [Gemma4DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = Gemma4RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Gemma4RotaryEmbedding(config)

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
        set_weight_loader(self.norm.weight, WL.passthrough_loader())

    def forward(self, input_ids, positions, attn_metadata, rank, is_prefill):
        # No sequence parallelism: replicate full hidden on all ranks.
        hidden_states = self.embed_tokens(input_ids, scatter_tokens=False, rank=rank)
        # Gemma scales embeddings by sqrt(hidden) in bf16.
        hidden_states = hidden_states.to(self.dtype)
        embed_scale = torch.tensor(
            self.embed_scale, dtype=hidden_states.dtype, device=hidden_states.device
        )
        hidden_states = hidden_states * embed_scale

        cos_sin = self.rotary_emb(positions, hidden_states.dtype)

        for layer in self.layers:
            ltype = (
                "sliding_attention"
                if layer.self_attn.is_sliding
                else "full_attention"
            )
            hidden_states = layer(
                hidden_states,
                positions,
                cos_sin[ltype],
                attn_metadata,
                is_prefill,
                rank,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


# =============================================================================
# Causal LM
# =============================================================================


class Gemma4ForCausalLM(nn.Module):
    def __init__(self, config: Gemma4Config):
        super().__init__()
        self.config = config
        self.model = Gemma4Model(config)
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.final_logit_softcapping = config.final_logit_softcapping

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
    ):
        if inputs_embeds is not None:
            raise ValueError("Input embeddings are not supported.")
        positions = positions.to(torch.int32)

        first = "layers.0.self_attn"
        max_query_len = attn_metadata[first]["max_query_len"]
        decode_token_threshold = attn_metadata[first]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.model(
            input_ids, positions, attn_metadata, rank, is_prefill
        )

        hidden_states = torch.index_select(hidden_states, 0, sampling_positions)
        hidden_states = hidden_states.to(self.config.torch_dtype)
        logits = self.lm_head(hidden_states)

        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap

        if self.on_device_sampling_config is None:
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )
        gathered_logits = None
        if self._gather_logits and self.tp_group is not None:
            gathered_logits = self.tp_group.all_gather(logits, dim=1)

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler

            return rejection_sampler(spec_decode_metadata, sampled_tokens)
        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = Gemma4Config.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV cache ─────────────────────────────────────────────────────────
    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            layers.append(
                LayerSpec(
                    name=f"layers.{i}.self_attn",
                    num_kv_heads=attn.kv_heads_per_rank,
                    head_size=attn.head_dim,
                    dtype=attn.dtype,
                    sliding_window_size=attn.sliding_window,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            name = f"layers.{i}.self_attn"
            if name not in kv_caches:
                raise Exception(f"KV cache for layer {name} not initialized")
            layer.self_attn.k_cache = kv_caches[name][0]
            layer.self_attn.v_cache = kv_caches[name][1]

    # ── Weight loading ───────────────────────────────────────────────────
    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        tp_rank = self.tp_group.rank_in_group
        tp_size = self.world_size
        P = "model.language_model"
        mappings = dict()

        for i, layer in enumerate(self.model.layers):
            lp = f"{P}.layers.{i}"
            attn = layer.self_attn
            mappings[f"model.layers.{i}.self_attn.q_proj_weight"] = (
                f"{lp}.self_attn.q_proj.weight"
            )
            mappings[f"model.layers.{i}.self_attn.k_proj_weight"] = (
                f"{lp}.self_attn.k_proj.weight"
            )
            if not attn.k_eq_v:
                mappings[f"model.layers.{i}.self_attn.v_proj_weight"] = (
                    f"{lp}.self_attn.v_proj.weight"
                )
            mappings[f"model.layers.{i}.self_attn.o_proj_weight"] = (
                f"{lp}.self_attn.o_proj.weight"
            )
            mappings[f"model.layers.{i}.self_attn.q_norm"] = (
                f"{lp}.self_attn.q_norm.weight"
            )
            mappings[f"model.layers.{i}.self_attn.k_norm"] = (
                f"{lp}.self_attn.k_norm.weight"
            )

            # Dense MLP
            mappings[f"model.layers.{i}.mlp.gate_proj_weight"] = (
                f"{lp}.mlp.gate_proj.weight"
            )
            mappings[f"model.layers.{i}.mlp.up_proj_weight"] = (
                f"{lp}.mlp.up_proj.weight"
            )
            mappings[f"model.layers.{i}.mlp.down_proj_weight"] = (
                f"{lp}.mlp.down_proj.weight"
            )

            # MoE
            mappings[f"model.layers.{i}.experts.router_weight"] = (
                f"{lp}.router.proj.weight"
            )
            mappings[f"model.layers.{i}.experts.router_gamma"] = f"{lp}.router.scale"
            mappings[f"model.layers.{i}.experts.gate_up_proj_weight"] = (
                f"{lp}.experts.gate_up_proj"
            )
            mappings[f"model.layers.{i}.experts.down_proj_weight"] = [
                f"{lp}.experts.down_proj",
                f"{lp}.router.per_expert_scale",
            ]

            # 7 norms
            for n in (
                "input_layernorm",
                "post_attention_layernorm",
                "pre_feedforward_layernorm",
                "post_feedforward_layernorm",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
                "pre_feedforward_layernorm_2",
            ):
                mappings[f"model.layers.{i}.{n}.weight"] = f"{lp}.{n}.weight"
            mappings[f"model.layers.{i}.layer_scalar"] = f"{lp}.layer_scalar"

        mappings["model.embed_tokens.weight"] = f"{P}.embed_tokens.weight"
        mappings["model.norm.weight"] = f"{P}.norm.weight"
        # Tied LM head.
        mappings["lm_head.weight"] = f"{P}.embed_tokens.weight"

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device
        ).state_dict
        self.load_state_dict(rank_sharded, strict=False, assign=True)
