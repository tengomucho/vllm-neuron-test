# SPDX-License-Identifier: Apache-2.0
"""Custom BF16 weight loaders for the Gemma4 text decoder.

Gemma4 checkpoints (text decoder under ``model.language_model.``) store:
  - self_attn.{q,k,v}_proj.weight : [out_heads*head_dim, hidden]  (v absent on
    full-attention layers, which reuse K as V via attention_k_eq_v)
  - self_attn.{q,k}_norm.weight   : [head_dim]
  - self_attn.o_proj.weight       : [hidden, num_heads*head_dim]
  - mlp.{gate,up}_proj.weight     : [intermediate, hidden]   (dense MLP)
  - mlp.down_proj.weight          : [hidden, intermediate]
  - experts.gate_up_proj          : [E, 2*moe_I, hidden]  (rows 0:I gate, I:2I up)
  - experts.down_proj             : [E, hidden, moe_I]
  - router.proj.weight            : [E, hidden]
  - router.scale                  : [hidden]
  - router.per_expert_scale       : [E]

These loaders shard for tensor parallelism and transpose into the layouts the
Neuron functional ops expect. ``per_expert_scale`` is folded into the expert
``down_proj`` (scaling an expert's output is equivalent to scaling its routing
weight).
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


# ── Attention ────────────────────────────────────────────────────────────


def attn_in_proj_loader(head_dim: int, heads_per_rank: int, head_start_for_rank):
    """q/k/v_proj: HF [num_param_heads*head_dim, hidden] -> [hidden, heads_per_rank*head_dim].

    ``head_start_for_rank(rank)`` returns the first head index this rank owns
    (handles GQA replication where several ranks share a KV head).
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        w = slices[0]
        hs = head_start_for_rank(rank)
        start = hs * head_dim
        end = (hs + heads_per_rank) * head_dim
        shard = w[start:end, :]  # [heads_per_rank*head_dim, hidden]
        return shard.T.contiguous()  # [hidden, heads_per_rank*head_dim]

    return SafetensorsWeightLoader(transform=transform)


def attn_o_proj_loader(head_dim: int, q_heads_per_rank: int, q_start_for_rank):
    """o_proj: HF [hidden, num_heads*head_dim] -> [q_heads_per_rank*head_dim, hidden]."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        w = slices[0]
        qs = q_start_for_rank(rank)
        start = qs * head_dim
        end = (qs + q_heads_per_rank) * head_dim
        shard = w[:, start:end]  # [hidden, q_heads_per_rank*head_dim]
        return shard.T.contiguous()  # [q_heads_per_rank*head_dim, hidden]

    return SafetensorsWeightLoader(transform=transform)


def passthrough_loader() -> SafetensorsWeightLoader:
    """Load a tensor unchanged (norms, router proj, replicated weights)."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        return slices[0][:].contiguous()

    return SafetensorsWeightLoader(transform=transform)


def router_gamma_loader(hidden_size: int) -> SafetensorsWeightLoader:
    """router.scale [hidden] -> effective RMSNorm gamma = scale * hidden^-0.5."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        scale = slices[0][:].to(torch.float32)
        return (scale * (hidden_size**-0.5)).contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ── Dense MLP ──────────────────────────────────────────────────────────────


def mlp_gate_up_loader(inter_per_rank: int) -> SafetensorsWeightLoader:
    """gate/up_proj: HF [intermediate, hidden] -> [hidden, inter_per_rank]."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        w = slices[0]
        start = rank * inter_per_rank
        end = start + inter_per_rank
        shard = w[start:end, :]  # [inter_per_rank, hidden]
        return shard.T.contiguous()  # [hidden, inter_per_rank]

    return SafetensorsWeightLoader(transform=transform)


def mlp_down_loader(inter_per_rank: int) -> SafetensorsWeightLoader:
    """down_proj: HF [hidden, intermediate] -> [inter_per_rank, hidden]."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        w = slices[0]
        start = rank * inter_per_rank
        end = start + inter_per_rank
        shard = w[:, start:end]  # [hidden, inter_per_rank]
        return shard.T.contiguous()  # [inter_per_rank, hidden]

    return SafetensorsWeightLoader(transform=transform)


# ── MoE experts ──────────────────────────────────────────────────────────


def expert_gate_up_loader(
    num_experts: int, moe_intermediate: int, inter_per_rank: int, hidden_pad: int
) -> SafetensorsWeightLoader:
    """experts.gate_up_proj: HF [E, 2*I, hidden] -> [E, hidden_pad, 2*inter_per_rank].

    Rows 0:I are gate, I:2I are up. Shards the intermediate dim across TP and
    concatenates the rank's gate/up shards as [gate_shard, up_shard]. The hidden
    dim is zero-padded to hidden_pad for the moe_cte kernel (no-op numerically)."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        w = slices[0]  # [E, 2*I, hidden]
        gstart = rank * inter_per_rank
        gend = gstart + inter_per_rank
        ustart = moe_intermediate + rank * inter_per_rank
        uend = ustart + inter_per_rank
        out = []
        for e in range(num_experts):
            gate = w[e][gstart:gend, :]  # [inter_per_rank, hidden]
            up = w[e][ustart:uend, :]
            fused = torch.cat([gate, up], dim=0)  # [2*inter_per_rank, hidden]
            t = fused.T.contiguous()  # [hidden, 2*inter_per_rank]
            if hidden_pad != t.shape[0]:
                t = torch.nn.functional.pad(t, (0, 0, 0, hidden_pad - t.shape[0]))
            out.append(t)  # [hidden_pad, 2*inter_per_rank]
        return torch.stack(out, dim=0)  # [E, hidden_pad, 2*inter_per_rank]

    return SafetensorsWeightLoader(transform=transform)


def expert_down_loader(
    num_experts: int, inter_per_rank: int, hidden_pad: int
) -> SafetensorsWeightLoader:
    """experts.down_proj: HF [E, hidden, I] -> [E, inter_per_rank, hidden_pad],
    with per_expert_scale folded in (slices[1] = router.per_expert_scale [E]).
    Hidden dim zero-padded to hidden_pad for the moe_cte kernel."""

    def transform(slices: list, rank: int) -> torch.Tensor:
        w = slices[0]  # [E, hidden, I]
        per_expert_scale = slices[1][:].to(torch.float32)  # [E]
        start = rank * inter_per_rank
        end = start + inter_per_rank
        out = []
        for e in range(num_experts):
            down = w[e][:, start:end]  # [hidden, inter_per_rank]
            down = down.T.contiguous().to(torch.float32)  # [inter_per_rank, hidden]
            down = down * per_expert_scale[e]
            down = down.to(torch.bfloat16)
            if hidden_pad != down.shape[1]:
                down = torch.nn.functional.pad(
                    down, (0, hidden_pad - down.shape[1])
                )
            out.append(down)  # [inter_per_rank, hidden_pad]
        return torch.stack(out, dim=0)  # [E, inter_per_rank, hidden_pad]

    return SafetensorsWeightLoader(transform=transform)
