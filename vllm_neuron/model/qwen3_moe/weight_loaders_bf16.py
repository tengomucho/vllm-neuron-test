# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for Qwen3 MoE BF16 checkpoint format.

Qwen3 MoE checkpoints store individual expert weights:
  - model.layers.{L}.mlp.experts.{E}.gate_proj.weight: [I, H]
  - model.layers.{L}.mlp.experts.{E}.up_proj.weight: [I, H]
  - model.layers.{L}.mlp.experts.{E}.down_proj.weight: [H, I]

These loaders handle fusing gate+up, TP sharding on intermediate dim,
and EP filtering of experts.
"""

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader


def expert_gate_up_weight_loader(
    num_experts: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load and fuse gate_proj + up_proj for all experts with TP sharding.

    Expects slices = [gate_0, ..., gate_{E-1}, up_0, ..., up_{E-1}].
    Returns [E, H, I*2/TP] (transposed for parameter layout).
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 2 * num_experts
        gate_slices = slices[:num_experts]
        up_slices = slices[num_experts:]

        tp_rank = rank % num_shards
        half = shard_size // 2
        start = tp_rank * half
        end = start + half

        result = []
        for i in range(num_experts):
            # gate_proj: [I, H], up_proj: [I, H] — shard on I dim
            gate = gate_slices[i][start:end, :]
            up = up_slices[i][start:end, :]
            # [I*2/TP, H] -> transpose to [H, I*2/TP]
            result.append(torch.cat([gate, up], dim=0).T.contiguous())

        return torch.stack(result, dim=0)  # [E, H, I*2/TP]

    return SafetensorsWeightLoader(transform=transform)


def expert_down_weight_loader(
    num_experts: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Load down_proj for all experts with TP sharding.

    Expects slices = [down_0, ..., down_{E-1}].
    Returns [E, I/TP, H] (transposed for parameter layout).
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == num_experts

        tp_rank = rank % num_shards
        start = tp_rank * shard_size
        end = start + shard_size

        result = []
        for i in range(num_experts):
            # down_proj: [H, I] — shard on I (dim 1)
            down = slices[i][:, start:end]
            # [H, I/TP] -> transpose to [I/TP, H]
            result.append(down.T.contiguous())

        return torch.stack(result, dim=0)  # [E, I/TP, H]

    return SafetensorsWeightLoader(transform=transform)
