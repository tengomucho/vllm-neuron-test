# SPDX-License-Identifier: Apache-2.0
"""
Distributed topk for tensor-parallel inference.

This module provides a distributed topk operation that works across
sharded tensors in a tensor-parallel setting.
"""

import logging

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed._functional_collectives import all_gather_tensor

logger = logging.getLogger(__name__)


def topk(
    tensor: Tensor,
    k: int,
    dim: int,
    gather_dim: int,
    process_group: ProcessGroup | None = None,
    rank: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Performs distributed topk on sharded tensors using 2-step algorithm.

    Args:
        tensor: Input tensor to perform topk on.
        k: Number of top elements to select.
        dim: Dimension along which to find topk.
        gather_dim: Dimension the tensor is sharded on.
        process_group: Process group for distributed operations.
        rank: Rank of this process within the group, as a scalar tensor
            (dtype int32). Required when ``process_group`` is set and
            ``gather_dim == dim``. Passing rank as a tensor avoids baking
            it as a constant during compilation, which would cause
            redundant per-rank compilations.

    Returns:
        Tuple of (values, indices) with global topk across all shards.

    Example:
        >>> import torch
        >>> import torch.distributed as dist
        >>> from vllm_neuron.functional.topk import topk
        >>>
        >>> # Tensor-parallel inference with TP=2, vocab_size=256
        >>> # Each rank has logits of shape (batch=1, vocab_per_rank=128)
        >>> logits = torch.randn(1, 128)  # Local shard on this rank
        >>> pg = dist.new_group([0, 1])
        >>> rank_t = torch.tensor(dist.get_rank(pg), dtype=torch.int32)
        >>>
        >>> # Compute global top-10 across both ranks
        >>> values, indices = topk(logits, k=10, dim=1, gather_dim=1,
        ...                        process_group=pg, rank=rank_t)
        >>> # Returns shapes (1, 10) with global values and indices
    """
    # Fast path for single device (no process group or tp_degree=1)
    if process_group is None:
        tp_degree = 1
    else:
        tp_degree = torch.distributed.get_world_size(group=process_group)

    if tp_degree == 1:
        if _can_use_nki_topk(tensor):
            # Use NKI kernel on Neuron
            return _topk_nki(tensor, k, dim)
        else:
            # CPU fallback
            return _topk_torch(tensor, k, dim)

    # Convert to float32 for kernel accuracy
    ## TODO: remove after migrating topk to NKIv2 frontend
    tensor = tensor.to(torch.float32)

    # Step 1: Local topk on each rank
    local_k = min(k, tensor.shape[gather_dim]) if gather_dim == dim else k
    if _can_use_nki_topk(tensor):
        local_value, local_index = _topk_nki(tensor, local_k, dim)
    else:
        local_value, local_index = _topk_torch(tensor, local_k, dim)

    # Apply sharding offset locally before gather (when gather_dim == dim)
    if gather_dim == dim:
        if rank is None:
            raise ValueError(
                "rank must be provided as a tensor when process_group is set "
                "and gather_dim == dim, to avoid baking rank as a compile-time "
                "constant. Pass torch.tensor(dist.get_rank(group), dtype=torch.int32)."
            )
        offset = rank * tensor.shape[gather_dim]
        local_index = local_index + offset

    # Step 2: Gather from all ranks
    global_values = all_gather_tensor(local_value, gather_dim, group=process_group)
    global_indices = all_gather_tensor(local_index, gather_dim, group=process_group)

    # Step 3: Global topk on gathered results
    if _can_use_nki_topk(global_values):
        values, global_max_local_index = _topk_nki(global_values, k, dim)
    else:
        values, global_max_local_index = _topk_torch(global_values, k, dim)

    # Convert indices to match global_indices dtype for gather
    global_max_local_index = global_max_local_index.to(dtype=global_indices.dtype)
    final_indices = torch.gather(global_indices, dim, global_max_local_index)

    return values, final_indices


def batch_sharded_topk(
    tensor: Tensor,
    k: int,
    dim: int,
    dp_degree: int,
    all2all_group: ProcessGroup,
    gather_group: ProcessGroup,
) -> tuple[Tensor, Tensor]:
    """Distributed topk with batch sharding for data-parallel sampling.

    Uses all-to-all to redistribute batch→vocab, performs local topk on larger
    vocab slices, then gathers results for final global topk.

    Flow:
        1. All-to-all: [batch, vocab/TP] → [batch/DP, vocab/TP × DP]
        2. Local topk on each rank's larger vocab slice
        3. Index correction: local indices → global vocab indices
        4. Gather topk results across gather_group
        5. Final topk on gathered results

    Args:
        tensor: Input tensor to perform topk on. Shape: [batch, vocab_per_rank].
        k: Number of top elements to select.
        dim: Dimension along which to find topk (typically -1 for vocab).
        dp_degree: Data-parallel degree (batch sharding factor).
        all2all_group: Process group for batch↔vocab redistribution.
        gather_group: Process group for gathering topk results.

    Returns:
        Tuple of (values, indices) with global topk across all shards.
        Shapes: [batch/DP, k] for both values and indices.

    Example:
        >>> # TP=8, DP=2, batch=8, vocab_per_rank=16K
        >>> logits = torch.randn(8, 16384)
        >>> values, indices = batch_sharded_topk(
        ...     logits, k=10, dim=-1, process_group=tp_group,
        ...     dp_degree=2, all2all_group=..., gather_group=..., all2all_mesh=...
        ... )
        >>> values.shape, indices.shape  # ([4, 10], [4, 10])
    """
    from vllm_neuron.functional import all_to_all

    # Step 1: All-to-all redistribution (batch → vocab)
    tensor = all_to_all(
        tensor,
        split_dim=0,
        concat_dim=dim,
        group=all2all_group,
    )

    bs = tensor.shape[0]  # batch/DP
    sharded_size = tensor.shape[dim] // dp_degree

    # Convert to float32 for kernel accuracy
    ## TODO: remove after migrating topk to nki frontend
    tensor = tensor.to(torch.float32)

    # Step 2: Local topk on larger vocab slice
    if _can_use_nki_topk(tensor):
        local_value, local_index = _topk_nki(tensor, k, dim)
    else:
        local_value, local_index = _topk_torch(tensor, k, dim)

    # Step 3: Index correction (local → global vocab indices)
    # After all-to-all, this rank's vocab is concatenated from the ranks in its
    # all2all group. The shard order matches the rank order in the group, so we
    # just need this rank's all2all group members to map shard ID → global rank.
    rank_map = torch.tensor(
        dist.get_process_group_ranks(all2all_group),
        dtype=torch.int32,
        device=tensor.device,
    )

    # Decompose local index: which shard + offset within shard
    quotient = local_index // sharded_size  # which shard (0, 1, 2, ...)
    remainder = local_index % sharded_size  # position within shard
    flat_quotient = quotient.flatten()
    shard_rank_id_flat = rank_map[flat_quotient]
    shard_rank_id = shard_rank_id_flat.view(
        quotient.size(0), quotient.size(1)
    )  # Dynamic shape

    # Convert to global vocab index
    global_index = shard_rank_id * sharded_size + remainder

    # Step 4: Gather topk results across gather_group
    topk_results = torch.cat([local_value, global_index.float()], dim=0)
    gathered_results = all_gather_tensor(topk_results, dim, group=gather_group)
    global_values, global_indices = torch.split(gathered_results, bs, dim=0)
    global_indices = global_indices.to(torch.int32)

    # Step 5: Final topk on gathered results
    if _can_use_nki_topk(global_values):
        values, final_local_idx = _topk_nki(global_values, k, dim)
    else:
        values, final_local_idx = _topk_torch(global_values, k, dim)
    final_indices = torch.gather(global_indices, dim, final_local_idx.to(torch.int32))

    return values, final_indices


def _can_use_nki_topk(tensor: Tensor) -> bool:
    """
    Check if we can use the NKI topk kernel.

    NOTE: Currently disabled. Will be updated when integrating with v2 topk kernel.
    """
    return False


def _topk_nki(tensor: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    """
    Compute topk using NKI rotated_topk_kernel.

    NOTE: Currently disabled. Will be updated when integrating with v2 topk kernel.

    Args:
        tensor: Input tensor.
        k: Number of top elements to select.
        dim: Dimension along which to find topk.

    Returns:
        Tuple of (values, indices) with top-k elements.
    """
    pass


def _topk_torch(tensor: Tensor, k: int, dim: int) -> tuple[Tensor, Tensor]:
    """
    Compute topk using torch.topk (CPU fallback).

    Args:
        tensor: Input tensor.
        k: Number of top elements to select.
        dim: Dimension along which to find topk.

    Returns:
        Tuple of (values, indices) with top-k elements.
    """
    return torch.topk(tensor, k, dim=dim)
