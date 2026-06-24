# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from safetensors import PySafeSlice

_WEIGHT_LOADER_ATTR = "weight_loader"


@dataclass
class ShardSpec:
    """Specification for sharding a tensor along a dimension.

    Args:
        dim: Dimension to shard along
        size: Size of each shard
        num_shards: Total number of shards (typically TP world size)
    """

    dim: int
    size: int
    num_shards: int


@dataclass
class SafetensorsWeightLoader:
    """Defines how to load checkpoint tensor(s) from a Safetensor slice into a parameter.

    SafetensorsWeightLoader allows customizing how checkpoint tensors are transformed before
    being loaded into a model parameter. This is useful for tensor parallelism
    (sharding), weight fusion (e.g., fused QKV), key remapping from checkpoint to
    parameter name, and custom transformations to support arbitrary layout requirements.

    Args:
        transform: Function (slices, rank) -> tensor. Receives a list of PySafeSlice
            objects (one per checkpoint key from mappings) and the current rank.
            Returns the final tensor to load into the parameter.
            If None, loads the first slice as-is (identity transform).

    Example:
        Column-parallel sharding (shard along dim 0)::
            SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][rank * 64:(rank + 1) * 64, :])

        Row-parallel sharding (shard along dim 1)::
            SafetensorsWeightLoader(transform=lambda slices, rank: slices[0][:, rank * 64:(rank + 1) * 64])

        Fusing multiple checkpoint tensors::
            SafetensorsWeightLoader(transform=lambda slices, rank: torch.cat([s[:] for s in slices], dim=0))
    """

    transform: Callable[[list["PySafeSlice"], int], torch.Tensor] = None

    def load(self, slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        """Load and transform checkpoint slices into a tensor.

        Args:
            slices: List of PySafeSlice objects from the checkpoint.
            rank: Current tensor parallel rank.

        Returns:
            Transformed tensor ready to be loaded into the parameter.
        """
        if self.transform:
            result = self.transform(slices, rank)
        else:
            # By default, load first slice as-is
            assert len(slices) == 1, (
                f"SafetensorsWeightLoader without transform should only take in a single slice but got {len(slices)}"
            )
            result = slices[0][:]

        return result.contiguous()


def set_weight_loader(param: nn.Parameter, loader: SafetensorsWeightLoader) -> None:
    """Attach a SafetensorsWeightLoader to a parameter.

    Args:
        param: The parameter to attach the loader to.
        loader: The SafetensorsWeightLoader defining how to load this parameter.

    Example::
        param = torch.nn.Parameter(torch.empty(256, 512))
        set_weight_loader(param, SafetensorsWeightLoader(...))
    """
    setattr(param, _WEIGHT_LOADER_ATTR, loader)


def get_weight_loader(param: nn.Parameter) -> SafetensorsWeightLoader:
    """Retrieve the SafetensorsWeightLoader attached to a parameter.

    Args:
        param: The parameter to get the loader from.

    Returns:
        The attached SafetensorsWeightLoader, or a default SafetensorsWeightLoader (identity transform) if none.
    """
    return getattr(param, _WEIGHT_LOADER_ATTR, SafetensorsWeightLoader())


def with_rank_override(
    loader: SafetensorsWeightLoader, rank: int
) -> SafetensorsWeightLoader:
    """Wrap a loader to use a specific rank instead of the passed rank.

    This is useful when using multiple tensor parallel groups where the global
    TP rank differs from the rank within a specific group. For example, with
    attention data parallelism, attention weights are sharded across a subset
    of ranks (e.g., 8 ranks per attention TP group), but the weight loading
    system passes the global TP rank (e.g., 0-15). This wrapper lets you
    specify the local rank within the group.

    Args:
        loader: The base loader to wrap.
        rank: The rank to use for loading (typically the local rank within a group).

    Returns:
        A new SafetensorsWeightLoader that uses the specified rank.

    Example::
        base_loader = sharding_weight_loader(shard_dim=0, shard_size=64, num_shards=8)
        # Load shard for local attention TP rank, not global TP rank
        loader = with_rank_override(base_loader, rank=attn_tp_rank)
    """
    return SafetensorsWeightLoader(
        transform=lambda slices, _: loader.load(slices, rank)
    )


##############################
# Common transform factories #
##############################


def sharding_weight_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    pad_shard: bool = False,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader that shards a tensor along a dimension.

    Args:
        shard_dim: Dimension to shard along in the final parameter shape.
        shard_size: Size of each shard along shard_dim.
        num_shards: Total number of shards (typically TP world size).
        is_storage_transposed: If True, checkpoint stores weights transposed
            (swaps dim 0 and 1) relative to the parameter layout.
        pad_shard: If True, zero-pad the returned shard to exactly shard_size
            when the checkpoint slice is smaller (e.g. last rank when
            total_dim % num_shards != 0). Defaults to False.

    Note:
        If using is_storage_transposed, we require checkpoint weight to be 2-dimensional.
        Please write your own weight loader if you need more complex functionality.

    Returns:
        SafetensorsWeightLoader configured for sharding.

    Example:
        param = torch.nn.Parameter(...)

        Column-parallel linear (shard output dim):
            # Checkpoint has [hidden_size, input_size], we want [hidden_size // tp, input_size]
            set_weight_loader(param, sharding_weight_loader(
                shard_dim=0,
                shard_size=hidden_size // tp_size,
                num_shards=tp_size
            ))

        Linear that is a transpose of checkpoint (shard input dim):
            # Checkpoint has [output_size, hidden_size], we want [hidden_size // tp, output_size]
            set_weight_loader(param, sharding_weight_loader(
                shard_dim=0,
                shard_size=hidden_size // tp_size,
                num_shards=tp_size,
                is_storage_transposed=True,
            ))

        Last rank needs padding (caller does not pre-zero the buffer):
            set_weight_loader(param, sharding_weight_loader(
                shard_dim=0,
                shard_size=shard_size,
                num_shards=tp_size,
                pad_shard=True,
            ))
    """

    def transform(slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            "sharding_weight_loader() only supports a single tensor"
        )

        slice_obj = slices[0]

        if is_storage_transposed:
            assert len(slice_obj.get_shape()) <= 2, (
                "sharding_weight_loader() does not support transposed storage for tensors with more than 2 dims"
            )

        # Determine which dim to slice in checkpoint
        storage_shard_dim = shard_dim
        if is_storage_transposed and shard_dim in (0, 1):
            storage_shard_dim = 1 - shard_dim

        # Compute slice range
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        # Build slice tuple
        slices = [slice(None)] * len(slice_obj.get_shape())
        slices[storage_shard_dim] = slice(start_idx, end_idx)
        result = slice_obj[tuple(slices)]

        if is_storage_transposed:
            result = result.T

        # Optionally pad to shard_size along shard_dim if the checkpoint
        # dimension is shorter (e.g. last rank when total_dim % num_shards != 0).
        if pad_shard:
            target_shape = list(result.shape)
            target_shape[shard_dim] = shard_size
            result = pad_to_shape(result, tuple(target_shape))

        return result

    return SafetensorsWeightLoader(transform=transform)


def sharding_weight_loader_with_padding(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    pad_dim: int | None = None,
    padded_size: int | None = None,
    unpadded_size: int | None = None,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader that shards and optionally pads a tensor.

    This loader combines sharding (for tensor parallelism) with optional padding
    (for models that pad hidden dimensions for hardware alignment).

    Args:
        shard_dim: Dimension to shard along in the final parameter shape.
        shard_size: Size of each shard along shard_dim.
        num_shards: Total number of shards (typically TP world size).
        is_storage_transposed: If True, checkpoint stores weights transposed.
        pad_dim: Dimension to pad after sharding (0 or 1). None means no padding.
        padded_size: Target size after padding.
        unpadded_size: Original size before padding.

    Returns:
        SafetensorsWeightLoader configured for sharding with optional padding.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1, (
            "sharding_weight_loader_with_padding() only supports a single tensor"
        )

        slice_obj = slices[0]

        if is_storage_transposed:
            assert len(slice_obj.get_shape()) <= 2, (
                "sharding_weight_loader_with_padding() does not support transposed storage for tensors with more than 2 dims"
            )

        # Determine which dim to slice in checkpoint
        storage_shard_dim = shard_dim
        if is_storage_transposed and shard_dim in (0, 1):
            storage_shard_dim = 1 - shard_dim

        # Compute slice range
        start_idx = (rank % num_shards) * shard_size
        end_idx = start_idx + shard_size

        # Build slice tuple
        sl = [slice(None)] * len(slice_obj.get_shape())
        sl[storage_shard_dim] = slice(start_idx, end_idx)
        result = slice_obj[tuple(sl)]

        if is_storage_transposed:
            result = result.T

        # Apply padding if needed
        if (
            pad_dim is not None
            and padded_size
            and unpadded_size
            and padded_size != unpadded_size
        ):
            pad_amount = padded_size - unpadded_size
            if pad_dim == 0:
                result = torch.nn.functional.pad(result, (0, 0, 0, pad_amount))
            else:  # pad_dim == 1
                result = torch.nn.functional.pad(result, (0, pad_amount))

        return result

    return SafetensorsWeightLoader(transform=transform)


def last_dim_padding_weight_loader(padded_size: int) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader that pads a tensor's last dimension.

    This loader is useful for models that pad hidden dimensions for hardware alignment.
    It handles the case where no padding is needed (padded_size <= tensor size).

    Args:
        padded_size: Target size for the last dimension.

    Returns:
        SafetensorsWeightLoader that pads the last dimension.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        tensor = slices[0][:]
        pad_amount = padded_size - tensor.shape[-1]
        if pad_amount <= 0:
            return tensor
        return torch.nn.functional.pad(tensor, (0, pad_amount))

    return SafetensorsWeightLoader(transform=transform)


def fused_qkv_weight_loader(
    q_size: int,
    kv_size: int,
    shard_dim: int,
    num_shards: int,
    is_storage_transposed: bool = False,
    num_kv_replicas: int = 1,
    padded_hidden_size: int | None = None,
    unpadded_hidden_size: int | None = None,
    attention_dp_rank: int = 0,
    attention_dp_size: int = 1,
    kv_sharded_across_attention_dp: bool = False,
) -> SafetensorsWeightLoader:
    """Create a SafetensorsWeightLoader that fuses Q, K, V weights with per-tensor sharding.

    This loader expects 3 checkpoint tensors (Q, K, V) specified via mappings,
    shards each according to rank, and concatenates them into a single fused tensor.

    Args:
        q_size: Size of Q projection per shard.
        kv_size: Size of K (and V) projection per shard.
        shard_dim: Dimension to shard along in the final parameter shape.
        num_shards: Total number of shards (TP world size).
        is_storage_transposed: If True, checkpoint stores weights transposed.
        num_kv_replicas: Number of KV replicas. For example with 8 KV heads and TP 16,
            we typically give rank 0 and 1 the 1st KV head, and so on.
        padded_hidden_size: Target hidden size after padding (for models with padded dimensions).
        unpadded_hidden_size: Original hidden size before padding.
        attention_dp_rank: This rank's position within its attention DP column group.
            Default 0 (no attention DP).
        attention_dp_size: Dependent DP degree. Default 1 (disabled).
            When > 1, Q is sharded across TP * attention_dp using an effective rank.
        kv_sharded_across_attention_dp: Whether KV weights are also sharded across
            attention DP (True when num_kv_heads > TP and divisible by TP * attention_dp).

    Returns:
        SafetensorsWeightLoader configured for fused QKV loading.

    Example:
        Fused QKV for attention with GQA::
            set_weight_loader(param, fused_qkv_weight_loader(
                q_size=num_heads * head_dim // tp_size,
                kv_size=num_kv_heads * head_dim // tp_size,
                shard_dim=0,
                num_shards=tp_size,
            ))
    """
    assert shard_dim == 0 or shard_dim == 1, (
        f"Shard dim must be 0 or 1, got {shard_dim}"
    )

    if is_storage_transposed:
        storage_shard_dim = 1 - shard_dim
    else:
        storage_shard_dim = shard_dim

    def transform(slices: list["PySafeSlice"], rank: int) -> torch.Tensor:
        assert len(slices) == 3, (
            "fused_qkv_weight_loader expects [Q, K, V] slices in order"
        )
        assert len(slices[0].get_shape()) == 2, "Q slice must be 2 dimensional"
        assert len(slices[0].get_shape()) == 2, "K slice must be 2 dimensional"
        assert len(slices[0].get_shape()) == 2, "V slice must be 2 dimensional"

        # With attention DP, Q uses effective rank across TP * attention_dp
        local_rank = rank % num_shards
        if attention_dp_size > 1:
            q_rank = attention_dp_rank + local_rank * attention_dp_size
            if kv_sharded_across_attention_dp:
                kv_rank = q_rank
            else:
                kv_rank = local_rank // num_kv_replicas
        else:
            q_rank = local_rank
            kv_rank = local_rank // num_kv_replicas

        qkv_tensors = []
        for slice_obj, shard_size, shard_rank in zip(
            slices, [q_size, kv_size, kv_size], [q_rank, kv_rank, kv_rank]
        ):
            start_idx = shard_rank * shard_size
            end_idx = start_idx + shard_size

            sl = [slice(None)] * len(slice_obj.get_shape())
            sl[storage_shard_dim] = slice(start_idx, end_idx)
            tensor = slice_obj[tuple(sl)]
            qkv_tensors.append(tensor)

        # Transpose each tensor first if needed
        if is_storage_transposed:
            qkv_tensors = [t.T for t in qkv_tensors]

        # Fused QKV concatenates along shard dimension (e.g. we shard Q, K, and V weights)
        result = torch.cat(qkv_tensors, dim=shard_dim)

        # Apply padding if needed (pad the hidden dimension, which is dim 0 after transpose)
        if (
            padded_hidden_size
            and unpadded_hidden_size
            and padded_hidden_size != unpadded_hidden_size
        ):
            pad_amount = padded_hidden_size - unpadded_hidden_size
            # After transpose, result is [hidden, qkv_size], pad first dim
            result = torch.nn.functional.pad(result, (0, 0, 0, pad_amount))

        return result

    return SafetensorsWeightLoader(transform=transform)


def scaled_bias_loader(
    scale: float,
    padded_size: int | None = None,
) -> SafetensorsWeightLoader:
    """Create a weight loader that scales a bias tensor (for all-reduce averaging).

    This is used for row-parallel biases where each rank contributes a partial sum
    that gets all-reduced. The bias must be divided by the number of ranks to produce
    the correct result after summation.

    Args:
        scale: Factor to divide by (typically world_size for row-parallel biases).
        padded_size: If provided, pad the last dimension to this size.

    Returns:
        SafetensorsWeightLoader that divides the tensor by scale and optionally pads.
    """

    def transform(slices: list, rank: int) -> torch.Tensor:
        assert len(slices) == 1
        tensor = slices[0][:] / scale

        if padded_size is not None:
            pad_amount = padded_size - tensor.shape[-1]
            if pad_amount > 0:
                tensor = torch.nn.functional.pad(tensor, (0, pad_amount))

        return tensor

    return SafetensorsWeightLoader(transform=transform)


def expert_parallel_weight_loader(
    local_expert_indices: list,
    original_loader: SafetensorsWeightLoader,
    expert_dim: int = 0,
) -> SafetensorsWeightLoader:
    """Wrap existing expert weight loader to load only local experts.

    This loader is useful for expert parallelism (EP) where different ranks
    are responsible for different subsets of experts. It wraps an existing
    weight loader (e.g., for tensor parallelism) and filters the loaded
    weights to only include the experts assigned to the current rank.

    Args:
        local_expert_indices: List of expert indices assigned to this rank.
            For example, with 16 experts and EP degree 4, rank 0 might get [0,1,2,3].
        original_loader: The original weight loader to wrap. This could be an
            identity loader, sharding loader, or any other SafetensorsWeightLoader.
        expert_dim: The dimension along which experts are organized. Defaults to 0.

    Returns:
        SafetensorsWeightLoader that loads only the specified experts

    Example:
        Basic usage with identity loader::

            # 16 experts total, EP degree 4, this rank gets experts [4,5,6,7]
            local_expert_indices = [4, 5, 6, 7]
            identity_loader = SafetensorsWeightLoader()
            ep_loader = expert_parallel_weight_loader(local_expert_indices, identity_loader)
            set_weight_loader(param, ep_loader)

        Combined with tensor parallelism::

            # 16 experts, EP degree 4, TP degree 2
            # This rank: EP rank 1 (experts [4,5,6,7]), TP rank 0 (first shard)
            local_expert_indices = [4, 5, 6, 7]
            tp_loader = sharding_weight_loader(shard_dim=1, shard_size=512, num_shards=2)
            ep_tp_loader = expert_parallel_weight_loader(local_expert_indices, tp_loader)
            set_weight_loader(param, ep_tp_loader)
            # Result: loads experts [4,5,6,7] with features [0:512] for TP rank 0
    """
    if len(local_expert_indices) == 0:
        raise ValueError("local_expert_indices cannot be empty")

    def transform(slices: list, rank: int) -> torch.Tensor:
        # Get the full tensor from the original loader
        full_tensor = original_loader.transform(slices, rank)

        # Extract only the experts assigned to this rank
        # We use basic slicing to support uint16 tensors (MxFP4 weights)
        expert_tensors = []
        for expert_idx in local_expert_indices:
            local_expert_slices = [slice(None)] * full_tensor.ndim
            local_expert_slices[expert_dim] = expert_idx
            expert_tensors.append(full_tensor[tuple(local_expert_slices)])

        # Stack the expert tensors back together along the expert dimension
        return torch.stack(expert_tensors, dim=expert_dim)

    return SafetensorsWeightLoader(transform=transform)


######################################
# Common functions for composability #
######################################


def get_shard(
    slice_obj: "PySafeSlice", dim: int, shard_size: int, num_shards: int, rank: int
) -> torch.Tensor:
    """Extract a shard from a Safetensors PySafeSlice without loading the full tensor.

    Only reads the shard portion from disk, avoiding full tensor materialization.
    Handles unpadded checkpoints where dim size < shard_size * num_shards.

    Args:
        slice_obj: Safetensors PySafeSlice object
        dim: Dimension to shard along
        shard_size: Size of each shard
        num_shards: Total number of shards
        rank: Current rank (0-indexed)

    Returns:
        Tensor of shape [..., shard_size, ...] on dim, or smaller if checkpoint
        is unpadded. Caller should pad if result.shape[dim] < shard_size.

    Example:
        # slice_obj shape: [8, 119, 256]
        # dim = 1
        # shard_size = 40
        # num_shards =

        # Rank 0: gets [:, 0:40, :]   -> shape [8, 40, 256]
        # Rank 1: gets [:, 40:80, :]  -> shape [8, 40, 256]
        # Rank 2: gets [:, 80:119, :] -> shape [8, 39, 256], needs 1 pad

        tensor = get_shard(slice_obj, dim=1, shard_size=40, rank=2, num_shards=3)
        # tensor.shape[1] == 39, caller pads to 40
    """
    shape = slice_obj.get_shape()

    start = (rank % num_shards) * shard_size
    end = start + shard_size

    slices = [slice(None)] * len(shape)
    slices[dim] = slice(start, end)

    return slice_obj[tuple(slices)]


def get_shuffled_shard(
    slice_obj: "PySafeSlice",
    dim: int,
    shard_size: int,
    rank: int,
    padded_dim_size: int | None = None,
) -> torch.Tensor:
    """Extract a shard equivalent to pad-then-shuffle-then-shard, without materializing the full tensor.

    The shuffle transform: [H] -> [H//4, 4] -> transpose -> [4, H//4] -> [H]
    For H=8: [0,1,2,3,4,5,6,7] -> [0,4,1,5,2,6,3,7]

    Args:
        slice_obj: Safetensors PySafeSlice object
        dim: Dimension to shuffle and shard
        shard_size: Size of each shard
        rank: Current rank (0-indexed)
        padded_dim_size: Size of dimension after padding (if None, no padding)

    Returns:
        Tensor equivalent to padding, shuffling, then sharding
    """
    shape = slice_obj.get_shape()
    H_orig = shape[dim]
    H = padded_dim_size if padded_dim_size is not None else H_orig
    stride = H // 4

    # Compute original indices for this shard's shuffled positions
    start_pos = rank * shard_size
    positions = torch.arange(shard_size)
    indices = ((start_pos + positions) % stride) * 4 + (
        (start_pos + positions) // stride
    )

    # Separate valid indices (< H_orig) from padding indices (>= H_orig)
    valid_mask = indices < H_orig
    valid_indices = indices[valid_mask]

    # Build output shape
    out_shape = list(shape)
    out_shape[dim] = shard_size
    output = torch.zeros(out_shape, dtype=slice_obj[tuple([0] * len(shape))].dtype)

    if valid_indices.numel() > 0:
        # Fetch valid data
        sl = [slice(None)] * len(shape)
        sl[dim] = valid_indices.tolist()
        valid_data = slice_obj[tuple(sl)]

        # Scatter into output at valid positions
        out_sl = [slice(None)] * len(shape)
        out_sl[dim] = valid_mask.nonzero().squeeze(-1).tolist()
        output[tuple(out_sl)] = valid_data

    return output


def get_shard_deinterleaved(
    slice_obj: "PySafeSlice",
    dim: int,
    shard_size: int,
    num_shards: int,
    rank: int,
) -> torch.Tensor:
    """Extract a shard of interleaved gate/up data, returning [gate_shard, up_shard] concatenated.

    For checkpoint data stored as [g0, u0, g1, u1, g2, u2, g3, u3] along dim,
    each rank gets its portion of both gate and up, concatenated.

    Args:
        slice_obj: Safetensors PySafeSlice object
        dim: Dimension with interleaved data
        shard_size: Size of each shard (should be 2 * I_per_rank, i.e., gate + up per rank)
        num_shards: Total number of shards (tp_degree)
        rank: Current rank (0-indexed)

    Returns:
        Tensor with [gate_shard, up_shard] concatenated along dim

    Example:
        # slice_obj data along dim: [g0, u0, g1, u1, g2, u2, g3, u3] (interleaved)
        # shard_size=4, num_shards=2

        # Rank 0: gets [g0, g1, u0, u1] (first half of gate + first half of up)
        # Rank 1: gets [g2, g3, u2, u3] (second half of gate + second half of up)
    """
    shape = slice_obj.get_shape()
    ndim = len(shape)

    start = (rank % num_shards) * shard_size
    end = start + shard_size

    # Gate: even indices [start*2 : end*2 : 2]
    sl_gate = [slice(None)] * ndim
    sl_gate[dim] = slice(start, end, 2)
    gate_shard = slice_obj[tuple(sl_gate)]

    # Up: odd indices [start*2 + 1 : end*2 + 1 : 2]
    sl_up = [slice(None)] * ndim
    sl_up[dim] = slice(start + 1, end + 1, 2)
    up_shard = slice_obj[tuple(sl_up)]

    return gate_shard, up_shard


def pad_to_shape(tensor: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Pad tensor to target shape. Pads at the end of each dimension.

    Args:
        tensor: Input tensor (possibly smaller than target on some dims)
        target_shape: Expected shape after padding

    Returns:
        Tensor padded to target_shape, or original if no padding needed.

    Example:
        # After sharding, assume rank X has shape (8, 39) but needs (8, 40)
        tensor = pad_to_shape(tensor, target_shape=(8, 40))
    """
    # Skip padding if shapes already match
    if tensor.shape == target_shape:
        return tensor

    pad = []
    for actual, target in zip(reversed(tensor.shape), reversed(target_shape)):
        pad.extend([0, target - actual])

    return torch.nn.functional.pad(tensor, pad)
