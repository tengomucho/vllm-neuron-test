# SPDX-License-Identifier: Apache-2.0
"""topk_reduce functional API."""

import torch
import torch.distributed as dist

from nkilib.experimental.subkernels.topk_reduce import topk_reduce as _topk_reduce_nki

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

_P_MAX = 128
_K_MAX = 8


def topk_reduce(
    input: torch.Tensor, T: int, K: int, is_sequence_parallel: bool = False
) -> torch.Tensor:
    """Compute sparse MoE Top-K reduction across all2all collective output buffer.

    Gathers scattered rows by packed global token index and reduces along the K dimension.
    Each token has between 1 and K rows scattered at arbitrary positions in the input, with remaining
    rows being padding.

    Token indices are 1-indexed. When is_sequence_parallel=False, indices are 1..T.
    When is_sequence_parallel=True, indices are global: rank r owns (r*T+1)..(r*T+T),
    and the offset rank_id*T is subtracted before indexing into the output.

    Padded rows must have index -1.

    Shapes: input [TK_padded, H+2] → output [T, H].
        TK_padded >= 8, H divisible by 2, T <= 128, K <= 8.

    Args:
        input (torch.Tensor): [TK_padded, H + 2] bf16/fp16. Sparse buffer with
            up to T*K routed rows. Final 2 bf16 columns encode a packed int32
            global token index (1-indexed, -1 for padding).
        T (int): Number of output tokens per rank (1 to 128). When
            is_sequence_parallel=True, T represents the global number of tokens / world_size.
        K (int): Max routed experts per token (1 to 8).
        is_sequence_parallel (bool): If True, token indices are expected to represent
            global token indices. Using local token indices with is_sequence_parallel=True,
            or using global token indices with is_sequence_parallel=False, will result in
            out of bounds at runtime. Defaults to False.

    Returns:
        torch.Tensor: [T, H] bf16/fp16. out[t] = sum of all rows with index
            (rank_id*T + t + 1) in SP mode, or (t + 1) in non-SP mode.

    Example:
        >>> # Non-SP, dense: T=2, K=2, H=4, input shape [4, 6]
        >>> # indices [1, 1, 2, 2] → out[0] = row0 + row1, out[1] = row2 + row3
        >>> out = topk_reduce(input, T=2, K=2)  # shape [2, 4]

        >>> # Non-SP, sparse: T=2, K=2, H=4, input shape [8, 6]
        >>> # indices [1, -1, 1, 2, -1, -1, 2, -1] → out[0] = row0 + row2, out[1] = row3 + row6
        >>> out = topk_reduce(input, T=2, K=2)  # shape [2, 4]

        >>> # SP: rank 1, T=8, K=2, H=4, input shape [16, 6], world_size=4
        >>> # global indices [9, 9, 10, 10, ..., 16, 16] (rank 1 owns 9..16)
        >>> # offset = rank * T = 1 * 8 = 8, local indices after subtract: [1, 1, 2, 2, ..., 8, 8]
        >>> out = topk_reduce(input, T=8, K=2, is_sequence_parallel=True)  # shape [8, 4]
    """
    _validate_inputs(T, K)

    sequence_parallel_rank_id = (
        torch.tensor([dist.get_rank()], dtype=torch.int32, device=input.device)
        if is_sequence_parallel
        else None
    )

    if _can_use_kernel(input, T, K):
        wrapped = wrap_nki(_topk_reduce_nki)
        return wrapped[2](input, T, K, sequence_parallel_rank_id)
    elif str(input.device) == "cpu":
        return _cpu_topk_reduce(input, T, K, sequence_parallel_rank_id)
    else:
        raise NotImplementedError(
            "topk_reduce only supports execution on Neuron hardware with NKI kernel."
        )


def _validate_inputs(T: int, K: int) -> None:
    """Validate topk_reduce arguments."""
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T=}")
    if K <= 0:
        raise ValueError(f"K must be > 0, got {K=}")


def _can_use_kernel(input: torch.Tensor, T: int, K: int) -> bool:
    """Check if the NKI topk_reduce kernel can be used.

    Kernel constraints:
        - Device must support NKI kernels (can_run_kernel)
        - 1 < T <= 128 (pmax)
        - K <= 8
        - input.shape[0] (TK_padded) >= 8
        - H (input.shape[1] - 2) must be even (divisible by LNC=2)
    """
    if not can_run_kernel(input):
        return False

    TK_padded = input.shape[0]
    H = input.shape[1] - 2

    if T > _P_MAX or T <= 1:
        return False
    if K > _K_MAX:
        return False
    if H % 2 != 0:
        return False

    return True


def _cpu_topk_reduce(
    input: torch.Tensor, T: int, K: int, sequence_parallel_rank_id: int = None
) -> torch.Tensor:
    """CPU-only reference implementation using scatter_add."""

    H = input.shape[1] - 2
    indices = input.view(torch.int32)[:, -1]  # (N,) packed token indices

    # In SP mode, subtract rank offset to get local indices
    if sequence_parallel_rank_id is not None:
        indices = indices - sequence_parallel_rank_id * T

    out = torch.zeros(T + 1, H, dtype=input.dtype)
    # Shift indices: valid tokens are 1..T → map to rows 1..T; <=0 → maps to 0
    # We accumulate into row 0 as a garbage bin, then discard it.
    bucket = indices.clamp(min=0).long()
    out.scatter_add_(0, bucket.unsqueeze(1).expand(-1, H), input[:, :H])
    return out[1:]  # discard row 0 (garbage from -1 and 0 indices)
