# SPDX-License-Identifier: Apache-2.0
"""argsort_unstable functional API."""

import torch

from nkilib.experimental.subkernels.argsort_unstable import (
    argsort_unstable as _argsort_unstable_nki,
)

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki


def argsort_unstable(
    data: torch.Tensor,
    descending: bool = False,
) -> torch.Tensor:
    """Perform unstable argsort on a 1D or [1, N] input tensor.

    Elements with equal values may appear in any order relative to their
    original positions. Vectorization along a batch dimension is not supported.

    Args:
        data (torch.Tensor): 1D [N] or 2D [1, N] int32/float32 tensor to sort.
        descending (bool): When True, return indices for descending order.
            Defaults to False (ascending).

    Returns:
        torch.Tensor: Same shape as input, int32 tensor of argsort indices.

    Example:
        >>> data = torch.tensor([[5, 2, 5, 3]], dtype=torch.int32)
        >>> indices = argsort_unstable(data)
        >>> # indices = [[1, 3, 0, 2]] or [[1, 3, 2, 0]] (unstable for ties)
    """
    _validate_inputs(data)

    if _can_use_kernel(data):
        wrapped = wrap_nki(_argsort_unstable_nki)
        return wrapped[2](data=data, descending=descending, output_in_sbuf=False)
    else:
        return _torch_argsort_unstable(data, descending)


def _validate_inputs(data: torch.Tensor) -> None:
    """Validate inputs for argsort_unstable."""
    assert data.ndim in (1, 2), (
        f"argsort_unstable only supports 1D [N] or 2D [1, N] input, got shape {data.shape}"
    )
    N = data.shape[-1]
    assert N <= 2**31 - 1, (
        f"argsort_unstable requires N <= int32 max ({2**31 - 1}), got {N=}"
    )


def _can_use_kernel(data: torch.Tensor) -> bool:
    """Check if the NKI kernel can be used.

    Constraints:
    - Data must be fp32 or int32
    """
    if not can_run_kernel(data):
        return False

    if data.dtype not in (torch.float32, torch.int32):
        return False

    return True


def _torch_argsort_unstable(
    data: torch.Tensor, descending: bool = False
) -> torch.Tensor:
    """Neuron-compilable torch argsort matching the NKI kernel behavior.

    Uses argmax on masked comparisons, with N/8 passes.

    Args:
        data: [N] or [1, N] tensor.
        descending: sort direction.

    Returns:
        Same shape as input, int32 tensor of argsort indices.
    """
    _ELEMS_PER_PASS = 8
    NEG_INF = float("-inf")

    is_1d = data.ndim == 1
    x = data.reshape(-1).to(torch.float32)
    N = x.shape[0]
    device = x.device

    # Pad to multiple of 8
    N_padded = ((N + _ELEMS_PER_PASS - 1) // _ELEMS_PER_PASS) * _ELEMS_PER_PASS
    if N_padded > N:
        sentinel = float("inf") if descending else NEG_INF
        x = torch.cat([x, torch.full((N_padded - N,), sentinel, device=device)])

    num_passes = N_padded // _ELEMS_PER_PASS
    indices = torch.zeros(N_padded, dtype=torch.int32, device=device)

    # In each pass, find the 8 largest values and then the top 8's indices
    for pass_idx in range(num_passes):
        top_vals, _ = torch.topk(x, _ELEMS_PER_PASS, largest=True, sorted=True)

        for val_idx in range(_ELEMS_PER_PASS - 1, -1, -1):
            val = top_vals[val_idx]
            # Find first position matching val (argmax on equality mask)
            match_mask = (x == val).to(torch.int32)
            pos = torch.argmax(match_mask).to(torch.int32)

            # Store index
            if descending:
                out_pos = _ELEMS_PER_PASS * pass_idx + val_idx
            else:
                out_pos = _ELEMS_PER_PASS * (num_passes - pass_idx) - 1 - val_idx
            indices[out_pos] = pos

            # Replace matched position with -inf
            x = x.scatter(
                0,
                pos.unsqueeze(0).to(torch.int32),
                torch.tensor([NEG_INF], device=device),
            )

    # Extract valid indices
    if descending:
        result = indices[:N]
    else:
        result = indices[N_padded - N :]

    return result if is_1d else result.unsqueeze(0)
