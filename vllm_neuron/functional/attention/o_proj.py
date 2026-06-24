# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from torch import Tensor

from nkilib.core.output_projection.output_projection_cte import (
    output_projection_cte as output_projection,
)
from nkilib.core.utils.common_types import QuantizationType

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki

P_MAX = 128
MAX_VALIDATED_H_SIZE = 16384 + 4321
MAX_VALIDATED_B_TIMES_S_SIZE = 128 * 1024
MAX_VALIDATED_N_SIZE = 17


def _can_use_kernel(
    active: Tensor,
    weight: Tensor,
) -> bool:
    """
    Check if the kernel can be used.

    Returns False when any NKI kernel constraint is violated or the tensors
    live on the CPU.
    """
    lnc_size = 2

    if not can_run_kernel(active):
        return False

    B, N, D, S = active.shape
    ND, H = weight.shape

    if N * D != ND:
        return False

    if H > MAX_VALIDATED_H_SIZE:
        return False

    if B * S > MAX_VALIDATED_B_TIMES_S_SIZE:
        return False

    if N > MAX_VALIDATED_N_SIZE:
        return False

    if D > P_MAX:
        return False

    if H % lnc_size != 0:
        return False

    return True


def _torch_o_proj_impl(
    active: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[Tensor] = None,
    weight_scales: Optional[Tensor] = None,
) -> Tensor:
    """
    PyTorch implementation of output projection.

    Args:
        active: Input tensor with shape [B, N, D, S]
        weight: Weight tensor with shape [N*D, H]
        bias: Optional bias tensor with shape [1, H]
        quantization_type: Must be QuantizationType.NONE — quantization is
            not supported in the PyTorch fallback path.
        input_scales: Must be None — not supported in the PyTorch fallback.
        weight_scales: Must be None — not supported in the PyTorch fallback.

    Returns:
        Output tensor with shape [B, S, H]

    Raises:
        AssertionError: If quantization parameters are provided.
    """
    assert quantization_type == QuantizationType.NONE, (
        f"Quantization is not supported in the PyTorch fallback path. "
        f"Got quantization_type={quantization_type}"
    )
    assert input_scales is None, (
        "input_scales must be None for the PyTorch fallback path — "
        "quantization is not supported."
    )
    assert weight_scales is None, (
        "weight_scales must be None for the PyTorch fallback path — "
        "quantization is not supported."
    )

    B, N, D, S = active.shape

    # Reshape: [B, N, D, S] -> [B, N*D, S] -> [B, S, N*D]
    x = active.reshape(B, N * D, S).transpose(1, 2)

    # Matmul: [B, S, N*D] @ [N*D, H] -> [B, S, H]
    output = x @ weight

    if bias is not None:
        output = output + bias

    return output


def o_proj(
    active: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    input_scales: Optional[Tensor] = None,
    weight_scales: Optional[Tensor] = None,
) -> Tensor:
    """
    Output Projection API that automatically selects between NKI kernels and PyTorch fallback.

    This function checks kernel constraints and dispatches to:
    - output_projection: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When constraints are violated or not on Neuron

    Computes: output = active @ weight + bias

    Dimensions:
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension size
        D: Head dimension size

    Args:
        active: Input tensor with shape [B, N, D, S]
        weight: Weight tensor with shape [N*D, H]
        bias: Optional bias tensor with shape [1, H]
        quantization_type: Type of quantization (NONE or STATIC for FP8).
            Quantization is only supported on the NKI kernel path; the PyTorch
            fallback will raise an AssertionError if a value other than NONE is
            supplied.
        input_scales: Optional input scale tensor with shape [128, 1] for FP8
            quantization.  Must be None when quantization_type is NONE.
        weight_scales: Optional weight scale tensor with shape [128, 1] for FP8
            quantization.  Must be None when quantization_type is NONE.

    Returns:
        Output tensor with shape [B, S, H]

    Constraints (falls back to PyTorch if violated):
        - active must be 4D tensor with shape [B, N, D, S]
        - weight must be 2D tensor with shape [N*D, H]
        - N * D must equal weight's first dimension
        - Head dimension D must be <= 128
        - H must be divisible by LNC size
        - bias (if provided) must have shape [1, H]
        - FP8 static quantization requires N or D to be even

    Notes:
        LNC sharding is performed on the weight tensor's hidden dimension.
    """

    if _can_use_kernel(active, weight):
        wrapped_output_projection = wrap_nki(output_projection)

        return wrapped_output_projection[2](
            active,
            weight,
            bias,
            quantization_type,
            input_scales,
            weight_scales,
        )

    return _torch_o_proj_impl(
        active,
        weight,
        bias=bias,
        quantization_type=quantization_type,
        input_scales=input_scales,
        weight_scales=weight_scales,
    )
