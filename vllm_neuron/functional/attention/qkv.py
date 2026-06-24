# SPDX-License-Identifier: Apache-2.0
from typing import Optional
from torch import Tensor
import torch

import nki

from nkilib.core.qkv.qkv import qkv as nki_qkv_kernel
from nkilib.core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from nkilib.core.utils.allocator import SbufManager

from vllm_neuron.nki.nki_hop import can_run_kernel, wrap_nki


def _maybe_build_sbm(
    sbm_lower_bound: Optional[int],
    sbm_upper_bound: Optional[int],
    sbm_use_auto_alloc: bool,
    sbm_default_stack_alloc: bool,
) -> Optional[SbufManager]:
    """Reconstruct SbufManager only when bounds are explicitly provided."""

    if sbm_lower_bound is None and sbm_upper_bound is None:
        return None

    return SbufManager(
        sb_lower_bound=sbm_lower_bound,
        sb_upper_bound=sbm_upper_bound,
        use_auto_alloc=sbm_use_auto_alloc,
        default_stack_alloc=sbm_default_stack_alloc,
    )


@nki.jit
def _torch_compatible_qkv_kernel(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    use_dma_transpose: bool = True,
    store_output_in_sbuf: bool = False,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[Tensor] = None,
    qkv_in_scale: Optional[Tensor] = None,
    fused_residual_add: Optional[bool] = False,
    fused_rope: Optional[bool] = False,
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = False,
    sbm_default_stack_alloc: bool = True,
    use_auto_allocation: bool = False,
):
    sbm = _maybe_build_sbm(
        sbm_lower_bound=sbm_lower_bound,
        sbm_upper_bound=sbm_upper_bound,
        sbm_use_auto_alloc=sbm_use_auto_alloc,
        sbm_default_stack_alloc=sbm_default_stack_alloc,
    )

    return nki_qkv_kernel(
        # Core inputs
        input=hidden,
        fused_qkv_weights=qkv_weights,
        output_layout=output_layout,
        # Bias
        bias=bias,
        # Quantization
        quantization_type=quantization_type,
        qkv_w_scale=qkv_w_scale,
        qkv_in_scale=qkv_in_scale,
        # Fused Residual Add
        fused_residual_add=fused_residual_add,
        mlp_prev=mlp_prev,
        attention_prev=attn_prev,
        # Fused Norm
        fused_norm_type=norm_type,
        gamma_norm_weights=norm_weights,
        layer_norm_bias=norm_b,
        norm_eps=eps,
        hidden_actual=hidden_actual,
        # Fused RoPE
        fused_rope=fused_rope,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        # Output storage
        store_output_in_sbuf=store_output_in_sbuf,
        # SBUF management
        sbm=sbm,
        use_auto_allocation=use_auto_allocation,
        # DMA
        load_input_with_DMA_transpose=use_dma_transpose,
    )


def _torch_qkv_impl(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
) -> Tensor:
    """
    PyTorch implementation of QKV projection (unchanged from original).
    """
    B, S, H = hidden.shape
    _, fused_qkv_dim = qkv_weights.shape

    # Step 1: Fused add
    if mlp_prev is not None and attn_prev is not None:
        hidden = hidden + mlp_prev + attn_prev

    # Step 2: Apply normalization
    if hidden_actual is None:
        hidden_actual = H

    if norm_type == NormType.RMS_NORM:
        variance = hidden.pow(2).sum(-1, keepdim=True) / hidden_actual
        hidden = hidden * torch.rsqrt(variance + eps)
        if norm_weights is not None:
            hidden = hidden * norm_weights

    elif norm_type == NormType.LAYER_NORM:
        mean = hidden.mean(-1, keepdim=True)
        variance = hidden.var(-1, keepdim=True, unbiased=False)
        hidden = (hidden - mean) * torch.rsqrt(variance + eps)
        if norm_weights is not None:
            hidden = hidden * norm_weights
        if norm_b is not None:
            hidden = hidden + norm_b

    elif norm_type == NormType.RMS_NORM_SKIP_GAMMA:
        variance = hidden.pow(2).sum(-1, keepdim=True) / hidden_actual
        hidden = hidden * torch.rsqrt(variance + eps)

    # Step 3: Matmul
    output = hidden @ qkv_weights

    # Step 4: Add bias
    if bias is not None:
        output = output + bias

    # Step 5: Apply RoPE if cos/sin caches provided
    fused_rope = (
        cos_cache is not None
        and sin_cache is not None
        and num_q_heads is not None
        and num_kv_heads is not None
    )

    if fused_rope:
        rope_d_head = (
            d_head
            if d_head is not None
            else fused_qkv_dim // (num_q_heads + 2 * num_kv_heads)
        )
        num_rope_heads = num_q_heads + num_kv_heads

        for head_idx in range(num_rope_heads):
            start_idx = head_idx * rope_d_head
            end_idx = start_idx + rope_d_head
            half_d = rope_d_head // 2

            head_output = output[:, :, start_idx:end_idx]
            x1 = head_output[:, :, :half_d]
            x2 = head_output[:, :, half_d:]

            cos = cos_cache
            sin = sin_cache[:, :, :half_d]

            rotated_x1 = x1 * cos[:, :, :half_d] - x2 * sin
            rotated_x2 = x2 * cos[:, :, half_d:] + x1 * sin

            output[:, :, start_idx : start_idx + half_d] = rotated_x1
            output[:, :, start_idx + half_d : end_idx] = rotated_x2

    # Step 6: Handle output layout
    if output_layout == QKVOutputLayout.BSD:
        return output

    elif output_layout == QKVOutputLayout.NBSd:
        if d_head is None:
            raise ValueError("d_head must be specified for NBSd output layout")

        num_heads = fused_qkv_dim // d_head
        output = output.reshape(B, S, num_heads, d_head)
        output = output.permute(2, 0, 1, 3)
        return output

    elif output_layout == QKVOutputLayout.NBdS:
        if d_head is None:
            raise ValueError("d_head must be specified for NBdS output layout")

        num_heads = fused_qkv_dim // d_head
        output = output.reshape(B, S, num_heads, d_head)
        output = output.permute(2, 0, 3, 1)
        return output
    else:
        raise ValueError(f"Unsupported output layout: {output_layout}")


def _can_use_qkv_kernel(
    hidden: Tensor,
    qkv_weights: Tensor,
    mlp_prev: Optional[Tensor],
    attn_prev: Optional[Tensor],
    d_head: Optional[int],
    output_layout: QKVOutputLayout,
    norm_type: NormType,
    norm_b: Optional[Tensor],
) -> bool:
    """
    Check if the NKI QKV kernel can be used based on constraints.

    Returns True if all constraints are satisfied, False otherwise.
    """
    if not can_run_kernel(hidden):
        return False

    _, _, H = hidden.shape
    _, fused_qkv_dim = qkv_weights.shape

    if H > 24576:
        return False

    if H % 128 != 0:
        return False

    if fused_qkv_dim > 4096:
        return False

    # The kernel produces incorrect results when fused_qkv_dim is large
    # relative to H (e.g., vision TP=1: H=1280, fused_qkv_dim=3*H=3840).
    # Validated configurations have fused_qkv_dim <= H. Fall back to PyTorch
    # when this is exceeded until the kernel is validated for larger ratios.
    if fused_qkv_dim > H:
        return False

    # Residual add: both must be provided or neither
    if (mlp_prev is None) != (attn_prev is None):
        return False

    # norm_b only supported for LAYER_NORM
    if norm_b is not None and norm_type != NormType.LAYER_NORM:
        return False

    # NBdS layout is not supported in the kernel
    if output_layout == QKVOutputLayout.NBdS:
        return False

    # For NBSd layout, d_head must be 128
    if output_layout == QKVOutputLayout.NBSd:
        if d_head is None or d_head != 128:
            return False

    return True


def qkv_proj(
    hidden: Tensor,
    qkv_weights: Tensor,
    norm_weights: Optional[Tensor] = None,
    mlp_prev: Optional[Tensor] = None,
    attn_prev: Optional[Tensor] = None,
    d_head: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.NO_NORM,
    use_dma_transpose: bool = True,
    store_output_in_sbuf: bool = False,
    bias: Optional[Tensor] = None,
    norm_b: Optional[Tensor] = None,
    hidden_actual: Optional[int] = None,
    cos_cache: Optional[Tensor] = None,
    sin_cache: Optional[Tensor] = None,
    num_q_heads: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    qkv_w_scale: Optional[Tensor] = None,
    qkv_in_scale: Optional[Tensor] = None,
    sbm_lower_bound: int = None,
    sbm_upper_bound: int = None,
    sbm_use_auto_alloc: bool = False,
    sbm_default_stack_alloc: bool = True,
    use_auto_allocation: bool = False,
) -> Tensor:
    """
    QKV Projection API that automatically selects between NKI kernel and PyTorch fallback.

    This function checks kernel constraints and dispatches to:
    - nki_qkv_kernel: When all constraints are satisfied and running on Neuron
    - PyTorch implementation: When constraints are violated or not on Neuron

    This kernel implements QKV projection with optional normalization fusion used in
    transformer models: output = norm(hidden + mlp_prev + attn_prev) @ qkv_weights + bias

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension
        fused_qkv_dim: (num_q_heads + 2 * num_kv_heads) * d_head

    Args:
        hidden: Input tensor with shape [B, S, H]
        qkv_weights: Weight tensor with shape [H, fused_qkv_dim]
        norm_weights: Optional normalization weights (gamma) with shape [1, H]
        mlp_prev: Optional previous MLP output for residual with shape [B, S, H]
        attn_prev: Optional previous attention output for residual with shape [B, S, H]
        d_head: Head dimension. Required for NBSd output layout and RoPE
        output_layout: Output tensor layout:
            - BSD: [B, S, fused_qkv_dim]
            - NBSd: [num_heads, B, S, d_head]
        eps: Epsilon for numerical stability in normalization. Default: 1e-6
        norm_type: Normalization type:
            - NO_NORM: No normalization
            - RMS_NORM: Root Mean Square normalization
            - LAYER_NORM: Layer normalization
            - RMS_NORM_SKIP_GAMMA: RMSNorm with gamma pre-multiplied in weights
        use_dma_transpose: Use DMA transpose for weight loading. Default: True
        store_output_in_sbuf: Store output in SBUF instead of HBM. Default: False
        bias: Optional bias tensor with shape [1, fused_qkv_dim]
        norm_b: Optional LayerNorm bias with shape [1, H]
        hidden_actual: Actual hidden dimension for padded tensors
        cos_cache: Cosine cache for RoPE with shape [B, S, d_head]
        sin_cache: Sine cache for RoPE with shape [B, S, d_head]
        num_q_heads: Number of query heads for RoPE
        num_kv_heads: Number of key/value heads for RoPE
        quantization_type: Type of quantization (NONE, ROW, STATIC). Default: NONE
        qkv_w_scale: QKV weight scale tensor for quantization
        qkv_in_scale: QKV input scale tensor for static quantization
        sbm: Optional SBUF manager for memory allocation control
        use_auto_allocation: Whether to use automatic SBUF allocation. Default: False

    Returns:
        Output tensor with shape determined by output_layout

    Kernel Constraints (falls back to PyTorch if violated):
        - hidden must be 3D tensor with shape [B, S, H]
        - qkv_weights must be 2D tensor with shape [H, fused_qkv_dim]
        - Hidden dimension H must match weight's first dimension
        - Hidden dimension H must be divisible by 128
        - Hidden dimension H must be <= 24576
        - fused_qkv_dim must be <= 4096
        - mlp_prev and attn_prev must both be provided or neither
        - norm_b is only supported for LAYER_NORM
        - For NBSd layout, d_head must be 128
        - NBdS layout is not supported

    Notes:
        - The kernel automatically selects between decode (TKG) and prefill (CTE)
          implementations based on sequence length threshold
        - RoPE fusion requires cos_cache, sin_cache, num_q_heads, and num_kv_heads
    """

    # Check if kernel can be used
    can_use_kernel = _can_use_qkv_kernel(
        hidden=hidden,
        qkv_weights=qkv_weights,
        mlp_prev=mlp_prev,
        attn_prev=attn_prev,
        d_head=d_head,
        output_layout=output_layout,
        norm_type=norm_type,
        norm_b=norm_b,
    )

    if can_use_kernel:
        # Compute derived boolean flags
        fused_residual_add = mlp_prev is not None and attn_prev is not None
        fused_rope = (
            cos_cache is not None
            and sin_cache is not None
            and num_q_heads is not None
            and num_kv_heads is not None
        )

        wrapped_kernel = wrap_nki(_torch_compatible_qkv_kernel)

        return wrapped_kernel[2](
            hidden=hidden,
            qkv_weights=qkv_weights,
            norm_weights=norm_weights,
            mlp_prev=mlp_prev,
            attn_prev=attn_prev,
            d_head=d_head,
            output_layout=output_layout,
            eps=eps,
            norm_type=norm_type,
            use_dma_transpose=use_dma_transpose,
            store_output_in_sbuf=store_output_in_sbuf,
            bias=bias,
            norm_b=norm_b,
            hidden_actual=hidden_actual,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            quantization_type=quantization_type,
            qkv_w_scale=qkv_w_scale,
            qkv_in_scale=qkv_in_scale,
            fused_residual_add=fused_residual_add,
            fused_rope=fused_rope,
            sbm_lower_bound=sbm_lower_bound,
            sbm_upper_bound=sbm_upper_bound,
            sbm_use_auto_alloc=sbm_use_auto_alloc,
            sbm_default_stack_alloc=sbm_default_stack_alloc,
            use_auto_allocation=use_auto_allocation,
        )
    else:
        if quantization_type != QuantizationType.NONE:
            raise NotImplementedError(
                "QKV Projection torch fallback currently does not support quantization"
            )

        return _torch_qkv_impl(
            hidden=hidden,
            qkv_weights=qkv_weights,
            norm_weights=norm_weights,
            mlp_prev=mlp_prev,
            attn_prev=attn_prev,
            d_head=d_head,
            output_layout=output_layout,
            eps=eps,
            norm_type=norm_type,
            bias=bias,
            norm_b=norm_b,
            hidden_actual=hidden_actual,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
        )
