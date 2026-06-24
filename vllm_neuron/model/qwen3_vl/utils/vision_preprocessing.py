# SPDX-License-Identifier: Apache-2.0
"""CPU preprocessing functions for the Qwen3-VL vision encoder.

These functions run on CPU before the compiled vision model. They compute:
- 2D rotary position embeddings (RoPE) from grid_thw
- Bilinear-interpolated position embeddings from learned embedding table
- Attention bounds (bound_min, bound_max) from grid_thw

All functions are pure tensor operations with no learnable parameters
(position embedding lookup uses pre-loaded weights passed as arguments).

Reference: HF ``Qwen3VLVisionModel.rot_pos_emb``,
``fast_pos_embed_interpolate``, and ``cu_seqlens`` computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_rotary_pos_emb(
    grid_thw: torch.Tensor,
    head_dim: int,
    spatial_merge_size: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute 2D rotary position embeddings for vision attention.

    Each token gets a 2D position (row, col) within its image. The RoPE
    frequencies are looked up from a precomputed table and concatenated
    for the two spatial dimensions.

    Matches HF ``Qwen3VLVisionModel.rot_pos_emb`` + the ``emb = cat(...)``
    and ``(emb.cos(), emb.sin())`` in the forward method.

    Args:
        grid_thw: ``[num_images, 3]`` — (temporal, height, width) per image.
        head_dim: Attention head dimension.
        spatial_merge_size: Spatial merge factor (patches are grouped in
            ``merge_size × merge_size`` blocks before the merger).
        theta: RoPE base frequency. Default: 10000.0.

    Returns:
        ``(cos, sin)`` each ``[total_tokens, head_dim]``.
    """
    half_dim = head_dim // 2

    # Frequency table: [max_hw, half_dim]
    max_hw = int(grid_thw[:, 1:].max().item())
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, half_dim, 2, dtype=torch.float32) / half_dim)
    )
    seq = torch.arange(max_hw, dtype=torch.float32)
    freq_table = torch.outer(seq, inv_freq)  # [max_hw, half_dim // 2]

    total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
    pos_ids = torch.empty((total_tokens, 2), dtype=torch.long)

    offset = 0
    for num_frames, height, width in grid_thw:
        num_frames = int(num_frames.item())
        height = int(height.item())
        width = int(width.item())
        merged_h = height // spatial_merge_size
        merged_w = width // spatial_merge_size

        block_rows = torch.arange(merged_h)
        block_cols = torch.arange(merged_w)
        intra_row = torch.arange(spatial_merge_size)
        intra_col = torch.arange(spatial_merge_size)

        # Full-resolution positions within the merge grid
        row_idx = (
            block_rows[:, None, None, None] * spatial_merge_size
            + intra_row[None, None, :, None]
        )
        col_idx = (
            block_cols[None, :, None, None] * spatial_merge_size
            + intra_col[None, None, None, :]
        )

        row_idx = row_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        col_idx = col_idx.expand(
            merged_h, merged_w, spatial_merge_size, spatial_merge_size
        ).reshape(-1)

        coords = torch.stack((row_idx, col_idx), dim=-1)  # [h*w, 2]

        if num_frames > 1:
            coords = coords.repeat(num_frames, 1)

        num_tokens = coords.shape[0]
        pos_ids[offset : offset + num_tokens] = coords
        offset += num_tokens

    # Lookup and flatten: [total_tokens, 2, half_dim//2] → [total_tokens, half_dim]
    embeddings = freq_table[pos_ids]  # [total_tokens, 2, half_dim//2]
    rotary_pos_emb = embeddings.flatten(1)  # [total_tokens, half_dim]

    # Double and compute cos/sin (matches HF: emb = cat(rotary, rotary))
    emb = torch.cat(
        (rotary_pos_emb, rotary_pos_emb), dim=-1
    )  # [total_tokens, head_dim]
    cos = emb.cos()
    sin = emb.sin()

    return cos, sin


def compute_position_embeddings(
    grid_thw: torch.Tensor,
    pos_embed_weight: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """Compute bilinear-interpolated position embeddings.

    Interpolates from a learned ``[num_position_embeddings, hidden_size]``
    embedding table to the actual image grid size using bilinear weights.
    Then permutes tokens into the spatial merge order.

    Matches HF ``Qwen3VLVisionModel.fast_pos_embed_interpolate``.

    Args:
        grid_thw: ``[num_images, 3]`` — (temporal, height, width) per image.
        pos_embed_weight: ``[num_position_embeddings, hidden_size]`` — learned
            position embedding table (from checkpoint ``model.visual.pos_embed.weight``).
        num_grid_per_side: ``int(sqrt(num_position_embeddings))`` — grid side length.
        spatial_merge_size: Spatial merge factor.

    Returns:
        ``[total_tokens, hidden_size]`` — position embeddings for all tokens.
    """
    grid_ts = grid_thw[:, 0]
    grid_hs = grid_thw[:, 1]
    grid_ws = grid_thw[:, 2]

    idx_list: list[list[int]] = [[] for _ in range(4)]
    weight_list: list[list[float]] = [[] for _ in range(4)]

    for t, h, w in zip(grid_ts, grid_hs, grid_ws):
        h, w = int(h.item()), int(w.item())
        h_idxs = torch.linspace(0, num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * num_grid_per_side
        base_h_ceil = h_idxs_ceil * num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    device = pos_embed_weight.device
    dtype = pos_embed_weight.dtype
    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=dtype, device=device)

    # Bilinear interpolation: lookup 4 corners and blend
    pos_embeds = pos_embed_weight[idx_tensor] * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    # Split by image and permute into spatial merge order
    patch_pos_embeds = patch_pos_embeds.split(
        [int(h.item()) * int(w.item()) for h, w in zip(grid_hs, grid_ws)]
    )

    patch_pos_embeds_permute = []
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
        t, h, w = int(t.item()), int(h.item()), int(w.item())
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(
                t,
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)

    return torch.cat(patch_pos_embeds_permute)


def compute_attention_bounds(
    grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token attention bounds from grid_thw.

    Each token attends only to tokens from the same image/frame group.
    Converts the ``cu_seqlens`` representation (used by HF) to per-token
    ``(bound_min, bound_max)`` format (used by our attention module).

    Args:
        grid_thw: ``[num_images, 3]`` — (temporal, height, width) per image.

    Returns:
        ``(bound_min, bound_max)`` each ``[total_tokens, 1]`` int32.
        Token ``i`` attends to positions ``[bound_min[i], bound_max[i])``.
    """
    # Compute cu_seqlens (same as HF)
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    total_tokens = int(cu_seqlens[-1].item())
    bound_min = torch.zeros(total_tokens, 1, dtype=torch.int32)
    bound_max = torch.zeros(total_tokens, 1, dtype=torch.int32)

    for i in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[i].item())
        end = int(cu_seqlens[i + 1].item())
        bound_min[start:end] = start
        bound_max[start:end] = end

    return bound_min, bound_max
