# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL vision encoder (BF16).

Complete vision transformer implementation: PatchEmbed → Blocks × depth → Merger.
Produces main embeddings and deepstack features at specified intermediate layers.

torch_neuronx patches F.gelu/nn.GELU with a C extension that torch.compile
(Dynamo) cannot trace through. This module provides erf-based equivalents
decorated with @torch.compiler.allow_in_graph.

All tensors flow as 3D ``[1, seq_len, hidden_size]`` through the vision encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import vllm_neuron.functional as NF
from vllm_neuron.parallel.neuron_parallel_state import get_neuron_vision_tp_group
from vllm_neuron.utils.weight_loader import set_weight_loader, sharding_weight_loader

from .weight_loaders import vis_qkv_bias_loader, vis_qkv_weight_loader


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------


@torch.compiler.allow_in_graph
def gelu(x: torch.Tensor) -> torch.Tensor:
    """GELU activation via erf, traceable by torch.compile on Neuron."""
    return 0.5 * x * (1.0 + torch.erf(x / 1.4142135623730951))


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class Qwen3VLVisionMLP(nn.Module):
    """Vision transformer MLP block with tensor-parallel sharding.

    Architecture: ``fc1(x) → GELU → fc2(x)``

    Parallelism:
    - fc1 weight is sharded on the intermediate dim (dim=1) across TP ranks.
    - fc2 weight is sharded on the intermediate dim (dim=0) across TP ranks.
    - TP all-reduce after fc2 matmul, then fc2 bias is added.

    Args:
        hidden_size: Input and output dimension.
        intermediate_size: Hidden dimension of the MLP.
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()

        self.tp_group = get_neuron_vision_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = hidden_size
        self.intermediate_size_per_rank = intermediate_size // self.world_size

        self.fc1_weight = nn.Parameter(
            torch.empty(hidden_size, self.intermediate_size_per_rank, dtype=dtype)
        )
        self.fc1_bias = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, dtype=dtype)
        )
        self.fc2_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, hidden_size, dtype=dtype)
        )
        self.fc2_bias = nn.Parameter(torch.empty(hidden_size, dtype=dtype))

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        """Attach weight loaders for checkpoint → parameter transformation."""
        fc1_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        fc2_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        fc1_bias_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=False,
        )

        set_weight_loader(self.fc1_weight, fc1_loader)
        set_weight_loader(self.fc1_bias, fc1_bias_loader)
        set_weight_loader(self.fc2_weight, fc2_loader)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """MLP forward: fc1 → GELU → fc2 → all-reduce → + bias.

        Args:
            hidden_states: Input tensor of shape ``[1, seq_len, hidden_size]``.

        Returns:
            Output tensor of shape ``[1, seq_len, hidden_size]``.
        """
        output = hidden_states @ self.fc1_weight + self.fc1_bias
        output = gelu(output)
        output = output @ self.fc2_weight

        if self.world_size > 1:
            out_dtype = output.dtype
            output = output.float()
            self.tp_group.all_reduce(output)
            output = output.to(out_dtype)

        output = output + self.fc2_bias

        return output


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved (rotate_half) RoPE: split into second/first half, negate first."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Qwen3VLVisionAttention(nn.Module):
    """Vision transformer self-attention with 2D RoPE and bounds masking.

    Architecture:
    - Fused QKV projection (with bias) via raw weight parameters
    - 2D RoPE applied to Q and K (interleaved rotate_half style)
    - Bidirectional attention with bounds masking (no causal mask)
    - Output projection (with bias) via raw weight parameters + TP all-reduce

    Parallelism:
    - QKV weight sharded via vis_qkv_weight_loader (interleaved head sharding)
    - Output projection weight sharded on input dim, all-reduced across TP ranks
    - No SP — vision encoder processes full sequence on all TP ranks
    - No GQA — num_heads Q = num_heads K = num_heads V

    Args:
        num_heads: Total number of attention heads.
        head_dim: Dimension per attention head.
        hidden_size: Total hidden dimension (num_heads * head_dim).
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = head_dim**-0.5
        self.dtype = dtype

        self.tp_group = get_neuron_vision_tp_group()
        self.world_size = self.tp_group.world_size

        assert num_heads % self.world_size == 0, (
            f"num_heads {num_heads} must be divisible by tp world_size {self.world_size}"
        )
        self.num_heads_per_rank: int = num_heads // self.world_size

        qkv_size_per_rank = 3 * self.num_heads_per_rank * self.head_dim
        self.qkv_weight = nn.Parameter(
            torch.empty(hidden_size, qkv_size_per_rank, dtype=dtype)
        )
        self.qkv_bias = nn.Parameter(torch.empty(qkv_size_per_rank, dtype=dtype))

        o_proj_in = self.num_heads_per_rank * self.head_dim
        self.proj_weight = nn.Parameter(
            torch.empty(o_proj_in, hidden_size, dtype=dtype)
        )
        self.proj_bias = nn.Parameter(torch.empty(hidden_size, dtype=dtype))

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        """Attach weight loaders for checkpoint → parameter transformation."""
        set_weight_loader(
            self.qkv_weight,
            vis_qkv_weight_loader(
                self.num_heads_per_rank, self.head_dim, self.hidden_size
            ),
        )
        set_weight_loader(
            self.qkv_bias,
            vis_qkv_bias_loader(
                self.num_heads_per_rank, self.head_dim, self.hidden_size
            ),
        )
        set_weight_loader(
            self.proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=self.num_heads_per_rank * self.head_dim,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """Attention forward with bounds masking.

        Args:
            hidden_states: ``[1, seq_len, hidden_size]``.
            cos: ``[seq_len, head_dim]`` — RoPE cosine embeddings.
            sin: ``[seq_len, head_dim]`` — RoPE sine embeddings.
            bound_min: ``[seq_len, 1]`` int32 — inclusive lower bound per token.
            bound_max: ``[seq_len, 1]`` int32 — exclusive upper bound per token.

        Returns:
            ``[1, seq_len, hidden_size]``
        """
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.to(self.dtype)

        qkv = NF.qkv_proj(
            hidden=hidden_states,
            qkv_weights=self.qkv_weight,
            bias=self.qkv_bias.unsqueeze(0),
        )

        hd = self.num_heads_per_rank * self.head_dim
        q = qkv[0, :, :hd]
        k = qkv[0, :, hd : 2 * hd]
        v = qkv[0, :, 2 * hd :]

        q = q.view(seq_len, self.num_heads_per_rank, self.head_dim)
        k = k.view(seq_len, self.num_heads_per_rank, self.head_dim)
        v = v.view(seq_len, self.num_heads_per_rank, self.head_dim)

        # 2D RoPE (interleaved rotate_half style)
        cos = cos.to(dtype=hidden_states.dtype)
        sin = sin.to(dtype=hidden_states.dtype)
        cos_sq = cos.unsqueeze(1)
        sin_sq = sin.unsqueeze(1)
        q = q * cos_sq + _rotate_half(q) * sin_sq
        k = k * cos_sq + _rotate_half(k) * sin_sq

        # [seq_len, heads_per_rank, head_dim] → [heads_per_rank, seq_len, head_dim]
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        # Flash attention with bounds masking (bidirectional, no causal mask)
        attn_bound_min = bound_min.unsqueeze(0).expand(self.num_heads_per_rank, -1, -1)
        attn_bound_max = bound_max.unsqueeze(0).expand(self.num_heads_per_rank, -1, -1)

        attn_output = NF.flash_attention(
            q,
            k,
            v,
            scale=self.scale,
            causal_mask=False,
            tp_q=True,
            tp_k=True,
            bound_min=attn_bound_min,
            bound_max=attn_bound_max,
        )

        # Output projection
        attn_output = attn_output.transpose(1, 2).unsqueeze(0)
        output = NF.o_proj(attn_output, self.proj_weight)

        if self.world_size > 1:
            out_dtype = output.dtype
            output = output.float()
            self.tp_group.all_reduce(output)
            output = output.to(out_dtype)

        output = output + self.proj_bias

        return output


# ---------------------------------------------------------------------------
# Block
# ---------------------------------------------------------------------------


class Qwen3VLVisionBlock(nn.Module):
    """Single vision transformer block: norm1 → attention → norm2 → MLP.

    Standard pre-norm ViT block layout with residual connections.

    Args:
        num_heads: Total number of attention heads.
        head_dim: Dimension per attention head.
        hidden_size: Vision hidden dimension.
        intermediate_size: MLP intermediate dimension.
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
        self.attn = Qwen3VLVisionAttention(num_heads, head_dim, hidden_size, dtype)
        self.mlp = Qwen3VLVisionMLP(hidden_size, intermediate_size, dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> torch.Tensor:
        """Block forward with pre-norm residual connections.

        Args:
            hidden_states: ``[1, seq_len, hidden_size]``
            cos: ``[seq_len, head_dim]`` — RoPE cosine embeddings.
            sin: ``[seq_len, head_dim]`` — RoPE sine embeddings.
            bound_min: ``[seq_len, 1]`` int32
            bound_max: ``[seq_len, 1]`` int32

        Returns:
            ``[1, seq_len, hidden_size]``
        """
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cos, sin, bound_min, bound_max
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------


class Qwen3VLVisionPatchEmbed(nn.Module):
    """Convert raw pixel patches to patch embeddings via Conv3d.

    Input:  ``[1, seq_len, C * temporal_patch_size * patch_size * patch_size]``
    Output: ``[1, seq_len, vis_hidden_size]``

    Args:
        in_channels: Number of input image channels (e.g. 3 for RGB).
        temporal_patch_size: Temporal patch size (e.g. 2).
        patch_size: Spatial patch size (e.g. 16).
        hidden_size: Vision encoder hidden dimension (output channels).
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        in_channels: int = 3,
        temporal_patch_size: int = 2,
        patch_size: int = 16,
        hidden_size: int = 1152,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal_patch_size
        self.patch_size = patch_size
        self.embed_dim = hidden_size

        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels,
            self.embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
            dtype=dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project flattened patches to vis_hidden_size.

        Args:
            hidden_states: Flattened pixel patches of shape
                ``[1, seq_len, C * temporal_patch_size * patch_size * patch_size]``.

        Returns:
            Patch embeddings of shape ``[1, seq_len, vis_hidden_size]``.
        """
        target_dtype = self.proj.weight.dtype
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(dtype=target_dtype)).view(
            1, seq_len, self.embed_dim
        )


# ---------------------------------------------------------------------------
# Patch Merger
# ---------------------------------------------------------------------------


class Qwen3VLVisionPatchMerger(nn.Module):
    """Spatial patch merger with two-layer MLP projection.

    Merges ``spatial_merge_size²`` adjacent patches by concatenating their
    hidden states, then projects through ``fc1 → GELU → fc2`` to produce
    tokens of dimension ``out_hidden_size``.

    Two variants controlled by ``use_postshuffle_norm``:
    - ``False`` (main merger): norm on ``[hidden_size]`` *before* spatial reshape
    - ``True`` (deepstack mergers): norm on ``[merged_hidden_size]`` *after* spatial reshape

    Parallelism:
    - fc1 weight sharded on output dim (ColumnParallel style) across TP ranks.
    - fc2 weight sharded on input dim (RowParallel style) across TP ranks.
    - TP all-reduce after fc2.
    - fc2 bias added after all-reduce (not sharded).

    Args:
        hidden_size: Vision encoder hidden dimension.
        out_hidden_size: Text model hidden dimension (output of merger).
        spatial_merge_size: Number of patches merged per spatial dimension.
        use_postshuffle_norm: If True, apply LayerNorm after spatial reshape.
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
        use_postshuffle_norm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.use_postshuffle_norm = use_postshuffle_norm
        self.merged_hidden_size = hidden_size * (spatial_merge_size**2)

        self.tp_group = get_neuron_vision_tp_group()
        self.world_size = self.tp_group.world_size

        self.merged_hidden_per_rank = self.merged_hidden_size // self.world_size

        norm_size = self.merged_hidden_size if use_postshuffle_norm else hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6, dtype=dtype)

        self.fc1_weight = nn.Parameter(
            torch.empty(
                self.merged_hidden_size, self.merged_hidden_per_rank, dtype=dtype
            )
        )
        self.fc1_bias = nn.Parameter(
            torch.empty(self.merged_hidden_per_rank, dtype=dtype)
        )

        self.fc2_weight = nn.Parameter(
            torch.empty(self.merged_hidden_per_rank, out_hidden_size, dtype=dtype)
        )
        self.fc2_bias = nn.Parameter(torch.empty(out_hidden_size, dtype=dtype))

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        """Attach weight loaders for checkpoint → parameter transformation."""
        fc1_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.merged_hidden_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        fc2_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.merged_hidden_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        fc1_bias_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.merged_hidden_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=False,
        )

        set_weight_loader(self.fc1_weight, fc1_loader)
        set_weight_loader(self.fc1_bias, fc1_bias_loader)
        set_weight_loader(self.fc2_weight, fc2_loader)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Merge spatial patches and project to text hidden dimension.

        Args:
            hidden_states: ``[num_patches, hidden_size]`` where
                ``num_patches`` is divisible by ``spatial_merge_size²``.

        Returns:
            ``[num_patches // spatial_merge_size², out_hidden_size]``
        """
        if self.use_postshuffle_norm:
            x = self.norm(hidden_states.view(-1, self.merged_hidden_size)).view(
                -1, self.merged_hidden_size
            )
        else:
            x = self.norm(hidden_states).view(-1, self.merged_hidden_size)

        x = (x @ self.fc1_weight) + self.fc1_bias
        x = gelu(x)

        x = x @ self.fc2_weight

        if self.world_size > 1:
            out_dtype = x.dtype
            x = x.float()
            self.tp_group.all_reduce(x)
            x = x.to(out_dtype)

        x = x + self.fc2_bias

        return x


# ---------------------------------------------------------------------------
# Vision Model (top-level)
# ---------------------------------------------------------------------------


class Qwen3VLVisionModel(nn.Module):
    """Complete Qwen3-VL vision encoder (ViT + mergers).

    Architecture::

        pixel_values → PatchEmbed → + pos_embeds
        → Block_0 → Block_1 → ... → Block_{depth-1}
                      ↓ (at deepstack_visual_indexes)
                      DeepstackMerger_i → deepstack_features[i]
        → Merger → main_embeddings

        Returns: (main_embeddings, deepstack_features)

    Args:
        hidden_size: Vision hidden dimension (e.g. 1152 for 8B).
        intermediate_size: MLP intermediate dimension (e.g. 4304).
        out_hidden_size: Text model hidden dimension (e.g. 4096).
        num_heads: Number of attention heads (e.g. 16).
        depth: Number of transformer blocks (e.g. 27).
        patch_size: Spatial patch size (e.g. 16).
        temporal_patch_size: Temporal patch size (e.g. 2).
        in_channels: Input image channels (e.g. 3).
        spatial_merge_size: Patches merged per spatial dim (e.g. 2).
        deepstack_visual_indexes: Layer indices for deepstack features.
        dtype: Weight dtype. Default: torch.bfloat16.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        out_hidden_size: int,
        num_heads: int,
        depth: int,
        patch_size: int = 16,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        spatial_merge_size: int = 2,
        deepstack_visual_indexes: list[int] | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.deepstack_visual_indexes = deepstack_visual_indexes or []
        head_dim = hidden_size // num_heads

        self.patch_embed = Qwen3VLVisionPatchEmbed(
            in_channels=in_channels,
            temporal_patch_size=temporal_patch_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            dtype=dtype,
        )

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    dtype=dtype,
                )
                for _ in range(depth)
            ]
        )

        self.merger = Qwen3VLVisionPatchMerger(
            hidden_size=hidden_size,
            out_hidden_size=out_hidden_size,
            spatial_merge_size=spatial_merge_size,
            use_postshuffle_norm=False,
            dtype=dtype,
        )

        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    hidden_size=hidden_size,
                    out_hidden_size=out_hidden_size,
                    spatial_merge_size=spatial_merge_size,
                    use_postshuffle_norm=True,
                    dtype=dtype,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pos_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        bound_min: torch.Tensor,
        bound_max: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Full vision encoder forward.

        Args:
            pixel_values: ``[1, num_patches, in_channels * temporal_patch_size * patch_size²]``
            pos_embeds: ``[1, num_patches, hidden_size]``
            cos: ``[seq_len, head_dim]`` — Pre-computed 2D RoPE cosine embeddings.
            sin: ``[seq_len, head_dim]`` — Pre-computed 2D RoPE sine embeddings.
            bound_min: ``[seq_len, 1]`` int32
            bound_max: ``[seq_len, 1]`` int32

        Returns:
            Tuple of:
            - main_embeddings: ``[num_merged_tokens, out_hidden_size]``
            - deepstack_embeddings: ``[num_deepstack_layers, num_merged_tokens, out_hidden_size]``
        """
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + pos_embeds

        deepstack_features: list[torch.Tensor] = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cos, sin, bound_min, bound_max)

            if layer_idx in self.deepstack_visual_indexes:
                ds_idx = self.deepstack_visual_indexes.index(layer_idx)
                ds_feature = self.deepstack_merger_list[ds_idx](
                    hidden_states.squeeze(0)
                )
                deepstack_features.append(ds_feature)

        main_embeddings = self.merger(hidden_states.squeeze(0))

        if deepstack_features:
            deepstack_embeddings = torch.stack(deepstack_features, dim=0)
        else:
            deepstack_embeddings = torch.empty(
                0, dtype=main_embeddings.dtype, device=main_embeddings.device
            )

        return main_embeddings, deepstack_embeddings

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @staticmethod
    def build_weight_mappings(
        depth: int, deepstack_visual_indexes: list[int]
    ) -> dict[str, str]:
        """Build parameter name → checkpoint key mappings for the full vision model."""
        mappings: dict[str, str] = {}

        mappings["patch_embed.proj.weight"] = "model.visual.patch_embed.proj.weight"
        mappings["patch_embed.proj.bias"] = "model.visual.patch_embed.proj.bias"

        for i in range(depth):
            prefix = f"blocks.{i}"
            ckpt_prefix = f"model.visual.blocks.{i}"

            mappings[f"{prefix}.attn.qkv_weight"] = f"{ckpt_prefix}.attn.qkv.weight"
            mappings[f"{prefix}.attn.qkv_bias"] = f"{ckpt_prefix}.attn.qkv.bias"
            mappings[f"{prefix}.attn.proj_weight"] = f"{ckpt_prefix}.attn.proj.weight"
            mappings[f"{prefix}.attn.proj_bias"] = f"{ckpt_prefix}.attn.proj.bias"

            mappings[f"{prefix}.mlp.fc1_weight"] = (
                f"{ckpt_prefix}.mlp.linear_fc1.weight"
            )
            mappings[f"{prefix}.mlp.fc1_bias"] = f"{ckpt_prefix}.mlp.linear_fc1.bias"
            mappings[f"{prefix}.mlp.fc2_weight"] = (
                f"{ckpt_prefix}.mlp.linear_fc2.weight"
            )
            mappings[f"{prefix}.mlp.fc2_bias"] = f"{ckpt_prefix}.mlp.linear_fc2.bias"

            mappings[f"{prefix}.norm1.weight"] = f"{ckpt_prefix}.norm1.weight"
            mappings[f"{prefix}.norm1.bias"] = f"{ckpt_prefix}.norm1.bias"
            mappings[f"{prefix}.norm2.weight"] = f"{ckpt_prefix}.norm2.weight"
            mappings[f"{prefix}.norm2.bias"] = f"{ckpt_prefix}.norm2.bias"

        mappings["merger.fc1_weight"] = "model.visual.merger.linear_fc1.weight"
        mappings["merger.fc1_bias"] = "model.visual.merger.linear_fc1.bias"
        mappings["merger.fc2_weight"] = "model.visual.merger.linear_fc2.weight"
        mappings["merger.fc2_bias"] = "model.visual.merger.linear_fc2.bias"
        mappings["merger.norm.weight"] = "model.visual.merger.norm.weight"
        mappings["merger.norm.bias"] = "model.visual.merger.norm.bias"

        for j in range(len(deepstack_visual_indexes)):
            prefix = f"deepstack_merger_list.{j}"
            ckpt_prefix = f"model.visual.deepstack_merger_list.{j}"
            mappings[f"{prefix}.fc1_weight"] = f"{ckpt_prefix}.linear_fc1.weight"
            mappings[f"{prefix}.fc1_bias"] = f"{ckpt_prefix}.linear_fc1.bias"
            mappings[f"{prefix}.fc2_weight"] = f"{ckpt_prefix}.linear_fc2.weight"
            mappings[f"{prefix}.fc2_bias"] = f"{ckpt_prefix}.linear_fc2.bias"
            mappings[f"{prefix}.norm.weight"] = f"{ckpt_prefix}.norm.weight"
            mappings[f"{prefix}.norm.bias"] = f"{ckpt_prefix}.norm.bias"

        return mappings

    def load_weights(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        *,
        cpu_mode: bool = True,
    ) -> None:
        """Load weights from a HuggingFace safetensors checkpoint."""
        from vllm_neuron.parallel.neuron_parallel_state import (
            get_neuron_vision_tp_group,
        )
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tp_group = get_neuron_vision_tp_group()
        tp_rank = tp_group.rank_in_group
        tp_size = tp_group.world_size

        mappings = self.build_weight_mappings(self.depth, self.deepstack_visual_indexes)
        checkpoint = SafetensorsCheckpoint(checkpoint_path)

        if cpu_mode:
            sd = checkpoint.load_sharded(
                rank=tp_rank,
                world_size=tp_size,
                model=self,
                mappings=mappings,
                device=device,
            ).state_dict
        else:
            sd = checkpoint.load_sharded_pipelined(
                rank=tp_rank,
                world_size=tp_size,
                model=self,
                mappings=mappings,
                device=device,
            ).state_dict

        self.load_state_dict(sd, strict=False, assign=True)
