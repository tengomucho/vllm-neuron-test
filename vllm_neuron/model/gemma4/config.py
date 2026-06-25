# SPDX-License-Identifier: Apache-2.0
"""Gemma4 (text) Configuration.

Parses the ``text_config`` of a Gemma4 HuggingFace config (model_type
``gemma4_text``) into a flat dataclass consumed by the Neuron implementation.

<-- MODEL-SPECIFIC: Gemma4 is an MoE, hybrid sliding/full attention model with
per-layer-type RoPE, qk-norm, embedding scaling and final logit softcapping.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Gemma4Config:
    """Configuration for the Gemma4 text decoder."""

    # ── Core architecture ────────────────────────────────────────────────
    vocab_size: int = 262144
    hidden_size: int = 2816
    num_hidden_layers: int = 30
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    hidden_activation: str = "gelu_pytorch_tanh"

    # ── Hybrid attention (sliding vs full) ──────────────────────────────
    # Sliding-window layers
    head_dim: int = 256
    num_key_value_heads: int = 8
    sliding_window: int = 1024
    # Full-attention layers
    global_head_dim: int = 512
    num_global_key_value_heads: int = 2
    # Per-layer type ("sliding_attention" | "full_attention")
    layer_types: list[str] | None = None
    # Per-type RoPE parameters, e.g.
    #   {"sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
    #    "full_attention": {"rope_theta": 1e6, "rope_type": "proportional",
    #                       "partial_rotary_factor": 0.25}}
    rope_parameters: dict | None = None

    # ── MoE ──────────────────────────────────────────────────────────────
    intermediate_size: int = 2112  # dense MLP intermediate
    num_experts: int = 128
    top_k_experts: int = 8
    moe_intermediate_size: int = 704
    norm_topk_prob: bool = True

    # ── Gemma-specific extras ────────────────────────────────────────────
    final_logit_softcapping: float | None = 30.0

    # ── Special tokens ───────────────────────────────────────────────────
    pad_token_id: int | None = 0
    bos_token_id: int = 2
    eos_token_id: int | list[int] = 1

    # ── Framework config (not model-specific) ────────────────────────────
    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.layer_types is None:
            # Default Gemma4: full attention every 6th layer, sliding elsewhere.
            self.layer_types = [
                "full_attention" if (i + 1) % 6 == 0 else "sliding_attention"
                for i in range(self.num_hidden_layers)
            ]

    # ── Per-layer helpers ────────────────────────────────────────────────
    def is_sliding(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "sliding_attention"

    def layer_head_dim(self, layer_idx: int) -> int:
        return self.head_dim if self.is_sliding(layer_idx) else self.global_head_dim

    def layer_num_kv_heads(self, layer_idx: int) -> int:
        return (
            self.num_key_value_heads
            if self.is_sliding(layer_idx)
            else self.num_global_key_value_heads
        )

    def layer_rope(self, layer_idx: int) -> dict:
        key = "sliding_attention" if self.is_sliding(layer_idx) else "full_attention"
        return self.layer_rope_for_type(key)

    def layer_rope_for_type(self, key: str) -> dict:
        if self.rope_parameters and key in self.rope_parameters:
            return self.rope_parameters[key]
        # Fallback defaults
        if key == "sliding_attention":
            return {"rope_theta": 10000.0, "rope_type": "default"}
        return {
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
        }

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        """Build from a HF config. Accepts either the composite Gemma4 config
        (with a ``text_config``) or the text config directly, or a path/dict."""
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
        else:
            config_dict = dict(hf_config)

        # Descend into text_config if this is the composite multimodal config.
        if "text_config" in config_dict and isinstance(config_dict["text_config"], dict):
            text = dict(config_dict["text_config"])
            # carry tie_word_embeddings from top level if absent
            text.setdefault(
                "tie_word_embeddings", config_dict.get("tie_word_embeddings", True)
            )
            config_dict = text

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])
        # HF uses "dtype" in newer configs
        if "torch_dtype" not in filtered_dict and isinstance(
            config_dict.get("dtype"), str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, config_dict["dtype"])

        filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
