# SPDX-License-Identifier: Apache-2.0
"""Qwen3 MoE Configuration."""

import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3MoeConfig:
    """Configuration for Qwen3 MoE model."""

    vocab_size: int = 151936
    hidden_size: int = 2048
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int | None = None
    head_dim: int | None = None
    num_experts: int = 64
    num_experts_per_tok: int = 8
    moe_intermediate_size: int = 1408
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 40960
    norm_topk_prob: bool = True
    torch_dtype: torch.dtype = torch.bfloat16

    pad_token_id: int | None = None
    bos_token_id: int = 151643
    eos_token_id: int = 151645

    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            if (
                hasattr(hf_config, "quantization_config")
                and hf_config.quantization_config is None
            ):
                delattr(hf_config, "quantization_config")
                config_dict = hf_config.to_dict()
                hf_config.quantization_config = None
            else:
                config_dict = hf_config.to_dict()
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
