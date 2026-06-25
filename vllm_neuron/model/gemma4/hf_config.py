# SPDX-License-Identifier: Apache-2.0
"""Register a minimal Gemma4 text config with HuggingFace transformers.

The vLLM-Neuron container ships an older transformers (4.57.x) that does not
know the ``gemma4`` / ``gemma4_text`` model types. vLLM calls
``AutoConfig.from_pretrained`` during model loading, which fails on an unknown
``model_type``. Importing this module registers a lightweight config class for
``gemma4_text`` so a text-only ``config.json`` (model_type ``gemma4_text``,
architectures ``["Gemma4ForCausalLM"]``) can be parsed.

This is intentionally minimal: it only needs to expose the attributes vLLM and
our ``Gemma4Config.from_configs`` read. The actual per-layer math lives in the
Neuron implementation, not here.
"""

import logging

from transformers import AutoConfig, PretrainedConfig

logger = logging.getLogger(__name__)


class Gemma4TextConfig(PretrainedConfig):
    model_type = "gemma4_text"

    def __init__(
        self,
        vocab_size: int = 262144,
        hidden_size: int = 2816,
        intermediate_size: int = 2112,
        num_hidden_layers: int = 30,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 8,
        head_dim: int = 256,
        global_head_dim: int = 512,
        num_global_key_value_heads: int = 2,
        sliding_window: int = 1024,
        layer_types: list[str] | None = None,
        rope_parameters: dict | None = None,
        rms_norm_eps: float = 1e-6,
        max_position_embeddings: int = 262144,
        hidden_activation: str = "gelu_pytorch_tanh",
        num_experts: int = 128,
        top_k_experts: int = 8,
        moe_intermediate_size: int = 704,
        enable_moe_block: bool = True,
        attention_k_eq_v: bool = True,
        final_logit_softcapping: float | None = 30.0,
        tie_word_embeddings: bool = True,
        hidden_size_per_layer_input: int = 0,
        pad_token_id: int | None = 0,
        bos_token_id: int = 2,
        eos_token_id=1,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.global_head_dim = global_head_dim
        self.num_global_key_value_heads = num_global_key_value_heads
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.rope_parameters = rope_parameters
        self.rms_norm_eps = rms_norm_eps
        self.max_position_embeddings = max_position_embeddings
        self.hidden_activation = hidden_activation
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.enable_moe_block = enable_moe_block
        self.attention_k_eq_v = attention_k_eq_v
        self.final_logit_softcapping = final_logit_softcapping
        self.hidden_size_per_layer_input = hidden_size_per_layer_input
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


def register_gemma4_configs() -> None:
    """Register Gemma4 text config with transformers (idempotent)."""
    try:
        AutoConfig.register("gemma4_text", Gemma4TextConfig)
    except ValueError:
        # Already registered (e.g. native gemma4 support or a second import).
        pass
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not register gemma4_text config: %s", e)


register_gemma4_configs()
