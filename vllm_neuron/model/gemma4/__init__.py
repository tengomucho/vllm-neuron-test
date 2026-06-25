# SPDX-License-Identifier: Apache-2.0
# Importing hf_config registers the gemma4_text config with transformers so
# vLLM can parse a text-only Gemma4 config.json on older transformers.
from . import hf_config  # noqa: F401
from .config import Gemma4Config
from .factory import Gemma4ForCausalLM

__all__ = [
    "Gemma4Config",
    "Gemma4ForCausalLM",
]
