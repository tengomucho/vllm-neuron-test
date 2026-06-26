#!/usr/bin/env python3
"""Build a synthetic Gemma4 model with head_dim=256/512 (same arch as 26B, tiny dims).

Run in the host .venv-hf (transformers 5.x):
    source .venv-hf/bin/activate
    python scripts/build_gemma4_hd256.py

Produces /tmp/g4-hd256/ with a text-only config and tiny random weights.
Then docker cp it into the container for fast compile iteration (~1-2 min vs 10-15).

NOTE: I_per_rank = moe_intermediate_size(128) / tp_degree(4) = 32 which is below the
moe_cte kernel minimum of 128. That's intentional — the decode path falls back to the
vectorized einsum, avoiding moe_cte entirely so we isolate the attention trigger.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

OUTPUT_DIR = Path("/tmp/g4-hd256")

try:
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    import torch
except ImportError:
    sys.exit("Run in .venv-hf: source .venv-hf/bin/activate && python " + __file__)

print("Building synthetic Gemma4 model with head_dim=256/512 ...")

cfg = Gemma4TextConfig(
    hidden_size=256,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=256,              # triggers the head_dim>128 code path
    global_head_dim=512,       # full-attention layers (every 6th -> layer_types[1])
    num_global_key_value_heads=2,
    num_experts=8,
    top_k_experts=2,
    moe_intermediate_size=128, # keep small; I_per_rank<128 at TP=4 -> einsum fallback, not moe_cte
    intermediate_size=512,
    vocab_size=1000,
    layer_types=["sliding_attention", "full_attention"],
    enable_moe_block=True,
    attention_k_eq_v=True,
    hidden_size_per_layer_input=0,
    rms_norm_eps=1e-6,
    final_logit_softcapping=30.0,
    sliding_window=512,
    rope_parameters={
        "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        "full_attention": {
            "rope_theta": 1000000.0,
            "rope_type": "proportional",
            "partial_rotary_factor": 0.25,
        },
    },
)

model = Gemma4ForCausalLM(cfg).to(torch.bfloat16)
model.eval()

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True)

model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
print(f"Saved HF model to {OUTPUT_DIR}")

# Remap safetensors keys: add 'model.language_model.' prefix so our vLLM-Neuron
# weight loader (which expects the 26B checkpoint key structure) can load them.
# The HF Gemma4ForCausalLM saves as 'model.*'; the real 26B has 'model.language_model.*'.
from safetensors.torch import load_file, save_file

st_path = OUTPUT_DIR / "model.safetensors"
tensors = load_file(st_path)
remapped = {}
for k, v in tensors.items():
    if k.startswith("model."):
        new_k = "model.language_model." + k[len("model."):]
    elif k.startswith("lm_head."):
        # lm_head is tied to embed_tokens — just skip; load_weights handles tie.
        continue
    else:
        new_k = k
    remapped[new_k] = v
save_file(remapped, st_path)
print(f"Remapped {len(tensors)} → {len(remapped)} keys (added model.language_model. prefix)")

# Rewrite config.json so the vLLM-Neuron plugin can load it.
config_path = OUTPUT_DIR / "config.json"
hf_cfg = json.loads(config_path.read_text())

# The HF save produces a nested text_config — flatten it.
text_cfg = hf_cfg.get("text_config", hf_cfg)
text_cfg["architectures"] = ["Gemma4ForCausalLM"]
text_cfg["model_type"] = "gemma4_text"
text_cfg["tie_word_embeddings"] = True
# Remove multimodal wrapper keys if present.
for k in ("text_config", "vision_config", "audio_config"):
    text_cfg.pop(k, None)

config_path.write_text(json.dumps(text_cfg, indent=2))
print("Rewrote config.json for vLLM-Neuron (gemma4_text, Gemma4ForCausalLM)")

# Copy tokenizer from the tiny-random model if available, else create a minimal one.
# We'll let the container use the tokenizer from gemma4-tiny-random.
print(f"\nDone. To push into the container:")
print(f"  docker cp {OUTPUT_DIR}/. vllm_neuron:/root/gemma4-hd256/")
print(f"  # then copy the tokenizer from gemma4-tiny-random:")
print(f"  docker exec vllm_neuron bash -c 'cp /root/gemma4-tiny-random/tokenizer* /root/gemma4-hd256/'")
print(f"  docker exec vllm_neuron bash -c 'cp /root/gemma4-tiny-random/special_tokens_map.json /root/gemma4-hd256/'")
