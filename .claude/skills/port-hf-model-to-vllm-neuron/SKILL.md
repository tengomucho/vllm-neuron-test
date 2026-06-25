---
name: port-hf-model-to-vllm-neuron
description: >-
  Port a HuggingFace text decoder to the vLLM-Neuron plugin (AWS Trainium/Inferentia)
  by implementing it under vllm_neuron/model/<arch>/, registering it, and validating
  in CPU mode then on-device. Use when the user says "port <model> to vLLM Neuron",
  "onboard a model to vLLM on Neuron", or asks to run a new HF architecture on Trainium
  with the vllm-neuron plugin. NOT for NXDI (use neuron-framework-autoport for that).
---

# Port a HuggingFace model to vLLM on Neuron

This skill operationalizes the onboarding guide ("How to onboard a model to vLLM on
Neuron"). It targets the **text decoder** of an HF model and produces a working
`vllm_neuron/model/<arch>/` implementation that loads, compiles, and generates
coherent text.

## Mental model

The vLLM-Neuron plugin implements each model directly (it does NOT reuse HF modeling
code). You write PyTorch that the Neuron compiler can trace, using the plugin's
functional ops (`vllm_neuron.functional` as `NF`) and parallel modules where they fit.
A model dir has: `config.py` (dataclass + `from_configs`), `model_bf16.py` (the
implementation), optional `weight_loaders.py`, `factory.py` (ModelRegistry entry),
`__init__.py`. Register the architecture string (must equal `architectures[0]` in the
checkpoint's config.json) in `vllm_neuron/model/registry.py`.

## Workflow

### 1. Get ground truth FIRST (do not skip)
- Fetch the model's `config.json` and `model.safetensors.index.json` from the Hub.
  Record: architecture string, every hyperparameter, and the **exact checkpoint key
  names** for one decoder layer (attention, MLP/experts, every norm, embeddings, lm_head).
- Get the **authoritative modeling source**. If the container's transformers is too old
  for the arch, `uv venv` a host venv and `uv pip install` a recent transformers, then
  READ `transformers/models/<arch>/modeling_*.py` and `configuration_*.py`. Do not guess
  the math (RoPE variant, norm convention, attention scaling, MoE routing, activation).
  Capture every quirk verbatim — these are where ports break.
- Pick reference templates by shape: dense→`llama3`, MoE→`qwen3_moe`/`gpt_oss`,
  sliding-window→`gpt_oss`, qk-norm→`qwen3_moe`. Read their real source for exact NF op
  signatures (`NF.router`, `NF.moe_cte`, `NF.flash_attention`, `NF.attention_decode`,
  `NF.mlp`) and weight-loader patterns.

### 2. Config-registration workaround (older transformers)
vLLM calls `AutoConfig.from_pretrained` BEFORE the model registry loads. If the
container's transformers lacks the arch, define a minimal `PretrainedConfig` subclass
(`model_type = "<arch>_text"`) and call `AutoConfig.register(...)` from the plugin's
`register()` in `vllm_neuron/__init__.py` (runs early enough). See
`references/config_registration.md`.

### 3. Implement the model
- Match the HF math EXACTLY. Common gotchas (verify each against the source):
  - RMSNorm convention: `(x/rms)*weight` vs `(x/rms)*(1+weight)`. **Check which** — the
    wrong one compounds across layers and saturates any logit softcapping → gibberish.
  - Attention softmax scale (some models use 1.0 with qk-norm, not 1/sqrt(d)).
  - RoPE: interleaved vs split-half (rotate_half); partial/proportional variants; per-
    layer-type theta for hybrid sliding/full attention.
  - Embedding scaling (e.g. `* sqrt(hidden)`), final logit softcapping, tied lm_head.
  - MoE routing order (softmax-before vs after topk, renorm), per-expert scales (fold a
    per-expert output scale into `down_proj` at load time), expert tensor layout.
- Buffers computed in `__init__` (e.g. RoPE `inv_freq`) MUST use `device="cpu"` — the
  model is built under a `torch.device("meta")` context.
- `flash_attention`/`attention_decode` kernels cap head_dim at 128. For larger head dims,
  implement attention with plain torch ops (matmul/softmax) — compiles for any head dim.
- Provide `forward`, `get_kv_spec`, `bind_kv_cache`, `load_weights`, `from_configs`.
  Mirror an existing model's `forward` signature exactly.

### 4. Text-only model dir (bypass multimodal)
Don't register a multimodal `*ForConditionalGeneration` arch. Build a local dir with the
original safetensors (+ index, tokenizer) and a rewritten text-only `config.json`
(top-level = the text_config; `architectures=["<Arch>ForCausalLM"]`). Patch tokenizer
quirks for the older transformers if needed (e.g. `extra_special_tokens` list→dict).

### 5. Validate — CPU mode FIRST, then device
- Sync code into the running container. CAUTION: `docker cp dir vllm:dest/dir` NESTS if
  `dest/dir` exists — use `docker cp dir/. vllm:dest/dir/` and clear `__pycache__`.
- Fast shakeout: a 2-layer config (same weights, `num_hidden_layers` reduced) to catch
  import/config/weight-loading/shape bugs in seconds.
- CPU correctness: `VLLM_NEURON_CPU_MODE=1` + `enforce_eager=True` in `LLM(...)` runs
  eager (no torch.compile), avoiding patched-op tracing errors. Full model, short prompt.
- **Debug gibberish by comparison, not guessing.** Run the real HF model on CPU (host
  venv) for the same input ids; capture top-k logits + per-layer hidden norms (and
  per-sub-block norms via forward hooks). Feed identical `prompt_token_ids` to the
  vLLM model and compare. Localize the diverging layer/sub-block, fix, repeat. See
  `references/debugging.md`.
- On-device: drop `enforce_eager`, set `tensor_parallel_size` to the core count. First
  run compiles NEFFs (cached). Validate output matches a reference (HF Inference
  Providers or local HF).

### 6. Reference for accuracy
HF Inference Providers (`huggingface_hub.InferenceClient(provider=...)`) gives a quick
chat-level sanity reference. For rigorous checks, compare greedy token/logits against
local HF.

## Files in this skill
- `references/config_registration.md` — the AutoConfig workaround pattern.
- `references/debugging.md` — the CPU-mode comparison-against-HF methodology.
- `references/checklist.md` — per-component HF-vs-impl verification checklist.
