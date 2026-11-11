# vLLM Neuron (torch-native) + Transformers modeling backend

Findings from setting up vLLM Neuron 0.29.0.1.2.0, the first release on the
native PyTorch / `torch.compile` route (libtorch-neuronx-lite), and from
prototyping the **Transformers modeling backend** in it. AWS Annapurna asked
for that prototype, to match upstream vLLM's GPU experience
(`--model-impl transformers`).

- Date: 2026-11-10/11.
- Host: a trn2.48xlarge container exposing 4 Neuron devices (16 logical
  cores, LNC=2), Ubuntu 24.04 (glibc 2.39), driver 2.31.6, runtime 2.35.30, no
  EFA devices visible.
- Prototype code: branch `transformers-backend` of `~/Dev/vllm-neuron`, 6
  commits on top of `main` (`a080ebf`). The same commits are in
  [`patches/vllm-neuron-transformers-backend/`](patches/vllm-neuron-transformers-backend/)
  (`git am`-able).

## TL;DR

- The transformers backend works on Trainium through vLLM Neuron, and the core
  is small: ~400 lines in a new `vllm_neuron/model/transformers/` package, plus
  a ~40-line registry hook. The runner, scheduler, KV-cache manager, bucketing
  and compile path are untouched. Dynamo traces transformers' modeling code
  as is.
- Verified on device against HF transformers. Every token generated is HF
  fp32's argmax or within 0.08 logits of it, the same accuracy as vLLM
  Neuron's native implementations:
  - Llama (TinyLlama, Llama-3.2-1B) and Qwen3-0.6B, at TP 1, 2 and 4, with
    `--model-impl transformers`;
  - **Granite-3.1-2b, Gemma-3-270m, Qwen2.5-0.5B, SmolLM3-3B and a tiny
    Qwen3-MoE, which vLLM Neuron has no implementation of**, served
    automatically through the backend (Model Runner V2 for the dense models,
    V1 for the MoE);
  - `vllm serve` plus OpenAI chat completions.
- Speed: decode is 10-30% slower than the native NKI implementation at
  TinyLlama size, at both 256 and 2048 context. Attention uses plain PyTorch
  ops, so NKI attention kernels are the next step.
- Found along the way, worth reporting to AWS:
  - the public torch-neuronx 2.33 wheels require glibc 2.43;
  - libtorch-neuronx-lite and vllm-neuron disagree on the default compile
    route;
  - a PyTorch precompile-cache bug.

  See [Issues found](#issues-found).

## 1. Environment

One venv (`.venv`), Python 3.14, torch **2.13.0** (pinned by `vllm==0.29.0`):

| package | version |
|---|---|
| vllm / vllm-neuron | 0.29.0 / 0.29.0.1.2.0 (editable, `~/Dev/vllm-neuron`) |
| libtorch-neuronx-lite | 2.13.0.1.0.3112 |
| torch-neuronx | 2.13.3.0.284 (only for the plain transformers test) |
| neuronx-cc / nki | 2.28.7631.0 / 0.7.0 |
| transformers / accelerate | 5.22.2 / 1.15.0 |

```bash
uv pip install --python .venv/bin/python \
  --extra-index-url https://pip.repos.neuron.amazonaws.com --index-strategy unsafe-best-match \
  --override <(echo transformers==5.22.2) \
  -e ~/Dev/vllm-neuron "torch-neuronx==2.13.*" transformers==5.22.2 accelerate==1.15.0 huggingface_hub
python scripts/patch_glibc_verneed.py .venv/lib/python3.14/site-packages/torch_neuronx/_C.so  # issue 1
source scripts/env.sh                         # issues 4, 5, 7
export TORCH_DEVICE_BACKEND_AUTOLOAD=0        # for vLLM processes only (issue 6)
```

## 2. Results

### 2.1 transformers on torch-neuronx (no vLLM)

[`scripts/run_qwen3_transformers.py`](scripts/run_qwen3_transformers.py) is the
`run.py` of Qwen/Qwen3-0.6B from the `aws-neuron/torch-neuronx-model-support`
sweep: `torch.compile(backend="neuron")`, static cache, chunked prefill. It
**passes**, and its 32 generated tokens match CPU exactly (`cpu match: Y`).

### 2.2 vLLM Neuron, native model

TinyLlama-1.1B at TP=2, the setup guide's configuration, works. The engine is
ready in 204 s cold, including compilation. One prompt matches HF CPU
exactly. The other diverges at token 8 on a near-tie (HF fp32 logits: "The"
10.497 vs "C" 10.479).

### 2.3 vLLM Neuron, Transformers backend

All runs use greedy decoding on 2 prompts, 16 tokens, max_model_len 128, and
are compared with HF transformers on CPU (bf16 greedy, plus an fp32
teacher-forced re-scoring):

| model | arch in vllm-neuron? | selected by | TP | vs HF CPU greedy | max fp32 logit gap |
|---|---|---|---|---|---|
| TinyLlama-1.1B | yes (Llama) | `--model-impl transformers` | 1 | **exact** | 0.019 |
| TinyLlama-1.1B | yes (Llama) | `--model-impl transformers` | 2 | 1 exact, 1 near-tie at token 8 | 0.019 |
| Qwen3-0.6B | yes (Qwen3) | `--model-impl transformers` | 1 | near-ties at tokens 5 and 2 | 0.024 |
| Qwen3-0.6B | yes (Qwen3) | `--model-impl transformers` | 4 | 1 exact, 1 near-tie at token 5 | 0.036 |
| granite-3.1-2b-instruct | **no** | default (auto fallback) | 1 | 1 exact, 1 near-tie at token 3 | 0.063 |
| gemma-3-270m-it | **no** | default (auto fallback) | 1 | **exact** | 0.000 |
| Qwen2.5-0.5B (QKV bias) | **no** | default (auto fallback) | 2 | **exact** | 0.000 |
| SmolLM3-3B (NoPE layers) | **no** | default (auto fallback) | 4 | 1 exact, 1 near-tie at token 2 | 0.011 |
| Llama-3.2-1B-Instruct (tied emb.) | yes (Llama) | `--model-impl transformers` | 4 | 1 exact, 1 near-tie at token 7 | 0.054 |
| TinyLlama-1.1B (4 KV heads, replicated) | yes (Llama) | `--model-impl transformers` | 8 | 1 exact, 1 near-tie at token 8 | 0.039 |
| tiny-random Qwen3-MoE (8 experts, top-2) | **no** | default (auto fallback), Model Runner V1 | 1 | 1 exact, 1 near-tie at token 9 (vs fp32) | 0.055 |

"Max fp32 logit gap" comes from
[`scripts/teacher_forced_logprobs.py`](scripts/teacher_forced_logprobs.py),
which re-scores every token vLLM generated with HF fp32 under teacher forcing.
The gap is how far each generated token is below HF's argmax, maximised over
all tokens. Greedy string matching is a weak test on these small models,
which hit near-ties within a few tokens, so this is the number to look at.
The native TinyLlama run scores 0.039 by the same measure.

**Online serving:** `vllm serve Qwen/Qwen3-0.6B --model-impl transformers`
([`scripts/serve_check.sh`](scripts/serve_check.sh)). An OpenAI
`/v1/chat/completions` request returns "The capital of France is Paris." with
`finish_reason: stop`.

**Native vs Transformers backend** (TinyLlama, TP=2, 2 prompts × 128 tokens,
steady state):

| | max_model_len | generate | per decode step | max fp32 logit gap | mean / max abs(logprob − HF) |
|---|---|---|---|---|---|
| native (`LlamaForCausalLM`, NKI kernels) | 256 | 0.8-1.0 s | ~7 ms | 0.047 | 0.004-0.008 / 0.060 |
| Transformers backend | 256 | 1.1-1.2 s | ~9 ms | 0.058 | 0.005-0.009 / 0.075 |
| native | 2048 | 1.0 s | ~8 ms | — | — |
| Transformers backend | 2048 | 1.1 s | ~9 ms | 0.047 | 0.004-0.008 / 0.049 |

At this model size the gap stays at 10-30%, and it does not grow from 256 to
2048 context, even though the backend gathers the whole block table on every
decode step: fixed per-step costs dominate. Larger models and much longer
contexts are untested.

**Compile and start-up:**

- Cold compile (one prefill and one decode bucket) takes 130-360 s.
- With the precompile cache, a warm start is ready in 18 s (TinyLlama) and
  gives identical outputs.
- The first `generate` after start-up pays a one-time 9-27 s with the
  precompile cache on, native models included.

## 3. Design

On GPU, vLLM's Transformers backend builds the HF model with
`attn_implementation="vllm"`, so each attention layer calls vLLM's paged
attention through the transformers `AttentionInterface`. It then swaps
`nn.Linear` and other layers for vLLM's parallel and fused layers. vLLM Neuron
cannot reuse those classes. Its runner has its own model contract:
`from_configs`, `get_kv_spec`, `bind_kv_cache`, `load_weights`, and
`forward(input_ids, positions, attn_metadata, sampling_positions,
sampling_params, ...)`, compiled per bucket with `torch.compile`. The
prototype implements that contract around an HF model.

`vllm_neuron/model/transformers/model.py`:

- **`neuron_paged_attention_forward`**, registered as
  `attn_implementation="neuron_paged"`. transformers calls it with the
  projected, rotated states of all the step's tokens, flattened into a batch
  of one. It writes K/V into the paged cache at `slot_mapping` (padding slots
  go to the null block, through `paged_slot_indices`). Then:
  - **prefill** (one sequence, no prior cache): causal attention over the
    step's own tokens. The runner pads after the real tokens, so no real
    token sees the padding;
  - **decode and segmented prefill**: each sequence gathers its blocks
    through its block table (unused entries are `-1`) and masks keys by
    position.

  It also handles sliding window, soft-cap and attention sinks, which
  transformers passes as kwargs. Plain PyTorch, no NKI.
- **`TransformersForCausalLM`**:
  - builds the model with `AutoModelForCausalLM.from_config` on `meta`, and
    loads it with `from_pretrained` in `load_weights`, so transformers applies
    its own key renames and weight conversions;
  - passes the runner's `attn_metadata` and positions to the attention through
    the transformers `**kwargs`;
  - computes logits only at `sampling_positions` (`logits_to_keep`), so the
    model's own head stays intact (Gemma's final soft-cap, Granite's logit
    scaling);
  - samples with vLLM Neuron's on-device `Sampler`, which Model Runner V2
    requires.
- **Tensor parallelism** from the HF `base_model_tp_plan`:
  - `colwise` linears are sliced on output features;
  - `rowwise` linears are sliced on input features, followed by
    `get_tp_group().all_reduce`, which compiles into the NEFF;
  - `replicated_with_grad_allreduce` modules are left as they are;
  - with fewer KV heads than ranks, KV heads are replicated, as vLLM does:
    each rank keeps the KV head of its query heads;
  - embeddings and the LM head stay replicated.
- **Mixture of experts**: MoE configs request transformers'
  `experts_implementation="batched_mm"`, which gathers each token's expert
  weights and runs a batched matmul. Its shapes do not depend on the routing,
  so it compiles into a static graph, unlike the default `grouped_mm` and its
  `eager` fallback. Memory grows with tokens × top-k × expert size, so this
  only suits small models; a Neuron experts function (`ExpertsInterface`)
  using vLLM Neuron's MoE kernels is the scalable path.

`vllm_neuron/model/registry.py`:

- Registers `TransformersForCausalLM`, the name vLLM resolves to with
  `--model-impl transformers`.
- `resolve_model_cls()` is used by both runners (V1 and V2). For an
  architecture vLLM Neuron does not implement, vLLM resolves its own GPU
  class, which the Neuron runner cannot run. The hook serves the model with
  the Transformers backend instead, like upstream's `model_impl="auto"`
  fallback.

CPU unit tests (`VLLM_NEURON_CPU_MODE=1 pytest test/unit/model/`):

- `test_transformers_backend.py` checks the attention function against
  reference causal attention: padded prefill, batched decode with a padded
  row, sliding window, and segmented prefill.
- `test_transformers_backend_tp.py` shards a tiny random Llama over simulated
  ranks (threads, in-process all-reduce) and checks its logits against the
  unsharded model at TP 2, 4 and 8, with plain and replicated KV heads.

Commits (in `patches/vllm-neuron-transformers-backend/`):

1. `feat(model): add a Transformers modeling backend` (TP=1)
2. `feat(model): shard the Transformers backend from the HF tp_plan`
3. `feat(model): serve unimplemented archs with the Transformers backend`
4. `test(model): check the Transformers backend attention on CPU`
5. `feat(model): replicate KV heads when TP exceeds them`
6. `feat(model): run MoE models with batched_mm experts`

## 4. Issues found

1. **All public torch-neuronx 2.33 wheels need glibc 2.43** (Ubuntu 26.04),
   for cp312, cp313 and cp314 alike (GitHub release `v2.33.0` and the Neuron
   pip repo). `torch_neuronx/_C.so` fails to load on Ubuntu 24.04 (glibc
   2.39): `` version `GLIBC_2.43' not found ``. Only 9 float libm symbols
   (`acosf`, `asinf`, `atan2f`, `sqrtf`, `log10f`, ...) need the `GLIBC_2.43`
   version, and glibc 2.39 exports the same functions as `GLIBC_2.2.5`.
   Workaround: [`scripts/patch_glibc_verneed.py`](scripts/patch_glibc_verneed.py)
   rewrites the version-needed entry; the original is kept as `_C.so.orig`.
   libtorch-neuronx-lite, neuronx-cc and nki need at most glibc 2.34, so vLLM
   Neuron itself is not affected.
2. The Neuron pip repo serves an ancient `islpy`. With uv's default
   (first-index) strategy it hides the `islpy>=2026.1` that neuronx-cc 2.28
   needs, and the install fails. `--index-strategy unsafe-best-match` fixes
   it.
3. vllm-neuron pins `transformers<5.18.0`, which only guards video-processor
   parsing. It is overridden to 5.22.2 here, which the Transformers backend
   wants anyway.
4. `torch_xla`'s `_XLAC`, pulled in by libtorch-neuronx-lite, links
   `libpython3.14.so`. With interpreters that keep it off the loader path
   (uv-managed Pythons), workers fail at start-up. Fix: add the interpreter's
   `sysconfig` `LIBDIR` to `LD_LIBRARY_PATH` (`scripts/env.sh`).
5. Without EFA devices, workers fail on the EFA CPU-affinity setup. The error
   message says what to do: `NEURON_SKIP_EFA_AFFINITY=1`.
6. As documented, vLLM processes need `TORCH_DEVICE_BACKEND_AUTOLOAD=0` when
   torch-neuronx is installed. Plain torch-neuronx scripts must leave it unset.
7. **The compile route defaults disagree.** With `VLLM_NEURON_BACKEND` unset,
   vllm-neuron takes the native route (`envs.is_native_backend()`), but
   libtorch-neuronx-lite (`_is_native_route()`) only recognizes `neuron` and
   `neuron_native`. So lite:
   - installs its XLA-route torch overrides (`F.gelu`, `F.softmax`,
     `torch.argmax`, `torch.topk`, `nn.Embedding`);
   - leaves the precompile cache off.

   Its `F.gelu` wrapper calls the raw builtin `torch._C._nn.gelu`, which
   Dynamo cannot trace (`Attempted to call function marked as skipped`), so
   every GELU model fails to compile (Gemma3, GPT-2, Phi, ...). Fix:
   `export VLLM_NEURON_BACKEND=neuron_native`. Suggested plugin fix: have
   vllm-neuron `setdefault` the variable before it imports lite.
8. **A PyTorch precompile-cache bug**, hit once issue 7 enables the cache.
   When Dynamo serializes guard state, its pickler records every non-tensor
   module attribute it does not guard on as missing, keyed by `id()`
   (`torch/_dynamo/guards.py`, `GuardsStatePickler`). A module attribute
   holding a `torch.dtype`, which is a singleton, therefore turns the dtype of
   every tensor serialized after it into `_Missing`. Reloading the package
   then fails with `empty_strided(): argument 'dtype' must be torch.dtype, not
   _Missing`. The backend avoids it by not keeping a `self.dtype` attribute.
   Torch should exempt immutable singletons (dtypes, devices, enums).
9. In CPU mode (`VLLM_NEURON_CPU_MODE=1`), the KV cache is sized from host
   RAM: 528 GiB and 25M tokens on this host, which takes forever to allocate.
   Pass `num_gpu_blocks_override`.
10. vllm-neuron 0.29 runs dense models on Model Runner V2. Its forward adapter
    reads the sampled tokens from the model output (`result[0]`), so a model
    must follow the on-device sampling return convention (tokens, gathered
    logits). The Transformers backend reuses vLLM Neuron's `Sampler` for that.
    Disabling ODS was not tested.

## 5. Reproduce

```bash
cd ~/Dev/vllm-test && source scripts/env.sh && export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd scripts
# Transformers backend forced on an architecture vllm-neuron implements:
python vllm_offline.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --model-impl transformers --tp 2 --out /tmp/out.json
# Automatic fallback for an architecture it does not implement:
python vllm_offline.py --model unsloth/gemma-3-270m-it --out /tmp/out.json
# Accuracy: HF fp32 teacher forcing, and the CPU greedy reference:
python teacher_forced_logprobs.py --model unsloth/gemma-3-270m-it /tmp/out.json
python hf_cpu_reference.py --model unsloth/gemma-3-270m-it --out /tmp/ref.json && python compare_outputs.py /tmp/ref.json /tmp/out.json
# Several models in a row (name model impl tp):
echo "qwen3 Qwen/Qwen3-0.6B transformers 4" | bash run_matrix.sh /tmp
# Online serving:
bash serve_check.sh /tmp/serve.log Qwen/Qwen3-0.6B --model-impl transformers
# Unit tests (CPU: attention, TP sharding), from ~/Dev/vllm-neuron:
VLLM_NEURON_CPU_MODE=1 python -m pytest -q test/unit/model/
# Plain transformers on torch-neuronx (leave TORCH_DEVICE_BACKEND_AUTOLOAD unset):
env -u TORCH_DEVICE_BACKEND_AUTOLOAD NEURON_RT_VISIBLE_CORES=8 python run_qwen3_transformers.py
```

## 6. Gaps and next steps

- **Performance**: decode gathers each sequence's whole block table and
  computes attention with plain ops. vLLM Neuron's existing
  `decode_context_length_buckets` option trims the block table to the decode
  bucket, so the gather then reads only that bucket's blocks, not
  max_model_len (untested with this backend). Next:
  - NKI attention kernels: `flash_attention` for prefill, and a paged decode
    kernel that takes Q/K/V (the current `attention_decode` fuses the QKV and
    output projections, so it doesn't fit the attention interface);
  - fused QKV and gate/up projections, and a vocab-sharded LM head
    (`ColumnParallelLinear`), as upstream's fusers do.
- **TP**: the number of attention heads must be divisible by the TP size, and
  KV heads and TP size must divide one another. Still missing:
  `packed_colwise` (fused gate_up/qkv, e.g. Phi3), `colwise_gather_output`,
  the MoE/EP styles, and sequence parallelism.
- **Sliding window**: a full-length cache is allocated and the window is
  applied by masking. Declaring `sliding_window_size` in the KV spec would
  save memory.
- **MoE**: works with `batched_mm` on a tiny model only. For real MoE sizes,
  register a Neuron experts function in transformers' `ExpertsInterface`,
  the counterpart of the attention one, backed by vLLM Neuron's MoE kernels
  (`moe_cte`/`moe_tkg`). Also missing: expert parallelism, and TP of the
  experts (the `moe_tp_experts`/`ep_*` plan styles).
- **Not covered**: multimodal (`TransformersMultiModalForCausalLM`), pooling,
  speculative decoding, quantization, LoRA, and prefix caching / chunked
  prefill on device (segmented prefill is only unit-tested on CPU).
- **Upstreaming**: the partner asked for a working prototype on a fork first.
  The commits are ready for a fork of `vllm-project/vllm-neuron`; none was
  created yet, since that makes the code public.
