# vLLM Neuron (torch-native) + Transformers modeling backend

Findings from setting up vLLM Neuron 0.29.0.1.2.0, the first release on the
native PyTorch / `torch.compile` route (libtorch-neuronx-lite), and from
prototyping the **Transformers modeling backend** in it. AWS Annapurna asked
for that prototype, to match upstream vLLM's GPU experience
(`--model-impl transformers`).

- Date: 2026-11-10 to 13.
- Host: a trn2.48xlarge container exposing 4 Neuron devices (16 logical
  cores, LNC=2), Ubuntu 24.04 (glibc 2.39), driver 2.31.6, runtime 2.35.30, no
  EFA devices visible.
- Prototype code: branch
  [`transformers-backend`](https://github.com/tengomucho/vllm-neuron/tree/transformers-backend)
  of the fork [`tengomucho/vllm-neuron`](https://github.com/tengomucho/vllm-neuron),
  signed-off commits on top of upstream `main` (`fe7bf43`), checked out in
  `~/Dev/vllm-neuron`
  ([diff against upstream](https://github.com/vllm-project/vllm-neuron/compare/main...tengomucho:vllm-neuron:transformers-backend)).

## TL;DR

- The transformers backend works on Trainium through vLLM Neuron, and the core
  is small: ~400 lines in a new `vllm_neuron/model/transformers/` package
  (~740 with the LFM2 and MoE additions), plus a ~40-line registry hook. The
  runner, scheduler, KV-cache manager, bucketing and compile path are
  untouched. Dynamo traces transformers' modeling code as is.
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
- **A 24B model up to 128k context**: LiquidAI/LFM2.5-24B-A2B, a hybrid of
  short convolutions and attention with 64 experts, runs at TP 4 (13 GiB of
  weights per rank) through the backend. It needed a paged state for the
  convolutions, an experts function, expert sharding, and NKI segmented
  attention for long prompts. See [2.4](#24-a-24b-hybrid-moe-lfm25-24b-a2b).
  - 4k context: TTFT 0.74 s for 3968 tokens, decode 38 ms/token (104
    tokens/s with 4 sequences), GSM8K 93/100.
  - 128k context: TTFT 66 s for 131k tokens, decode 47 ms/token, and
    15/15 needles found at 32k, 64k and 128k.
  - Its logprobs are as close to HF fp32 as HF's own bf16 run: 16 and 15
    tokens of 128 off fp32's argmax.
- **A Docker image** on Ubuntu 26.04, [`docker/Dockerfile`](docker/Dockerfile),
  needs none of this host's workarounds. Its entrypoint is `vllm`, and it
  serves Qwen3-0.6B and LFM2.5-24B-A2B. See [Docker image](#docker-image).
- Speed: decode is 10-30% slower than the native NKI implementation at
  TinyLlama size, at both 256 and 2048 context. Attention uses plain PyTorch
  ops (but for segmented prefill), so NKI attention kernels are the next
  step.
- Found along the way, worth reporting to AWS:
  - the public torch-neuronx 2.33 wheels require glibc 2.43;
  - libtorch-neuronx-lite and vllm-neuron disagree on the default compile
    route;
  - a PyTorch precompile-cache bug;
  - neuronx-cc needs `libarchive.so.13`, which no package declares;
  - the compile cache spans three directories that point to one another:
    keeping only some of them makes start-up fail instead of recompiling.

  See [Issues found](#4-issues-found).

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

### Docker image

[`docker/Dockerfile`](docker/Dockerfile) builds the same stack on Ubuntu 26.04.
Its glibc 2.43 loads the torch-neuronx wheels as they are, and its system
Python keeps `libpython3.14.so` on the loader path, so neither the glibc patch
(issue 1) nor `scripts/env.sh` is needed. The image holds:

- the Neuron runtime, collectives and tools (`neuron-ls`, `neuron-top`), from
  the `resolute` suite of the Neuron apt repo, at this host's versions. The
  driver stays on the host;
- the system Python 3.14 and a uv venv in `/opt/venv`, with the packages of
  the table above. vllm-neuron comes from the fork's `transformers-backend`
  branch;
- `libpython3.14` and `libarchive13t64`, which torch_xla and neuronx-cc link
  but no package pulls in (issue 14);
- `VLLM_NEURON_BACKEND=neuron_native`, `TORCH_DEVICE_BACKEND_AUTOLOAD=0` and
  `NEURON_SKIP_EFA_AFFINITY=1` (issues 5 to 7), and
  `NKI_COMPILE_CACHE_URL=/root/.cache/nki` (issue 15);
- `vllm` as its entrypoint.

It takes 12.2 GB, most of it the CUDA libraries that vLLM's torch pulls in.

```bash
docker build -t vllm-neuron-transformers docker/
docker run --rm -p 8000:8000 --shm-size=16g \
  $(for d in /dev/neuron*; do echo --device=$d; done) \
  -v neuron-cache:/root/.cache -e HF_TOKEN \
  vllm-neuron-transformers serve Qwen/Qwen3-0.6B --model-impl transformers \
  --tensor-parallel-size 4 --max-model-len 256 --max-num-seqs 2 \
  --max-num-batched-tokens 256 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256], "num_seqs_buckets": [2]}}'
# LFM2.5-24B-A2B at 4k context, 4 sequences (the setup of 2.4), same docker flags:
docker run ... vllm-neuron-transformers serve LiquidAI/LFM2.5-24B-A2B \
  --tensor-parallel-size 4 --max-model-len 4096 --max-num-seqs 4 \
  --max-num-batched-tokens 4096 --no-enable-prefix-caching \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [4096], "num_seqs_buckets": [4]}}'
```

- **Cache volume**: keep all of `/root/.cache` in one volume. It holds the
  models (`huggingface/`) and the whole compile cache: the compiled graphs
  (`vllm/neuron/`), their NEFFs (`neuron_libtorch/`) and the NKI kernels
  (`nki/`). Keeping only some of them makes start-up fail instead of
  recompiling (issue 15).
- **Fork version**: the branch is fetched when the image is built, and Docker
  reuses that layer afterwards. To pick up a newer commit, rebuild with
  `--no-cache`, or pass one with `--build-arg VLLM_NEURON_REF=<commit>`.

Tested on this host, whose Docker daemon sees neither the host's files nor its
network: the models went to a named volume, and the requests were sent from
inside the container with `docker exec ... curl` (`-p` is untested). Each run
sent one chat completion ("What is the capital of France?"). Start-up times
leave out downloading the model, which was already in the volume:

| model | TP | cold start | warm start | answer |
|---|---|---|---|---|
| Qwen3-0.6B (`--model-impl transformers`) | 4 | 184 s | 70 s | reasons its way to Paris (cut at 64 tokens) |
| LFM2.5-24B-A2B (4k context, 4 sequences) | 4 | 509 s | 121 s | "The capital of France is Paris.", after 50 tokens of reasoning |

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

All runs but the last use greedy decoding on 2 prompts, 16 tokens,
max_model_len 128, and are compared with HF transformers on CPU (bf16 greedy, plus an fp32
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
| LFM2.5-24B-A2B (hybrid conv/attention, 64 experts) | **no** | default (auto fallback), Model Runner V1 | 4 | 16/128 tokens off fp32's argmax (HF bf16: 15/128) | 1.691 (HF bf16: 1.691) |

"Max fp32 logit gap" comes from
[`scripts/teacher_forced_logprobs.py`](scripts/teacher_forced_logprobs.py),
which re-scores every token vLLM generated with HF fp32 under teacher forcing.
The gap is how far each generated token is below HF's argmax, maximised over
all tokens. Greedy string matching is a weak test on these small models,
which hit near-ties within a few tokens, so this is the number to look at.
The native TinyLlama run scores 0.039 by the same measure.

The LFM2.5-24B-A2B row comes from [2.4](#24-a-24b-hybrid-moe-lfm25-24b-a2b):
4 prompts (one of 7936 tokens), 32 tokens each, max_model_len 8192,
re-scored by [`scripts/bf16_baseline.py`](scripts/bf16_baseline.py). This
model is sensitive to bf16, so it is compared with HF's own bf16 run rather
than with a CPU greedy reference, and its gap is the same as HF bf16's.

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

### 2.4 A 24B hybrid MoE: LFM2.5-24B-A2B

[LiquidAI/LFM2.5-24B-A2B](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B)
(`Lfm2MoeForCausalLM`) has 40 layers: 30 gated short convolutions and 10 GQA
attention layers (32 query heads, 8 KV heads). After 2 dense layers, the
feed-forwards are 64-expert MoEs (top-4, sigmoid routing with an fp32 expert
bias). That is 24B parameters, about 2B active per token, and 128k context.
vLLM Neuron has no implementation of it, so the backend serves it, on Model
Runner V1 (hybrid and MoE models).

It needed a paged state for the short convolutions, a Neuron experts
function, and expert sharding (see [Design](#3-design)). At TP 4, a rank holds
13.08 GiB of weights out of 24 GiB of HBM. Only the experts are sharded: the
HF plan of this model shards nothing else.

All runs: bf16, TP 4, greedy, one 4096-token prefill bucket, prefix caching
off ([`scripts/bench_offline.py`](scripts/bench_offline.py)).

**4k context** (max_model_len 4096, 4 sequences), measured with the first
LFM2 version, which kept one conv entry per token:

| | |
|---|---|
| start-up, cold (compiles 1 prefill and 1 decode graph) | 335 s |
| start-up, warm (compiled graphs cached) | 95 s, about 49 s of it loading weights |
| TTFT, 3968-token prompt | 0.735 s |
| decode, 1 sequence | 38.0 ms/token |
| decode, 4 sequences | 38.6 ms/step, 104 tokens/s |
| KV cache | 49,600 tokens |

Each rank loads the whole 48 GB checkpoint, then keeps its shard: hence the
time spent loading weights.

**128k context** (max_model_len 131072, 1 sequence, decode graphs for 32k, 64k
and 128k of context through `decode_context_length_buckets`):

| prompt tokens | TTFT | prefill | decode |
|---|---|---|---|
| 32,512 | 10.5 s | 3.1k tokens/s | 35.6 ms/token |
| 65,280 | 24.9 s | 2.6k tokens/s | 40.5 ms/token |
| 130,816 | 65.9 s | 2.0k tokens/s | 47.3 ms/token |

With max_model_len 8192 (and logprobs on), a 7936-token prompt (2 chunks)
takes 2.3 s.

- Start-up: 136 s warm, 924 s cold (compiling the 4 graphs takes 842 s).
- The KV cache holds 198,432 tokens, 1.5 sequences of 128k. A token takes
  28 KB: 20 KB of attention K/V and 7.5 KB of conv state. With one conv entry
  per token (the first LFM2 version), it took 143 KB, which left room for
  about 49k tokens.
- Prompts longer than 4096 tokens are prefilled in 4096-token chunks. Each
  chunk attends over all the chunks before it, so TTFT grows faster than the
  prompt.
- Decode gathers the whole context of each attention layer with PyTorch ops,
  so a decode step slows down with the context. The context buckets keep a
  32k sequence from paying for 128k.
- TP 4 is enough for 128k: no need to split further.

**Accuracy:**

- [`scripts/bf16_baseline.py`](scripts/bf16_baseline.py) re-scores vLLM's
  greedy tokens under teacher forcing with HF fp32 and HF bf16 (32 tokens per
  prompt). This model is sensitive to bf16 (sigmoid routing with an fp32
  expert bias), so the bar is HF's own bf16 run, not fp32. With the final
  code and max_model_len 8192, prompts go through the NKI segmented attention,
  and the 7936-token prompt takes 2 chunks:

  | prompt | vLLM: tokens not fp32's argmax | vLLM: mean \|logprob − fp32\| | HF bf16: argmax ≠ fp32's | HF bf16: mean \|logprob − fp32\| |
  |---|---|---|---|---|
  | "The capital of France is" | 6/32 | 0.133 | 6/32 | 0.158 |
  | "The future of AI is" | 2/32 | 0.123 | 1/32 | 0.147 |
  | "def fibonacci(n):" | 8/32 | 0.191 | 8/32 | 0.161 |
  | 7936-token prompt | 0/32 | 0.015 | 0/32 | 0.007 |
  | **all** | **16/128** | **0.115** | **15/128** | **0.118** |

  The first LFM2 version, at 4k (3968-token prompt, one chunk), scored the
  same way: 18/128 and 0.23 for vLLM, 17/128 and 0.25 for HF bf16.
- [`scripts/gsm8k_eval.py`](scripts/gsm8k_eval.py), first 100 GSM8K test
  questions, greedy chat, up to 3584 tokens (first LFM2 version, 4k):
  **93/100**. The 7 misses all hit the token limit while still reasoning. It
  took 14.7 min for 82.6k tokens (93 tokens/s, 4 sequences at a time).
- [`scripts/needle_eval.py`](scripts/needle_eval.py), needle in a haystack: a
  7-digit passcode hidden in installed packages' READMEs (dist-info
  METADATA), at depths 10%, 30%, 50%, 70% and 90%, then asked for:

  | context | prompt tokens | passcode in the final answer |
  |---|---|---|
  | 32k | 30,636 | 5/5 |
  | 64k | 63,404 | 5/5 |
  | 128k | 128,939 | 5/5 |

  The model reasons for 200 to 1,600 tokens before it answers, so it gets
  2048. With 512, 8 of 15 answers were cut while the model was still
  reasoning, 7 of them after it had quoted the passcode.
- The 130,816-token prompt (README.md repeated) is continued exactly: the 32
  greedy tokens are the README text that follows it.

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
  - **segmented prefill** (a prompt longer than `max_num_batched_tokens`,
    prefilled in chunks): vLLM Neuron's `NF.segmented_attention`, an NKI
    kernel, reads the cache in segments of `kv_segment_size` tokens. Gathering
    the whole context instead would take 17 GB of fp32 scores per layer for a
    4096-token chunk at 32k;
  - **decode**: each sequence gathers its blocks through its block table
    (unused entries are `-1`) and masks keys by position. Each KV head
    attends for its group of query heads, without repeating K/V.

  It also handles sliding window, soft-cap and attention sinks, which
  transformers passes as kwargs. Plain PyTorch, except segmented prefill
  (soft-capped attention keeps the PyTorch path there too).
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
- **Mixture of experts**: MoE configs request
  `experts_implementation="neuron"`, `neuron_experts_forward` registered in
  transformers' `ExpertsInterface`. Its shapes do not depend on the routing,
  so it compiles into a static graph, unlike the default `grouped_mm` and its
  `eager` fallback:
  - for a few tokens (2 × tokens × top-k ≤ experts, as in decode), it runs
    transformers' `batched_mm`, which gathers each (token, expert) pair's
    weights;
  - past that, the gathered copies outgrow the experts (a 4096-token prefill
    of LFM2.5-24B-A2B would gather 77 GB per layer at TP 4). Every token then
    goes through every expert, as two dense matmuls, scaled by its routing
    weights (zero for the experts it is not routed to).

  No NKI MoE kernel yet.
- **Expert sharding**, from the HF plan: `packed_colwise` slices both halves
  of the fused gate/up weights, `rowwise` slices the input dimension of the
  down projections, and `moe_tp_experts` all-reduces the experts' output.
- **LFM2 short convolutions**: `_PagedShortConv` replaces each gated short
  convolution (`in_proj` → B, C, x; `out_proj(C · conv(B · x))`, kernel 3).
  Its state, the last 2 inputs B · x of a sequence, lives in a key-only paged
  cache that vLLM allocates and manages like K/V: one more `LayerSpec` per
  conv layer, the same block table. Each block keeps 2 entries, position p at
  entry p mod 2 of its block, 1/16 of a per-token cache with 32-token blocks.
  A step takes the window of its own tokens from the step, reads the earlier
  positions from the cache, then writes the last 2 tokens of each sequence.
  Prefill, chunked prefill and batched decode share this one path, and
  vLLM's per-sequence (Mamba-style) state is not needed.

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
- `test_transformers_backend_hybrid.py` runs a tiny random LFM2-MoE through
  the backend (prefill with null-block and `-1` padding, batched decode,
  chunked prefill then decode) and checks its logits against transformers. It
  also checks both experts paths against `eager`, and the sharded experts at
  TP 2 and 4.

The commits are on the
[`transformers-backend` branch of the fork](https://github.com/tengomucho/vllm-neuron/commits/transformers-backend).

## 4. Issues found

1. **All public torch-neuronx 2.33 wheels need glibc 2.43** (Ubuntu 26.04),
   for cp312, cp313 and cp314 alike (GitHub release `v2.33.0` and the Neuron
   pip repo). `torch_neuronx/_C.so` fails to load on Ubuntu 24.04 (glibc
   2.39): `` version `GLIBC_2.43' not found ``. Only 9 float libm symbols
   (`acosf`, `asinf`, `atan2f`, `sqrtf`, `log10f`, ...) need the `GLIBC_2.43`
   version, and glibc 2.39 exports the same functions as `GLIBC_2.2.5`.
   Workaround: [`scripts/patch_glibc_verneed.py`](scripts/patch_glibc_verneed.py)
   rewrites the version-needed entry; the original is kept as `_C.so.orig`.
   The clean fix is a container built from `ubuntu:26.04`, which ships glibc
   2.43: [`docker/Dockerfile`](docker/Dockerfile), tested here (see
   [Docker image](#docker-image)). The Neuron apt repo has a `resolute`
   (26.04) suite with the same runtime (`aws-neuronx-runtime-lib` 2.35.30),
   and the driver runs on the host, so only the container changes.
   Installing 26.04's `libc6` on 24.04 instead would mix releases
   (`libc-bin`, `locales`, the gcc runtime) and is not recommended.
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
11. vLLM replaces the checkpoint's config with its own `Lfm2MoeConfig`
    (`vllm/transformers_utils/configs`), which has no `base_model_tp_plan`.
    The backend falls back to the plan of the transformers model's config
    class.
12. **Model Runner V1 returns no logprobs with async scheduling**, the
    default: requests that ask for logprobs get empty lists. Pass
    `async_scheduling=False`, and set `max_logprobs` in `neuron_config` so that
    the model returns logits. `logprobs=0` returns nothing either: ask for 1.
13. Padding slots differ by side: Model Runner V1 pads `slot_mapping` (writes)
    with the null block, slot 0, and block tables (reads) with -1. The
    `paged_slot_indices` docstring only mentions -1 for writes. The LFM2 conv
    state first took null-block slots for real tokens, which only short
    prompts showed: their decoding degenerated after the first token.
14. Two system libraries are needed but not declared. On a bare Ubuntu 26.04
    they are missing:
    - neuronx-cc's `walrus_driver` links `libarchive.so.13`. Without it, the
      first compilation fails with `error while loading shared libraries:
      libarchive.so.13: cannot open shared object file`;
    - torch_xla's `_XLAC` links `libpython3.14.so.1.0` (see issue 4).

    The Dockerfile installs `libarchive13t64` and `libpython3.14`.
15. **The compile cache is spread over three directories** that point to one
    another:
    - vLLM Neuron keeps the compiled graphs under `VLLM_CACHE_ROOT`
      (`~/.cache/vllm/neuron/`);
    - libtorch-neuronx-lite keeps their NEFFs, and an index of the NKI kernels
      it compiled, in `~/.cache/neuron_libtorch/`;
    - nki keeps the kernel binaries in `/var/tmp/nki-intermediate-cache`
      (`NKI_COMPILE_CACHE_URL`).

    Keeping only some of them makes start-up fail instead of recompiling. Both
    failures showed up in containers:
    - with the graphs but not the NEFFs (only `~/.cache/vllm` kept), a warm
      start fails on every rank with `NEFF for cache_key=... is neither on
      disk at ... nor resident in the compilation cache`;
    - with the kernel index but not the binaries (`~/.cache` kept, not
      `/var/tmp`), the next model that uses the same kernel (here the
      sampler's `cumsum`) fails to compile with `[NCC_EVRF059] Kernel file
      '/var/tmp/nki-intermediate-cache/...colz' ... does not exist on the
      host`.

    The Dockerfile moves the NKI cache to `~/.cache/nki`, so that one volume
    on `~/.cache` keeps all three.

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
# LFM2.5-24B-A2B at TP 4: speed at 4k, then at 32k/64k/128k context:
python bench_offline.py --model LiquidAI/LFM2.5-24B-A2B --tp 4 --out /tmp/lfm.json
python bench_offline.py --model LiquidAI/LFM2.5-24B-A2B --tp 4 --max-model-len 131072 --max-num-seqs 1 \
  --prompt-tokens 32512 65280 130816 --decode-context-buckets 32768 65536
# Accuracy: logprobs against HF fp32 and HF bf16, GSM8K, needles up to 128k:
python bench_offline.py --model LiquidAI/LFM2.5-24B-A2B --tp 4 --logprobs --out /tmp/lfm_lp.json
python bf16_baseline.py --model LiquidAI/LFM2.5-24B-A2B /tmp/lfm_lp.json
curl -o /tmp/gsm8k.json "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset=0&length=100"
python gsm8k_eval.py --model LiquidAI/LFM2.5-24B-A2B --tp 4 /tmp/gsm8k.json
python needle_eval.py --model LiquidAI/LFM2.5-24B-A2B --tp 4 --max-model-len 131072 \
  --context-tokens 32768 65536 131072 --decode-context-buckets 32768 65536
# Unit tests (CPU: attention, TP sharding, LFM2 hybrid), from ~/Dev/vllm-neuron:
VLLM_NEURON_CPU_MODE=1 python -m pytest -q test/unit/model/
# Plain transformers on torch-neuronx (leave TORCH_DEVICE_BACKEND_AUTOLOAD unset):
env -u TORCH_DEVICE_BACKEND_AUTOLOAD NEURON_RT_VISIBLE_CORES=8 python run_qwen3_transformers.py
# Docker image, from ~/Dev/vllm-test (`docker run` commands in 1, Docker image):
docker build -t vllm-neuron-transformers docker/
```

## 6. Gaps and next steps

- **Performance**: decode gathers each sequence's whole block table and
  computes attention with plain ops. vLLM Neuron's
  `decode_context_length_buckets` trims the block table to the decode bucket,
  so the gather reads only that bucket's blocks (used for the 128k runs).
  Next:
  - NKI attention kernels: `flash_attention` for single-step prefill, and a
    paged decode kernel that takes Q/K/V (the current `attention_decode` fuses
    the QKV and output projections, so it doesn't fit the attention
    interface). Segmented prefill already uses the NKI kernel;
  - NKI MoE kernels (`moe_cte`/`moe_tkg`) behind the experts function. LFM2.5
    decode at 38 ms/token is far from what its 2B active parameters need;
  - fused QKV and gate/up projections, and a vocab-sharded LM head
    (`ColumnParallelLinear`), as upstream's fusers do;
  - load only each rank's shard: every rank reads the whole checkpoint (49 s
    of the 95 s warm start-up of LFM2.5-24B-A2B).
- **TP**: the number of attention heads must be divisible by the TP size, and
  KV heads and TP size must divide one another. The backend follows the HF
  plan, so models whose plan shards only the experts (LFM2-MoE) replicate
  their attention, convolutions and caches on every rank. Still missing:
  `packed_colwise` on linear layers (fused gate_up/qkv, e.g. Phi3),
  `colwise_gather_output`, expert parallelism (`ep_*`), and sequence
  parallelism.
- **Sliding window**: a full-length cache is allocated and the window is
  applied by masking. Declaring `sliding_window_size` in the KV spec would
  save memory.
- **Hybrids**: the short-conv state follows LFM2 (`in_proj`, `conv`,
  `out_proj`). Other state-space layers (Mamba, linear attention) would need
  their own paged state.
- **Not covered**: multimodal (`TransformersMultiModalForCausalLM`), pooling,
  speculative decoding, quantization, LoRA, and prefix caching on device.
- **Upstreaming**: the partner asked for a working prototype on a fork first,
  then a discussion of the PR and integration process. The fork is published;
  no PR is open yet.
