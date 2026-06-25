# Next steps: get `google/gemma-4-26B-A4B-it` serving on vLLM-Neuron

## TL;DR for the next session

A Gemma4 text-decoder is **implemented and validated correct** in this repo
(`vllm_neuron/model/gemma4/`, branch `feat/gemma4-vllm-neuron`). It runs **token-identical
to HF/CPU on Neuron hardware** for the small same-architecture model `tiny-random/gemma-4-moe`
(at TP=4 and at TP=8 with expert parallelism).

The **only** thing preventing `vllm serve google/gemma-4-26B-A4B-it` from working today is that
**neuronx-cc fails to compile the 26B model** with an internal error:

```
[NCC_IGCA024] Illegal IR, encountered undefined use: inst__I-417-0:_mem_...
neuronx-cc compilation failed with 70.
```

This is **not** a correctness bug and **not** an OOM (expert parallelism fixed memory). It is a
compiler-lowering failure that is **shape-triggered** (max_model_len 256 and 128 both fail at the
same early instruction, so it is not graph-size). The **leading hypothesis** is the 26B model's
**large attention head dims (256 / 512)**, handled by plain-torch ("manual") attention because the
Neuron flash kernel caps head_dim at 128 — the tiny model (head_dim 32/64) compiles and runs fine.
**But this is NOT yet confirmed**: the other big-shape difference is `moe_cte` at `I_per_rank=704`
/ `hidden_pad=3072`. **Step 1 below is to confirm which one it is** — do not assume.

**Primary task:** localize the op that produces the illegal IR, then make it compile (most likely an
NKI attention for head_dim>128, or a torch reformulation of the offending op). Once it compiles, the
26B should serve. Everything else (weights, MoE, EP, norms, RoPE, sampling) is already correct and
validated on-device via the tiny model.

---

## What works today (validated)

- **CPU mode, real 26B weights** vs HF reference: coherent, correct greedy chat output
  (e.g. "Paris"; "The three primary colors are red, yellow, and blue"); per-layer hidden states
  match HF.
- **On-device (Neuron), `tiny-random/gemma-4-moe`** (identical architecture, tiny dims): produces
  **token-identical** output to CPU mode at **TP=4** and at **TP=8 + `enable_expert_parallel`**.
  This exercises the full on-device path: NKI kernels (`moe_cte`, `router`), manual attention,
  hidden-dim padding, decode MoE, `layer_scalar`, RoPE, and EP collectives.
- Three commits on `feat/gemma4-vllm-neuron`:
  - `feat: add gemma4 (gemma-4-moe) text decoder to vllm-neuron`
  - `feat: expert parallelism + on-device support for gemma4`
  - `docs: note neuronx-cc head_dim>128 / decode-MoE pitfalls in porting skill`

## Files
- `vllm_neuron/model/gemma4/` — config.py, hf_config.py, weight_loaders.py, model_bf16.py,
  factory.py, __init__.py.
- Modified plugin files: `vllm_neuron/model/registry.py`, `vllm_neuron/model/__init__.py`,
  `vllm_neuron/__init__.py` (early config registration), `vllm_neuron/functional/moe/moe_cte.py`
  (traceable gelu_tanh in the torch fallback).
- Skill: `.claude/skills/port-hf-model-to-vllm-neuron/` (SKILL.md + references). The skill should
  be finalized after the 26B serves end-to-end.

## Environment notes (carry over)
- Container `vllm_neuron` (trn2.48xlarge, 4 neuron devices × 4 cores = 16 NeuronCores, 96 GB
  device memory shared by 4 cores → ~24 GB usable HBM per core), transformers 4.57.6 (no gemma4).
- The container's bind mounts to host `/home` (NFS) and `/tmp` are **stale/broken**; do NOT rely on
  them. Put model weights on the container's local disk (`/root/...`) via `docker cp`.
- Host venv `.venv-hf` has transformers 5.12.1 (+ CPU torch) with the gemma4 source for reference.
- Sync code into the container with `docker cp vllm_neuron/model/gemma4/. vllm_neuron:<dest>/gemma4/`
  (trailing `/.` — a plain dir copy NESTS as gemma4/gemma4). Also copy any edited plugin files
  (registry.py, model/__init__.py, __init__.py, functional/moe/moe_cte.py). Clear the NEFF cache
  `/root/.cache/vllm/*` between model/shape changes (stale graphs cause confusing errors).

---

## THE blocker to fix: attention for head_dim > 128 (NCC_IGCA024)

Gemma4 attention dims:
- sliding-attention layers: head_dim 256, 8 KV heads.
- full-attention layers (every 6th): head_dim (global) 512, 2 KV heads, `attention_k_eq_v` (V=K),
  proportional/partial RoPE.

`vllm_neuron.functional.flash_attention` / `attention_decode` cap head_dim at 128, so the current
implementation (`Gemma4Attention._attn_prefill` / `_attn_decode` in model_bf16.py) uses plain torch
ops (matmul/softmax/where). That compiles for the tiny model (head_dim 32/64) but the 26B's 256/512
head dims trigger `NCC_IGCA024`.

### Step 1 (DO THIS FIRST): localize the offending op — don't assume head_dim
Each 26B compile is ~10–15 min, so confirm the trigger before rewriting anything. Iterate on a
**small model with head_dim ≥ 256** (recipe in the harness section) so the head_dim>128 path
compiles in ~1–2 min instead of 10–15.
- Run with `XLA_HLO_DEBUG=1 XLA_IR_DEBUG=1` env vars and capture the HLO around the failing
  instruction (`inst__I-…`). This usually names the op.
- Bisect by stubbing: temporarily replace `_attn_prefill`/`_attn_decode` with a trivial stub
  (e.g. `return v.repeat_interleave(groups,1)` reshaped, no QK^T/softmax). If the small-head≥256
  model then compiles → attention is the trigger. If it still fails → the trigger is `moe_cte`
  (`I_per_rank=704`/`hidden_pad=3072`) or another op; bisect there instead.
- The SDK message asks you to file an AWS Neuron issue; worth doing in parallel.

### If attention is the trigger — fixing head_dim > 128
The Neuron flash/decode kernels cap head_dim at 128 (confirmed:
`vllm_neuron/functional/attention/attention_cte.py` docstring "D: Head dimension (max 128)").
**Naive head-dim splitting onto the flash kernel does NOT work**: flash attention fuses the softmax,
so you cannot run it per head_dim chunk and recombine — `softmax(Σ partial QK scores)` is not a
function of per-chunk softmaxed outputs (you would need an online-softmax merge over chunks, which
the fused kernel does not expose). Realistic options, in order of preference:
1. **Find and fix the specific torch op** that lowers to illegal IR (from Step 1). The current
   manual attention is plain matmul/softmax/where; one op/shape is the culprit. A reformulation
   (e.g. avoid the `torch.where` scalar-broadcast mask → use an additive bias tensor; or reshape
   differently) may compile without an NKI kernel. Cheapest if it works.
2. **Write an NKI attention kernel** supporting head_dim 256/512 (use the `neuron-nki-*` agents/
   skills here). Cleanest, heaviest.
3. **Online-softmax head-dim tiling** in torch (accumulate running max/sum across head_dim chunks):
   correct but complex and may hit the same compiler issue.
Whatever you choose, it must preserve the Gemma4 attention contract: **softmax scale = 1.0** (q/k-norm
replaces 1/sqrt(d)); q_norm/k_norm (Gemma RMSNorm `*weight`) before RoPE; **parameterless v_norm on
values** (`gemma_rms_norm(v_raw, None, eps)`); per-layer-type RoPE; **full layers reuse K as V**
(`attention_k_eq_v`: `v_raw = k_raw` the RAW k_proj output, then v_norm, no rope); sliding-window
masking on sliding layers; index-based causal mask. See `model_bf16.py:Gemma4Attention` and the
`gemma4-neuron-architecture` memory file for exact details.

### If `moe_cte` is the trigger
Try `shard_on_i` instead of `shard_on_block`, or smaller `hidden_pad`, or compare against the tiny
model's working moe_cte invocation; the kernel may dislike `I_per_rank=704` or the 3072 pad.

---

## Reproduction harness (the test scripts are NOT in the repo — recreate them)

The session's test scripts lived in an ephemeral scratchpad and are gone. But the **prebuilt model
dirs already exist in the running container** (verify: `docker exec vllm_neuron ls -d
/root/gemma4-text /root/gemma4-tiny-random`):
- `/root/gemma4-text` — 26B text-only dir (real weights, text-only config.json).
- `/root/gemma4-tiny-random` — `tiny-random/gemma-4-moe`, identical architecture, tiny dims.

If the container was recreated, rebuild these per "Build the text-only model dir" below.

**Sync edited code into the container** (the plugin lives at
`/opt/conda/lib/python3.12/site-packages/vllm_neuron`):
```bash
D=/opt/conda/lib/python3.12/site-packages/vllm_neuron
docker exec vllm_neuron rm -rf $D/model/gemma4/__pycache__
docker cp vllm_neuron/model/gemma4/. vllm_neuron:$D/model/gemma4/        # trailing /. — avoids nesting
docker cp vllm_neuron/__init__.py vllm_neuron:$D/__init__.py
docker cp vllm_neuron/model/registry.py vllm_neuron:$D/model/registry.py
docker cp vllm_neuron/model/__init__.py vllm_neuron:$D/model/__init__.py
docker cp vllm_neuron/functional/moe/moe_cte.py vllm_neuron:$D/functional/moe/moe_cte.py
docker exec vllm_neuron sh -c 'rm -rf /root/.cache/vllm/*'                 # clear NEFF cache after changes
```

**Offline test script** (`/tmp/g4.py` in the container) — env vars select mode:
```python
import os
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
tp  = int(os.environ.get("TP", "1"))
cpu = os.environ.get("CPU", "0") == "1"
ep  = os.environ.get("EP", "0") == "1"
kw = dict(model=os.environ["MODEL"], tensor_parallel_size=tp, max_model_len=64,
          max_num_seqs=1, gpu_memory_utilization=0.5, enable_expert_parallel=ep,
          additional_config={"neuron_config": {"num_batched_tokens_buckets": [64],
                                               "num_seqs_buckets": [1]}})
if cpu: kw["enforce_eager"] = True            # CPU mode requires eager (avoids torch.compile)
llm = LLM(**kw)
out = llm.generate([TokensPrompt(prompt_token_ids=[10,20,30,40,50,60,70,80])],
                   SamplingParams(temperature=0.0, max_tokens=8))
print("TOKIDS:", list(out[0].outputs[0].token_ids))
```
Run it:
```bash
# CPU reference (tiny):     deterministic, fast
docker exec vllm_neuron bash -lc 'cd /root && MODEL=/root/gemma4-tiny-random CPU=1 TP=1 VLLM_NEURON_CPU_MODE=1 python /tmp/g4.py'
# On-device tiny, TP=4:     must token-match CPU
docker exec vllm_neuron bash -lc 'cd /root && MODEL=/root/gemma4-tiny-random TP=4 python /tmp/g4.py'
# On-device tiny, TP=8+EP:  must token-match CPU
docker exec vllm_neuron bash -lc 'cd /root && MODEL=/root/gemma4-tiny-random TP=8 EP=1 python /tmp/g4.py'
# 26B on-device (after the fix):
docker exec vllm_neuron bash -lc 'cd /root && MODEL=/root/gemma4-text TP=8 EP=1 GPU_MEM=0.85 python /tmp/g4.py'
```
On a compile error, dump full output (not grep): `docker exec vllm_neuron bash -lc '... python /tmp/g4.py 2>&1' | tail -60`, and look for `NCC_…` / "Top three largest input tensors" / dynamo tracebacks.

**HF reference (host `.venv-hf`)** — load the FULL multimodal class so weights actually load:
```python
import glob, torch
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
SNAP = glob.glob("/home/.../models--google--gemma-4-26B-A4B-it/snapshots/*")[0]
m = Gemma4ForConditionalGeneration.from_pretrained(SNAP, dtype=torch.bfloat16); m.eval()
lm = m.model.language_model                                   # text decoder
out = lm(torch.tensor([[818,5279,529,7001,563]]), output_hidden_states=True)
logits = m.lm_head(out.last_hidden_state)[0,-1]               # compare top-k / per-layer hidden norms
```
Pitfall: loading `Gemma4ForCausalLM` (the text class) with this checkpoint leaves ALL weights MISSING
(key prefix is `model.language_model.…`) → silently random weights → useless reference.

**Synthetic small model with head_dim ≥ 256** (for fast head_dim>128 compile iteration):
```python
import torch, json
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM   # in .venv-hf (transformers 5.x)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
cfg = Gemma4TextConfig(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, head_dim=256, global_head_dim=512,
                       num_global_key_value_heads=2, num_experts=8, top_k_experts=2,
                       moe_intermediate_size=256, intermediate_size=512, vocab_size=1000,
                       layer_types=["sliding_attention","full_attention"], enable_moe_block=True,
                       attention_k_eq_v=True, hidden_size_per_layer_input=0,
                       rope_parameters={"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"},
                                        "full_attention":{"rope_theta":1e6,"rope_type":"proportional","partial_rotary_factor":0.25}})
Gemma4ForCausalLM(cfg).to(torch.bfloat16).save_pretrained("/tmp/g4-hd256")
```
Then build a text-only dir from it like the 26B (flatten not needed — it's already a text config;
just set `architectures=["Gemma4ForCausalLM"]`, `model_type="gemma4_text"`, add a single-file
`model.safetensors.index.json`, copy a tokenizer). NOTE `moe_intermediate_size` must be a multiple
that keeps `I_per_rank ≥ 128` after `/tp_degree`, OR rely on the decode einsum + check moe_cte
fallback — keep it simple for a *compile* test (correctness is checked separately on the real model).

## Memory / context
A fresh session in THIS repo dir auto-loads two memory files with deeper detail:
`gemma4-neuron-architecture` (exact per-op math/quirks) and `gemma4-neuron-port-decisions`
(scope, env, every fix). Read them.

---

## Serving recipe (once attention compiles)

`vllm serve google/gemma-4-26B-A4B-it` will NOT work directly because:
- the HF `architectures` is `Gemma4ForConditionalGeneration` (multimodal) — we register the
  text-only `Gemma4ForCausalLM`; and
- the container's transformers can't parse `model_type: gemma4`.

So serve a **text-only model dir** (build script below) and pass EP + HBM flags:

```bash
# inside the container
vllm serve /root/gemma4-text \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85
```

Why each:
- `--enable-expert-parallel` + `--tensor-parallel-size 8`: the 128 experts (~22.8B of 26.5B params)
  are replicated per rank without EP; EP shards the expert COUNT across ranks. With ep=tp_total=8,
  experts ≈ 6 GB/rank and the MoE kernel's `I_per_rank = moe_intermediate(704)/tp_degree(1) = 704`
  stays ≥128. (At plain TP=4 the model OOMs by ~2 GB; the MoE kernel needs `I_per_rank≥128` which
  caps TP≤4, hence EP is required.)
- `--gpu-memory-utilization 0.85`: vLLM over-allocates the KV cache to fill HBM, and the compiled
  graph includes the KV tensors as inputs; cap it so weights+KV+activations fit ~24 GB.
  (`--num-gpu-blocks-override` is IGNORED by the neuron runner — use gpu-memory-utilization.)

### Build the text-only model dir (`/root/gemma4-text`)
This was already built this session, but here is the recipe to recreate it:
1. Download text-relevant files of `google/gemma-4-26B-A4B-it` (safetensors, index, tokenizer,
   generation_config) — already in the HF cache; `docker cp` the resolved files to the container's
   LOCAL disk (mounts are broken).
2. Write `config.json` = the original `text_config` flattened to top level, with
   `architectures=["Gemma4ForCausalLM"]`, `model_type="gemma4_text"`, `tie_word_embeddings=true`.
3. Patch `tokenizer_config.json`: `extra_special_tokens` is a list (newer transformers) → set to
   `{}` for transformers 4.57.6.
4. **Chat template**: the tokenizer ships no chat template that vLLM 4.57-era accepts, so
   `llm.chat()` / the OpenAI chat endpoint fails. Provide one: either add a Gemma turn-marker Jinja
   `chat_template` to `tokenizer_config.json` / a `chat_template.jinja`, or pass
   `--chat-template <file>` to `vllm serve`. The format is
   `<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n`.

### Verify the server
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/root/gemma4-text",
  "messages": [{"role":"user","content":"What is the capital of France? Answer in one word."}],
  "max_tokens": 8, "temperature": 0 }'
# expect "Paris"
```

---

## Verification procedure (mirrors how this was validated — feeds the skill)

1. **CPU correctness**: `VLLM_NEURON_CPU_MODE=1` + `enforce_eager=True`, `LLM(model=/root/gemma4-text,
   tensor_parallel_size=1, ...)`. Compare greedy output / per-layer hidden states to the real HF
   model run in `.venv-hf` (load the FULL `Gemma4ForConditionalGeneration` so weights actually load;
   loading `Gemma4ForCausalLM` with this checkpoint silently leaves all weights MISSING). Feed
   identical `prompt_token_ids` via `TokensPrompt`. Per-layer cosine ≈ 1.0; first greedy token matches
   HF top-2 (bf16 noise). Already PASSING.
2. **On-device kernel-path correctness (tiny)**: run `tiny-random/gemma-4-moe` on-device and confirm
   token-identical to its CPU output, at TP=4 and TP=8+EP. Already PASSING.
3. **On-device 26B**: after the attention fix, serve and confirm coherent output and (ideally)
   greedy-token agreement with HF on short factual prompts. **This is the remaining gate.**

The architecture details and the debugging methodology are captured in the skill at
`.claude/skills/port-hf-model-to-vllm-neuron/` — once the 26B serves, finalize the skill with the
head_dim>128 attention guidance (the checklist already flags it).

## Pitfalls already solved (don't re-debug)
- RMSNorm is `(x/rms)·weight` (NOT Gemma1/2/3 `·(1+weight)`).
- Per-layer learned `layer_scalar` output multiply (`hidden *= layer_scalar`) — easy to miss; its
  absence gives cos≈1.0 with a constant ~14× norm ratio.
- Prefill causal mask must be index-based (`j<=i`), not position-based (bucket padding leaks in).
- Decode MoE: `moe_cte`/`build_blockwise_mapping` is prefill-only (`f_len=min(128,total_tokens//16)`
  → modulo-by-zero for few decode tokens) → use the vectorized local-expert einsum (already done).
- `moe_cte` needs hidden a multiple of 512 → zero-pad expert hidden 2816→3072 (already done).
- moe_cte torch fallback must use elementary-op gelu_tanh, not the patched `F.gelu` (dynamo-skipped).
- per_expert_scale folded into `down_proj`; router uses `gamma = router.scale * hidden^-0.5`.
- Attention masks use `-1e30`, not `float("-inf")`.
- RoPE `inv_freq` buffers built with `device="cpu"` (model is constructed under `torch.device("meta")`).
