# Next steps: get `google/gemma-4-26B-A4B-it` serving on vLLM-Neuron

## TL;DR for the next session

A Gemma4 text-decoder is **implemented and validated correct** in this repo
(`vllm_neuron/model/gemma4/`, branch `feat/gemma4-vllm-neuron`). The two compiler
blockers that prevented on-device serving of the 26B model have been fixed
(commit `78259a3`).  The 26B model now compiles past the `NCC_IGCA024` step and the
`moe_cte OOB` step (observed in run2 log).  **The final gate is whether the full
26B NEFF compile and inference runs end-to-end.**

---

## What works today (validated)

- **CPU mode, real 26B weights** vs HF reference: coherent, correct greedy chat output.
- **On-device (Neuron), `tiny-random/gemma-4-moe`**: token output on-device TP=4 and
  TP=8+EP (small numeric drift from CPU at decode step 6+ is expected — bf16 vs
  float32 greedy threshold with random tiny weights; both Neuron runs agree).
- **On-device, synthetic `gemma4-hd256`** (head_dim=256/512): **CPU-identical output**
  `[80, 337, 337, ...]` at TP=4 after the `torch.where` mask fix — confirms the
  attention fix works for large head dims.
- **26B model**: compilation reached neuronx-cc backend ("Compiling...") — both
  NCC_IGCA024 and moe_cte OOB errors are resolved. Full compile still in progress.

## Files changed (commit 78259a3)

- `vllm_neuron/model/gemma4/model_bf16.py` — two fixes (see below).
- `scripts/build_gemma4_hd256.py` — build synthetic head_dim=256/512 model for
  fast compile testing.
- `scripts/g4_test.py` — unified test script (MODEL/TP/EP/CPU env vars).
- `scripts/sync_gemma4.sh` — sync plugin + test script into container.

---

## Fixes applied (commit 78259a3)

### Fix 1: attention mask — `NCC_IGCA024` (DONE)

**Root cause**: `torch.where(cond, scalar_0_tensor, scalar_-1e30_tensor)` with 0-dim
scalar tensors lowered to undefined IR in neuronx-cc at head_dim 256/512.

**Fix**: replaced with `(~cond).to(float32) * -1e30` in both `_attn_prefill` and
`_attn_decode`. Mathematically identical, but compiles cleanly.

**Verified**: synthetic hd256 model (head_dim=256/512) compiles and produces
CPU-identical output at TP=4.

### Fix 2: moe_cte token count — NKI OOB assertion (DONE)

**Root cause**: The `moe_cte` NKI kernel tiles the token dimension in 128-element NKI
partitions; it asserts `num_tokens >= 128`. With the 64-token test bucket, it fires
`Out-of-bound access ... index range [0, 127] exceed dimension size of 64`.

**Fix**: extended the existing decode einsum fallback to also cover small prefill
buckets (`num_tokens < 128`). In production serving (buckets ≥ 128) the NKI moe_cte
path runs as before. The constant `_NKI_PARTITION = 128` is at the top of the
relevant block.

**Note**: this changes the 64-token bucket from using `shard_on_block` moe_cte to the
einsum, which is numerically identical (validated via the tiny model). Production
buckets of 128+ are unaffected.

### Debug helper (stays in code)

`GEMMA4_STUB_ATTN=1` env var replaces the full attention computation with zero tensors.
Useful for bisecting future compiler issues (if attn stubs compile but the full model
doesn't, attn is the trigger).

---

## Outstanding (if the 26B compile completes successfully)

1. **Run the serving recipe** (see below) with `vllm serve /root/gemma4-text ...`.
2. **Verify coherent output**: curl the chat endpoint, expect "Paris" for the capital
   of France question.
3. **Finalize the skill** at `.claude/skills/port-hf-model-to-vllm-neuron/` with the
   head_dim>128 guidance and moe_cte token-count guidance.

---

## Environment notes (carry over)

- Container `vllm_neuron` (trn2.48xlarge, 4 neuron devices × 4 cores = 16 NeuronCores,
  96 GB device memory shared by 4 cores → ~24 GB usable HBM per core).
- Container bind mounts to host `/home` (NFS) and `/tmp` are **stale/broken**; do NOT
  rely on them. Put model weights on the container's local disk (`/root/...`) via
  `docker cp`.
- Host venv `.venv-hf` has transformers 5.12.1 (+ CPU torch) with the gemma4 source.

**Sync code into the container:**
```bash
bash scripts/sync_gemma4.sh   # clears pycache + NEFF cache, copies all plugin files + test script
```

**Model dirs in the container** (verify: `docker exec vllm_neuron ls -d /root/gemma4-text /root/gemma4-tiny-random /root/gemma4-hd256`):
- `/root/gemma4-text` — 26B text-only dir (real weights, text-only config.json).
- `/root/gemma4-tiny-random` — `tiny-random/gemma-4-moe`, identical architecture,
  tiny dims; fast iteration (compile in ~30s).
- `/root/gemma4-hd256` — synthetic model with head_dim=256/512, tiny weights; fast
  compile test for the head_dim>128 attention path (~1-2 min).

---

## THE blocker (was): two compiler errors (now FIXED)

### NCC_IGCA024 (FIXED)
Gemma4 attention uses plain torch ops at head_dim 256/512 (flash kernel caps at 128).
The `torch.where(cond, 0_scalar, -1e30_scalar)` pattern in `_attn_prefill` and
`_attn_decode` triggered an undefined IR reference in neuronx-cc. Fixed in commit
`78259a3` by using `(~allowed).to(float32) * -1e30` instead.

### moe_cte OOB (FIXED)
With EP=8 (ep_degree=8), tp_degree=1, I_per_rank=704 ≥ 128 → moe_cte NKI kernel is
selected. The NKI kernel tiles the token dimension in 128-element partitions. With the
64-token test bucket, it asserts num_tokens ≥ 128 and fires OOB. Fixed in commit
`78259a3` by falling back to the einsum path when `num_tokens < 128`.

---

## Test commands (in the container via docker exec)

```bash
# After syncing with: bash scripts/sync_gemma4.sh

# CPU reference (tiny, fast):
docker exec vllm_neuron bash -lc 'MODEL=/root/gemma4-tiny-random CPU=1 TP=1 VLLM_NEURON_CPU_MODE=1 python /tmp/g4.py'

# On-device tiny, TP=4:
docker exec vllm_neuron bash -lc 'MODEL=/root/gemma4-tiny-random TP=4 python /tmp/g4.py'

# On-device tiny, TP=8+EP:
docker exec vllm_neuron bash -lc 'MODEL=/root/gemma4-tiny-random TP=8 EP=1 python /tmp/g4.py'

# Synthetic hd256, on-device TP=4 (fast head_dim>128 test):
docker exec vllm_neuron bash -lc 'MODEL=/root/gemma4-hd256 TP=4 python /tmp/g4.py'

# 26B on-device (TP=8+EP, after compile ~10-15 min):
docker exec vllm_neuron bash -lc 'MODEL=/root/gemma4-text TP=8 EP=1 GPU_MEM=0.85 python /tmp/g4.py'

# Serving:
docker exec -it vllm_neuron vllm serve /root/gemma4-text \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85
```

---

## Serving recipe (once the 26B compile is confirmed)

`vllm serve google/gemma-4-26B-A4B-it` will NOT work directly because:
- the HF `architectures` is `Gemma4ForConditionalGeneration` (multimodal) — we
  register the text-only `Gemma4ForCausalLM`; and
- the container's transformers can't parse `model_type: gemma4`.

Serve the **text-only model dir** (`/root/gemma4-text`) and pass EP + HBM flags:

```bash
# inside the container
vllm serve /root/gemma4-text \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85
```

### Verify the server
```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "/root/gemma4-text",
  "messages": [{"role":"user","content":"What is the capital of France? Answer in one word."}],
  "max_tokens": 8, "temperature": 0 }'
# expect "Paris"
```

---

## Pitfalls already solved (don't re-debug)

- RMSNorm is `(x/rms)·weight` (NOT Gemma1/2/3 `·(1+weight)`).
- Per-layer learned `layer_scalar` output multiply (`hidden *= layer_scalar`) — its
  absence gives cos≈1.0 with constant ~14× norm ratio.
- Prefill causal mask must be index-based (`j<=i`), not position-based (bucket
  padding leaks in).
- Decode MoE: use the vectorized local-expert einsum (prefill-only `moe_cte`
  blockwise mapping divides by total_tokens//16, which is 0 for few decode tokens).
- Also use einsum for small prefill buckets (num_tokens < 128): moe_cte NKI
  tiles in 128-element partitions.
- `moe_cte` needs hidden a multiple of 512 → zero-pad expert hidden 2816→3072.
- moe_cte torch fallback must use elementary-op gelu_tanh, not `F.gelu`.
- per_expert_scale folded into `down_proj`; router uses `gamma = router.scale * hidden^-0.5`.
- Attention masks use `-1e30` not `float("-inf")`, AND use additive mask not
  `torch.where(cond, scalar_0, scalar_-1e30)` — the scalar-broadcast form triggers
  NCC_IGCA024 at head_dim > 128.
- RoPE `inv_freq` buffers built with `device="cpu"` (model is constructed under
  `torch.device("meta")`).
- Checkpoint weight keys: the 26B HF checkpoint uses `model.language_model.*`
  prefix (it's a `Gemma4ForConditionalGeneration` checkpoint); the synthetic test
  model saved with `Gemma4ForCausalLM.save_pretrained()` needs a key-remapping step
  (see `scripts/build_gemma4_hd256.py`).

## Memory / context
A fresh session in THIS repo dir auto-loads two memory files with deeper detail:
`gemma4-neuron-architecture` (exact per-op math/quirks) and `gemma4-neuron-port-decisions`
(scope, env, every fix). Read them.
