# Gemma4 vLLM-Neuron: COMPLETE

## Status: DONE ✓

`google/gemma-4-26B-A4B-it` is **fully serving** on trn2.48xlarge via `vllm serve`
(branch `feat/gemma4-vllm-neuron`). The OpenAI-compatible REST API returns coherent,
correct text. All compiler blockers and runtime issues are resolved.

**Validated output** (France question, `vllm serve` via curl):
```
"The capital of France is Paris."
```

**No outstanding blockers** for single-sequence serving (max_num_seqs=1).
Multi-sequence (batch > 1) requires extending `_attn_decode` to handle per-sequence
k_ctx — not needed for current deployment.

---

## What works today (validated)

- **CPU mode, real 26B weights** vs HF reference: coherent, correct greedy chat output.
- **On-device (Neuron), `tiny-random/gemma-4-moe`**: token output on-device TP=4 and
  TP=8+EP (small numeric drift from CPU at decode step 6+ is expected — bf16 vs
  float32 greedy threshold with random tiny weights; both Neuron runs agree).
- **On-device, synthetic `gemma4-hd256`** (head_dim=256/512): **CPU-identical output**
  `[80, 337, 337, ...]` at TP=4 after the `torch.where` mask fix — confirms the
  attention fix works for large head dims.
- **26B model, g4_test.py (Run 15)**: `TEXT: 'th\n\nThe capital of France is Paris'`
  — Paris confirmed in output. No OOB, no degenerate repetition.
- **26B model, `vllm serve` (curl)**: `"The capital of France is Paris."` — REST API
  serving confirmed working end-to-end.

## Files changed

- `vllm_neuron/model/gemma4/model_bf16.py` — all fixes (attention mask, k_ctx linear buffer, padding guard).
- `vllm_neuron/vllm/worker/neuron_model_runner.py` — `_pregather_decode_kv` no-op; `bind_kv_cache` passes `max_ctx_len`.
- `scripts/build_gemma4_hd256.py` — build synthetic head_dim=256/512 model for fast compile testing.
- `scripts/g4_test.py` — unified test script (MODEL/TP/EP/CPU env vars).
- `scripts/sync_gemma4.sh` — sync plugin + test script into container.

---

## Fixes applied

### Fix 1: attention mask — `NCC_IGCA024`

`torch.where(cond, scalar_0_tensor, scalar_-1e30_tensor)` with 0-dim scalar tensors
lowered to undefined IR in neuronx-cc at head_dim 256/512. Replaced with
`(~cond).to(float32) * -1e30` (additive mask, mathematically identical).

### Fix 2: moe_cte token count — NKI OOB assertion

The `moe_cte` NKI kernel tiles the token dimension in 128-element NKI partitions.
Small prefill buckets (< 128 tokens) cause OOB. Extended the existing decode einsum
fallback to also cover small prefill buckets; production buckets ≥ 128 still use NKI.

### Fix 3: decode HBM OOM (`NCC_EOOM002`)

`self.k_cache[block_table]` inside the NEFF created ~629 MB intermediate per layer
(37.6 GB total). Fixed with per-layer `k_ctx`/`v_ctx` linear buffers `[max_ctx_len+1,
nkv, hd]` written by `_write_cache` and read directly by `_attn_decode` — no gather
inside the NEFF. The +1 is a scratch slot for padding (see Fix 5).

### Fix 4: DGE out-of-bound (slot_mapping as k_ctx index)

`slot_mapping` holds PHYSICAL block-pool addresses (can be >> max_ctx_len=64).
Using it as k_ctx scatter index → DGE scatter OOB at runtime.
Fixed by using `positions` (logical, always in [0, max_model_len-1]) with
`clamp(max=max_ctx-1)` for decode warmup where context_len=256.

### Fix 5: padding k_ctx corruption

Padding tokens have `slot_mapping == -1` but `positions = last_real_token_pos`.
Without a guard, `k_ctx[last_pos]` was overwritten → degenerate repetitive output.
Fix: `torch.where(slot_mapping == -1, scratch_idx, positions.clamp(...))` redirects
padding writes to `k_ctx[max_ctx_len]` (the scratch element). `_attn_decode` reads
only `k_ctx[:max_ctx_len]` (excludes scratch).

### Debug helper

`GEMMA4_STUB_ATTN=1` env var replaces attention with zeros — useful for bisecting
compiler issues.

### After fixing the KV pre-gather
1. **Verify coherent output**: run the 26B test again, expect coherent (not garbage) text.
2. **Run the serving recipe** with `vllm serve /root/gemma4-text ...`.
3. **Verify serving**: curl the chat endpoint, expect "Paris" for the capital question.
4. **Finalize the skill** at `.claude/skills/port-hf-model-to-vllm-neuron/`.

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

# 26B on-device (after ~10-15 min compile, uses cached NEFFs after first run):
docker exec vllm_neuron bash -lc 'MODEL=/root/gemma4-text TP=8 EP=1 GPU_MEM=0.85 python /tmp/g4.py'
```

---

## Serving recipe

`vllm serve google/gemma-4-26B-A4B-it` will NOT work directly because:
- the HF `architectures` is `Gemma4ForConditionalGeneration` (multimodal) — we
  register the text-only `Gemma4ForCausalLM`; and
- the container's transformers can't parse `model_type: gemma4`.

Serve the **text-only model dir** (`/root/gemma4-text`) and pass EP + HBM flags:

```bash
# inside the container (max-num-seqs=1; batch>1 requires per-seq k_ctx extension)
vllm serve /root/gemma4-text \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --max-model-len 64 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 \
  --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [64], "num_seqs_buckets": [1]}}'
```

### Verify the server
```bash
curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "/root/gemma4-text",
  "prompt": "What is the capital of France? Answer in one word.",
  "max_tokens": 16, "temperature": 0}'
# expect "The capital of France is Paris."
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
- **Decode HBM OOM (NCC_EOOM002)**: `self.k_cache[block_table]` inside the compiled
  decode graph creates a full-pool intermediate (~629 MB per layer → 37.6 GB total).
  Fix: per-layer `k_ctx`/`v_ctx` linear buffers `[max_ctx_len+1, nkv, hd]`. Written
  by `_write_cache` (using logical `positions`, clamped — not `slot_mapping` which is
  PHYSICAL block-pool address and can be >> max_ctx_len). Read by `_attn_decode`
  directly, no gather inside NEFF. The +1 scratch element absorbs padding writes
  (`slot_mapping==-1` guard with `torch.where`); `_attn_decode` reads only
  `k_ctx[:max_ctx_len]`.
- **k_cache cross-dtype view (historical)**: in the non-native backend, k_cache is an
  int8 raw tensor reinterpreted as bfloat16 on `neuron:0`. Fancy indexing from Python
  host fails. This was bypassed by the linear context buffer approach (Fix 3 above)
  which avoids host-side gather entirely.
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
