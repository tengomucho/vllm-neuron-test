# Per-component HF-vs-implementation checklist

Verify each against the authoritative HF `modeling_*.py` — do not assume conventions
carry over between model families.

## Norms
- [ ] RMSNorm scale convention: `(x/rms)·weight` vs `(x/rms)·(1+weight)`? (Gemma1/2/3 use
      `1+weight`; Gemma4 uses plain `weight`.) Wrong choice compounds → saturates softcap.
- [ ] Variance in fp32; eps inside or outside the sqrt; over which dim.
- [ ] Parameterless norms (e.g. a value-norm with `with_scale=False`) — no weight in ckpt.
- [ ] How many norms per layer and exactly where each is applied (pre/post attn, pre/post FFN,
      sandwich norms, separate norms for parallel branches).

## Attention
- [ ] Softmax scale: `1/sqrt(head_dim)` or `1.0` (some qk-norm models use 1.0).
- [ ] q_norm / k_norm / v_norm — applied before/after RoPE? over head_dim?
- [ ] Hybrid layer types (sliding vs full) with different head_dim, kv-head count, RoPE.
- [ ] RoPE style: interleaved vs split-half (rotate_half); partial rotary; proportional/yarn;
      per-layer-type theta. Position 0 is RoPE-identity, so RoPE bugs only show at pos>0.
- [ ] GQA grouping order must match (repeat_kv contiguous vs interleaved).
- [ ] KV reuse tricks (e.g. attention_k_eq_v: V = normed raw-K, no v_proj).
- [ ] **Causal mask by token INDEX (`j<=i`), not by the `positions` tensor** — bucket padding
      tokens otherwise leak in at non-zero positions.

## MLP / MoE
- [ ] Activation (silu / gelu / gelu_tanh / swish) and whether the on-device kernel supports it.
- [ ] Dense MLP and MoE in parallel (summed) vs one-or-the-other.
- [ ] Router: norm? extra scale vector? `* hidden^-0.5`? softmax-before-topk + renorm vs after?
      Fold any router pre-scale into `NF.router`'s `gamma`.
- [ ] per-expert output scale → fold into `down_proj` at load (≡ scaling the routing weight).
- [ ] Expert tensor layout in checkpoint (stacked 3D `[E,2I,H]`/`[E,H,I]` vs per-expert keys);
      gate/up split orientation; transpose for the kernel layout `[E,H,2,I_TP]` / `[E,I_TP,H]`.

## Embeddings / head / output
- [ ] Embedding scaling (e.g. `* sqrt(hidden)`), done in the checkpoint dtype (bf16 rounding).
- [ ] Tied lm_head (map `lm_head.weight` → embedding key).
- [ ] Final logit softcapping (`tanh(logits/cap)*cap`).
- [ ] **Per-layer output scalar** (e.g. Gemma4 `layer_scalar`, applied `hidden *= layer_scalar`).
      Easy to miss; cosine=1.0 with a constant norm ratio is the signature.
- [ ] PLE / per-layer-input embeddings — only if `hidden_size_per_layer_input > 0`.

## Device/compile (separate from CPU correctness)
- [ ] Attention head_dim > 128 → can't use the flash/decode NKI kernel; plain torch ops work
      on CPU and for head_dim<=128 on-device, but at large head dims (256/512) neuronx-cc may
      fail with `[NCC_IGCA024] Illegal IR, encountered undefined use`. Localize the offending op
      first (XLA_HLO_DEBUG=1 + stub-out bisection); fixes are an NKI attention for head_dim>128 or
      a torch reformulation of the bad op — NOT naive head-dim splitting onto the flash kernel
      (flash fuses softmax, so per-chunk scores can't be recombined without an online-softmax merge
      the kernel doesn't expose). Validate any kernel path on a tiny same-architecture model first
      (e.g. tiny-random/<arch>), and reproduce the >128 path on a small synthetic head_dim=256 model.
- [ ] MoE: moe_cte/build_blockwise_mapping is PREFILL-ONLY (`f_len=min(128,total_tokens//16)`
      → modulo-by-zero for the few decode tokens). Use a vectorized local-expert einsum for decode.
- [ ] MoE `moe_cte` kernel requires the hidden dim be a multiple of 512 → zero-pad the expert
      hidden dim (and unpad the output); a numerical no-op. (gpt_oss pads hidden globally.)
- [ ] Buffers built in `__init__` (RoPE inv_freq) must use `device="cpu"` (model on meta device).
- [ ] CPU-mode debugging: `VLLM_NEURON_CPU_MODE=1` + `enforce_eager=True`. Device run: neither.
- [ ] Attention masks: use a large finite negative (`-1e30`), not `float("-inf")` (Neuron-friendlier).
- [ ] HBM budget: per-rank graph must fit ~24GB on Trn2 (`NCC_EVRF009` = OOM). MoE experts
      replicated per rank unless EP is enabled → raise TP (shards weights/intermediate) or
      implement expert parallelism (ep_degree) to shard the expert count. Roughly weights/rank
      ≈ total_params·2 / TP bytes.
- [ ] **Decode HBM OOM (NCC_EOOM002)**: `self.k_cache[block_table]` inside the compiled decode
      graph forces neuronx-cc to create a full-pool intermediate (~num_blocks × nkv × block_size × hd).
      For 26B at TP=8 this is ~629 MB per layer; 30 layers × 2 (k+v) = 37.6 GB → OOM.
      **Fix (linear context buffer)**: Add per-layer `k_ctx`/`v_ctx` tensors of shape
      `[max_ctx_len + 1, nkv, hd]` to the attention module (the +1 is a scratch slot for padding).
      In `bind_kv_cache`, allocate these on the same device/dtype as the paged cache.
      In `_write_cache`, fill them using logical `positions` as the scatter index (NOT
      `slot_mapping` which is a PHYSICAL block-pool address and can be >> max_ctx_len).
      In `_attn_decode`, read `k_ctx[:max_ctx_len]` directly — no gather inside the NEFF.
- [ ] **Padding token KV corruption**: Padding tokens have `slot_mapping == PAD_SLOT_ID (-1)`
      but `positions == last_real_token_position`. Without a guard, `_write_cache` overwrites
      real token KVs in k_ctx, producing degenerate repetitive output at decode.
      Fix: `is_pad = (slot_mapping == -1)`; compute
      `ctx_pos = torch.where(is_pad, torch.full_like(positions, max_ctx_len), positions.clamp(max=max_ctx_len-1)).long()`
      and scatter-write to `ctx_pos`. The scratch element at index `max_ctx_len` is never
      read by `_attn_decode` (which reads `k_ctx[:max_ctx_len]`).
- [ ] **Decode warmup positions OOB**: `_build_decode_synthetic_inputs` uses `context_len=256`
      by default → `positions=[255]` during warmup. If `max_ctx_len=64`, writing to
      k_ctx[255] would OOB. The `clamp(max=max_ctx_len-1)` in the ctx_pos computation above
      prevents this — verify it is in place before the `torch.where`.
