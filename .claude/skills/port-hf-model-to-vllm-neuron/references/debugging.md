# Debugging accuracy by comparison against HF (the method that works)

Gibberish output almost always means a numerical bug, not a crash. Do NOT guess —
compare against the real HF model layer by layer. This method localized 3 distinct
bugs in the Gemma4 port in a few iterations.

## Setup
- **Run the real HF model on CPU** in a host venv with a recent transformers + CPU torch
  (`uv pip install transformers torch --index-url https://download.pytorch.org/whl/cpu`).
- **Load it correctly.** For a multimodal checkpoint, load the FULL
  `*ForConditionalGeneration` (its keys match the checkpoint) and use
  `model.model.language_model` + `model.lm_head`. Loading the text-only `*ForCausalLM`
  with the multimodal checkpoint silently leaves ALL weights MISSING (key prefix
  mismatch) → it runs with random weights → a useless reference (tell-tale sign: every
  RMSNorm output norm is identical to `sqrt(hidden)` because norm weights defaulted to 1).
- **Feed identical token ids to both.** In vLLM use
  `TokensPrompt(prompt_token_ids=[...])` so positions/tokenization match HF exactly.
- Run vLLM in CPU mode eager: `VLLM_NEURON_CPU_MODE=1` + `enforce_eager=True`.

## What to capture and compare
1. **Per-layer hidden vectors** (not just norms): dump `hidden_states[layer][0, :5]` from
   HF (`output_hidden_states=True`) and the same from your model (collect each decoder
   layer output into a list during forward; guard the dump with `is_prefill` so a later
   decode step doesn't overwrite it). Save to `.pt`, `docker cp` out, compare per-layer
   **cosine similarity** and norm.
2. Interpreting the cosine table:
   - **cos ≈ 1.0 but norm off by factor k** → a pure scalar bug (missing/extra
     per-layer or per-tensor multiply). Find the constant k in the checkpoint (e.g.
     Gemma4's `layer_scalar` ≈ 1/k).
   - **cos drops at layer N** → a real computation bug introduced at layer N. If N is
     the first layer of a special type (e.g. first full-attention layer), suspect that
     type's RoPE/head-dim/kv handling.
   - **cos drops only at non-zero token positions** (position 0 fine) → attention bug
     (RoPE or causal mask). Position 0 has position_id 0 ⇒ RoPE identity ⇒ trivial
     attention, so it masks attention bugs.
3. **Sub-block norms** at a fixed position to localize within a layer (attn vs dense MLP
   vs MoE vs each norm). Caveat: HF norm modules output 3D `[B,S,H]`, but some modules
   (e.g. MoE experts) return 2D `[T,H]` — index a token *vector* (`t[0]`), not a scalar
   element (`t[0,0]` on a 2D tensor), or your "norm" is one number.
4. **Final check**: top-k logits and greedy continuation. Note instruct models often
   **degenerate/loop under greedy on raw (non-chat) prompts** — verify the reference HF
   loops too before calling it a bug. Use the chat template (or hand-built turn markers)
   for a coherent demo.

## Bugs this caught (Gemma4)
- RMSNorm `·(1+weight)` vs `·weight` (compounds across many norms → logits saturate softcap).
- Position-based causal mask let bucket-padding leak into attention at non-zero positions
  → use index-based causal (`key_index <= query_index`) for a contiguous prefill.
- Missing per-layer learned output scalar `layer_scalar` → every layer ~1/scalar too big
  (cos 1.0, norm 14× off).
