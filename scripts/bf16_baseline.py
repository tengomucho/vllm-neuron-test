#!/usr/bin/env python3
"""Compare a vLLM run's accuracy with HF bf16's own, both against HF fp32, on identical prefixes.

teacher_forced_logprobs.py measures how far each vLLM token is from HF fp32's
argmax. Some models (e.g. sigmoid-routed MoE with an expert bias) drift that
far in bf16 already. This script teacher-forces each vLLM sequence through HF
fp32 and HF bf16, and reports, per prompt:

- off-argmax: the tokens where vLLM's token, or HF bf16's argmax, is not
  HF fp32's argmax;
- max gap: how far below HF fp32's best logit that token is;
- |logprob - fp32|: the error of vLLM's logprobs (run with --logprobs), and of
  HF bf16's logprobs, for the same tokens.

Usage:
    python scripts/bf16_baseline.py --model LiquidAI/LFM2.5-24B-A2B run.json
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    models = {
        dtype: AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, dtype)).eval()
        for dtype in ("float32", "bfloat16")
    }
    for r in json.load(open(args.run)):
        prompt_ids = tok(r["prompt"], return_tensors="pt").input_ids
        gen = torch.tensor([r["token_ids"]])
        logits = {}
        with torch.no_grad():
            for dtype, model in models.items():
                out = model(torch.cat([prompt_ids, gen], 1)).logits
                logits[dtype] = out[0, prompt_ids.shape[1] - 1 : -1].float()
        ref = logits["float32"]
        ref_lp = torch.log_softmax(ref, -1).gather(1, gen.T).squeeze(1)
        best = ref.max(-1).values
        bf16_tokens = logits["bfloat16"].argmax(-1, keepdim=True)
        rows = {
            "vllm": (gen.T, torch.tensor(r["logprobs"]) if r.get("logprobs") else None),
            "hf_bf16": (
                bf16_tokens,
                torch.log_softmax(logits["bfloat16"], -1).gather(1, gen.T).squeeze(1),
            ),
        }
        name = r["prompt"] if len(r["prompt"]) < 40 else f"<{prompt_ids.shape[1]}-token prompt>"
        for who, (tokens, lp) in rows.items():
            gap = best - ref.gather(1, tokens).squeeze(1)
            line = (
                f"{name!r:32} {who:8} off-argmax {(gap > 0).sum().item():2d}/{len(gap)}"
                f"  max gap {gap.max().item():.3f}"
            )
            if lp is not None:
                err = (lp - ref_lp).abs()
                line += f"  |logprob - fp32| mean {err.mean():.4f} max {err.max():.4f}"
            print(line, flush=True)


if __name__ == "__main__":
    main()
