#!/usr/bin/env python3
"""Score a vLLM run's generated tokens with HF transformers (CPU, teacher forcing).

For each prompt, compares the logprob vLLM reported for each token it generated
with the logprob HF gives the same token after the same prefix, and checks that
each vLLM token is HF's argmax or within `--tie` logits of it.

Usage:
    python scripts/teacher_forced_logprobs.py --model Qwen/Qwen3-0.6B run.json
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--tie", type=float, default=0.1, help="logit gap counted as a tie")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).eval()
    ok = True
    for r in json.load(open(args.run)):
        prompt_ids = tok(r["prompt"], return_tensors="pt").input_ids
        gen = torch.tensor([r["token_ids"]])
        with torch.no_grad():
            logits = model(torch.cat([prompt_ids, gen], 1)).logits[0, prompt_ids.shape[1] - 1 : -1].float()
        logprobs = torch.log_softmax(logits, -1)
        hf_lp = logprobs.gather(1, gen.T).squeeze(1)
        gap = logits.max(-1).values - logits.gather(1, gen.T).squeeze(1)
        worst = gap.max().item()
        line = f"{r['prompt']!r}: max HF logit gap to the vLLM token {worst:.3f} (tokens off-argmax: {(gap > 0).sum().item()})"
        if r.get("logprobs"):
            diff = (torch.tensor(r["logprobs"]) - hf_lp).abs()
            line += f"; |logprob vLLM - HF| mean {diff.mean():.4f} max {diff.max():.4f}"
        ok &= worst <= args.tie
        print(("OK   " if worst <= args.tie else "FAIL ") + line)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
