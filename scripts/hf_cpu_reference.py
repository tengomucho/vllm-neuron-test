#!/usr/bin/env python3
"""CPU greedy reference with plain transformers, same prompts as vllm_offline.py.

Usage:
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python scripts/hf_cpu_reference.py --model Qwen/Qwen3-0.6B --out ref.json
"""
import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_offline import PROMPTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype))
    model.eval()
    results = []
    for prompt in PROMPTS:
        enc = tok(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False)
        ids = out[0, enc["input_ids"].shape[1]:].tolist()
        text = tok.decode(ids, skip_special_tokens=True)
        print(f"[ref] prompt: {prompt!r}\n[ref] output: {text!r}", flush=True)
        results.append({"prompt": prompt, "token_ids": ids, "text": text})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
