#!/usr/bin/env python3
"""Offline vLLM Neuron smoke test: greedy-generate a few prompts.

Usage:
    python scripts/vllm_offline.py --model Qwen/Qwen3-0.6B --tp 1
    python scripts/vllm_offline.py --model Qwen/Qwen3-0.6B --model-impl transformers
"""
import argparse
import json
import time

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "The future of AI is",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--model-impl", default="auto")
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=2)
    parser.add_argument("--max-num-batched-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--num-blocks", type=int, default=None, help="num_gpu_blocks_override")
    parser.add_argument("--repeat", type=int, default=1, help="generate passes (timing)")
    parser.add_argument("--no-ods", action="store_true", help="disable on-device sampling")
    parser.add_argument("--out", default=None, help="write generated token ids as JSON")
    args = parser.parse_args()

    neuron_config = {
        "num_batched_tokens_buckets": [args.max_num_batched_tokens],
        "num_seqs_buckets": [args.max_num_seqs],
    }
    if args.no_ods:
        neuron_config["on_device_sampling_config"] = None

    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        model_impl=args.model_impl,
        num_gpu_blocks_override=args.num_blocks,
        additional_config={"neuron_config": neuron_config},
    )
    print(f"[smoke] engine ready in {time.time() - t0:.1f}s", flush=True)

    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0, logprobs=0)
    for _ in range(args.repeat):
        t0 = time.time()
        outputs = llm.generate(PROMPTS, params)
        print(f"[smoke] generated in {time.time() - t0:.1f}s", flush=True)
    results = []
    for out in outputs:
        text = out.outputs[0].text
        print(f"[smoke] prompt: {out.prompt!r}\n[smoke] output: {text!r}", flush=True)
        token_ids = list(out.outputs[0].token_ids)
        logprobs = None
        if out.outputs[0].logprobs:
            logprobs = [lp[t].logprob for t, lp in zip(token_ids, out.outputs[0].logprobs)]
        results.append({"prompt": out.prompt, "token_ids": token_ids, "text": text, "logprobs": logprobs})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
