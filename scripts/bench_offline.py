#!/usr/bin/env python3
"""Offline vLLM Neuron benchmark: start-up, TTFT and decode time, plus greedy outputs.

Prefill is compiled for one bucket of --max-num-batched-tokens tokens and decode
for one bucket of --max-num-seqs sequences. After a warm-up pass:

- TTFT: a long prompt (--prompt-tokens tokens) with max_tokens=1;
- decode: the same prompt with 1 + --decode-tokens tokens (EOS ignored): the
  time past TTFT, per token, for 1 sequence and for --max-num-seqs at once;
- greedy outputs of short prompts and of the long prompt, with logprobs if --logprobs, for
  scripts/teacher_forced_logprobs.py (--out).

Usage:
    python scripts/bench_offline.py --model LiquidAI/LFM2.5-24B-A2B --tp 4 --out run.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

SHORT_PROMPTS = [
    "The capital of France is",
    "The future of AI is",
    "def fibonacci(n):",
]
LONG_TEXT = Path(__file__).resolve().parent.parent / "README.md"


def long_prompt(tokenizer, num_tokens: int) -> str:
    """A prefix of README.md that tokenizes to about num_tokens tokens."""
    text = LONG_TEXT.read_text() * 4
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return text[: enc["offset_mapping"][num_tokens - 1][1]]


def timed(llm, prompts, params, repeat):
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        times.append(time.perf_counter() - t0)
    return statistics.median(times), outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--model-impl", default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--prompt-tokens", type=int, default=3968)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--gen-tokens", type=int, default=32, help="greedy tokens per accuracy prompt")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--logprobs", action="store_true", help="return logprobs (gathers logits on device)")
    parser.add_argument("--out", default=None, help="write the greedy outputs as JSON")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = long_prompt(tokenizer, args.prompt_tokens)
    num_prompt_tokens = len(tokenizer(prompt).input_ids)

    neuron_config = {
        "num_batched_tokens_buckets": [args.max_num_batched_tokens],
        "num_seqs_buckets": [args.max_num_seqs],
    }
    if args.logprobs:
        neuron_config["max_logprobs"] = -1

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        model_impl=args.model_impl,
        # Model Runner V1 drops logprobs with async scheduling.
        async_scheduling=not args.logprobs,
        additional_config={"neuron_config": neuron_config},
    )
    startup = time.perf_counter() - t0
    print(f"[bench] engine ready in {startup:.1f}s", flush=True)

    first = SamplingParams(max_tokens=1, temperature=0.0)
    decode = SamplingParams(max_tokens=1 + args.decode_tokens, temperature=0.0, ignore_eos=True)
    t0 = time.perf_counter()
    llm.generate([prompt], decode, use_tqdm=False)
    print(f"[bench] first generate (warm-up) in {time.perf_counter() - t0:.1f}s", flush=True)

    ttft, _ = timed(llm, [prompt], first, args.repeat)
    t_one, _ = timed(llm, [prompt], decode, args.repeat)
    t_all, _ = timed(llm, [prompt] * args.max_num_seqs, decode, args.repeat)
    ttft_all, _ = timed(llm, [prompt] * args.max_num_seqs, first, args.repeat)
    step_one = (t_one - ttft) / args.decode_tokens
    step_all = (t_all - ttft_all) / args.decode_tokens
    results = {
        "prompt_tokens": num_prompt_tokens,
        "startup_s": startup,
        "ttft_s": ttft,
        "decode_ms_per_token_bs1": step_one * 1e3,
        f"decode_ms_per_step_bs{args.max_num_seqs}": step_all * 1e3,
        f"decode_tokens_per_s_bs{args.max_num_seqs}": args.max_num_seqs / step_all,
    }
    print(f"[bench] {json.dumps(results)}", flush=True)

    params = SamplingParams(max_tokens=args.gen_tokens, temperature=0.0, logprobs=1)
    outputs = llm.generate(SHORT_PROMPTS + [prompt], params, use_tqdm=False)
    runs = []
    for out in outputs:
        token_ids = list(out.outputs[0].token_ids)
        logprobs = [lp[t].logprob for t, lp in zip(token_ids, out.outputs[0].logprobs)]
        text = out.outputs[0].text
        print(f"[bench] prompt: {out.prompt[-60:]!r}\n[bench] output: {text!r}", flush=True)
        runs.append({"prompt": out.prompt, "token_ids": token_ids, "text": text, "logprobs": logprobs})
    if args.out:
        with open(args.out, "w") as f:
            json.dump(runs, f, indent=1)


if __name__ == "__main__":
    main()
