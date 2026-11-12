#!/usr/bin/env python3
"""Small GSM8K check through vLLM Neuron: greedy chat completions, exact match on the final number.

The questions come from the datasets-server JSON of openai/gsm8k:
    curl -o gsm8k.json "https://datasets-server.huggingface.co/rows?dataset=openai/gsm8k&config=main&split=test&offset=0&length=100"

Usage:
    python scripts/gsm8k_eval.py --model LiquidAI/LFM2.5-24B-A2B --tp 4 gsm8k.json
"""
import argparse
import json
import re
import time

from vllm import LLM, SamplingParams

INSTRUCTION = "Solve the problem. Give the final answer as a number in \\boxed{}."


def final_number(text: str) -> str | None:
    """The last \\boxed{} content, else the last number, of the answer (after the reasoning)."""
    answer = text.split("</think>")[-1]
    boxed = re.findall(r"\\boxed\{([^}]*)\}", answer)
    candidates = boxed or re.findall(r"-?[\d,]*\.?\d+", answer)
    if not candidates:
        return None
    number = re.findall(r"-?[\d,]*\.?\d+", candidates[-1])
    return number[-1].replace(",", "").rstrip(".") if number else None


def same(a: str | None, b: str) -> bool:
    try:
        return a is not None and float(a) == float(b)
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=3584)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    rows = [r["row"] for r in json.load(open(args.rows))["rows"]][: args.limit]
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [args.max_num_batched_tokens],
                "num_seqs_buckets": [args.max_num_seqs],
            }
        },
    )
    messages = [[{"role": "user", "content": f"{r['question']}\n\n{INSTRUCTION}"}] for r in rows]
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    t0 = time.perf_counter()
    outputs = llm.chat(messages, params, use_tqdm=False)
    elapsed = time.perf_counter() - t0

    results, correct, truncated, tokens = [], 0, 0, 0
    for row, out in zip(rows, outputs):
        completion = out.outputs[0]
        expected = row["answer"].split("####")[-1].strip().replace(",", "")
        got = final_number(completion.text)
        ok = same(got, expected)
        correct += ok
        truncated += completion.finish_reason == "length"
        tokens += len(completion.token_ids)
        results.append({"expected": expected, "got": got, "ok": ok,
                        "finish_reason": completion.finish_reason, "text": completion.text})
    print(
        f"[gsm8k] {correct}/{len(rows)} correct ({100 * correct / len(rows):.1f}%), "
        f"{truncated} truncated at {args.max_tokens} tokens, {tokens} tokens generated "
        f"in {elapsed:.0f}s ({tokens / elapsed:.0f} tok/s)",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
