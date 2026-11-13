#!/usr/bin/env python3
"""Needle in a haystack through vLLM Neuron: find a passcode hidden in a long context.

The haystack is the text of the installed packages' METADATA files (their
READMEs), in a fixed order. A sentence with a random 7-digit passcode is put
at several depths of haystacks of several lengths, and the model is asked for
it (chat template, greedy). A needle is found when the passcode is in the final
answer, after the reasoning; an answer cut by --max-tokens is reported apart.

Usage:
    python scripts/needle_eval.py --model LiquidAI/LFM2.5-24B-A2B --tp 4 --max-model-len 131072 \\
        --context-tokens 32768 65536 131072 --decode-context-buckets 32768 65536 --out needle.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

QUESTION = "What is the secret passcode mentioned in the text above? Reply with the passcode only."


def haystack() -> str:
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    files = sorted(site_packages.glob("*.dist-info/METADATA"))
    return "\n\n".join(f.read_text(errors="ignore") for f in files)


def needle_prompt(text: str, offsets, num_tokens: int, depth: float, code: str) -> str:
    """The first num_tokens tokens of the haystack, the needle at a line break near depth."""
    end = offsets[num_tokens - 1][1]
    cut = text.rfind("\n", 0, offsets[int(num_tokens * depth)][0]) + 1
    needle = f"The secret passcode is {code}. Remember it.\n"
    return text[:cut] + needle + text[cut:end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--context-tokens", type=int, nargs="+", default=[32768, 65536, 131072])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--decode-context-buckets", type=int, nargs="*", default=None,
        help="decode_context_length_buckets: decode graphs that read only this much context",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    text = haystack()
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    rng = random.Random(0)
    cases = []
    for context in args.context_tokens:
        # Room for the chat template, the question and the answer.
        num_tokens = context - args.max_tokens - 128
        for depth in args.depths:
            code = str(rng.randrange(10**6, 10**7))
            body = needle_prompt(text, enc["offset_mapping"], num_tokens, depth, code)
            cases.append({"context": context, "depth": depth, "code": code,
                          "messages": [{"role": "user", "content": f"{body}\n\n{QUESTION}"}]})

    neuron_config = {
        "num_batched_tokens_buckets": [args.max_num_batched_tokens],
        "num_seqs_buckets": [1],
    }
    if args.decode_context_buckets:
        neuron_config["decode_context_length_buckets"] = args.decode_context_buckets
    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        additional_config={"neuron_config": neuron_config},
    )
    print(f"[needle] engine ready in {time.perf_counter() - t0:.1f}s", flush=True)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=0.0)
    results = []
    for case in cases:
        t0 = time.perf_counter()
        out = llm.chat([case["messages"]], params, use_tqdm=False)[0]
        elapsed = time.perf_counter() - t0
        text = out.outputs[0].text
        finished = "</think>" in text
        found = finished and case["code"] in text.split("</think>")[-1]
        verdict = "found" if found else "MISSED" if finished else "CUT (still reasoning)"
        print(
            f"[needle] context {case['context']:>6} ({len(out.prompt_token_ids)} prompt tokens) "
            f"depth {case['depth']:.1f}: {verdict} in {elapsed:.1f}s, "
            f"{len(out.outputs[0].token_ids)} tokens, ends {text.strip()[-60:]!r}",
            flush=True,
        )
        results.append({"context": case["context"], "depth": case["depth"], "code": case["code"],
                        "prompt_tokens": len(out.prompt_token_ids), "finished": finished,
                        "found": found, "seconds": elapsed, "text": text})
    print(
        f"[needle] {sum(r['found'] for r in results)}/{len(results)} found, "
        f"{sum(not r['finished'] for r in results)} cut at {args.max_tokens} tokens",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
