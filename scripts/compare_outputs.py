#!/usr/bin/env python3
"""Compare the greedy token ids of two runs (JSON from vllm_offline.py / hf_cpu_reference.py).

Usage:
    python scripts/compare_outputs.py ref.json candidate.json
"""
import json
import sys


def main() -> None:
    ref = {r["prompt"]: r for r in json.load(open(sys.argv[1]))}
    cand = json.load(open(sys.argv[2]))
    all_match = True
    for c in cand:
        r = ref[c["prompt"]]
        n = min(len(r["token_ids"]), len(c["token_ids"]))
        diverge = next((i for i in range(n) if r["token_ids"][i] != c["token_ids"][i]), None)
        if diverge is None and len(r["token_ids"]) != len(c["token_ids"]):
            diverge = n
        status = "MATCH" if diverge is None else f"DIVERGE at token {diverge}"
        all_match &= diverge is None
        print(f"{status:22s} {c['prompt']!r}\n    ref : {r['text']!r}\n    cand: {c['text']!r}")
    sys.exit(0 if all_match else 1)


if __name__ == "__main__":
    main()
