#!/usr/bin/env python3
"""Gemma4 on-device test script for the vllm_neuron container.

Copy to /tmp/g4.py in the container, then run via docker exec.

Environment variables:
  MODEL   path to the model dir (required)
  TP      tensor_parallel_size (default 4)
  EP      enable_expert_parallel 0/1 (default 0)
  CPU     CPU mode 0/1 (default 0)
  MAX_LEN max_model_len (default 64)
  GPU_MEM gpu_memory_utilization (default 0.5)

  GEMMA4_STUB_ATTN  set to 1 to replace attention with zeros (bisect test)

Example runs (inside container):
  # CPU reference, tiny model:
  MODEL=/root/gemma4-tiny-random CPU=1 TP=1 VLLM_NEURON_CPU_MODE=1 python /tmp/g4.py

  # On-device tiny TP=4:
  MODEL=/root/gemma4-tiny-random TP=4 python /tmp/g4.py

  # On-device tiny TP=8 + EP:
  MODEL=/root/gemma4-tiny-random TP=8 EP=1 python /tmp/g4.py

  # Fast head_dim-256 compile test (synthetic model):
  MODEL=/root/gemma4-hd256 TP=4 python /tmp/g4.py

  # Fast head_dim-256 compile test, STUBBED attention (bisect):
  MODEL=/root/gemma4-hd256 TP=4 GEMMA4_STUB_ATTN=1 python /tmp/g4.py

  # 26B model (after the fix):
  MODEL=/root/gemma4-text TP=8 EP=1 GPU_MEM=0.85 python /tmp/g4.py
"""
import os
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

tp          = int(os.environ.get("TP", "4"))
cpu         = os.environ.get("CPU", "0") == "1"
ep          = os.environ.get("EP", "0") == "1"
max_len     = int(os.environ.get("MAX_LEN", "64"))
gpu_mem     = float(os.environ.get("GPU_MEM", "0.5"))
model       = os.environ["MODEL"]
# Extra neuronx-cc flags (e.g. "--hbm-scratchpad-page-size=1024" to avoid the
# page-boundary intermediate-copy overhead that inflates decode peak HBM).
cc_flags    = os.environ.get("CC_FLAGS", "")

neuron_cfg: dict = {
    "num_batched_tokens_buckets": [max_len],
    "num_seqs_buckets": [1],
}
if cc_flags:
    neuron_cfg["compiler_args"] = cc_flags

kw = dict(
    model=model,
    tensor_parallel_size=tp,
    max_model_len=max_len,
    max_num_seqs=1,
    gpu_memory_utilization=gpu_mem,
    enable_expert_parallel=ep,
    additional_config={"neuron_config": neuron_cfg},
)
if cpu:
    kw["enforce_eager"] = True  # CPU mode: avoids torch.compile

print(f"LLM config: model={model}, TP={tp}, EP={ep}, CPU={cpu}, max_len={max_len}")
llm = LLM(**kw)

prompt_ids = [10, 20, 30, 40, 50, 60, 70, 80]
out = llm.generate(
    [TokensPrompt(prompt_token_ids=prompt_ids)],
    SamplingParams(temperature=0.0, max_tokens=8),
)
print("TOKIDS:", list(out[0].outputs[0].token_ids))
print("TEXT:  ", repr(out[0].outputs[0].text))
