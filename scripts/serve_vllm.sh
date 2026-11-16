#!/bin/bash
# Serve a model on port $PORT (8000) with vLLM Neuron, in the Docker image of
# docker/Dockerfile ($IMAGE, vllm-neuron-transformers).
#
# Usage: scripts/serve_vllm.sh <model_id> <tp> <max_model_len> [max_num_seqs] [vllm serve args...]
#
# - Prompts are prefilled in chunks of up to 4096 tokens: one prefill graph.
# - Decode compiles one graph for max_num_seqs (default 1) sequences at
#   max_model_len, and past 32k one more per power of two from 32k
#   (decode_context_length_buckets), so that shorter sequences read less
#   context.
# - The models go in $CACHE (neuron-cache, on /root/.cache) and the compile
#   cache in $COMPILE_CACHE (neuron-compile-cache, on
#   /root/.cache/neuron-compile): Docker volume names or host paths.
# - HF_TOKEN and NEURON_COMPILE_CACHE_BUCKET are passed on: with the bucket
#   set, the container pulls the compile cache from it first.
set -eu
if [ $# -lt 3 ]; then
    echo "usage: $0 <model_id> <tp> <max_model_len> [max_num_seqs] [vllm serve args...]" >&2
    exit 1
fi
MODEL=$1; TP=$2; MAX_MODEL_LEN=$3; shift 3
MAX_NUM_SEQS=1
if [[ ${1:-} =~ ^[0-9]+$ ]]; then MAX_NUM_SEQS=$1; shift; fi

CHUNK=$(( MAX_MODEL_LEN < 4096 ? MAX_MODEL_LEN : 4096 ))
NEURON_CONFIG="\"num_batched_tokens_buckets\": [$CHUNK], \"num_seqs_buckets\": [$MAX_NUM_SEQS]"
DECODE_BUCKETS=""
for (( n = 32768; n < MAX_MODEL_LEN; n *= 2 )); do DECODE_BUCKETS+="${DECODE_BUCKETS:+, }$n"; done
if [ -n "$DECODE_BUCKETS" ]; then
    NEURON_CONFIG+=", \"decode_context_length_buckets\": [$DECODE_BUCKETS]"
fi

exec docker run --rm -p "${PORT:-8000}:8000" --shm-size=16g \
    $(for d in /dev/neuron*; do echo "--device=$d"; done) \
    -v "${CACHE:-neuron-cache}:/root/.cache" \
    -v "${COMPILE_CACHE:-neuron-compile-cache}:/root/.cache/neuron-compile" \
    -e HF_TOKEN -e NEURON_COMPILE_CACHE_BUCKET \
    "${IMAGE:-vllm-neuron-transformers}" serve "$MODEL" \
    --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$CHUNK" \
    --no-enable-prefix-caching \
    --additional-config "{\"neuron_config\": {$NEURON_CONFIG}}" \
    "$@"
