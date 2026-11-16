#!/bin/bash
# Serve a model on port $PORT (8000) with vLLM Neuron, in the Docker image of
# docker/Dockerfile ($IMAGE, vllm-neuron-transformers).
#
# Usage: scripts/serve_vllm.sh [--warmup-bucket] <model_id> <tp> <max_model_len> [max_num_seqs] [vllm serve args...]
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
# - --warmup-bucket: instead of serving, start the server, wait until it has
#   compiled its graphs, send it one request, push the compile cache to
#   NEURON_COMPILE_CACHE_BUCKET, then stop it. HF_TOKEN must be able to write
#   to the bucket.
set -eu
WARMUP=
if [ "${1:-}" = --warmup-bucket ]; then WARMUP=1; shift; fi
if [ $# -lt 3 ]; then
    echo "usage: $0 [--warmup-bucket] <model_id> <tp> <max_model_len> [max_num_seqs] [vllm serve args...]" >&2
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

DOCKER_ARGS=(--shm-size=16g $(for d in /dev/neuron*; do echo "--device=$d"; done)
    -v "${CACHE:-neuron-cache}:/root/.cache"
    -v "${COMPILE_CACHE:-neuron-compile-cache}:/root/.cache/neuron-compile"
    -e HF_TOKEN -e NEURON_COMPILE_CACHE_BUCKET)
SERVE_ARGS=("${IMAGE:-vllm-neuron-transformers}" serve "$MODEL"
    --tensor-parallel-size "$TP" --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$CHUNK"
    --no-enable-prefix-caching
    --additional-config "{\"neuron_config\": {$NEURON_CONFIG}}"
    "$@")

if [ -z "$WARMUP" ]; then
    exec docker run --rm -p "${PORT:-8000}:8000" "${DOCKER_ARGS[@]}" "${SERVE_ARGS[@]}"
fi

: "${NEURON_COMPILE_CACHE_BUCKET:?--warmup-bucket pushes to NEURON_COMPILE_CACHE_BUCKET, which is not set}"
: "${HF_TOKEN:?--warmup-bucket needs an HF_TOKEN that can write to the bucket}"
NAME=vllm-neuron-warmup-$$
docker run -d --name "$NAME" "${DOCKER_ARGS[@]}" "${SERVE_ARGS[@]}" > /dev/null
trap 'docker rm -f "$NAME" > /dev/null' EXIT
echo "Warming up $MODEL in container $NAME (docker logs -f $NAME)"
# The server answers /health once it has compiled, or loaded, all its graphs.
until docker exec "$NAME" curl -sf localhost:8000/health > /dev/null 2>&1; do
    if [ "$(docker inspect -f '{{.State.Running}}' "$NAME")" != true ]; then
        docker logs --tail 40 "$NAME"
        echo "The server exited before it was ready." >&2
        exit 1
    fi
    sleep 5
done
docker exec "$NAME" curl -sf localhost:8000/v1/completions -H "Content-Type: application/json" \
    -d '{"prompt": "The capital of France is", "max_tokens": 8}' > /dev/null
echo "Pushing the compile cache to $NEURON_COMPILE_CACHE_BUCKET"
docker exec "$NAME" bash -c \
    'HF_HUB_DISABLE_UPDATE_CHECK=1 hf buckets sync "$NEURON_COMPILE_CACHE" "$NEURON_COMPILE_CACHE_BUCKET" --format quiet'
echo "Done"
