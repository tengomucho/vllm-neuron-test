#!/bin/bash
# Start `vllm serve`, wait for it, send one chat completion, stop it.
# Usage: scripts/serve_check.sh <log_file> <model> [extra vllm serve args...]
set -u
LOG=$1; MODEL=$2; shift 2
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
PORT=${PORT:-8000}

vllm serve "$MODEL" --port "$PORT" \
    --max-model-len 256 --max-num-seqs 2 --max-num-batched-tokens 256 \
    --no-enable-prefix-caching \
    --additional-config '{"neuron_config": {"num_batched_tokens_buckets": [256], "num_seqs_buckets": [2]}}' \
    "$@" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for _ in $(seq 1800); do
    curl -sf "http://localhost:$PORT/health" > /dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "server died, see $LOG"; exit 1; }
    sleep 1
done
curl -s "http://localhost:$PORT/v1/chat/completions" -H "Content-Type: application/json" -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France? Answer in one sentence.\"}],
  \"max_tokens\": 48, \"temperature\": 0,
  \"chat_template_kwargs\": {\"enable_thinking\": false}
}"
echo
