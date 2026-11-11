#!/bin/bash
# Run vllm_offline.py for a list of "<name> <model> <model-impl> <tp>" configs, one after the other.
# Usage: scripts/run_matrix.sh <log_dir> <<EOF
# tinyllama_tfm_tp2 TinyLlama/TinyLlama-1.1B-Chat-v1.0 transformers 2
# EOF
set -u
LOG_DIR=$1
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPTS/env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
cd "$SCRIPTS"
while read -r name model impl tp; do
    [ -z "$name" ] && continue
    timeout 3600 python vllm_offline.py --model "$model" --model-impl "$impl" --tp "$tp" \
        --max-model-len "${MAX_MODEL_LEN:-128}" --max-num-seqs 2 \
        --max-num-batched-tokens "${MAX_BATCHED:-128}" --max-tokens "${MAX_TOKENS:-16}" \
        --repeat "${REPEAT:-1}" --out "$LOG_DIR/$name.json" > "$LOG_DIR/$name.log" 2>&1
    echo "$name rc=$?"
done
