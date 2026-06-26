#!/usr/bin/env bash
# Sync the gemma4 plugin code into the vllm_neuron container and clear the NEFF cache.
# Run from the repo root: bash scripts/sync_gemma4.sh

set -euo pipefail

CONTAINER=vllm_neuron
D=/opt/conda/lib/python3.12/site-packages/vllm_neuron

echo "==> Clearing pycache ..."
docker exec $CONTAINER rm -rf "$D/model/gemma4/__pycache__"

echo "==> Syncing gemma4 model files ..."
docker cp vllm_neuron/model/gemma4/. $CONTAINER:"$D/model/gemma4/"

echo "==> Syncing plugin files ..."
docker cp vllm_neuron/__init__.py        $CONTAINER:"$D/__init__.py"
docker cp vllm_neuron/model/registry.py  $CONTAINER:"$D/model/registry.py"
docker cp vllm_neuron/model/__init__.py  $CONTAINER:"$D/model/__init__.py"
docker cp vllm_neuron/functional/moe/moe_cte.py $CONTAINER:"$D/functional/moe/moe_cte.py"

echo "==> Syncing test script ..."
docker cp scripts/g4_test.py $CONTAINER:/tmp/g4.py

echo "==> Clearing NEFF cache ..."
docker exec $CONTAINER sh -c 'rm -rf /root/.cache/vllm/*'

echo ""
echo "Done. Example runs:"
echo "  # CPU reference (tiny):"
echo "  docker exec $CONTAINER bash -lc 'MODEL=/root/gemma4-tiny-random CPU=1 TP=1 VLLM_NEURON_CPU_MODE=1 python /tmp/g4.py'"
echo ""
echo "  # On-device tiny TP=4:"
echo "  docker exec $CONTAINER bash -lc 'MODEL=/root/gemma4-tiny-random TP=4 python /tmp/g4.py'"
echo ""
echo "  # On-device tiny TP=8+EP:"
echo "  docker exec $CONTAINER bash -lc 'MODEL=/root/gemma4-tiny-random TP=8 EP=1 python /tmp/g4.py'"
echo ""
echo "  # Synthetic hd256 model (need to build + docker cp it first):"
echo "  docker exec $CONTAINER bash -lc 'MODEL=/root/gemma4-hd256 TP=4 python /tmp/g4.py'"
echo ""
echo "  # Synthetic hd256 with STUB attn (bisect — should compile):"
echo "  docker exec $CONTAINER bash -lc 'GEMMA4_STUB_ATTN=1 MODEL=/root/gemma4-hd256 TP=4 python /tmp/g4.py'"
echo ""
echo "  # 26B on-device (EP+TP=8):"
echo "  docker exec $CONTAINER bash -lc 'MODEL=/root/gemma4-text TP=8 EP=1 GPU_MEM=0.85 python /tmp/g4.py'"
