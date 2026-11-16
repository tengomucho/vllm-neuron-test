#!/bin/bash
# Runs `vllm "$@"`. With NEURON_COMPILE_CACHE_BUCKET set to an
# hf://buckets/<namespace>/<bucket>[/<prefix>] URL, first pulls the compile cache
# from it into NEURON_COMPILE_CACHE, so that the graphs it holds are not
# compiled again. A failed pull only warns: vLLM then compiles what is missing.
if [ -n "${NEURON_COMPILE_CACHE_BUCKET:-}" ]; then
    HF_HUB_DISABLE_UPDATE_CHECK=1 hf buckets sync "$NEURON_COMPILE_CACHE_BUCKET" "$NEURON_COMPILE_CACHE" --format quiet \
        || echo "warning: could not pull the compile cache from $NEURON_COMPILE_CACHE_BUCKET" >&2
fi
exec vllm "$@"
