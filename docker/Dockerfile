# vLLM Neuron (torch-native route) with the Transformers modeling backend, from
# the tengomucho/vllm-neuron fork. Ubuntu 26.04 ships glibc 2.43, which the
# torch-neuronx wheels need, so nothing is patched (README, issue 1).
#
#   docker build -t vllm-neuron-transformers docker/
#   docker run --rm -p 8000:8000 --shm-size=16g \
#     $(for d in /dev/neuron*; do echo --device=$d; done) \
#     -v neuron-cache:/root/.cache -e HF_TOKEN \
#     vllm-neuron-transformers serve <model> ...
FROM ubuntu:26.04

ARG DEBIAN_FRONTEND=noninteractive

# Python 3.14 is Ubuntu 26.04's python3. torch_xla's _XLAC links
# libpython3.14.so.1.0, and neuronx-cc's walrus_driver links libarchive. The
# Neuron driver runs on the host: the image only needs the runtime, the
# collectives and the tools (neuron-ls, neuron-top), at the versions tested
# with driver 2.31.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git gnupg python3 libpython3.14 libarchive13t64 \
 && curl -fsSL https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
      | gpg --dearmor -o /etc/apt/keyrings/neuron.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/neuron.gpg] https://apt.repos.neuron.amazonaws.com resolute main" \
      > /etc/apt/sources.list.d/neuron.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      aws-neuronx-runtime-lib=2.35.30.0-d2725ccb8 \
      aws-neuronx-collectives=2.35.43.0-354ad3064 \
      aws-neuronx-tools=2.33.38.0-f874d5618 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /usr/local/bin/uv

ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/opt/aws/neuron/bin:$PATH \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1
RUN uv venv --python /usr/bin/python3 /opt/venv

# The fork's branch is fetched when this layer is built: pass a commit as
# VLLM_NEURON_REF (or build with --no-cache) to pick up a newer one.
# - unsafe-best-match: the Neuron index serves an ancient islpy that would
#   hide the one neuronx-cc needs (issue 2);
# - vllm-neuron pins transformers<5.18.0, overridden (issue 3).
ARG VLLM_NEURON_REPO=https://github.com/tengomucho/vllm-neuron.git
ARG VLLM_NEURON_REF=transformers-backend
RUN echo "transformers==5.22.2" > /tmp/overrides.txt \
 && uv pip install \
      --extra-index-url https://pip.repos.neuron.amazonaws.com \
      --index-strategy unsafe-best-match \
      --override /tmp/overrides.txt \
      "vllm-neuron @ git+${VLLM_NEURON_REPO}@${VLLM_NEURON_REF}" \
      torch-neuronx==2.13.3.0.284 \
      libtorch-neuronx-lite==2.13.0.1.0.3112 \
      neuronx-cc==2.28.7631.0 \
      nki==0.7.0 \
      transformers==5.22.2 \
      accelerate==1.15.0 \
 && rm /tmp/overrides.txt

# - VLLM_NEURON_BACKEND: makes libtorch-neuronx-lite take the native route too
#   (issue 7);
# - TORCH_DEVICE_BACKEND_AUTOLOAD=0: keeps torch-neuronx from loading in vLLM
#   (issue 6); unset it to run plain torch-neuronx scripts;
# - NEURON_SKIP_EFA_AFFINITY=1: without EFA devices in the container, workers
#   fail on the EFA CPU-affinity setup (issue 5); set it to 0 when passing them;
# - NKI_COMPILE_CACHE_URL: keeps the NKI kernel binaries (/var/tmp by default)
#   in ~/.cache with the rest of the compile cache, which points to them
#   (issue 15). Mount a volume on /root/.cache to keep them all.
ENV VLLM_NEURON_BACKEND=neuron_native \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    NEURON_SKIP_EFA_AFFINITY=1 \
    NKI_COMPILE_CACHE_URL=/root/.cache/nki

ENTRYPOINT ["vllm"]
CMD ["--help"]
