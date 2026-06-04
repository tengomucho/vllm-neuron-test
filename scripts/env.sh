# Source me: environment for the Python 3.14 Neuron venv on this (Ubuntu 24.04) host.
VLLM_TEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$VLLM_TEST_DIR/.venv/bin/activate"

# torch_xla's _XLAC (pulled in by libtorch-neuronx-lite) links libpython3.14.so,
# which the uv-managed interpreter keeps in its own lib dir.
PY_LIBDIR="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')"
export LD_LIBRARY_PATH="$PY_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# This container does not expose the EFA devices: skip the EFA CPU-affinity setup
# (a performance optimization only).
export NEURON_SKIP_EFA_AFFINITY=1

# vllm-neuron takes the native route when this is unset, but libtorch-neuronx-lite
# only when it is set: unset, lite installs its XLA-route torch overrides (F.gelu,
# F.softmax, torch.argmax, ...), which break torch.compile of GELU models, and
# leaves its precompile cache off.
export VLLM_NEURON_BACKEND=neuron_native

# vLLM Neuron runs on libtorch-neuronx-lite: torch-neuronx must not autoload in
# vLLM processes. Leave it unset for plain torch-neuronx scripts.
#   export TORCH_DEVICE_BACKEND_AUTOLOAD=0
