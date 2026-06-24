# SPDX-License-Identifier: Apache-2.0
"""
NeuronWorker - Worker implementation for vLLM integration

This provides the main worker class that implements vLLM's WorkerBase interface
for AWS Neuron/Trainium hardware integration through vLLM Neuron.
"""

import logging
import os
import re
import time
from typing import Any

import torch
import vllm.distributed.parallel_state as parallel_state
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer import (
    ensure_kv_transfer_initialized,
    has_kv_transfer_group,
    get_kv_transfer_group,
)
from vllm.utils.torch_utils import set_random_seed
from vllm.tasks import SupportedTask
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.worker.worker_base import WorkerBase
from vllm.v1.outputs import (
    DraftTokenIds,
    ModelRunnerOutput,
    AsyncModelRunnerOutput,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.v1.core.sched.output import GrammarOutput

from vllm_neuron import envs
from vllm_neuron.metrics import STARTUP_TIME
from vllm_neuron.utils.hardware_config import (
    get_efa_interface,
    parse_range_list,
)
from vllm.utils.network_utils import get_open_port
from vllm_neuron.parallel.neuron_parallel_state import tp_barrier
from vllm_neuron.vllm.platform import NeuronPlatform
from vllm_neuron.vllm.worker.neuron_profiler import (
    NeuronProfilerConfig,
    NeuronProfiler,
)

logger = logging.getLogger(__name__)


def rendezvous_ccom_bootstrap():
    """Broadcast NEURON_RT_ROOT_COMM_ID from rank 0 via gloo.

    Must be called after ``init_distributed_environment()`` (gloo is up) and
    before the first NEFF execution (which triggers CCOM bootstrap).

    Rank 0 constructs the CCOM address from ``MASTER_ADDR`` and a free port,
    then broadcasts it to all workers. This ensures every worker — regardless
    of which DP engine spawned it — uses the same CCOM bootstrap endpoint.

    Works for all DP modes:
      - **No DP / external DP**: each process has its own gloo world with its
        own rank 0, so each independently picks its own port.
      - **Internal DP** (``--data-parallel-size > 1``): one gloo world spans
        all DP engines. Rank 0 (from engine DP0) broadcasts its address, so
        all TP*DP workers share one CCOM communicator.

    No-op in CPU mode (no CCOM needed).
    """
    import torch.distributed as dist

    if envs.VLLM_NEURON_CPU_MODE or envs.VLLM_NEURON_CPU_COMPILE:
        return

    if dist.get_rank() == 0:
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        port = get_open_port()
        addr = f"{master_addr}:{port}"
    else:
        addr = None

    addr_list = [addr]
    dist.broadcast_object_list(addr_list, src=0, device=torch.device("cpu"))
    addr = addr_list[0]

    os.environ["NEURON_RT_ROOT_COMM_ID"] = addr
    logger.info(
        "CCOM bootstrap rendezvous: NEURON_RT_ROOT_COMM_ID=%s (rank %d/%d)",
        addr,
        dist.get_rank(),
        dist.get_world_size(),
    )


class _SuppressModelRegistryOverwrite(logging.Filter):
    """Filter to suppress expected model registration overwrite warnings from vLLM.

    This filter is used when registering vLLM Neuron's Neuron-optimized model implementations
    with vLLM's ModelRegistry. Since we intentionally override vLLM's default models,
    the "already registered" warnings are expected and not actionable.
    """

    # Matches: "Model architecture LlamaForCausalLM is already registered, and will be overwritten..."
    _PATTERN = re.compile(
        r"Model architecture \w+ is already registered.*will be overwritten"
    )

    def filter(self, record: logging.LogRecord) -> bool:
        """Return False to suppress matching messages, True to allow them through."""
        return not self._PATTERN.search(record.getMessage())


class NeuronWorker(WorkerBase):
    """
    Worker implementation for AWS Neuron/Trainium hardware.
    This class implements the vLLM WorkerBase interface.
    """

    def __init__(
        self,
        vllm_config: Any | None = None,
        local_rank: int = 0,
        rank: int = 0,
        distributed_init_method: str | None = None,
        is_driver_worker: bool = True,
    ):
        """
        Initialize the Neuron worker.

        Args:
            vllm_config: vLLM configuration object
            local_rank: Local rank of this worker
            rank: Global rank of this worker
            distributed_init_method: Method for distributed initialization
            is_driver_worker: Whether this is the driver worker
        """
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        self.use_async_scheduling = vllm_config.scheduler_config.async_scheduling

        # NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS means max number of neffs queued up in libnrt.
        # Since there is always 1 neff executing on hardware, the total number of inflight neffs is actually +1.
        # A common misunderstanding is NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=1 means it is not async.
        # NOTE: We intentionally always set this env variable to maintain the consistency of the runtime environment.
        # Our worker controls whether we actually schedules ahead by choosing when to read the output of the model.
        if not envs.VLLM_NEURON_LIBTORCH_NEURONX_LITE:
            os.environ["NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS"] = "1"

        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker
        self._startup_start_time = time.perf_counter()

        # Model runner will be created in init_device()
        # Following GPU/TPU worker pattern
        self.model_runner = None
        self.device = None

        # Register models with ModelRegistry
        # Put it in the worker as vLLM's model registry overwrites vllm_neuron model registry
        # if we put it too early in the platform.py.
        # We need to use or create a platform API for OOT platform model registry.
        # Tracked as TODO below.
        # TODO [CHRYS-72]: Investigate vLLM model registry
        from vllm import ModelRegistry
        from vllm_neuron.model import registry

        # Suppress the vLLM warning about overwriting existing model registrations.
        # This is expected behavior - we intentionally override vLLM's default model
        # implementations with vLLM Neuron's Neuron-optimized versions.
        vllm_registry_logger = logging.getLogger("vllm.model_executor.models.registry")
        suppress_filter = _SuppressModelRegistryOverwrite()
        vllm_registry_logger.addFilter(suppress_filter)
        try:
            for arch, model_cls in registry.get_models():
                ModelRegistry.register_model(arch, model_cls)
        finally:
            vllm_registry_logger.removeFilter(suppress_filter)

        self._profiler = self._init_profiler(vllm_config)

        logger.info(
            "Initialized NeuronWorker with rank %s, local_rank %s", rank, local_rank
        )

    def _set_efa_affinity(self, visible_devices) -> str:
        """
        Set EFA interface affinity for optimal network performance.

        Delegates the full device-to-EFA-interface resolution to
        hardware_config.get_efa_interface().

        Args:
            visible_devices: List of visible Neuron device indices

        Returns:
            str: Name of the EFA interface assigned to this host worker

        Raises:
            RuntimeError: If the Neuron device cannot be accessed or EFA interface not found
            IndexError: If the computed device index is out of bounds
        """
        efa = get_efa_interface(self.local_rank, visible_devices)
        os.environ["FI_EFA_IFACE"] = efa
        logger.debug(
            "Set EFA interface to %s for local_rank %s",
            efa,
            self.local_rank,
        )
        return efa

    def _set_cpu_affinity(self) -> None:
        """
        Set CPU affinity for all threads based on EFA's NUMA node.

        Pins all threads in the current process to CPUs in the same NUMA node as
        the EFA interface to minimize interconnection overhead and improve performance.
        This is critical for both single-instance and disaggregated inference workloads.

        Queries the EFA interface name from the FI_EFA_IFACE environment variable.
        """
        efa = os.environ.get("FI_EFA_IFACE")
        numa_node_path = (
            f"/sys/class/infiniband/{efa}/device/numa_node" if efa else None
        )

        if efa and numa_node_path and os.path.exists(numa_node_path):
            with open(f"/sys/class/infiniband/{efa}/device/numa_node") as f:
                numa = int(f.read())
            with open(f"/sys/devices/system/node/node{numa}/cpulist") as f:
                cpulist = parse_range_list(f.read())

            # Set affinity for all existing threads in the process
            # This handles any vLLM threads that may have been created before init_device()
            # e.g., ZeroMQ I/O or reaper threads
            threads_set = 0
            try:
                thread_ids = os.listdir("/proc/self/task")
                logger.debug(
                    "Setting affinity for %s threads to NUMA node %s",
                    len(thread_ids),
                    numa,
                )
                for tid in thread_ids:
                    try:
                        os.sched_setaffinity(int(tid), cpulist)
                        threads_set += 1
                    except OSError as ose:
                        # Thread may have exited or we don't have permission
                        logger.warning(
                            "Failed to set affinity for thread %s: %s", tid, ose
                        )
            except FileNotFoundError as fnfe:
                logger.warning(
                    "Could not enumerate threads (/proc/self/task not found): %s", fnfe
                )
                os.sched_setaffinity(0, cpulist)
                threads_set = 1

            logger.debug(
                "Set CPU affinity for %s thread(s) to NUMA node %s, CPUs: %s",
                threads_set,
                numa,
                cpulist,
            )
        else:
            logger.warning(
                "EFA device %s not found, skipping CPU affinity settings", efa
            )

    def init_device(self) -> None:
        """
        Initialize the device for this worker.

        Sets up Neuron environment variables, device, and creates model runner.
        Follows the vLLM GPU/TPU worker pattern.

        Initialization order:
        1. ``_init_neuron_distributed_environment_and_runtime()`` – affinity,
           env vars, gloo init, rendezvous_ccom_bootstrap, runtime.initialize(),
           ensure_kv_transfer_initialized
        2. Create torch device and model runner
        """
        if envs.VLLM_NEURON_CPU_MODE:
            self.device = torch.device("cpu")
        elif envs.VLLM_NEURON_CPU_COMPILE:
            self.device = torch.device("meta")

        self._init_neuron_distributed_environment_and_runtime(
            self.vllm_config, self.rank, self.distributed_init_method, self.local_rank
        )

        if self._use_neuron_device():
            self.device = torch.device(f"neuron:{self.local_rank}")

        # Create model runner with vllm_config and device
        from .neuron_model_runner import NeuronModelRunner

        self.model_runner = NeuronModelRunner(
            vllm_config=self.vllm_config, device=self.device
        )

        # Set random seed
        set_random_seed(self.model_config.seed)

        logger.debug(
            "Device initialized: device=%s, local_rank=%s, rank=%s",
            self.device,
            self.local_rank,
            self.rank,
        )

    def _get_visible_devices(self):
        if "NEURON_RT_VISIBLE_CORES" in os.environ:
            raise RuntimeError(
                "NEURON_RT_VISIBLE_CORES cannot be used with multi-processing execution on vLLM. "
                "Set NEURON_VISIBLE_DEVICES to control device visibility."
            )
        world_size = self.vllm_config.parallel_config.world_size
        nnodes = self.vllm_config.parallel_config.nnodes
        num_local_ranks = world_size // nnodes

        if "NEURON_VISIBLE_DEVICES" not in os.environ:
            # NEURON_VISIBLE_DEVICES must be set for libtorchneuron
            os.environ["NEURON_VISIBLE_DEVICES"] = ",".join(
                str(i) for i in range(num_local_ranks)
            )

        neuron_visible_devices = os.getenv("NEURON_VISIBLE_DEVICES")
        neuron_visible_devices = parse_range_list(neuron_visible_devices)
        assert len(neuron_visible_devices) == num_local_ranks
        return neuron_visible_devices

    def get_kv_connector_handshake_metadata(self) -> dict | None:
        """Get KV connector metadata from this worker if available."""

        if not has_kv_transfer_group():
            return None

        connector = get_kv_transfer_group()
        # Return None for connectors that don't need to exchange handshake
        # metadata across workers.
        if (metadata := connector.get_handshake_metadata()) is None:
            return None

        tp_rank = get_tp_group().rank_in_group
        return {tp_rank: metadata}

    def _patch_in_same_node_as_function(self):
        # WORKAROUND: Patch in_the_same_node_as() to avoid PyTorch barrier() bug with Neuron device backend
        #
        # THE PROBLEM:
        # vLLM's parallel_state.in_the_same_node_as() function calls torch.distributed.barrier(group=pg)
        # to synchronize ranks and find ranks that share the same shared memory.
        # However, PyTorch's barrier() implementation
        # (in torch/distributed/distributed_c10d.py, line ~4792) unconditionally calls:
        #     device = torch._C._get_accelerator()
        # BEFORE checking the backend type or dispatching to the backend-specific barrier implementation.
        #
        # For Neuron (which uses PyTorch's PrivateUse1 extension mechanism), _get_accelerator() requires
        # PrivateUse1HooksInterface to be registered. Since we haven't registered it, this causes:
        #     RuntimeError: Please register PrivateUse1HooksInterface by `RegisterPrivateUse1HooksInterface` first.
        #
        # This error occurs even though:
        # 1. The process group (pg) is using the Gloo backend (CPU-based)
        # 2. Gloo's barrier implementation doesn't need device/accelerator information
        # 3. The barrier would work fine if PyTorch didn't try to detect the device first
        #
        # CALL CHAIN WHERE THIS BREAKS:
        # init_model_parallel_group()
        #   → GroupCoordinator.__init__()
        #   → MessageQueue.create_from_process_group()
        #   → in_the_same_node_as(pg)
        #   → torch.distributed.barrier(group=pg)
        #   → torch._C._get_accelerator()  ← CRASH HERE
        #
        # THIS WORKAROUND:
        # We patch in_the_same_node_as() to bypass the barrier() call entirely and calculate
        # the result based on ParallelConfig, rather than probe the system. This allows:
        # - MessageQueue to use shared memory for inter-rank communication
        # - Process group initialization to complete successfully
        # - Single-node and multi-node distributed inference/training to work
        #
        # PROPER SOLUTIONS (either one will work):
        # 1. Register PrivateUse1HooksInterface for Neuron:
        #    - Will come with consolidation with Torch Eager
        # 2. Register vLLM Neuron plugin from a separate python package
        #    - Will mean that PrivateUse1 is not bound and hence error isn't hit.
        def patched_in_the_same_node_as(pg, source_rank=0):
            world_size = torch.distributed.get_world_size(group=pg)
            nnodes = self.vllm_config.parallel_config.nnodes
            ranks_per_node = world_size // nnodes
            source_node = source_rank // ranks_per_node
            return [rank // ranks_per_node == source_node for rank in range(world_size)]

        parallel_state.in_the_same_node_as = patched_in_the_same_node_as

    def _init_neuron_distributed_environment_and_runtime(
        self,
        vllm_config: VllmConfig,
        rank: int,
        distributed_init_method: str | None,
        local_rank: int,
    ) -> None:
        """Initialize the distributed environment and Neuron runtime.

        For non-CPU mode the full sequence is:
        1. Set EFA affinity for network performance
        2. Set CPU affinity to match EFA's NUMA node (all threads including main)
        3. Set visible cores, device count, and HBM mapping
        4. Initialize gloo distributed backend
        5. rendezvous_ccom_bootstrap  →  sets unique COMM_ID per server
        6. runtime.initialize()       →  latches COMM_ID, maps HBM
        7. ensure_kv_transfer_initialized  →  NIXL registers HBM memory
        """

        # --- Prepare runtime environment (affinity, env vars) ---------------
        if self._use_neuron_device():
            visible_devices = self._get_visible_devices()

            if os.environ.get("NEURON_SKIP_EFA_AFFINITY", "0") == "0":
                self._set_efa_affinity(visible_devices)
            self._set_cpu_affinity()

            # Set visible cores and platform device count. The worker thread
            # has one visible core, while the platform device count is the
            # number of visible cores on the local node.
            NeuronPlatform.set_device_count(len(visible_devices))
            os.environ["NEURON_RT_VISIBLE_CORES"] = (
                f"{visible_devices[self.local_rank]}"
            )

            # Enable HBM mapping (required for RDMA in disaggregated
            # inference and other use cases)
            os.environ["NEURON_RT_MAP_HBM"] = "1"

        # --- Distributed environment init -----------------------------------
        parallel_config = vllm_config.parallel_config

        self._patch_in_same_node_as_function()

        logger.debug(
            f"Initializing distributed environment: world_size={parallel_config.world_size}, {rank=}, {local_rank=}"
        )

        from vllm_neuron.envs import get_dist_backend
        from vllm_neuron.parallel.neuron_parallel_state import (
            init_neuron_distributed_environment,
        )

        neuron_config_dict = vllm_config.additional_config.get("neuron_config", {})
        dist_backend = get_dist_backend()

        ods_config = neuron_config_dict.get("on_device_sampling_config", {})
        sampling_dp_degree = (
            int(ods_config.get("sampling_dp_degree", 1)) if ods_config else 1
        )

        # When EP is enabled, default ep_degree to world_size (TP * DP) so
        # experts span all ranks including across DP replicas.
        ep_degree = neuron_config_dict.get("ep_degree", 1)
        if not parallel_config.enable_expert_parallel and ep_degree > 1:
            raise ValueError(
                f"ep_degree={ep_degree} requires --enable-expert-parallel to be set."
            )
        if parallel_config.enable_expert_parallel and ep_degree == 1:
            ep_degree = parallel_config.world_size_across_dp
        if (
            parallel_config.enable_expert_parallel
            and parallel_config.data_parallel_size > 1
            and ep_degree != parallel_config.world_size_across_dp
        ):
            raise ValueError(
                f"With DP, ep_degree must equal TP * DP "
                f"({parallel_config.world_size_across_dp}), got {ep_degree}."
            )

        attention_dp_size = neuron_config_dict.get("attention_dp_size", 1)
        embedding_dp_size = neuron_config_dict.get("embedding_dp_size", 1)
        lm_head_dp_size = neuron_config_dict.get("lm_head_dp_size", 1)
        mlp_dp_size = neuron_config_dict.get("mlp_dp_size", 1)

        init_neuron_distributed_environment(
            world_size=parallel_config.world_size,
            rank=rank,
            local_rank=local_rank,
            distributed_init_method=distributed_init_method,
            backend=dist_backend,
            tensor_parallel_size=parallel_config.tensor_parallel_size,
            pipeline_parallel_size=parallel_config.pipeline_parallel_size,
            decode_context_parallel_size=parallel_config.decode_context_parallel_size,
            ep_degree=ep_degree,
            sampling_dp_degree=sampling_dp_degree,
            attention_dp_size=attention_dp_size,
            embedding_dp_size=embedding_dp_size,
            lm_head_dp_size=lm_head_dp_size,
            mlp_dp_size=mlp_dp_size,
        )

        # Rendezvous CCOM bootstrap address via gloo (now that dist is up).
        rendezvous_ccom_bootstrap()

        # Initialize the Neuron runtime NOW — after rendezvous has set the
        # correct NEURON_RT_ROOT_COMM_ID but before NIXL/libfabric tries to
        # register device memory.  The runtime latches COMM_ID during
        # initialize() and never re-reads it, so the ordering is critical:
        #   1. gloo init  (above)
        #   2. rendezvous_ccom_bootstrap  →  sets unique COMM_ID per server
        #   3. runtime.initialize()       →  latches COMM_ID, maps HBM
        if self._use_neuron_device():
            runtime = torch.classes.neuron.Runtime()
            runtime.initialize()

    def load_model(self) -> None:
        """
        Load the model on this worker.

        This initializes the model runner and loads the model.
        After loading, attempts to fetch custom prefill buckets from
        the model if available.
        """
        from vllm.config import set_current_vllm_config

        logger.info("Loading model on NeuronWorker")
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()
            self.model_runner.init_tensor_replacement()

    def _get_local_worker_count(self) -> int:
        """Return number of worker processes sharing host memory on this node."""
        world_size = self.vllm_config.parallel_config.world_size
        nnodes = self.vllm_config.parallel_config.nnodes
        if nnodes <= 0:
            return 1
        return max(world_size // nnodes, 1)

    def _query_host_runtime_memory(self) -> int:
        """Query host available memory and return this worker's fair-share bytes."""
        from vllm_neuron.utils.memory import get_available_memory_bytes

        host_available = get_available_memory_bytes()

        if host_available <= 0:
            raise RuntimeError(
                f"Non-positive host available memory detected: {host_available} bytes."
            )

        local_workers = self._get_local_worker_count()
        fair_share = host_available // local_workers
        if fair_share <= 0:
            raise RuntimeError(
                "Host memory fair-share per worker is non-positive. "
                f"host_available={host_available} bytes, local_workers={local_workers}."
            )

        logger.info(
            "Host memory: %.2f GiB available, %d local worker(s), %.2f GiB fair-share per worker",
            host_available / (1024**3),
            local_workers,
            fair_share / (1024**3),
        )
        return fair_share

    def _query_runtime_memory(self) -> int:
        """Query Neuron runtime for available HBM (raw bytes_free)."""
        _, bytes_free = self._query_runtime_memory_stats()
        return bytes_free

    def _query_runtime_memory_stats(self) -> tuple[int, int]:
        """Query Neuron runtime for used/free HBM bytes."""
        runtime = torch.classes.neuron.Runtime()
        bytes_used, bytes_free = runtime.get_vnc_memory_stats()

        if bytes_used < 0 or bytes_free <= 0:
            raise RuntimeError(
                f"Neuron runtime reported unusable memory stats: "
                f"used={bytes_used}, free={bytes_free}"
            )

        logger.info(
            "Neuron HBM: %.2f GiB used, %.2f GiB free",
            bytes_used / (1024**3),
            bytes_free / (1024**3),
        )
        return bytes_used, bytes_free

    def _get_kv_cap_fraction(self) -> float:
        """Return validated KV cap fraction shared by CPU and Neuron modes."""
        cap_fraction = envs.VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION
        if cap_fraction <= 0 or cap_fraction > 1:
            raise RuntimeError(
                "VLLM_NEURON_KV_GMU_BUDGET_CAP_FRACTION must be in (0, 1]. "
                f"Got {cap_fraction}."
            )
        return cap_fraction

    def _determine_available_memory_cpu(self, gpu_mem_util: float) -> int:
        """Compute CPU-mode KV memory budget from fair-share host memory."""
        bytes_free = self._query_host_runtime_memory()
        total_budget = int(bytes_free * gpu_mem_util)
        cap_fraction = self._get_kv_cap_fraction()
        available = int(total_budget * cap_fraction)

        logger.debug(
            "CPU mode KV cache memory: %.2f GiB free-share × %.1f%% gpu_memory_utilization = %.2f GiB total budget, %.2f GiB available (cap_fraction=%.2f)",
            bytes_free / (1024**3),
            gpu_mem_util * 100,
            total_budget / (1024**3),
            available / (1024**3),
            cap_fraction,
        )
        return available

    def _compute_kv_budget(
        self,
        total_hbm_bytes: int,
        bytes_used: int,
        gpu_mem_util: float,
    ) -> int:
        """Shared KV cache budget logic for both estimate and runtime paths.

        Applies gpu_memory_utilization, cap-fraction guardrail, and min-KV
        threshold check, then returns the available bytes for KV cache.
        """
        total_budget = int(total_hbm_bytes * gpu_mem_util)
        # Preserve vLLM semantics: GMU applies to total device memory first.
        user_budget = max(total_budget - bytes_used, 0)

        # Apply a fixed cap to the GMU-scaled total HBM budget.
        # TODO: Replace this cap with a cleaner solution later.
        cap_fraction = self._get_kv_cap_fraction()
        heuristic_cap = int(total_budget * cap_fraction)
        available = max(min(user_budget, heuristic_cap), 0)

        logger.debug(
            "KV cache heuristic: user_budget=%.2f GiB, cap=%.2f GiB "
            "(total_budget=%.2f GiB, total_hbm=%.2f GiB, bytes_used=%.2f GiB, "
            "cap_fraction=%.2f), "
            "effective=%.2f GiB",
            user_budget / (1024**3),
            max(heuristic_cap, 0) / (1024**3),
            total_budget / (1024**3),
            total_hbm_bytes / (1024**3),
            bytes_used / (1024**3),
            cap_fraction,
            available / (1024**3),
        )

        min_kv_gib = envs.VLLM_NEURON_MIN_KV_BUDGET_GIB
        if min_kv_gib > 0:
            min_kv_bytes = int(min_kv_gib * (1024**3))
            if available < min_kv_bytes:
                raise RuntimeError(
                    "Computed KV cache budget is below minimum threshold. "
                    f"effective={available / (1024**3):.2f} GiB, "
                    f"minimum={min_kv_gib:.2f} GiB. "
                    "Increase gpu_memory_utilization or lower "
                    "VLLM_NEURON_MIN_KV_BUDGET_GIB."
                )

        logger.debug(
            "KV cache memory: %.2f GiB total × %.1f%% gpu_memory_utilization "
            "- %.2f GiB used = %.2f GiB user budget, %.2f GiB available",
            total_hbm_bytes / (1024**3),
            gpu_mem_util * 100,
            bytes_used / (1024**3),
            user_budget / (1024**3),
            available / (1024**3),
        )
        return available

    def _get_byte_used_from_model(self) -> int:
        """
        Use the model params and buffers to calculate the size of tensors
        allocated on device.
        """
        bytes_used_params = sum(p.nbytes for p in self.model_runner.model.parameters())
        bytes_used_buffers = sum(b.nbytes for b in self.model_runner.model.buffers())
        # Add spec decode model weights.
        if self.model_runner.drafter is not None:
            bytes_used_params += sum(
                p.nbytes for p in self.model_runner.drafter.model.parameters()
            )
            bytes_used_buffers += sum(
                b.nbytes for b in self.model_runner.drafter.model.buffers()
            )
        return bytes_used_params + bytes_used_buffers

    def _estimate_available_memory_neuron(self, gpu_mem_util: float) -> int:
        """
        Estimate KV budget from static HBM size and model parameter bytes.
        This is used during CPU Compilation flow and computes bytes used
        based on the model params and buffer sizes.
        """
        from vllm_neuron.compile.platform import get_total_available_memory

        total_hbm_bytes = get_total_available_memory() * 1024 * 1024 * 1024
        bytes_used = self._get_byte_used_from_model()
        return self._compute_kv_budget(total_hbm_bytes, bytes_used, gpu_mem_util)

    def _determine_available_memory_neuron(self, gpu_mem_util: float) -> int:
        """
        Compute Neuron-mode KV memory budget using static HBM size and model size

        Uses model parameter and buffer sizes for bytes_used (same as the CPU
        compile estimate path) instead of runtime.get_vnc_memory_stats().
        This ensures the KV cache block count is identical between CPU compile
        and execution, which is required for compilation cache hits.

        Without this, mismatch happens because runtime.get_vnc_memory_stats()
        itself uses ~3MB of device memory which goes unaccounted in CPU compile.
        Ignoring this is safe as the total memory is in order of GBs.
        """
        bytes_used, bytes_free = self._query_runtime_memory_stats()
        total_hbm_bytes = bytes_used + bytes_free
        bytes_used = self._get_byte_used_from_model()
        return self._compute_kv_budget(total_hbm_bytes, bytes_used, gpu_mem_util)

    @torch.inference_mode()
    def determine_available_memory(self) -> int:
        """
        Determine the amount of memory available for KV cache.

        In CPU mode, queries host available memory and uses a fair-share per
        local worker process. On Neuron device, queries runtime free HBM and
        applies gpu_memory_utilization.

        Note: In vLLM v1, gpu_memory_utilization is applied inside the worker
        (not by vLLM core). The GPU worker does the same pattern.

        Returns:
            int: Available memory in bytes (already scaled by gpu_memory_utilization)
        """
        gpu_mem_util = self.cache_config.gpu_memory_utilization
        if envs.VLLM_NEURON_CPU_COMPILE:
            return self._estimate_available_memory_neuron(gpu_mem_util)
        if envs.VLLM_NEURON_CPU_MODE:
            return self._determine_available_memory_cpu(gpu_mem_util)
        return self._determine_available_memory_neuron(gpu_mem_util)

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """
        Initialize the KV cache with the given number of blocks.

        Args:
            num_gpu_blocks: Number of GPU blocks to allocate
            num_cpu_blocks: Number of CPU blocks to allocate
        """
        logger.debug(
            "Initializing cache with %s GPU blocks, %s CPU blocks",
            num_gpu_blocks,
            num_cpu_blocks,
        )
        # TODO: Determine if initialize_cache() needs implementation for Neuron
        # Why required: Clarify whether explicit cache initialization is needed or handled by vLLM Neuron
        # Current: Empty method implementation
        # Target: Either implement cache allocation or document that it's handled elsewhere

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache from configuration object.

        This is the vLLM v1 API method that replaces initialize_cache.
        Required by vLLM v1 WorkerBase interface.

        Args:
            kv_cache_config: KV cache configuration object
        """
        logger.debug("Initializing KV cache from config (NeuronWorker)")

        # Initialize KV transfer connector before KV cache so register_kv_caches
        # can find the connector (has_kv_transfer_group() must be True)
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)
        # Delegate to model runner to initialize KV cache
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> float:
        """
        Compile or warm up the model for optimal performance.

        Pre-compiles the model for all prefill bucket sizes and decode batch sizes
        to ensure optimal performance on Neuron hardware. Each bucket/batch size
        triggers a separate compilation that is cached by the Neuron compiler.

        In disaggregated inference (DI) mode, only the relevant warmup phase
        runs: prefill-only servers skip decode warmup, and decode-only servers
        skip prefill warmup. Servers with kv_role="kv_both" run both.

        This method calls the model directly without using SchedulerOutput or
        execute_model(), avoiding unnecessary side effects on internal state.

        Returns:
            float: Compilation/warmup time in seconds. vllm 0.19.0 executor
            aggregates return values via max() to set compilation_config.compilation_time.
        """
        if self.model_runner is None or self.model_runner.model is None:
            logger.warning(
                "Model not loaded yet. Skipping warmup. "
                "This may result in compilation happening during first inference."
            )
            return 0.0

        # SyntheticNeuronModel: skip warmup (no compiled graphs needed)
        from vllm_neuron.model.synthetic import SyntheticNeuronModel

        if isinstance(self.model_runner.model, SyntheticNeuronModel):
            logger.info("SyntheticNeuronModel detected — skipping warmup")
            return 0.0

        # CPU eager mode: skip warmup (no NEFFs to compile, shapes are dynamic)
        if envs.VLLM_NEURON_CPU_MODE and self.vllm_config.model_config.enforce_eager:
            logger.info("CPU eager mode — skipping warmup")
            return 0.0

        # Determine which warmup phases to run based on kv_role.
        # kv_producer = prefill server → only prefill warmup
        # kv_consumer = decode server  → only decode warmup
        # kv_both / no DI             → both
        warmup_start = time.perf_counter()
        kv_cfg = getattr(self.vllm_config, "kv_transfer_config", None)
        skip_prefill_warmup = (
            kv_cfg is not None and kv_cfg.is_kv_consumer and not kv_cfg.is_kv_producer
        )
        skip_decode_warmup = (
            kv_cfg is not None and kv_cfg.is_kv_producer and not kv_cfg.is_kv_consumer
        )

        # === Prefill warmup ===
        if skip_prefill_warmup:
            logger.info(
                "Skipping prefill warmup (kv_role=%s, this is a decode-only server)",
                kv_cfg.kv_role,
            )
        else:
            if self.model_runner.capture_backend_model is not None:
                self._extract_prefill_graphs()
                logger.info(
                    "Barrier: waiting for all ranks to finish prefill graph extraction"
                )
                tp_barrier()
                self.model_runner.parallel_compile()
                logger.info(
                    "Barrier: waiting for all ranks to finish prefill compilation"
                )
                tp_barrier()
            if not envs.VLLM_NEURON_CPU_COMPILE:
                self._warmup_prefill()

        # === Decode warmup ===
        if skip_decode_warmup:
            logger.info(
                "Skipping decode warmup (kv_role=%s, this is a prefill-only server)",
                kv_cfg.kv_role,
            )
        else:
            if self.model_runner.capture_backend_model is not None:
                self._extract_decode_graphs()
                logger.info(
                    "Barrier: waiting for all ranks to finish decode graph extraction"
                )
                tp_barrier()
                self.model_runner.parallel_compile()
                logger.info(
                    "Barrier: waiting for all ranks to finish decode compilation"
                )
                tp_barrier()
            if not envs.VLLM_NEURON_CPU_COMPILE:
                self._warmup_decode()

        # Enable tensor capture after warmup (skip warmup captures)
        if self.model_runner._capture_registry is not None:
            self.model_runner.enable_capture()
            logger.info("Tensor capture enabled after warmup")

        num_batched_tokens_buckets = (
            self.model_runner.neuron_config.num_batched_tokens_buckets
        )
        num_seqs_buckets = self.model_runner.neuron_config.num_seqs_buckets
        logger.info("=" * 80)
        logger.info(
            "Model warmup completed: %s num_batched_tokens_buckets, %s num_seqs_buckets",
            len(num_batched_tokens_buckets),
            len(num_seqs_buckets) if num_seqs_buckets else 0,
        )
        logger.info("=" * 80)

        STARTUP_TIME.labels(
            model_name=self.vllm_config.model_config.model,
        ).set(time.perf_counter() - self._startup_start_time)

        return time.perf_counter() - warmup_start

    def _prefill_buckets(self) -> tuple[list[int], list[int]]:
        num_batched_tokens_buckets = (
            self.model_runner.neuron_config.num_batched_tokens_buckets
        )
        kv_segment_size_buckets = (
            self.model_runner.neuron_config.kv_segment_size_buckets
        )
        # Use [0] as a sentinel for non-segmented mode (cached_seq_len=0)
        effective_kv_buckets = (
            kv_segment_size_buckets if kv_segment_size_buckets else [0]
        )
        return num_batched_tokens_buckets, effective_kv_buckets

    def _extract_prefill_graphs(self) -> None:
        """
        Extract all the prefill HLO graphs for all bucket and KV segment size combinations.
        """
        num_batched_tokens_buckets, effective_kv_buckets = self._prefill_buckets()
        logger.info(
            "Extracting graphs for %s num_batched_tokens_buckets: %s",
            len(num_batched_tokens_buckets),
            num_batched_tokens_buckets,
        )

        total_extractions = len(effective_kv_buckets) * len(num_batched_tokens_buckets)
        extraction_count = 0

        for kv_seg_size in effective_kv_buckets:
            for bucket_size in num_batched_tokens_buckets:
                extraction_count += 1
                logger.info(
                    "\n[%s/%s] Extracting for prefill: bucket_size=%s, "
                    "kv_segment_size=%s",
                    extraction_count,
                    total_extractions,
                    bucket_size,
                    kv_seg_size,
                )
                try:
                    self.model_runner.extract_prefill_graphs(bucket_size, kv_seg_size)
                    logger.info(
                        "  Successfully extracted graph for prefill bucket %s "
                        "with kv_segment_size %s",
                        bucket_size,
                        kv_seg_size,
                    )
                except Exception as e:
                    logger.error(
                        "  Failed to extract graph for prefill bucket %s "
                        "with kv_segment_size %s: %s",
                        bucket_size,
                        kv_seg_size,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Model graph extraction failed for prefill bucket {bucket_size} "
                        f"with kv_segment_size {kv_seg_size}. Error: {e}"
                    ) from e

    def _warmup_prefill(self) -> None:
        """Run prefill warmup for all bucket and KV segment size combinations."""
        num_batched_tokens_buckets, effective_kv_buckets = self._prefill_buckets()

        logger.info(
            "Warming up model for %s num_batched_tokens_buckets: %s",
            len(num_batched_tokens_buckets),
            num_batched_tokens_buckets,
        )

        total_warmups = len(effective_kv_buckets) * len(num_batched_tokens_buckets)
        warmup_count = 0

        for kv_seg_size in effective_kv_buckets:
            for bucket_size in num_batched_tokens_buckets:
                warmup_count += 1
                logger.info(
                    "\n[%s/%s] Warming up for prefill: bucket_size=%s, "
                    "kv_segment_size=%s",
                    warmup_count,
                    total_warmups,
                    bucket_size,
                    kv_seg_size,
                )
                try:
                    self.model_runner.warmup_prefill(bucket_size, kv_seg_size)
                    logger.info(
                        "  Successfully warmed up for prefill bucket %s "
                        "with kv_segment_size %s",
                        bucket_size,
                        kv_seg_size,
                    )
                except Exception as e:
                    logger.error(
                        "  Failed to warmup for prefill bucket %s "
                        "with kv_segment_size %s: %s",
                        bucket_size,
                        kv_seg_size,
                        e,
                        exc_info=True,
                    )
                    raise RuntimeError(
                        f"Model warmup failed for prefill bucket {bucket_size} "
                        f"with kv_segment_size {kv_seg_size}. Error: {e}"
                    ) from e

    def _decode_compile_targets(self) -> list[tuple[int, int]]:
        """Enumerate (batch_bucket, ctx_bucket) pairs to compile.

        Without ``decode_context_length_buckets``: one pair per batch bucket,
        seq=max_model_len (today's behavior).

        With ``decode_context_length_buckets`` set: cartesian product of batch
        buckets × (configured seq buckets + max_model_len fallback).

        Returns:
            List of (batch_bucket, ctx_bucket) tuples. Empty when
            ``num_seqs_buckets`` is None.
        """
        num_seqs_buckets = self.model_runner.neuron_config.num_seqs_buckets
        if not num_seqs_buckets:
            return []

        decode_ctx_buckets = (
            self.model_runner.neuron_config.decode_context_length_buckets
        )
        max_model_len = self.model_runner.max_model_len

        if decode_ctx_buckets is None:
            return [(b, max_model_len) for b in num_seqs_buckets]

        ctx_buckets_with_fallback = list(decode_ctx_buckets) + [max_model_len]
        return [(b, s) for b in num_seqs_buckets for s in ctx_buckets_with_fallback]

    def _extract_decode_graphs(self) -> None:
        """Extract decode HLO graphs for all (batch, seq) compile targets."""
        targets = self._decode_compile_targets()
        if not targets:
            return
        logger.info(
            "\nExtracting %d decode compile targets: %s",
            len(targets),
            targets,
        )

        for i, (batch_size, ctx_bucket) in enumerate(targets, 1):
            logger.info(
                "\n[%s/%s] Extracting graph for decode (batch=%s, seq=%s)",
                i,
                len(targets),
                batch_size,
                ctx_bucket,
            )
            try:
                self.model_runner.extract_decode_graphs(
                    batch_size, ctx_bucket=ctx_bucket
                )
                logger.info(
                    "  Extracted graph for decode (batch=%s, seq=%s)",
                    batch_size,
                    ctx_bucket,
                )
            except Exception as e:
                logger.error(
                    "  Failed to extract graph for decode (batch=%s, seq=%s): %s",
                    batch_size,
                    ctx_bucket,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Model graph extraction failed for decode "
                    f"(batch={batch_size}, seq={ctx_bucket}). Error: {e}"
                ) from e

    def _warmup_decode(self) -> None:
        """Run decode warmup for all (batch, seq) compile targets."""
        targets = self._decode_compile_targets()
        if not targets:
            return
        logger.info(
            "\nWarming up %d decode compile targets: %s",
            len(targets),
            targets,
        )

        for i, (batch_size, ctx_bucket) in enumerate(targets, 1):
            logger.info(
                "\n[%s/%s] Warming up decode (batch=%s, seq=%s)",
                i,
                len(targets),
                batch_size,
                ctx_bucket,
            )
            try:
                self.model_runner.warmup_decode(batch_size, ctx_bucket=ctx_bucket)
                logger.info(
                    "  Warmed up decode (batch=%s, seq=%s)",
                    batch_size,
                    ctx_bucket,
                )
            except Exception as e:
                logger.error(
                    "  Failed to warmup decode (batch=%s, seq=%s): %s",
                    batch_size,
                    ctx_bucket,
                    e,
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Model warmup failed for decode "
                    f"(batch={batch_size}, seq={ctx_bucket}). Error: {e}"
                ) from e

    def execute_model(
        self,
        scheduler_output: Any,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """
        Execute the model with the given scheduler output.

        If this method returns None, sample_tokens should be called immediately after
        to obtain the ModelRunnerOutput.

        NOTE (vllm) Note that this design may be changed in future if/when structured outputs
        parallelism is re-architected.

        Args:
            scheduler_output: Output from the vLLM scheduler

        Returns:
            - ModelRunnerOutput
            - or AsyncModelRunnerOutput
            - or None
        """
        if self._profiler is not None:
            self._profiler.step()

        logger.debug("Executing model on NeuronWorker")
        output = self.model_runner.execute_model(scheduler_output)

        return output

    def _init_profiler(self, vllm_config) -> NeuronProfiler | None:
        """Create the profiler wrapper if this rank should profile."""
        profiler_config = vllm_config.profiler_config
        if profiler_config is not None and profiler_config.profiler == "cuda":
            neuron_profiler_config = NeuronProfilerConfig(
                vllm_config.additional_config.get("neuron_profiler")
            )
        else:
            neuron_profiler_config = None

        if not self._should_profile(neuron_profiler_config):
            return None

        return NeuronProfiler(profiler_config, neuron_profiler_config)

    def _should_profile(
        self, neuron_profiler_config: NeuronProfilerConfig | None
    ) -> bool:
        """Return True if this worker's local rank should participate in profiling."""
        if neuron_profiler_config is None:
            return False
        neuron_cores = neuron_profiler_config.neuron_cores
        if neuron_cores is None:
            return self.local_rank == 0
        return self.local_rank in neuron_cores

    def profile(self, is_start: bool = True, profile_prefix: str | None = None):
        """Toggle NRT profiling on/off via /start_profile and /stop_profile."""
        if self._profiler is None:
            return

        if is_start:
            self._profiler.start()
        else:
            self._profiler.stop()

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        return self.model_runner.sample_tokens(grammar_output)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Get KV cache specifications for all model layers.

        This method is required by vLLM v1 WorkerBase interface.

        Returns:
            Dictionary mapping layer names to their KV cache specifications
        """
        logger.debug("Getting KV cache spec from NeuronWorker")
        return self.model_runner.get_kv_cache_spec()

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """
        Get draft token ids from the model runner.

        This method is required by vLLM v1 WorkerBase interface.
        Delegates to the model runner to get draft token ids.

        Returns:
            DraftTokenIds or None if not available
        """
        logger.debug("Getting draft token ids from NeuronWorker")
        return self.model_runner.take_draft_token_ids()

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """
        Get the tasks supported by this worker.

        This method is required by vLLM v1 WorkerBase interface.
        Delegates to the model runner to determine supported tasks.

        Returns:
            Tuple of supported tasks
        """
        logger.debug("Getting supported tasks from NeuronWorker")
        return self.model_runner.get_supported_tasks()

    def execute_dummy_batch(self) -> None:
        """
        Execute a dummy batch for Data Parallel synchronization.

        This method is called by the vLLM engine when Data Parallel workload
        imbalance is detected (i.e., this rank has no real requests but other
        DP ranks do). It runs a minimal dummy forward pass to keep all ranks
        synchronized during collective operations.

        The implementation simply delegates to the model runner's
        execute_dummy_batch method, which handles the actual dummy execution.
        """
        logger.debug("Executing dummy batch on NeuronWorker")
        self.model_runner.execute_dummy_batch()

    def shutdown(self) -> None:
        """Clean up resources held by the worker."""
        if self._profiler is not None:
            self._profiler.stop()

        # Disable tensor capture to prevent writes during teardown
        if self.model_runner is not None:
            self.model_runner.disable_capture()

        self.model_runner.ensure_kv_transfer_shutdown()

        # This triggers an unexpected state error. Lets try avoiding unsafe_close()
        #
        # if not envs.VLLM_NEURON_CPU_MODE:
        #     # force the neuron profiles to dump the profiles
        #     torch.classes.neuron.Runtime().unsafe_close()

    def get_kv_caches(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Extract KV cache tensors from this worker's model (CPU tensors for serialization).

        Example:
            >>> kv = worker.get_kv_caches()
        """
        return self.model_runner.get_kv_caches()

    def get_block_table_info(self) -> dict:
        """Extract block table and sequence info for KV cache reconstruction.

        Example:
            >>> info = worker.get_block_table_info()
        """
        return self.model_runner.get_block_table_info()

    def get_kv_cache_config(self) -> dict:
        """Return KVCacheConfig for model-agnostic KV reconstruction.

        Example:
            >>> cfg = worker.get_kv_cache_config()
        """
        return self.model_runner.get_kv_cache_config()

    def get_block_tables(self) -> list[bytes]:
        """Return block tables for all KV cache groups.

        Example:
            >>> tables = worker.get_block_tables()
        """
        return self.model_runner.get_block_tables()

    def clear_kv_snapshot(self) -> None:
        """Release block table snapshot memory and disable snapshotting.

        Example:
            >>> worker.clear_kv_snapshot()
        """
        self.model_runner.clear_kv_snapshot()

    def get_async_scheduling_stats(self) -> dict:
        """Return async scheduling step counters from the model runner.

        Returns a dict with ``async_steps`` and ``sync_fallback_steps``.
        Returns zeros when async scheduling is not enabled.

        Example:
            >>> stats = worker.get_async_scheduling_stats()
            >>> stats["async_steps"]
            42
        """
        if not getattr(self.model_runner, "use_async_scheduling", False):
            return {"async_steps": 0, "sync_fallback_steps": 0}
        return {
            "async_steps": self.model_runner._async_steps,
            "sync_fallback_steps": self.model_runner._sync_fallback_steps,
        }

    def _use_neuron_device(self) -> bool:
        """
        VLLM_NEURON_CPU_MODE forces cpu as the device with vLLM.
        VLLM_NEURON_CPU_COMPILE forces meta as the device type with vLLM.

        This check returns if neuron is the device type being used with vLLM.
        """
        return not envs.VLLM_NEURON_CPU_MODE and not envs.VLLM_NEURON_CPU_COMPILE
