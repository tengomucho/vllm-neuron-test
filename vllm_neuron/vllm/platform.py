# SPDX-License-Identifier: Apache-2.0
"""
Neuron Platform implementation for vLLM

This module provides the NeuronPlatform class that integrates vLLM Neuron with vLLM's
platform plug-in system.
"""

import logging
import os
import time as _time
from multiprocessing.process import BaseProcess
from typing import TYPE_CHECKING

import torch
from vllm_neuron import envs

from vllm.platforms import Platform, PlatformEnum
from vllm.v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from vllm.attention.selector import AttentionSelectorConfig
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = logging.getLogger(__name__)


# Relax DCP config validation for prefill-side DCP where TP <= num_kv_heads.
# Upstream asserts TP > num_kv_heads when DCP is enabled, but prefill DCP
# with apply_prefill_dcp=True shards attention across the DCP sub-group, not full TP.
_dcp_config_patched = False


def _patch_dcp_config_validation():
    global _dcp_config_patched
    if _dcp_config_patched:
        return
    _dcp_config_patched = True
    try:
        from vllm.config.model import ModelConfig

        _orig_verify = ModelConfig.verify_with_parallel_config

        def _patched_verify(self, parallel_config):
            dcp = parallel_config.decode_context_parallel_size
            if dcp > 1 and not self.use_mla:
                # DCP prefill uses weights-replicated sharding which doesn't
                # require dcp <= tp // kv_heads. Bypass the DCP validation
                # entirely by temporarily setting DCP=1.
                saved_dcp = parallel_config.decode_context_parallel_size
                parallel_config.decode_context_parallel_size = 1
                _orig_verify(self, parallel_config)
                parallel_config.decode_context_parallel_size = saved_dcp
                return
            _orig_verify(self, parallel_config)

        ModelConfig.verify_with_parallel_config = _patched_verify
    except Exception as e:
        logger.warning("Failed to patch DCP config validation: %s", e)


class NeuronPlatform(Platform):
    # OOT (Out-Of-Tree) refers to hardware platforms that are integrated with vLLM
    # through plugin projects.
    _enum = PlatformEnum.OOT
    # Note we need to make sure torch_neuronx has been imported before
    # getting here. This is to guarantee the privateuseone binding and
    # neuron renaming has been registered with torch.
    device_name: str = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron"
    device_type: str = "cpu" if envs.VLLM_NEURON_CPU_MODE else "neuron"
    ray_device_key: str = "neuron_cores"
    supported_quantization: list[str] = ["neuron_quant", "compressed-tensors"]
    device_control_env_var: str = "NEURON_VISIBLE_DEVICES"
    _device_count: int = -1
    _termination_timeout_patched: bool = False

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        # TODO: Return based on hardware generation like GPUs and TPUs
        # It is only used for logging and debugging - hence not critical
        return f"neuron:{device_id}"

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        """Register Neuron model architectures before ModelConfig validation."""
        import os

        if os.environ.get("VLLM_NEURON_SYNTHETIC_MODEL") == "1":
            from vllm.model_executor.models.registry import ModelRegistry

            ModelRegistry.register_model(
                "SyntheticNeuronModel",
                "vllm_neuron.model.synthetic:SyntheticNeuronModel",
            )

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """Configure vLLM for vLLM Neuron platform."""
        # Patches for bidirectional KV transfer and streaming KV (DI features).
        from vllm_neuron.vllm.patches import apply_patches

        apply_patches()

        cls._patch_termination_timeouts()
        cls._register_neuron_all2all_backend()

        model_config = vllm_config.model_config
        if model_config is None:
            return

        # Component DP on dense models needs the MoE/DP engine path to
        # preserve data_parallel_size across engine core subprocesses.
        if cls._has_neuron_component_dp(vllm_config):
            cls._validate_component_dp_config(vllm_config)
        cls._maybe_enable_moe_for_component_dp(vllm_config)

        cls._validate_prefill_dcp_config(vllm_config)
        cls._validate_decode_dcp_config(vllm_config)
        cls._auto_set_neuron_connector_module_path(vllm_config)
        cls._validate_quantization_config(model_config)

        parallel_config = vllm_config.parallel_config
        # TODO: Implement a CPU fallback based on the value from DeviceConfig
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = (
                "vllm_neuron.vllm.worker.neuron_worker.NeuronWorker"
            )
            # Set default block_size for Neuron if not specified by user
            cache_config = vllm_config.cache_config
            if cache_config.block_size is None:
                cache_config.block_size = 32
            # TODO: Move all Neuron specific config overrides into a separate function
            # TODO: Can vLLM to vLLM Neuron config mapping be done here after the overrides are applied?

        allowed_backends = ("mp", "uni")
        if parallel_config.distributed_executor_backend not in allowed_backends:
            # multi-process executor is now the default executor in vLLM. But want
            # to explicitly set it here for clarity
            parallel_config.distributed_executor_backend = "mp"

        logger.warning(
            f"Neuron Platform only supports the following distributed executor backends: {allowed_backends}, using {parallel_config.distributed_executor_backend}"
        )

        # Set custom scheduler for Neuron platform
        scheduler_config = vllm_config.scheduler_config

        if scheduler_config.enable_chunked_prefill:
            logger.warning(
                "Chunked prefill is enabled on Neuron platform. Currently Neuron only supports"
                " chunking prefills with batch size of 1. Mixing prefill and decode in the same"
                " batch is not supported"
            )

        speculative_config = getattr(vllm_config, "speculative_config", None)
        if speculative_config and scheduler_config.async_scheduling:
            # TODO: Remove this override once speculative decoding is
            # stabilized with async scheduling.
            logger.warning(
                "Speculative decoding is enabled, disabling async scheduling."
            )
            scheduler_config.async_scheduling = False

        # Override the default vLLM scheduler with our custom Neuron scheduler
        # We need to override even if it's already set to the default scheduler
        if scheduler_config.scheduler_cls is None or scheduler_config.scheduler_cls in (
            "vllm.v1.core.sched.scheduler.Scheduler",
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler",
        ):
            if scheduler_config.async_scheduling:
                scheduler_config.scheduler_cls = (
                    "vllm_neuron.vllm.core.scheduler.NeuronAsyncScheduler"
                )
            else:
                scheduler_config.scheduler_cls = (
                    "vllm_neuron.vllm.core.scheduler.NeuronScheduler"
                )
        else:
            logger.warning(
                "scheduler_cls already set to non-default: %s, "
                "NOT overriding with custom Neuron scheduler",
                scheduler_config.scheduler_cls,
            )

    @classmethod
    def _validate_quantization_config(cls, model_config) -> None:
        """Validate quantization config. Only KV cache quantization
        (q_scale/k_scale/v_scale) is supported for compressed-tensors."""
        quant_cfg = getattr(model_config.hf_config, "quantization_config", None)
        if not quant_cfg or quant_cfg.get("quant_method") != "compressed-tensors":
            return
        for group_name, group in quant_cfg.get("config_groups", {}).items():
            if group.get("weights"):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Weight quantization "
                    f"in config_groups['{group_name}'] is not supported."
                )
            if group.get("output_activations"):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Output activation "
                    f"quantization in config_groups['{group_name}'] is not "
                    f"supported."
                )
            targets = group.get("targets", [])
            if group.get("input_activations") and not any(
                "Attention" in t for t in targets
            ):
                raise ValueError(
                    f"Neuron only supports compressed-tensors for KV cache "
                    f"quantization (q_scale/k_scale/v_scale). Input activation "
                    f"quantization on non-attention targets in "
                    f"config_groups['{group_name}'] is not supported."
                )

    @staticmethod
    def _has_neuron_component_dp(vllm_config) -> bool:
        """Check if any Neuron component DP size > 1 in additional_config."""
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        return (
            neuron_config.get("attention_dp_size", 1) > 1
            or neuron_config.get("embedding_dp_size", 1) > 1
            or neuron_config.get("mlp_dp_size", 1) > 1
            or neuron_config.get("lm_head_dp_size", 1) > 1
        )

    @classmethod
    def _maybe_enable_moe_for_component_dp(cls, vllm_config) -> None:
        """Mark model as MoE when component DP needs cross-DP process groups.

        vLLM's run_engine_core only keeps data_parallel_size > 1 for MoE models.
        Component DP creates cross-DP process groups that require all DP workers
        in the same torch distributed world. Setting num_experts=1 causes
        run_engine_core to take the DPEngineCoreProc path.
        """
        if (
            vllm_config.parallel_config.data_parallel_size > 1
            and not vllm_config.model_config.is_moe
            and cls._has_neuron_component_dp(vllm_config)
        ):
            vllm_config.model_config.model_arch_config.num_experts = 1
            logger.info(
                "Component DP enabled — setting num_experts=1 to preserve "
                "data_parallel_size for cross-DP process groups"
            )

    @classmethod
    def _validate_component_dp_config(cls, vllm_config) -> None:
        """Validate component DP (attention/embedding/mlp/lm_head DP) constraints.

        Component DP is decode-only and requires disaggregated inference (DI)
        with data_parallel_size > 1. Single-node non-DI serving is not supported.
        """
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        dp_configs = {
            "attention_dp_size": neuron_config.get("attention_dp_size", 1),
            "embedding_dp_size": neuron_config.get("embedding_dp_size", 1),
            "mlp_dp_size": neuron_config.get("mlp_dp_size", 1),
            "lm_head_dp_size": neuron_config.get("lm_head_dp_size", 1),
        }
        active = {k: v for k, v in dp_configs.items() if v > 1}
        if not active:
            return

        kv_config = vllm_config.kv_transfer_config
        if kv_config is None:
            raise ValueError(
                f"Component DP ({active}) requires disaggregated inference "
                f"(--kv-transfer-config). Single-node serving without DI is "
                f"not supported with component DP."
            )

        kv_role = kv_config.kv_role
        if kv_role not in ("kv_consumer", "kv_both"):
            raise ValueError(
                f"Component DP ({active}) is only supported on the decode "
                f"server (kv_role='kv_consumer' or 'kv_both'), got "
                f"kv_role='{kv_role}'. The prefill server should not set "
                f"component DP sizes > 1."
            )

        if vllm_config.parallel_config.data_parallel_size < 2:
            raise ValueError(
                f"Component DP ({active}) requires --data-parallel-size >= 2. "
                f"Component DP shards modules across DP replicas and needs "
                f"multiple DP ranks to function."
            )

    @classmethod
    def _validate_prefill_dcp_config(cls, vllm_config) -> None:
        """Validate apply_prefill_dcp constraints.

        DCP prefill requires disaggregated inference on the prefill server
        with decode_context_parallel_size > 1.
        """
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        if not neuron_config.get("apply_prefill_dcp", False):
            return

        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_size <= 1:
            raise ValueError(
                "apply_prefill_dcp requires --decode-context-parallel-size > 1"
            )

        kv_config = vllm_config.kv_transfer_config
        if kv_config is None:
            raise ValueError(
                "apply_prefill_dcp requires disaggregated inference "
                "(--kv-transfer-config must be set)"
            )

        if kv_config.kv_role != "kv_producer":
            raise ValueError(
                f"apply_prefill_dcp is only supported on the prefill server "
                f"(kv_role must be 'kv_producer', got '{kv_config.kv_role}')"
            )

        if kv_config.kv_connector != "NeuronNixlConnector":
            raise ValueError(
                "apply_prefill_dcp requires kv_connector='NeuronNixlConnector'. "
                'Set "kv_connector": "NeuronNixlConnector" in --kv-transfer-config'
            )

    @classmethod
    def _validate_decode_dcp_config(cls, vllm_config) -> None:
        """Validate DCP decode constraints.

        DCP decode requires:
        - tp > num_kv_heads (KV must be replicated across DCP group)
        - dcp <= tp // num_kv_heads
        - (num_q_heads // num_kv_heads) % dcp == 0
        """
        neuron_config = vllm_config.additional_config.get("neuron_config", {})
        if neuron_config.get("apply_prefill_dcp", False):
            return
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        if dcp_size <= 1:
            return
        tp_size = vllm_config.parallel_config.tensor_parallel_size
        num_kv_heads = vllm_config.model_config.get_total_num_kv_heads()
        num_q_heads = vllm_config.model_config.hf_config.num_attention_heads
        if tp_size <= num_kv_heads:
            raise ValueError(
                f"DCP decode requires tp > num_kv_heads "
                f"(tp={tp_size}, num_kv_heads={num_kv_heads}). "
                f"KV must be replicated across the DCP group."
            )
        max_dcp = tp_size // num_kv_heads
        if dcp_size > max_dcp:
            raise ValueError(
                f"DCP decode requires dcp <= tp // num_kv_heads "
                f"(dcp={dcp_size}, tp // kv_heads={max_dcp})"
            )
        gqa_ratio = num_q_heads // num_kv_heads
        if gqa_ratio % dcp_size != 0:
            raise ValueError(
                f"DCP decode requires (num_q_heads // num_kv_heads) % dcp == 0 "
                f"(gqa_ratio={gqa_ratio}, dcp={dcp_size})"
            )

        kv_config = vllm_config.kv_transfer_config
        if kv_config is not None and kv_config.kv_connector != "NeuronNixlConnector":
            raise ValueError(
                "DCP decode with DI requires kv_connector='NeuronNixlConnector'. "
                'Set "kv_connector": "NeuronNixlConnector" in --kv-transfer-config'
            )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_neuron.parallel.neuron_communicator.NeuronDeviceCommunicator"

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: AttentionBackendEnum,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        logger.info("Using NEURON custom attention backend.")
        return "vllm_neuron.vllm.attention.attn.NeuronAttentionBackend"

    @classmethod
    def get_nixl_supported_devices(cls) -> dict[str, tuple[str, ...]]:
        """
        Returns a mapping from device_type to a tuple of supported
        kv_buffer_device for nixl.
        """
        # "cuda" here means device. Neuron DI uses device-to-device
        # direct transfer.
        return {NeuronPlatform.device_type: {"cuda"}}

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """
        Whether the platform supports hybrid KV cache with multiple cache groups.

        When True, models with mixed attention types (e.g., full attention and
        sliding window attention) will use separate KV cache groups, enabling
        memory optimization by freeing blocks outside the sliding window.

        Returns:
            True to enable hybrid KV cache support on Neuron.
        """
        return True

    @classmethod
    def device_count(cls) -> int:
        """
        Gets the visible device count. This returns the configured device count
        if set, otherwise falls back to the number of visible devices returned
        by the torch runtime.
        """
        if cls._device_count != -1:
            return cls._device_count

        runtime = torch.classes.neuron.Runtime()
        nc_count = runtime.get_nc_count()
        if nc_count == -1:
            raise RuntimeError(
                "Neuron runtime cannot be initialized; cannot determine the number of available NeuronCores"
            )
        return nc_count

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        pass

    @classmethod
    def set_device_count(cls, device_count: int) -> None:
        """
        Sets the visible device count.

        In the new vLLM Neuron backend, each worker thread has a single visible device,
        so the torch runtime can only see one device. This is a problem because
        vLLM expects that the device count returns the number of devices visible
        to a node, such as when initializing MultiprocExecutor for a multi-node
        server. To fix that issue, each worker thread sets the device count to
        the number of visible devices on the local node.
        """
        cls._device_count = device_count

    @classmethod
    def _auto_set_neuron_connector_module_path(cls, vllm_config) -> None:
        """Auto-inject kv_connector_module_path for NeuronNixlConnector.

        The factory needs module_path to locate the class in worker subprocesses.
        We set it here so users only need to specify kv_connector name.
        """
        kv_config = vllm_config.kv_transfer_config
        if kv_config is None:
            return
        if (
            kv_config.kv_connector == "NeuronNixlConnector"
            and not kv_config.kv_connector_module_path
        ):
            kv_config.kv_connector_module_path = (
                "vllm_neuron.vllm.kv_connector.neuron_nixl_connector"
            )

    @classmethod
    def _register_neuron_all2all_backend(cls) -> None:
        """Register 'neuron' as an all2all_backend option in vLLM's ParallelConfig.

        vLLM's ParallelConfig.all2all_backend only accepts CUDA backends by default.
        This patches ParallelConfig's pydantic schema to also accept 'neuron'.

        See: https://docs.vllm.ai/en/latest/design/moe_kernel_features/#fused-moe-modular-all2all-backends
        """
        from pydantic_core import SchemaValidator
        from vllm.config import ParallelConfig

        schema = ParallelConfig.__pydantic_core_schema__

        for field in schema["schema"]["schema"]["schema"]["fields"]:
            if field.get("name") == "all2all_backend":
                literal_schema = field["schema"]["schema"]
                if "neuron" not in literal_schema["expected"]:
                    literal_schema["expected"] = list(literal_schema["expected"]) + [
                        "neuron"
                    ]
                break

        ParallelConfig.__pydantic_validator__ = SchemaValidator(schema)

    @classmethod
    def _patch_termination_timeouts(cls) -> None:
        """Patch vLLM process termination to use a configurable timeout.

        Upstream vLLM has two hardcoded SIGTERM-to-SIGKILL timeout paths:

        1. ``v1/utils.py:shutdown()`` (5 s) — Main process kills the
           EngineCore process tree via ``kill_process_tree()``.
        2. ``MultiprocExecutor._ensure_worker_termination()`` (4 s) —
           EngineCore kills individual worker processes.

        When Neuron profiling is enabled (``NEURON_RT_INSPECT_ENABLE=1``),
        the runtime needs extra time after SIGTERM to flush profiling data
        to disk.  This method monkey-patches both paths so the timeout is
        controlled by ``VLLM_NEURON_WORKER_TERMINATION_TIMEOUT`` (default
        5 s, matching upstream).

        For path (1), ``v1/engine/utils.py`` does
        ``from vllm.v1.utils import shutdown`` which creates a module-level
        binding later captured by ``weakref.finalize``.  We must replace
        the name in **both** modules so the finalizer gets our version.
        """
        if cls._termination_timeout_patched:
            return
        cls._termination_timeout_patched = True

        timeout = envs.VLLM_NEURON_WORKER_TERMINATION_TIMEOUT

        if timeout == 5 and os.getenv("NEURON_RT_INSPECT_ENABLE") == "1":
            logger.warning(
                "Neuron profiling is enabled but VLLM_NEURON_WORKER_TERMINATION_TIMEOUT "
                "is at default (5s). Profile data may be incomplete. "
                "Consider setting VLLM_NEURON_WORKER_TERMINATION_TIMEOUT=60."
            )
            return

        if timeout == 5:
            return

        logger.info(
            "Patching vLLM termination timeouts to %ss (pid=%s)", timeout, os.getpid()
        )
        cls._patch_shutdown(timeout)
        cls._patch_ensure_worker_termination(timeout)

    @classmethod
    def _patch_shutdown(cls, neuron_timeout: int) -> None:
        """Patch ``v1.utils.shutdown`` (5 s → *timeout*)."""
        import vllm.v1.engine.utils as engine_utils
        import vllm.v1.utils as v1_utils
        from vllm.utils.system_utils import kill_process_tree

        def _shutdown(procs: list[BaseProcess], timeout: float | None = None) -> None:
            logger.info(
                "patched shutdown: pid=%s, timeout=%ss, %s proc(s)",
                os.getpid(),
                timeout,
                len(procs),
            )
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            timeout = max(neuron_timeout, timeout) if timeout else neuron_timeout
            deadline = _time.monotonic() + timeout
            for proc in procs:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if proc.is_alive():
                    proc.join(remaining)

            for proc in procs:
                if proc.is_alive() and (pid := proc.pid) is not None:
                    logger.warning(
                        "%s (pid=%s) still alive after %ss — sending SIGKILL",
                        proc.name,
                        pid,
                        timeout,
                    )
                    kill_process_tree(pid)

        v1_utils.shutdown = _shutdown
        engine_utils.shutdown = _shutdown

    @classmethod
    def _patch_ensure_worker_termination(cls, timeout: int) -> None:
        """Patch ``MultiprocExecutor._ensure_worker_termination`` (4 s → *timeout*)."""
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        @staticmethod  # type: ignore[misc]
        def _ensure_worker_termination(worker_procs: list[BaseProcess]) -> None:
            active = [p for p in worker_procs if p.is_alive()]
            logger.info(
                "patched _ensure_worker_termination: pid=%s, timeout=%ss, %s worker(s)",
                os.getpid(),
                timeout,
                len(active),
            )
            for p in active:
                p.terminate()

            deadline = _time.monotonic() + timeout
            for p in active:
                remaining = deadline - _time.monotonic()
                if remaining <= 0:
                    break
                if p.is_alive():
                    p.join(remaining)

            still_alive = [p for p in active if p.is_alive()]
            if still_alive:
                for p in still_alive:
                    logger.warning(
                        "Worker %s (pid=%s) still alive after %ss — sending SIGKILL",
                        p.name,
                        p.pid,
                        timeout,
                    )
                    p.kill()

        MultiprocExecutor._ensure_worker_termination = _ensure_worker_termination

    @classmethod
    def device_id_to_physical_device_id(cls, device_id: int):
        """Map logical device ID to physical Neuron core ID."""
        if cls.device_control_env_var in os.environ:
            # Parse NEURON_VISIBLE_DEVICES range (e.g., "0-7")
            core_range = os.environ[cls.device_control_env_var]
            if "-" in core_range:
                start, end = map(int, core_range.split("-"))
                available_cores = list(range(start, end + 1))
            else:
                # Single core or comma-separated list
                available_cores = [int(c.strip()) for c in core_range.split(",")]
            logger.debug(
                "device id: %s, physical device id: %s",
                device_id,
                available_cores[device_id],
            )
            return available_cores[device_id]
        return device_id
