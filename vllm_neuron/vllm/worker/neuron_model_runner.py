# SPDX-License-Identifier: Apache-2.0
"""
NeuronModelRunner - Model runner implementation for vLLM integration with Neuron

This provides the execution engine that processes SchedulerOutput and returns
ModelRunnerOutput with proper request tracking, output reordering, and persistent
state management.
"""

import logging
import os
import threading
import time
from copy import copy, deepcopy
from typing import Any, NamedTuple, cast

import numpy as np
import torch
from vllm import ModelRegistry
from vllm.v1.attention.backend import AttentionMetadata
from vllm.distributed.kv_transfer import get_kv_transfer_group, has_kv_transfer_group
from vllm.distributed.parallel_state import get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.interfaces import supports_eagle3
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm_neuron.utils.dtype_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
    SamplerOutput,
)
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin,
    KVConnectorOutput,
)

from contextlib import contextmanager

from vllm_neuron import envs
from vllm_neuron.compile.backend import model_forward_context
from vllm_neuron.metrics import (
    COMPILATION_TIME,
    MODEL_LOAD_SIZE,
    MODEL_LOAD_TIME,
    NEFF_EXECUTION_COUNT,
)
from vllm_neuron.model.neuron_config import (
    NeuronConfig,
)
from vllm_neuron.compile.capture_backend import CaptureComplete
from vllm_neuron.vllm.sample.rejection_sampler import RejectionSampler
from vllm_neuron.vllm.spec_decode.eagle import EagleProposer
from vllm_neuron.utils.bucket_utils import (
    get_max_num_batched_tokens,
    get_decode_padded_batch_size,
)
from vllm_neuron.utils.spec_decode_utils import (
    extract_next_token_ids,
    replicate_per_seq_rows,
)
from vllm_neuron.accuracy.tensor_replacement import (
    TensorReplacer,
    set_active_context,
)

logger = logging.getLogger(__name__)


NULL_BLOCK_ID = 0
# used for block kv and slot mapping padding
PAD_SLOT_ID = -1


def _remap_null_block_to_sentinel(block_table: torch.Tensor) -> torch.Tensor:
    """Remap vLLM's null-block (0) in an int32 block table to -1 so the
    Neuron attention kernel can DMA-skip inactive slots via oob_mode.skip.

    Out-of-place so prior-step async work holding the previous tensor is
    unaffected. torch.compile-friendly (no python branches on tensor values).
    """
    return torch.where(
        block_table == NULL_BLOCK_ID,
        torch.full_like(block_table, PAD_SLOT_ID),
        block_table,
    )


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    positions: torch.Tensor | None = None
    logits_indices: torch.Tensor | None = None
    aux_hidden_states: list[torch.Tensor] | None = None
    input_ids: torch.Tensor | None = None
    attn_metadata: dict | None = None


class AsyncNeuronModelRunnerOutput(AsyncModelRunnerOutput):
    """Wraps a ModelRunnerOutput whose sampled_token_ids is still a device
    tensor future.  Exactly one instance is created per step in
    sample_tokens() and shared between the async output thread (via
    enqueue_output) and the worker main thread (via the pre-_update_states
    guard).  The per-instance lock in get_output() prevents both threads
    from materializing the same future concurrently."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        model_runner,
        partial_prefill_req_ids: set[str] | None = None,
    ):
        self.model_runner_output = model_runner_output
        self._model_runner = model_runner
        self._partial_prefill_req_ids = partial_prefill_req_ids or set()
        # Per-object lock: prevents double-materialization when both the async
        # output thread and the main thread call get_output() on the same object.
        self._lock = threading.Lock()

    def _discard_partial_prefill_samples(
        self, sampled_token_ids: list[list[int]]
    ) -> None:
        """Drop sampled tokens produced by intermediate chunked-prefill steps."""
        if not self._partial_prefill_req_ids:
            return
        for req_idx, req_id in enumerate(self.model_runner_output.req_ids):
            if req_id not in self._partial_prefill_req_ids:
                continue
            if req_idx < len(sampled_token_ids):
                sampled_token_ids[req_idx] = []

    def get_output(self) -> ModelRunnerOutput:
        """
        Get the ModelRunnerOutput for this async output.

        This is a blocking call that waits until the results are ready, which
        involves copying device tensors to the host.

        Thread-safety: Both the async output thread (via enqueue_output) and
        the worker main thread (via the pre-_update_states guard) may call
        this method concurrently on the same object. A lock ensures only one
        thread performs the materialization; the other sees the already-
        converted list and skips _update_batch_state_with_samples.
        """
        with self._lock:
            sampled_token_ids = self.model_runner_output.sampled_token_ids
            if torch.is_tensor(sampled_token_ids):
                # First call: materialize device future to CPU.
                sampled_token_ids = [[x] for x in sampled_token_ids.cpu().tolist()]
                self._discard_partial_prefill_samples(sampled_token_ids)
                # Use the snapshotted req_ids from the time of sampling so that
                # tokens are written to the correct batch slot even if condense()
                # has reordered the batch since then.
                self._model_runner._update_batch_state_with_samples(
                    sampled_token_ids,
                    snapshot_req_ids=self.model_runner_output.req_ids,
                )
                self.model_runner_output.sampled_token_ids = sampled_token_ids
                logger.debug(
                    "sampled_token_ids on cpu after moving: sampled_token_ids=%s",
                    sampled_token_ids,
                )
            else:
                logger.debug(
                    "sampled_token_ids has already been actualized. "
                    "Skip updating batch state."
                )

        # Logprobs are computed in _sample() when on-device logits are available
        return self.model_runner_output


def _compute_slot_mapping_cpu(
    block_table_cpu: np.ndarray,
    slot_mapping_np: np.ndarray,
    positions: np.ndarray,
    req_indices: np.ndarray,
    block_size: int,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    cp_kv_cache_interleave_size: int = 1,
) -> None:
    """Compute slot mapping on CPU using vectorized numpy.

    This was originally how vllm computed slot mappings (vectorized numpy on
    CPU). In vllm 0.19.0 (commit fafe76b), this was replaced with a Triton
    GPU kernel. Since Triton is not available on Neuron, we keep the original
    numpy approach here.

    Supports context parallelism (CP): when cp_world_size > 1, only local
    tokens get valid slot IDs; remote tokens are mapped to -1.
    """
    max_blocks_per_req = block_table_cpu.shape[1]
    virtual_block_size = block_size * cp_world_size

    block_indices = positions // virtual_block_size
    flat_indices = req_indices * max_blocks_per_req + block_indices
    block_numbers = block_table_cpu.ravel()[flat_indices]

    if cp_world_size > 1:
        virtual_block_offsets = positions % virtual_block_size
        is_local = (
            virtual_block_offsets // cp_kv_cache_interleave_size % cp_world_size
            == cp_rank
        )
        block_offsets = (
            virtual_block_offsets
            // (cp_world_size * cp_kv_cache_interleave_size)
            * cp_kv_cache_interleave_size
            + virtual_block_offsets % cp_kv_cache_interleave_size
        )
        slot_mapping = block_numbers * block_size + block_offsets
        slot_mapping_np[: len(positions)] = np.where(is_local, slot_mapping, -1)
    else:
        block_offsets = positions % block_size
        np.add(
            block_numbers * block_size,
            block_offsets,
            out=slot_mapping_np[: len(positions)],
        )


def build_sampling_params_tensor(
    sampling_metadata, num_reqs: int, device: torch.device
) -> torch.Tensor:
    """Build a (num_reqs, 3) tensor of [top_k, top_p, temperature] from vLLM sampling metadata.

    vLLM's InputBatch (gpu_input_batch.py) optimizes away sampling tensors when
    all requests share the same type: all_greedy → temperature=None, no_top_k →
    top_k=None, no_top_p → top_p=None. The provided tensors live on the InputBatch
    device (e.g. neuron:0), while absent ones need defaults.

    All tensors are placed on the caller's device to avoid mixed-device errors
    in torch.stack. Provided tensors are defensively moved to the target device
    in case they originate from a different device.

    Dtype handling: top_k is int32 in InputBatch but we need float32 for
    stacking. We use .float() instead of .to(torch.float32) because the latter
    fails on XLA devices with a dtype mismatch error.

    Args:
        sampling_metadata: vLLM SamplingMetadata with per-request params.
        num_reqs: Number of active requests in the batch.
        device: Device to create tensors on (must match InputBatch device).

    Returns:
        Tensor of shape (num_reqs, 3) with columns [top_k, top_p, temperature].
    """
    # Defaults: top_k=-1 (disabled), top_p=1.0 (disabled), temperature=0.0 (greedy).
    top_k = torch.full((num_reqs,), -1, dtype=torch.float32, device=device)
    top_p = torch.ones(num_reqs, dtype=torch.float32, device=device)
    temperature = torch.zeros(num_reqs, dtype=torch.float32, device=device)

    # Override with per-request values when vLLM provides them.
    # Defensive .to(device) in case tensors originate from a different device.
    if sampling_metadata.top_k is not None:
        # [CHRS-711] TODO: Find a way to do avoid round-trip.
        top_k = sampling_metadata.top_k.cpu().float().to(device)
    if sampling_metadata.top_p is not None:
        top_p = sampling_metadata.top_p.to(device)
    if sampling_metadata.temperature is not None:
        temperature = sampling_metadata.temperature.to(device)

    result = torch.stack([top_k, top_p, temperature], dim=1)
    logger.debug("On-device sampling params (top_k, top_p, temp): %s", result.tolist())
    return result


# TODO: Inherit from LoRAModelRunnerMixin to support LoRA
class NeuronModelRunner(KVConnectorModelRunnerMixin):
    """
    Model runner that executes the NeuronModel with proper state management.

    This class processes SchedulerOutput from vLLM and returns ModelRunnerOutput
    with correct request ordering and persistent state tracking.
    """

    # ------------------------------------------------------------------------
    # INITIALIZATION
    # ------------------------------------------------------------------------
    def __init__(
        self,
        vllm_config: Any,
        device: torch.device | None = None,
    ):
        """
        Initialize the model runner.

        Args:
            vllm_config: vLLM configuration object
            device: Device to use for execution (neuron:0 by default)
        """
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config

        # TODO: Initialize LORA configuration and manager
        # Why required: Required before LORA adapters can be used
        # Current: No LORA config or manager initialization
        # Target: self.lora_config = vllm_config.lora_config if vllm_config else None; self.lora_manager = None

        # TODO: Add prefix caching support check
        # Why required: Prevents silent failures when APC is enabled (not yet supported on Neuron)
        # Current: No validation, will fail silently or cause incorrect behavior
        # Target: Raise NotImplementedError if cache_config.enable_prefix_caching is True with clear error message
        self.model: Any | None = None
        self._is_synthetic_model: bool = False
        model_config = vllm_config.model_config
        self.is_pooling_model = False
        self.uses_mrope = model_config.uses_mrope
        self.uses_xdrope_dim = model_config.uses_xdrope_dim

        # Extract config from vllm_config
        # For multimodal models (e.g., Qwen3-VL), vocab_size is nested in text_config
        hf_config = vllm_config.model_config.hf_config
        if hasattr(hf_config, "text_config") and hasattr(
            hf_config.text_config, "vocab_size"
        ):
            self.vocab_size = hf_config.text_config.vocab_size
        else:
            self.vocab_size = hf_config.vocab_size
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        # Initialize persistent batch and request tracking
        # Use provided device or default to neuron:0
        # Device should be provided by worker after setting env vars
        self.device = device if device is not None else torch.device("neuron:0")
        self.pin_memory = False  # Default to False
        # Prompt-embeds config and per-step staging state.
        self.enable_prompt_embeds = model_config.enable_prompt_embeds
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        # Staged prompt_embeds for the current step (None => token-id-only step).
        self._current_inputs_embeds: torch.Tensor | None = None
        # Selection mask for the current step (True=token-id embedding).
        self._current_is_token_ids: torch.Tensor | None = None
        # Log state tracker to avoid repeated "path active" logs on every step.
        self._last_prompt_embeds_path_active: bool | None = None
        if self.enable_prompt_embeds:
            logger.info(
                "PROMPT_EMBEDS enabled in model config. "
                "Default path remains token IDs; pass `prompt_embeds` with shape "
                "[sequence_length, hidden_size] to activate prompt-embeds path."
            )
        self.max_num_batched_tokens = get_max_num_batched_tokens(
            self.max_model_len, vllm_config.scheduler_config.max_num_batched_tokens
        )

        # Initialize NeuronConfig from additional_config
        neuron_config_dict = vllm_config.additional_config.get("neuron_config", {})
        # TODO: Remove max_logprobs from NeuronConfig; use vllm_config directly in the model instead
        neuron_config_dict.setdefault(
            "max_logprobs", getattr(vllm_config.model_config, "max_logprobs", 0)
        )
        self.neuron_config = NeuronConfig.from_dict(neuron_config_dict)

        # Vision neuron config for multimodal models (None for text-only)
        vision_config_dict = vllm_config.additional_config.get("vision_neuron_config")
        if vision_config_dict is not None:
            from vllm_neuron.model.neuron_config import VisionNeuronConfig

            self.vision_neuron_config = VisionNeuronConfig.from_dict(vision_config_dict)
        else:
            self.vision_neuron_config = None

        self.on_device_sampling: bool = (
            self.neuron_config.on_device_sampling_config is not None
        )
        self._on_device_logits: torch.Tensor | None = None
        self._debug_logits_dir: str | None = self.neuron_config.debug_logits_dir
        self._debug_logits_step_counter: int = 0
        self._tensor_replacer = None

        # Context parallelism: DCP rank for KV cache sharding.
        parallel_config = vllm_config.parallel_config
        self._dcp_size = parallel_config.decode_context_parallel_size
        neuron_cfg = vllm_config.additional_config.get("neuron_config", {})
        self.cp_world_size = self._dcp_size
        self.cp_kv_cache_interleave_size = getattr(
            parallel_config, "cp_kv_cache_interleave_size", 1
        )
        if self._dcp_size > 1 and vllm_config.kv_transfer_config is not None:
            block_size = vllm_config.cache_config.block_size
            assert self.cp_kv_cache_interleave_size == block_size, (
                f"DI with DCP requires --cp-kv-cache-interleave-size={block_size} "
                f"(must equal block_size), got {self.cp_kv_cache_interleave_size}"
            )
        # cp_rank computed lazily on first access (process groups not yet initialized)
        self._cp_rank: int | None = None

        # Parse num_batched_tokens_buckets from config or compute defaults
        # This must happen before InputBatch creation to ensure buffers are sized correctly
        from vllm_neuron.utils.bucket_utils import (
            get_default_num_batched_tokens_buckets,
            get_default_num_seqs_buckets,
            resolve_segmented_prefill_config,
            validate_decode_context_length_buckets,
            validate_kv_segment_size_buckets,
            validate_num_batched_tokens_buckets,
            validate_num_seqs_buckets,
        )

        user_set_kv_segment_size_buckets = (
            "kv_segment_size_buckets" in neuron_config_dict
        )
        user_set_num_batched_tokens_buckets = (
            "num_batched_tokens_buckets" in neuron_config_dict
        )

        auto_kv_segment_size_buckets: list[int] | None = None
        auto_num_batched_tokens_buckets: list[int] | None = None
        if not user_set_kv_segment_size_buckets:
            (
                auto_kv_segment_size_buckets,
                auto_num_batched_tokens_buckets,
            ) = resolve_segmented_prefill_config(
                self.max_num_batched_tokens, self.max_model_len
            )

        if user_set_num_batched_tokens_buckets:
            buckets_from_config = neuron_config_dict.get("num_batched_tokens_buckets")
            self.neuron_config.num_batched_tokens_buckets = (
                validate_num_batched_tokens_buckets(
                    buckets_from_config, self.max_num_batched_tokens
                )
            )
            logger.info(
                "Using num_batched_tokens_buckets from config: %s",
                self.neuron_config.num_batched_tokens_buckets,
            )
        elif auto_num_batched_tokens_buckets is not None:
            self.neuron_config.num_batched_tokens_buckets = (
                auto_num_batched_tokens_buckets
            )
        else:
            self.neuron_config.num_batched_tokens_buckets = (
                get_default_num_batched_tokens_buckets(self.max_num_batched_tokens)
            )
            logger.info(
                "Using default num_batched_tokens_buckets: %s",
                self.neuron_config.num_batched_tokens_buckets,
            )

        # Parse num_seqs_buckets from config or compute defaults
        if "num_seqs_buckets" in neuron_config_dict:
            buckets_from_config = neuron_config_dict.get("num_seqs_buckets")
            self.neuron_config.num_seqs_buckets = validate_num_seqs_buckets(
                buckets_from_config, self.max_num_reqs
            )
            logger.info(
                "Using num_seqs_buckets from config: %s",
                self.neuron_config.num_seqs_buckets,
            )
        else:
            self.neuron_config.num_seqs_buckets = get_default_num_seqs_buckets(
                self.max_num_reqs
            )
            logger.info(
                "Using default num_seqs_buckets: %s",
                self.neuron_config.num_seqs_buckets,
            )

        # Parse decode_context_length_buckets — opt-in feature that compiles a
        # 2D bucket grid (batch, seq). Default None preserves today's
        # behavior of compiling decode at max_model_len.
        if "decode_context_length_buckets" in neuron_config_dict:
            self.neuron_config.decode_context_length_buckets = (
                validate_decode_context_length_buckets(
                    neuron_config_dict["decode_context_length_buckets"],
                    self.max_model_len,
                )
            )
            logger.info(
                "Using decode_context_length_buckets from config: %s",
                self.neuron_config.decode_context_length_buckets,
            )

        # Parse kv_segment_size_buckets: explicit opt-in takes precedence
        # over the auto-enable path.
        if user_set_kv_segment_size_buckets:
            buckets = neuron_config_dict.get("kv_segment_size_buckets")
            # Pass num_batched_tokens_buckets only if user explicitly set it
            explicit_num_batched_tokens_buckets = (
                self.neuron_config.num_batched_tokens_buckets
                if user_set_num_batched_tokens_buckets
                else None
            )
            self.neuron_config.kv_segment_size_buckets = (
                validate_kv_segment_size_buckets(
                    buckets,
                    explicit_num_batched_tokens_buckets,
                )
            )
            kv_segment_size = self.neuron_config.kv_segment_size_buckets[0]

            # Auto-set num_batched_tokens_buckets to match if not explicitly set
            if not user_set_num_batched_tokens_buckets:
                self.neuron_config.num_batched_tokens_buckets = (
                    self.neuron_config.kv_segment_size_buckets
                )
                logger.info(
                    "Auto-setting num_batched_tokens_buckets to match "
                    "kv_segment_size_buckets: %s",
                    self.neuron_config.kv_segment_size_buckets,
                )

            if self.max_num_batched_tokens < kv_segment_size:
                logger.warning(
                    "max_num_batched_tokens (%s) < segment size (%s). Adjusting to %s.",
                    self.max_num_batched_tokens,
                    kv_segment_size,
                    kv_segment_size,
                )
                self.max_num_batched_tokens = kv_segment_size

            logger.info(
                "Segmented prefill enabled explicitly with kv_segment_size_buckets: %s",
                self.neuron_config.kv_segment_size_buckets,
            )
        elif auto_kv_segment_size_buckets is not None:
            # Cross-check the resolver's segment size against the user's
            # explicit num_batched_tokens_buckets (if any), using the same
            # validation rule as the explicit opt-in path.
            explicit_num_batched_tokens_buckets = (
                self.neuron_config.num_batched_tokens_buckets
                if user_set_num_batched_tokens_buckets
                else None
            )
            self.neuron_config.kv_segment_size_buckets = (
                validate_kv_segment_size_buckets(
                    auto_kv_segment_size_buckets,
                    explicit_num_batched_tokens_buckets,
                )
            )
            logger.info(
                "Segmented prefill auto-enabled with "
                "kv_segment_size_buckets=%s, num_batched_tokens_buckets=%s",
                self.neuron_config.kv_segment_size_buckets,
                self.neuron_config.num_batched_tokens_buckets,
            )
        else:
            logger.info(
                "Segmented prefill disabled (single-shot prefill; "
                "max_num_batched_tokens == max_model_len)"
            )

        # Ensure max_num_batched_tokens is large enough for the largest prefill bucket
        # This is required because slot_mapping buffer is sized by max_num_batched_tokens
        max_bucket_size = max(self.neuron_config.num_batched_tokens_buckets)
        if self.max_num_batched_tokens < max_bucket_size:
            logger.warning(
                "max_num_batched_tokens (%s) is smaller than largest prefill bucket "
                "(%s). Adjusting to %s to ensure buffers are large enough for warmup "
                "compilation.",
                self.max_num_batched_tokens,
                max_bucket_size,
                max_bucket_size,
            )
            self.max_num_batched_tokens = max_bucket_size

        logger.debug("Neuron config: %s", self.neuron_config)

        self.use_async_scheduling: bool = vllm_config.scheduler_config.async_scheduling
        if self.use_async_scheduling:
            if not self.on_device_sampling:
                raise RuntimeError(
                    "On-device sampling must be enabled for async execution"
                )
            # The buffer to facilitate async execution.
            self.async_execution_buffer = dict()
            # Counters for observability: track how many forward steps used
            # the async path vs sync fallback. Logged periodically and useful
            # for tests to verify async scheduling is actually engaged.
            self._async_steps: int = 0
            self._sync_fallback_steps: int = 0
            self._batch_composition_changed: bool = False

        # Set up speculative decoding.
        self.drafter = None
        self.is_eagle3_spec = False
        self._draft_token_ids = None

        if self.speculative_config:
            # TODO make spec work with async + on-device sampling
            if self.use_async_scheduling:
                raise ValueError(
                    "Speculative decoding is currently not supported with async execution"
                )
            # TODO add more spec decode methods
            if self.speculative_config.method == "eagle3":
                self.is_eagle3_spec = True
                self.drafter = EagleProposer(
                    self.vllm_config, self.device, self.on_device_sampling
                )
                self.rejection_sampler = RejectionSampler()
            else:
                raise ValueError(
                    f"Unsupported speculative decoding method: {self.speculative_config.method}"
                )

        # InputBatch is initialized in initialize_kv_cache once KV cache groups are known
        self.input_batch = None
        self._kv_snapshot_enabled: bool = False
        self._block_table_snapshot: list[torch.Tensor] | None = None
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}
        # Encoder cache for multimodal models
        self.encoder_cache: dict[str, Any] = {}

        # Ephemeral state for execute_model() -> sample_tokens() flow
        self.execute_model_state: ExecuteModelState | None = None
        self.kv_connector_output: KVConnectorOutput | None = None
        # Full (2, ...) KV cache tensors for DI connector registration
        self._kv_cache_full_tensors: dict[str, torch.Tensor] = {}

        # Initialize vLLM Sampler for proper token sampling
        logprobs_mode = getattr(
            vllm_config.model_config, "logprobs_mode", "raw_logprobs"
        )
        self.sampler = Sampler(logprobs_mode=logprobs_mode)

        # Cache numpy arange for efficient tensor operations
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_batched_tokens),
            dtype=np.int64,
        )

        # We pass rank as a tensor input to avoid it becoming a constant
        # TODO: Update dist.get_rank() to avoid becoming a constant during lowering, and remove this
        # Use TP-local rank (not world rank) so DP doesn't affect model computation
        tp_group = get_tp_group()
        tp_rank = tp_group.rank_in_group if tp_group else 0
        self.rank_tensor = torch.tensor(tp_rank, dtype=torch.int32, device=self.device)

        logger.info(
            "Initialized NeuronModelRunner with config: max_num_reqs=%s, "
            "vocab_size=%s, max_model_len=%s, block_size=%s, device=%s",
            self.max_num_reqs,
            self.vocab_size,
            self.max_model_len,
            self.vllm_config.cache_config.block_size,
            self.device,
        )

    @property
    def cp_rank(self) -> int:
        if self._cp_rank is None:
            if self.cp_world_size <= 1:
                self._cp_rank = 0
            else:
                from vllm.distributed.parallel_state import get_tp_group

                tp_rank = get_tp_group().rank_in_group
                tp_size = get_tp_group().world_size
                neuron_cfg = self.vllm_config.additional_config.get("neuron_config", {})
                if neuron_cfg.get("apply_prefill_dcp", False):
                    # DCP prefill: cp_rank = token group index
                    self._cp_rank = tp_rank // (tp_size // self._dcp_size)
                else:
                    # DCP decode: cp_rank = position within DCP group
                    self._cp_rank = tp_rank % self._dcp_size
        return self._cp_rank

    def _reset_state(self) -> None:
        """
        Reset internal state to clean state.

        This is called after warmup to clear dummy requests and prepare
        the model runner for real inference.
        """
        # Remove all requests from input_batch
        for req_id in list(self.input_batch.req_id_to_index.keys()):
            self.input_batch.remove_request(req_id)
        self.input_batch.condense()

        # Clear request cache
        self.requests.clear()

        logger.debug("Model runner state reset successfully")

    def _get_grammar_bitmask(
        self,
        scheduler_output: SchedulerOutput,
    ) -> torch.Tensor | None:
        """
        Get grammar bitmask from scheduler_output for structured outputs.

        Builds a full logit-row-aligned bitmask by reordering the compact bitmask
        (which only contains rows for structured output requests) to match the
        order of requests in input_batch. Non-SO rows are filled with -1 (all tokens
        allowed). Handles speculative decode offsets.

        This mirrors vLLM's apply_grammar_bitmask() reordering logic but adapted
        for on-device sampling where the mask is passed into the model forward.

        Args:
            scheduler_output: Output from scheduler containing:
                - _grammar_bitmask: Compact numpy array [num_so_reqs, packed_vocab_size]
                - _structured_output_request_ids: List of request IDs with SO

        Returns:
            Boolean tensor [num_logit_rows, vocab_size] or None if no bitmask
        """
        packed_bitmask_np = getattr(scheduler_output, "_grammar_bitmask", None)
        struct_req_ids = getattr(
            scheduler_output, "_structured_output_request_ids", None
        )

        if packed_bitmask_np is None:
            logger.debug("[SO] model_runner: no grammar bitmask in scheduler_output")
            return None

        # Guard: bitmask present but request IDs missing indicates scheduler bug
        if not struct_req_ids:
            logger.warning(
                "[SO] Grammar bitmask present but structured_output_request_ids missing. "
                "This indicates a scheduler invariant violation."
            )
            return None

        # Convert numpy to torch on CPU once
        packed_bitmask = torch.from_numpy(packed_bitmask_np).to(torch.int32)
        # Rows = logit rows (SO reqs + spec), NOT prompt tokens / prefill bucket
        num_compact_rows = packed_bitmask.shape[0]
        packed_vocab_size = packed_bitmask.shape[1]

        # Build mapping: req_id -> logit_index (accounting for spec tokens)
        struct_out_req_batch_indices: dict[str, int] = {}
        cumulative_offset = 0
        spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        struct_out_req_ids_set = set(struct_req_ids) if struct_req_ids else set()

        for batch_index, req_id in enumerate(self.input_batch.req_ids):
            logit_index = batch_index + cumulative_offset
            cumulative_offset += len(spec_tokens.get(req_id, ()))
            if req_id in struct_out_req_ids_set:
                struct_out_req_batch_indices[req_id] = logit_index

        # Calculate total logit rows (requests + spec tokens)
        num_logit_rows = len(self.input_batch.req_ids) + sum(
            len(spec_tokens.get(r, ())) for r in self.input_batch.req_ids
        )

        # Full bitmask: -1 (all allowed) for non-SO rows, aligned to logit rows
        sorted_bitmask = torch.full(
            (num_logit_rows, packed_vocab_size),
            fill_value=-1,
            dtype=torch.int32,
        )

        # Reorder: copy SO request masks to correct positions
        cumulative_index = 0
        for req_id in struct_req_ids:
            num_spec = len(spec_tokens.get(req_id, ()))
            if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
                for i in range(1 + num_spec):
                    bitmask_index = logit_idx + i
                    sorted_bitmask[bitmask_index] = packed_bitmask[cumulative_index + i]
            cumulative_index += 1 + num_spec

        # Sanity check: ensure we consumed all rows in packed bitmask
        # Use explicit check instead of assert (asserts can be optimized away with -O)
        if cumulative_index != num_compact_rows:
            raise RuntimeError(
                f"Bitmask alignment error: consumed {cumulative_index} rows but "
                f"packed_bitmask has {num_compact_rows} rows. "
                "This indicates a mismatch between scheduler and model_runner."
            )

        # Unpack the bitmask: int32 -> 32 boolean bits per value
        packed_uint = sorted_bitmask.view(num_logit_rows, packed_vocab_size, 1)

        bit_positions = torch.arange(32, dtype=torch.int32)

        unpacked = ((packed_uint >> bit_positions) & 1).view(num_logit_rows, -1)
        unpacked = unpacked[:, : self.vocab_size]
        bitmask = unpacked.bool()

        # NOTE: TP sharding of the mask is handled in vLLM Neuron sampler (matches lm_head sharding)

        # Move to device only for on-device sampling path.
        # For CPU sampling, keep bitmask on CPU — it will be applied to logits
        # after they return from the model forward pass.
        if self.on_device_sampling:
            bitmask = bitmask.to(device=self.device)

        # Log bitmask statistics for structured output debugging
        num_allowed = bitmask.sum().item()
        total_tokens = bitmask.numel()
        pct_allowed = 100 * num_allowed / total_tokens if total_tokens > 0 else 0

        logger.debug(
            "[SO] model_runner: bitmask shape=%s, allowed=%d/%d (%.1f%%), reqs=%s",
            bitmask.shape,
            num_allowed,
            total_tokens,
            pct_allowed,
            list(struct_out_req_ids_set),
        )

        return bitmask

    def _apply_grammar_bitmask_cpu(
        self,
        logits: torch.Tensor,
        bitmask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply grammar bitmask to logits on CPU for structured output constraints.

        Used in the CPU sampling path where logits return from the device to CPU
        before sampling. This is the CPU-side equivalent of the on-device
        masked_fill performed by the vLLM Neuron sampler.

        Args:
            logits: [num_logit_rows, vocab_size] logits tensor on CPU.
            bitmask: [num_logit_rows, vocab_size] boolean mask on CPU.
                     True = token allowed, False = token disallowed (set to -inf).

        Returns:
            Logits tensor with disallowed tokens set to -inf.
        """
        num_allowed = bitmask.sum().item()
        total_tokens = bitmask.numel()
        pct_allowed = 100.0 * num_allowed / total_tokens if total_tokens > 0 else 0.0
        logger.debug(
            "[SO] CPU bitmask applied: logits=%s, allowed=%d/%d (%.1f%%)",
            logits.shape,
            num_allowed,
            total_tokens,
            pct_allowed,
        )
        return logits.masked_fill(~bitmask, float("-inf"))

    def load_model(self) -> None:
        """Load the NeuronModel."""
        logger.info("Starting NeuronModel loading process")

        model_cls, _ = ModelRegistry.resolve_model_cls(
            self.vllm_config.model_config.architecture,
            model_config=self.vllm_config.model_config,
        )

        # TODO: validate and ensure all vLLM configs are complied to

        # TODO: remove assert once eager mode on Neuron is supported
        cpu_mode = envs.VLLM_NEURON_CPU_MODE
        cpu_compile = envs.VLLM_NEURON_CPU_COMPILE
        eager_mode = self.vllm_config.model_config.enforce_eager
        skip_graph_capture_backend = envs.VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND
        assert not (eager_mode and not cpu_mode), (
            "Eager mode on Neuron is not yet supported."
        )
        if eager_mode:
            torch.compiler.set_stance("force_eager")

        from vllm_neuron.model.synthetic import SyntheticNeuronModel

        self._is_synthetic_model = issubclass(model_cls, SyntheticNeuronModel)
        if self._is_synthetic_model:
            self.model = model_cls.from_configs(
                self.vllm_config.model_config.hf_config,
            )
            logger.info(
                "SyntheticNeuronModel — skipping weight loading and compilation"
            )
            self._capture_model = None
            self._capture_registry = None
            return

        with torch.device("meta"):
            if self.vision_neuron_config is not None:
                # Init multimodal model with vision_neuron_config if provided
                self.model = model_cls.from_configs(
                    hf_config=self.vllm_config.model_config.hf_config,
                    text_neuron_config=self.neuron_config,
                    vision_neuron_config=self.vision_neuron_config,
                )
            else:
                # Use only (text) neuron_config to init model. Works for both text-only and multimodal model
                self.model = model_cls.from_configs(
                    self.vllm_config.model_config.hf_config,
                    self.neuron_config,
                )

        # TODO: Load LORA model wrapper if LORA config exists
        # Why required: Required to actually load LORA adapters onto the model
        # Current: Base model loaded without LORA support
        # Target: if self.lora_config: self.model = self.load_lora_model(self.model, self.vllm_config, self.device)

        # Load weights - let the model handle checkpoint loading
        if not cpu_compile:
            logger.info("Loading weights from %s", self.vllm_config.model_config.model)
            weight_load_start_time = time.perf_counter()
            self.model.load_weights(
                self.vllm_config.model_config.model,
                self.device,
                self.vllm_config.load_config.download_dir,
            )
            logger.info(
                "Weight loading completed in %.2f seconds",
                time.perf_counter() - weight_load_start_time,
            )

            # Move model to Neuron device
            logger.info("Moving model to device: %s", self.device)
            self.model = self.model.to(self.device)

            model_name = self.vllm_config.model_config.model
            MODEL_LOAD_TIME.labels(model_name=model_name).set(
                time.perf_counter() - weight_load_start_time,
            )
            MODEL_LOAD_SIZE.labels(model_name=model_name).set(
                sum(p.nbytes for p in self.model.parameters()),
            )
        else:
            if hasattr(self.model, "load_weights_lite"):
                self.model.load_weights_lite(
                    self.vllm_config.model_config.model,
                    torch.device("cpu"),  # cpu needed getting compile time constants
                    self.vllm_config.load_config.download_dir,
                )
                logger.info("Light weight loading complete for CPU compilation.")
            logger.info("CPU Compilation is enabled. Skipping full weight loading.")
            # Explicit call to force all buffers to use meta.
            self.model = self.model.to("meta")

        if self.is_eagle3_spec:
            if supports_eagle3(self.model):
                # First try to get aux_layers from draft model config, then fall back to model's default
                aux_layers = self._get_eagle3_aux_layers_from_config()
                if aux_layers:
                    logger.info(
                        "Using auxiliary layers from speculative config: %s", aux_layers
                    )
                else:
                    aux_layers = self.model.get_eagle3_aux_hidden_state_layers()
                    logger.info(
                        "Using model's default auxiliary layers: %s", aux_layers
                    )
                self.model.set_aux_hidden_state_layers(aux_layers)
            else:
                raise RuntimeError(
                    "Model does not support EAGLE3 interface but EAGLE3 spec decoding was requested"
                )

        # Tensor capture: imports and config
        from vllm_neuron.accuracy.tensor_capture import (
            TensorCaptureModel,
            CaptureWriter,
        )

        capture_config = self.neuron_config.tensor_capture
        self._capture_model = None
        self._capture_registry = None

        # Compile model with vllm_neuron backend for Neuron execution
        # This is where torch.compile uses the registered "vllm_neuron" backend
        logger.info("Compiling model with vllm_neuron backend")

        # Disable dynamo cache limit — recompilations are caught at runtime
        # by FailOnRecompileLimitHit, so a static limit adds no value.
        torch._dynamo.config.cache_size_limit = 2**62

        # Build compile options
        self.compile_options = {}

        # FIXME: -O1 and mac-threshold are temporary until NKI adds MAC count estimates for kernels.
        self.compile_options["compiler_args"] = [
            "--auto-cast=none",
            "--verbose=35",
            "-O1",
            "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10",
            "--internal-backend-options=--enable-verifier=false",
        ]

        # Check for debug mode environment variable
        # Set VLLM_NEURON_DEBUG_MODE=1 to disable fullgraph for debugging (allows print statements)
        debug_mode = envs.VLLM_NEURON_DEBUG_MODE
        fullgraph_enabled = not debug_mode

        if debug_mode:
            logger.info(
                "VLLM_NEURON_DEBUG_MODE enabled: fullgraph=False (allows print statements and graph breaks)"
            )

        # We should fail explicitly for graph cuts (unless debug mode is enabled).
        # Note: When eager mode is enabled, torch compile becomes a no op.
        from vllm_neuron.envs import get_compile_backend_name

        if eager_mode or cpu_mode or debug_mode or skip_graph_capture_backend:
            logger.debug("Graph capture and parallel compilation is disabled.")
            self.capture_backend_model = None
        else:
            # Graph extraction compile on raw model BEFORE tensor capture
            # wrapping to avoid BACKEND_MATCH conflict with the main backend.
            self.capture_backend_model = torch.compile(
                self.model,
                backend="vllm_neuron_graph_capture",
                fullgraph=fullgraph_enabled,
                options=self.compile_options,
            )

        # Tensor capture: wrap model BEFORE main compile so hooks are traced
        if capture_config:
            if capture_config.modules:
                logger.info(
                    "Tensor capture enabled for modules: %s", capture_config.modules
                )
            else:
                logger.info(
                    "modules not specified, enable tensor capture for manual tensors only"
                )
            self._capture_model = TensorCaptureModel(self.model, capture_config.modules)
            self.model = self._capture_model

            try:
                rank = (
                    torch.distributed.get_rank()
                    if torch.distributed.is_initialized()
                    else 0
                )
            except Exception:
                rank = 0

            tp_size = self.vllm_config.parallel_config.tensor_parallel_size
            dp_rank = self.vllm_config.parallel_config.data_parallel_rank
            tp_rank = rank % tp_size

            self._capture_registry = CaptureWriter(
                capture_dir=capture_config.capture_dir,
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                capture_filter=(
                    set(capture_config.capture_filter)
                    if capture_config.capture_filter is not None
                    else None
                ),
            )
            logger.info(
                "Tensor capture registry initialized, capture_dir=%s, rank=%s, dp_rank=%s, tp_rank=%s",
                capture_config.capture_dir,
                rank,
                dp_rank,
                tp_rank,
            )

        self.model = torch.compile(
            self.model,
            backend=get_compile_backend_name(),
            fullgraph=fullgraph_enabled,
            options=self.compile_options,
        )

        logger.info("NeuronModel loading complete (moved to device and compiled)")

        if self.drafter is not None:
            logger.info("Spec decode enabled. Loading draft model ...")
            # TODO: model loading logic could be extracted
            # Pass target model's padded hidden_size to draft model
            target_hidden_size = self.model.config.hidden_size
            self.drafter.load_model(target_hidden_size=target_hidden_size)
            logger.info("Draft model loading complete (moved to device and compiled)")

    def init_tensor_replacement(self) -> None:
        """Initialize tensor replacement from neuron_config."""
        replacement_config = self.neuron_config.tensor_replacement
        if not replacement_config:
            return

        tensors, prompt_token_ids = replacement_config.resolve()
        self._tensor_replacer = TensorReplacer(
            tensors, prompt_token_ids=prompt_token_ids
        )
        logger.info("Tensor replacement initialized")

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config.

        These indices specify which hidden states from the target model should
        be used as auxiliary inputs for the Eagle3 draft model during
        speculative decoding.

        Returns:
            Tuple of layer indices if found in draft model config,
            None otherwise.
        """
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    # ------------------------------------------------------------------------
    # STATE MANAGEMENT
    # ------------------------------------------------------------------------
    def get_model(self) -> Any:
        """Get the underlying model.

        Neuron does not support pooling models, so this is a simple accessor.
        Provided for API compatibility with GPUModelRunner._update_states.
        """
        if self.model is None:
            raise ValueError("Cannot get model before model has been initialized")
        return self.model

    def _may_reorder_batch(self, scheduler_output: SchedulerOutput) -> None:
        """Reorder batch to match scheduler output order if needed.

        This is a no-op for Neuron since we don't support reordering yet.
        """
        pass

    def _get_valid_sampled_token_count(self) -> list[int]:
        """Get the number of valid sampled tokens for each request.

        This is required by async scheduling + SD. On GPU, this returns
        per-request accepted token counts from the previous step. On Neuron,
        we return a single-element list with 0 because Neuron does not yet
        support speculative decoding with async scheduling, so there are no
        accepted/rejected tokens to track.

        Returns:
            List of valid token counts per request
        """
        # No-op: Neuron does not support async SD yet.
        return [0]

    def _init_mrope_positions(self, req_state: CachedRequestState) -> None:
        """Initialize M-RoPE positions for models like Qwen2-VL.

        This is a placeholder for future M-RoPE support.
        """
        pass

    def _init_xdrope_positions(self, req_state: CachedRequestState) -> None:
        """Initialize XD-RoPE positions for models like HunYuan-VL.

        This is a placeholder for future XD-RoPE support.
        """
        pass

    def _update_streaming_request(
        self, req_id: str, new_req_data: Any
    ) -> CachedRequestState:
        """Update a streaming request with new data.

        For streaming case only.
        """
        req_state = self.requests[req_id]
        # Update with new data as needed
        return req_state

    # !! NEURON MAINTAINER NOTE !!
    # The _update_states method below is copied VERBATIM from upstream
    # vllm/v1/worker/gpu_model_runner.py (GPUModelRunner._update_states).
    # DO NOT modify it directly. To update, copy the latest upstream
    # implementation and replace entirely. Neuron-specific behavior goes
    # in the helper stubs above (get_model, _may_reorder_batch,
    # _get_valid_sampled_token_count, _init_mrope_positions,
    # _init_xdrope_positions, _update_streaming_request).
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            # Register request for tensor replacement prompt matching
            if self._tensor_replacer is not None:
                self._tensor_replacer.register_request(
                    req_id, new_req_data.prompt_token_ids
                )
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if req_state.prev_num_draft_len and self.use_async_scheduling:
                if req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    new_token_ids: list[int] = []
                else:
                    new_token_ids = req_data.new_token_ids[i]
                    num_new_tokens = (
                        num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    )
                    if num_new_tokens == 1:
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(
                            new_token_ids[-num_new_tokens:]
                        )
            elif num_output_tokens < len(req_state.output_token_ids):
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids

            if req_index is None:
                if self.use_async_scheduling and num_output_tokens > 0:
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            if not is_last_rank:
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

        logger.debug(
            "Completed state update. Current batch size: %s",
            len(self.input_batch.req_ids),
        )

    def _compute_cached_seq_len(self) -> int:
        """
        Compute the maximum cached sequence length across all requests in the batch.
        Note that Neuron currently only supports a prefill batch size of 1.

        Returns:
            int: Maximum number of already-computed tokens across all requests.
                 Returns 0 if no tokens are cached.
        """
        cached_seq_len = 0
        for req_index in range(self.input_batch.num_reqs):
            num_computed = self.input_batch.num_computed_tokens_cpu[req_index]
            if num_computed > cached_seq_len:
                cached_seq_len = num_computed

        if cached_seq_len > 0:
            logger.debug("Computed cached_seq_len=%s", cached_seq_len)

        return cached_seq_len

    # ------------------------------------------------------------------------
    # INPUT PREPARATION
    # TODO: Methods here and elsewhere are shared with GPUs and TPUs.
    # There is opportunity to move this common code to vLLM
    # ------------------------------------------------------------------------
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.

        E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        Equivalent to but faster than:
        np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _save_block_table_to_device(self) -> None:
        """
        To avoid data dependencies that block async execution, we should store the
        full block table tensor on device instead of updating it in place with commit_block_table
        """
        for block_table in self.input_batch.block_table.block_tables:
            block_table.block_table.gpu = _remap_null_block_to_sentinel(
                block_table.block_table.cpu.to(self.device)
            )
        # Snapshot block tables for KV cache analysis (survives request freeing).
        # Apply remap on CPU tensor directly to avoid device→CPU transfer.
        if self._kv_snapshot_enabled:
            self._block_table_snapshot = [
                _remap_null_block_to_sentinel(bt.block_table.cpu)
                for bt in self.input_batch.block_table.block_tables
            ]

    def _save_slot_mapping_to_device(self) -> None:
        """
        To avoid data dependencies that block async execution, we should store the
        full slot mapping tensor on device instead of updating it in place with commit_slot_mapping
        """
        for block_table in self.input_batch.block_table.block_tables:
            block_table.slot_mapping.gpu = block_table.slot_mapping.cpu.to(self.device)

    def _prepare_model_input(
        self, scheduler_output: Any
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        AttentionMetadata | None,
        SpecDecodeMetadata | None,
    ]:
        """
        Prepare model inputs from scheduler output with padding for prefill requests.

        Args:
            scheduler_output: Output from vLLM scheduler

        Returns:
            Tuple of (input_ids, positions, logit_indices, attention_metadata, spec_decode_metadata)
        """
        import traceback

        try:
            return self._prepare_model_input_impl(scheduler_output)
        except Exception as e:
            logger.error("Error in _prepare_model_input: %s", e)
            logger.error("Full traceback:\n%s", traceback.format_exc())
            raise

    def _prepare_model_input_impl(
        self, scheduler_output: Any
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        AttentionMetadata | None,
        SpecDecodeMetadata | None,
    ]:
        """Actual implementation of _prepare_model_input."""
        assert (
            scheduler_output.total_num_scheduled_tokens > 0
        )  # this is actual token count, no padding
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table to neuron first.
        # This way, we can overlap the copy with the following CPU operations.
        self._save_block_table_to_device()

        logger.debug(
            "Starting vectorized model input preparation for %s requests, %s tokens",
            num_reqs,
            scheduler_output.total_num_scheduled_tokens,
        )

        # Get number of padded scheduled tokens for each request using vectorized approach
        req_ids = self.input_batch.req_ids
        tokens = [
            scheduler_output.num_scheduled_tokens_padded[req_id] for req_id in req_ids
        ]
        num_scheduled_tokens_padded = np.array(tokens, dtype=np.int64)
        total_num_scheduled_tokens_padded = int(num_scheduled_tokens_padded.sum())
        max_num_scheduled_tokens_padded = max(tokens)
        draft_tokens = (
            [0]
            if not scheduler_output.scheduled_spec_decode_tokens
            else [
                len(scheduler_output.scheduled_spec_decode_tokens[req_id])
                for req_id in req_ids
            ]
        )
        max_num_draft_tokens = max(draft_tokens)

        logger.debug("draft_tokens=%s", draft_tokens)
        logger.debug("max_num_draft_tokens=%s", max_num_draft_tokens)

        # Compute ACTUAL tokens to extract for each request (not padded count)
        # For prefill: actual = num_prompt_tokens - num_computed_tokens
        # For decode: actual = num_scheduled_tokens (scheduler knows the real count)
        # This is needed because:
        # - Scheduler pads prefill to bucket size (e.g., 17 -> 256)
        # - But token_ids_cpu only contains actual prompt tokens (e.g., 17)
        # - With prefix cache, num_computed_tokens > 0, so actual may be even smaller
        actual_tokens_list = []
        actual_tokens_list = [
            scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids
        ]

        actual_num_tokens = np.array(actual_tokens_list, dtype=np.int64)

        logger.debug(
            "Token counts: scheduled=%s, actual=%s",
            num_scheduled_tokens_padded.tolist(),
            actual_num_tokens.tolist(),
        )

        # Get request indices using ACTUAL token counts (not padded)
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], actual_num_tokens)

        # Get cumsum and arange efficiently using ACTUAL counts
        cu_num_tokens, arange = self._get_cumsum_and_arange(actual_num_tokens)

        # Calculate positions using vectorized operations
        positions_np = np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
        )

        # Get token indices efficiently using vectorized indexing
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # Use torch.index_select for fast token extraction
        input_ids = torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
        ).to(dtype=torch.int32)

        # Validate token IDs are within vocabulary range before device transfer
        self._validate_token_ids(input_ids)

        positions = torch.from_numpy(positions_np).to(dtype=torch.long)

        # Apply padding for prefill requests
        input_ids, positions, padding_map = self._create_padded_inputs(
            input_ids, positions, scheduler_output, req_ids, num_scheduled_tokens_padded
        )

        # Store padding map for later use in output extraction
        self._current_padding_map = padding_map

        # Move slot mapping to device
        # NOTE: After padding, positions array size = sum(num_scheduled_tokens)
        # We need to recompute req_indices to match the padded size
        padded_positions_np = positions.numpy()

        # Recompute req_indices for padded positions (since positions size changed after padding)
        if padding_map:
            padded_req_indices = np.repeat(
                self.arange_np[:num_reqs], num_scheduled_tokens_padded
            )
        else:
            padded_req_indices = req_indices

        # NOTE: We compute slot mappings for all positions (including padding) and then
        # overwrite padding slots with -1. An alternative would be to compute slot_mapping
        # for actual tokens only, then build the padded array by inserting -1s at padding
        # positions. We adopt the current approach as it's simpler to implement.
        # Block allocation in the scheduler is unaware of padding - blocks are allocated
        # for actual tokens only, which is correct since padding tokens don't need KV
        # cache storage.
        # Compute slot mapping on CPU (vllm 0.19.0 uses Triton kernel, not available on Neuron)
        for block_table in self.input_batch.block_table.block_tables:
            _compute_slot_mapping_cpu(
                block_table.block_table.cpu,
                block_table.slot_mapping.np,
                padded_positions_np,
                padded_req_indices,
                block_table.block_size,
                cp_world_size=self.cp_world_size,
                cp_rank=self.cp_rank,
                cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
            )
        # Set slot_mapping to PAD_SLOT_ID (-1) for padding tokens
        # Padding tokens should NOT write to KV cache
        if padding_map:
            # Calculate padding token indices
            # Token layout: [req0_real, req0_pad, req1_real, req1_pad, ...]
            current_idx = 0
            for i, req_id in enumerate(req_ids):
                scheduled = int(num_scheduled_tokens_padded[i])
                padding_count = padding_map.get(req_id, 0)
                actual = scheduled - padding_count

                if padding_count > 0:
                    # Padding tokens start at current_idx + actual
                    # and end at current_idx + scheduled
                    pad_start = current_idx + actual
                    pad_end = current_idx + scheduled

                    # Set slot_mapping to -1 for padding tokens in all block tables
                    for block_table in self.input_batch.block_table.block_tables:
                        block_table.slot_mapping.np[pad_start:pad_end] = PAD_SLOT_ID

                    logger.debug(
                        "SLOT_MAPPING: Request %s: set slots [%s:%s] "
                        "(%s padding tokens) to PAD_SLOT_ID (-1)",
                        req_id,
                        pad_start,
                        pad_end,
                        padding_count,
                    )

                current_idx += scheduled

        # Calculate decode batch padding once (used by slot_mapping, attention metadata, and input tensors)
        padded_num_reqs = get_decode_padded_batch_size(
            num_reqs,
            max_num_scheduled_tokens_padded,
            self.neuron_config.num_seqs_buckets,
            decode_token_threshold=self._decode_token_threshold(),
        )

        # DP+EP coordination: if other DP ranks have more tokens after their
        # own bucket padding, increase padded_num_reqs to match.  This must
        # happen before slot_mapping / attn_metadata are built so that
        # block_table rows, slot_mapping length, and input_ids all agree on
        # the same padded size.
        dp_pad, _ = self._get_dp_padding(padded_num_reqs)
        if dp_pad > 0:
            padded_num_reqs = padded_num_reqs + dp_pad
            logger.debug(
                "DP_PADDING: Increased padded_num_reqs by %s to %s",
                dp_pad,
                padded_num_reqs,
            )

        if padded_num_reqs != num_reqs:
            logger.debug(
                "DECODE_BATCH_PADDING: Need to pad %s requests to %s requests "
                "(num_seqs_buckets: %s)",
                num_reqs,
                padded_num_reqs,
                self.neuron_config.num_seqs_buckets,
            )

        # Apply decode batch padding to slot_mapping if needed
        total_num_scheduled_tokens_padded = self._apply_decode_slot_mapping_padding(
            num_reqs,
            padded_num_reqs,
            total_num_scheduled_tokens_padded,
            max_num_scheduled_tokens_padded,
        )

        self._save_slot_mapping_to_device()

        # Compute cached_seq_len for segmented prefill
        cached_seq_len = self._compute_cached_seq_len()

        # Build attention metadata
        attn_metadata = self._build_attention_metadata(
            padded_num_reqs,
            total_num_scheduled_tokens_padded,  # Includes decode padding
            max_num_scheduled_tokens_padded,
            max_num_draft_tokens,
            cached_seq_len,
        )

        # Spec decoding
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # Only sample logits for the last token of each request
            # Since padding was applied, we need to compute indices based on padded cumsum
            if padding_map:
                # Compute padded cumsum for correct indexing into output tensor
                padded_cu_num_tokens = np.cumsum(num_scheduled_tokens_padded)
                # Last real token is at: (end of request's padded segment) - 1 - padding_count
                logits_indices_list = []
                for i, req_id in enumerate(req_ids):
                    # Index of last position in this request's padded segment
                    end_idx = int(padded_cu_num_tokens[i]) - 1
                    # Subtract padding to get last real token
                    req_padding = padding_map.get(req_id, 0)
                    actual_last_token_idx = end_idx - req_padding
                    logits_indices_list.append(actual_last_token_idx)

                logits_indices = torch.tensor(logits_indices_list, dtype=torch.long)
                logger.debug(
                    "Computed logits_indices with padding: "
                    "padded_cu=%s, padding_map=%s, logits_indices=%s",
                    padded_cu_num_tokens.tolist(),
                    padding_map,
                    logits_indices.tolist(),
                )
            else:
                # No padding, use actual cumsum directly
                logits_indices = (
                    torch.tensor(cu_num_tokens.tolist(), dtype=torch.long) - 1
                )

            num_draft_tokens = None
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)

            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            # Build spec decode metadata, padded to the decode bucket so
            # tensor shapes match the warmed-up decode-with-spec NEFF.
            spec_decode_metadata = self._build_spec_decode_metadata(
                input_ids,
                num_draft_tokens,
                cu_num_tokens,
                padded_num_reqs=padded_num_reqs,
            )
            logits_indices = spec_decode_metadata.logits_indices

            logger.debug("spec_decode_metadata=%s", spec_decode_metadata)

        logger.debug(
            "Completed vectorized model input preparation: "
            "input_ids.shape=%s, positions.shape=%s",
            input_ids.shape,
            positions.shape,
        )

        # Apply decode batch padding if needed
        # This adds fake batch lines to match the compiled bucket size
        input_ids, positions, logits_indices = self._apply_decode_batch_padding(
            input_ids, positions, logits_indices, padded_num_reqs
        )

        self._current_inputs_embeds = None
        self._current_is_token_ids = None
        prompt_embeds_path_active = False
        if self.enable_prompt_embeds:
            req_prompt_embeds = getattr(self.input_batch, "req_prompt_embeds", None)
            if req_prompt_embeds:
                for req_id in req_ids:
                    req_idx = self.input_batch.req_id_to_index.get(req_id)
                    if req_idx is None or req_idx not in req_prompt_embeds:
                        continue
                    num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                    if num_sched <= 0:
                        continue
                    req_embeds = req_prompt_embeds[req_idx]
                    start_pos = int(self.input_batch.num_computed_tokens_cpu[req_idx])
                    if start_pos < req_embeds.shape[0]:
                        prompt_embeds_path_active = True
                        break
        if prompt_embeds_path_active:
            self._current_inputs_embeds, self._current_is_token_ids = (
                self._build_prompt_embeds_tensors(
                    input_ids,
                    token_indices_tensor,
                    req_ids,
                    scheduler_output,
                    padding_map,
                )
            )
        if (
            prompt_embeds_path_active
            and self._last_prompt_embeds_path_active != prompt_embeds_path_active
        ):
            logger.info(
                "PROMPT_EMBEDS_PATH active for current batch. "
                "Input format: pass `prompt_embeds` with shape [sequence_length, hidden_size]."
            )
        self._last_prompt_embeds_path_active = prompt_embeds_path_active

        return input_ids, positions, logits_indices, attn_metadata, spec_decode_metadata

    def _maybe_replicate_for_spec_decode(
        self,
        per_seq_params: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> torch.Tensor:
        """Replicate per-sequence sampling params for the spec-decode verify step.

        When speculative decoding is active, the sampler operates on
        ``num_seqs * (num_speculative_tokens + 1)`` logits rows (one per
        draft + bonus position), so the ``[num_seqs, ...]`` sampling params
        need to be replicated row-wise to match. Returns ``per_seq_params``
        unchanged when speculative decoding is not active.
        """
        if spec_decode_metadata is None or self.speculative_config is None:
            return per_seq_params
        return replicate_per_seq_rows(
            per_seq_params, self.speculative_config.num_speculative_tokens + 1
        )

    def _build_spec_decode_metadata(
        self,
        input_ids: torch.Tensor,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
        padded_num_reqs: int | None = None,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        #
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106, 206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]
        #
        # If `padded_num_reqs` > len(num_draft_tokens), fake full-spec reqs
        # are appended so the returned tensors have the shapes expected by
        # `_create_warmup_spec_decode_metadata` (bucket-sized decode batch).
        # The fake rows point at the last real token slot — rejection
        # sampling on them is a no-op and their outputs are trimmed
        # downstream.

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int64
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int64
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens)
        logits_indices = torch.from_numpy(logits_indices)
        target_logits_indices = torch.from_numpy(target_logits_indices)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        num_reqs = len(num_draft_tokens)
        if padded_num_reqs is None or padded_num_reqs <= num_reqs:
            return SpecDecodeMetadata(
                draft_token_ids=draft_token_ids,
                num_draft_tokens=num_draft_tokens.tolist(),
                cu_num_draft_tokens=cu_num_draft_tokens,
                cu_num_sampled_tokens=cu_num_sampled_tokens,
                target_logits_indices=target_logits_indices,
                bonus_logits_indices=bonus_logits_indices,
                logits_indices=logits_indices,
            )

        # Append fake full-spec reqs so shapes match the warmed-up NEFF.
        assert self.speculative_config is not None, (
            "padded_num_reqs requires speculative_config"
        )
        num_spec_tokens = self.speculative_config.num_speculative_tokens
        num_pad_reqs = padded_num_reqs - num_reqs
        pad_draft = num_pad_reqs * num_spec_tokens

        padded_num_draft = num_draft_tokens.tolist() + [num_spec_tokens] * num_pad_reqs

        last_draft = int(cu_num_draft_tokens[-1].item())
        last_sampled = int(cu_num_sampled_tokens[-1].item())
        cu_draft_pad = torch.tensor(
            [last_draft + (i + 1) * num_spec_tokens for i in range(num_pad_reqs)],
            dtype=cu_num_draft_tokens.dtype,
        )
        cu_sampled_pad = torch.tensor(
            [
                last_sampled + (i + 1) * (num_spec_tokens + 1)
                for i in range(num_pad_reqs)
            ],
            dtype=cu_num_sampled_tokens.dtype,
        )
        cu_num_draft_tokens = torch.cat([cu_num_draft_tokens, cu_draft_pad])
        cu_num_sampled_tokens = torch.cat([cu_num_sampled_tokens, cu_sampled_pad])

        pad_draft_ids = torch.zeros(
            pad_draft, dtype=draft_token_ids.dtype, device=draft_token_ids.device
        )
        draft_token_ids = torch.cat([draft_token_ids, pad_draft_ids])

        # Fake reqs reuse the last real draft index so gather is safe.
        safe_target_idx = (
            int(target_logits_indices[-1].item())
            if target_logits_indices.numel() > 0
            else 0
        )
        target_logits_indices = torch.cat(
            [
                target_logits_indices,
                torch.full(
                    (pad_draft,), safe_target_idx, dtype=target_logits_indices.dtype
                ),
            ]
        )

        # bonus_logits_indices: fake reqs point into the padded logits slots.
        bonus_logits_indices = torch.cat(
            [
                bonus_logits_indices,
                torch.tensor(
                    [
                        num_reqs * (num_spec_tokens + 1)
                        + i * (num_spec_tokens + 1)
                        + num_spec_tokens
                        for i in range(num_pad_reqs)
                    ],
                    dtype=bonus_logits_indices.dtype,
                ),
            ]
        )

        logits_indices = torch.cat(
            [
                logits_indices,
                torch.arange(
                    num_reqs * (num_spec_tokens + 1),
                    padded_num_reqs * (num_spec_tokens + 1),
                    dtype=logits_indices.dtype,
                ),
            ]
        )

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=padded_num_draft,
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _decode_token_threshold(self) -> int:
        """Largest `max_query_len` still classified as decode.

        With spec decode the target verifies `1 + num_speculative_tokens`
        tokens per request per step; otherwise decode is 1 token per request.

        Matches the `decode_token_threshold` field set on the per-step
        attention metadata by `_build_attention_metadata` whenever all
        scheduled requests carry the full `num_speculative_tokens` drafts
        — the invariant upheld by `NeuronScheduler` (prefill/decode
        separation, no mid-step new-req entries).
        """
        if self.speculative_config is not None:
            return 1 + self.speculative_config.num_speculative_tokens
        return 1

    def _create_padded_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        scheduler_output: Any,
        req_ids: list[str],
        num_scheduled_tokens_padded: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
        """
        Create padded input tensors for prefill requests.

        Args:
            input_ids: Original input token IDs [T]
            positions: Original position IDs [T]
            scheduler_output: Scheduler output with padding info
            req_ids: List of request IDs
            num_scheduled_tokens: Array of scheduled token counts per request

        Returns:
            Tuple of (padded_input_ids, padded_positions, padding_map)
            padding_map: dict mapping req_id to number of padding tokens
        """
        padding_map = {}

        # Check if self.requests exists
        if not hasattr(self, "requests") or not self.requests:
            logger.warning(
                "INPUT_PADDING: self.requests not available, skipping padding"
            )
            return input_ids, positions, padding_map

        # Check if any requests need padding (prefill requests)
        # NOTE: A request is in prefill if num_computed_tokens < num_prompt_tokens.
        # We cannot use "num_computed_tokens == 0" because prefix caching may have
        # already set num_computed_tokens to a non-zero value for cached tokens.
        has_prefill = any(
            req_id in self.requests
            and self.requests[req_id].num_computed_tokens
            < self.requests[req_id].num_prompt_tokens
            for req_id in req_ids
        )

        if not has_prefill:
            logger.debug("INPUT_PADDING: No prefill requests, skipping padding")
            return input_ids, positions, padding_map

        logger.debug(
            "=== INPUT_PADDING: Creating padded inputs for %s requests ===",
            len(req_ids),
        )

        # Calculate padding for each request
        padded_segments = []
        position_segments = []
        current_pos = 0
        total_padding_added = 0

        for i, req_id in enumerate(req_ids):
            # Check if request exists
            if req_id not in self.requests:
                logger.warning(
                    "INPUT_PADDING: Request %s not found in self.requests, "
                    "skipping padding",
                    req_id,
                )
                # Use original tokens without padding
                actual_tokens = num_scheduled_tokens_padded[i]
                actual_input_ids = input_ids[current_pos : current_pos + actual_tokens]
                actual_positions = positions[current_pos : current_pos + actual_tokens]
                padded_segments.append(actual_input_ids)
                position_segments.append(actual_positions)
                current_pos += actual_tokens
                continue

            req_state = self.requests[req_id]
            num_prompt_tokens = req_state.num_prompt_tokens
            # NOTE: Use same prefill detection as has_prefill above (handles prefix caching)
            is_prefill = req_state.num_computed_tokens < num_prompt_tokens

            scheduled_tokens_padded = scheduler_output.num_scheduled_tokens_padded[
                req_id
            ]
            actual_tokens = scheduler_output.num_scheduled_tokens[req_id]
            padding_needed = scheduled_tokens_padded - actual_tokens
            padding_map[req_id] = padding_needed

            # Extract actual tokens
            actual_input_ids = input_ids[current_pos : current_pos + actual_tokens]
            actual_positions = positions[current_pos : current_pos + actual_tokens]

            if padding_needed > 0:
                # Create padding tokens (use token_id=0)
                pad_ids = torch.zeros(padding_needed, dtype=torch.int32)
                # NOTE: Padding position values don't affect correctness because:
                # 1. Real tokens can't attend to padding (causal attention)
                # 2. Padding outputs are discarded (logits_indices points to real tokens)
                # 3. KV cache uses slot_mapping (set to -1 for padding), not positions
                # We use last_pos for simplicity, but any value would work.
                last_pos = (
                    actual_positions[-1].item() if len(actual_positions) > 0 else 0
                )
                pad_positions = torch.full(
                    (padding_needed,), last_pos, dtype=torch.long
                )

                # Concatenate actual + padding
                padded_input_ids = torch.cat([actual_input_ids, pad_ids])
                padded_positions = torch.cat([actual_positions, pad_positions])

                total_padding_added += padding_needed

                # Enhanced logging
                logger.debug(
                    "INPUT_PADDING: Request %s (%s): actual=%s, scheduled=%s, "
                    "padding=%s, input_ids: %s -> %s, positions: [%s..%s] + "
                    "%s padding tokens at position %s",
                    req_id,
                    "PREFILL" if is_prefill else "DECODE",
                    actual_tokens,
                    scheduled_tokens_padded,
                    padding_needed,
                    actual_input_ids.shape,
                    padded_input_ids.shape,
                    actual_positions[0].item(),
                    actual_positions[-1].item(),
                    padding_needed,
                    last_pos,
                )
            else:
                padded_input_ids = actual_input_ids
                padded_positions = actual_positions
                logger.debug(
                    "INPUT_PADDING: Request %s (%s): actual=%s, no padding needed",
                    req_id,
                    "PREFILL" if is_prefill else "DECODE",
                    actual_tokens,
                )

            padded_segments.append(padded_input_ids)
            position_segments.append(padded_positions)
            current_pos += actual_tokens

        # Concatenate all segments
        final_input_ids = torch.cat(padded_segments)
        final_positions = torch.cat(position_segments)

        original_size = input_ids.shape[0]
        final_size = final_input_ids.shape[0]
        overhead_pct = (
            ((final_size - original_size) / original_size * 100)
            if original_size > 0
            else 0
        )

        logger.debug(
            "INPUT_PADDING: Summary - Original: %s tokens, Final: %s tokens, "
            "Padding added: %s (%.1f%% overhead)",
            original_size,
            final_size,
            total_padding_added,
            overhead_pct,
        )

        return final_input_ids, final_positions, padding_map

    def _apply_decode_slot_mapping_padding(
        self, num_reqs: int, padded_num_reqs: int, total_tokens: int, max_query_len: int
    ) -> int:
        """
        Apply decode batch padding to slot_mapping in block tables.

        For decode phase, pads slot_mapping with -1 values to match the batch bucket size.

        Args:
            num_reqs: Number of actual requests
            padded_num_reqs: Padded number of requests (from get_decode_padded_batch_size)
            total_tokens: Total tokens (should equal num_reqs for decode)
            max_query_len: Max query length (should be 1 for decode)

        Returns:
            Total token count after padding for decode batch
        """

        # Only apply to decode phase. tokens_per_req = max_query_len:
        #   1 for non-spec decode, 1+num_spec_tokens for spec decode.
        # Prefill (large max_query_len) is skipped.
        tokens_per_req = max_query_len
        decode_token_threshold = self._decode_token_threshold()
        if (
            max_query_len < 1
            or max_query_len > decode_token_threshold
            or total_tokens != num_reqs * tokens_per_req
        ):
            logger.debug(
                "DECODE_SLOT_MAPPING_PADDING: Skipping - not decode phase "
                "(max_query_len=%s, total_tokens=%s, num_reqs=%s)",
                max_query_len,
                total_tokens,
                num_reqs,
            )
            return total_tokens

        if padded_num_reqs == num_reqs:
            logger.debug(
                "DECODE_SLOT_MAPPING_PADDING: No padding needed (num_reqs=%s)",
                num_reqs,
            )
            return total_tokens

        num_pad_reqs = padded_num_reqs - num_reqs

        logger.debug(
            "DECODE_SLOT_MAPPING_PADDING: Padding slot_mapping from %s to %s requests "
            "(num_seqs_buckets: %s, adding %s padding slots, tokens_per_req=%s)",
            num_reqs,
            padded_num_reqs,
            self.neuron_config.num_seqs_buckets,
            num_pad_reqs,
            tokens_per_req,
        )

        # Set padding slots to -1. With spec decode, each request occupies
        # tokens_per_req contiguous slots in the flattened slot_mapping.
        pad_start = num_reqs * tokens_per_req
        pad_end = padded_num_reqs * tokens_per_req

        for block_table in self.input_batch.block_table.block_tables:
            block_table.slot_mapping.np[pad_start:pad_end] = PAD_SLOT_ID
            logger.debug(
                "DECODE_SLOT_MAPPING_PADDING: Set slot_mapping[%s:%s] = %s",
                pad_start,
                pad_end,
                PAD_SLOT_ID,
            )

        padded_total_tokens = padded_num_reqs * tokens_per_req

        logger.debug(
            "DECODE_SLOT_MAPPING_PADDING: Padded total_tokens: %s -> %s",
            total_tokens,
            padded_total_tokens,
        )

        return padded_total_tokens

    def _pad_logit_mask_for_decode(
        self, logit_mask: torch.Tensor, padded_batch_size: int
    ) -> torch.Tensor:
        """Pad logit_mask to match decode batch padding size.

        During decode, input_ids may be padded to a bucket size larger than
        the number of active requests. The logit_mask must match so that
        masked_fill inside the compiled model sees matching batch dimensions.

        Padding rows are all-True (no constraints on dummy sequences).

        Args:
            logit_mask: [num_reqs, vocab_size] boolean mask.
            padded_batch_size: Target batch size after decode padding.

        Returns:
            Padded logit_mask if decode padding was applied, otherwise unchanged.
        """
        if (
            padded_batch_size <= self.max_num_reqs
            and logit_mask.shape[0] < padded_batch_size
        ):
            pad_rows = padded_batch_size - logit_mask.shape[0]
            return torch.nn.functional.pad(logit_mask, (0, 0, 0, pad_rows), value=True)
        return logit_mask

    def _apply_decode_batch_padding(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        padded_num_reqs: int,
    ):
        """
        Apply decode batch padding to input tensors to match compiled bucket sizes.

        For decode phase (1 token per request), pads input_ids, positions, and logits_indices
        with fake requests to match the next nearest bucket size.

        Note: slot_mapping padding is handled separately in _apply_decode_slot_mapping_padding
        before _build_attention_metadata is called.

        Args:
            input_ids: Input token IDs [num_tokens]
            positions: Token positions [num_tokens]
            logits_indices: Indices for logits extraction [num_reqs]
            padded_num_reqs: Padded number of requests

        Returns:
            Tuple of (padded_input_ids, padded_positions, padded_logits_indices)
        """

        # Only apply to decode phase (tokens_per_req tokens per request).
        # With spec decode: tokens_per_req = 1 + num_spec_tokens; otherwise 1.
        num_reqs = len(self.input_batch.req_ids)
        num_tokens = input_ids.shape[0]
        tokens_per_req = self._decode_token_threshold()

        if num_tokens != num_reqs * tokens_per_req:
            logger.debug(
                "DECODE_BATCH_PADDING: Skipping - not decode phase "
                "(num_tokens=%s, num_reqs=%s, tokens_per_req=%s)",
                num_tokens,
                num_reqs,
                tokens_per_req,
            )
            return input_ids, positions, logits_indices

        # Check if we have decode buckets configured
        if not self.neuron_config.num_seqs_buckets:
            logger.debug("DECODE_BATCH_PADDING: No buckets configured, skipping")
            return input_ids, positions, logits_indices

        if padded_num_reqs == num_reqs:
            logger.debug(
                "DECODE_BATCH_PADDING: No padding needed (num_reqs=%s already matches bucket)",
                num_reqs,
            )
            return input_ids, positions, logits_indices

        # Calculate padding needed (in requests and tokens)
        num_pad_reqs = padded_num_reqs - num_reqs
        num_pad_tokens = num_pad_reqs * tokens_per_req

        logger.debug(
            "DECODE_BATCH_PADDING: Padding batch from %s to %s requests "
            "(bucket: %s, adding %s fake requests, %s fake tokens, tokens_per_req=%s)",
            num_reqs,
            padded_num_reqs,
            self.neuron_config.num_seqs_buckets,
            num_pad_reqs,
            num_pad_tokens,
            tokens_per_req,
        )

        pad_input_ids = torch.zeros(num_pad_tokens, dtype=input_ids.dtype)
        pad_positions = torch.zeros(num_pad_tokens, dtype=positions.dtype)

        padded_input_ids = torch.cat([input_ids, pad_input_ids])
        padded_positions = torch.cat([positions, pad_positions])

        # Extend logits_indices to cover the fake token slots. Only pad when
        # caller passed the unpadded view (length num_reqs * tokens_per_req).
        if logits_indices.shape[0] == num_reqs * tokens_per_req:
            pad_logits_indices = torch.arange(
                num_reqs * tokens_per_req,
                padded_num_reqs * tokens_per_req,
                dtype=torch.long,
            )
            padded_logits_indices = torch.cat([logits_indices, pad_logits_indices])
        else:
            padded_logits_indices = logits_indices

        logger.debug(
            "DECODE_BATCH_PADDING: Applied - input_ids: %s -> %s, "
            "positions: %s -> %s, logits_indices: %s -> %s",
            input_ids.shape,
            padded_input_ids.shape,
            positions.shape,
            padded_positions.shape,
            logits_indices.shape,
            padded_logits_indices.shape,
        )

        return padded_input_ids, padded_positions, padded_logits_indices

    def _validate_token_ids(self, input_ids: torch.Tensor) -> None:
        """
        Validate that all token IDs are within the vocabulary range.

        Args:
            input_ids: Input token IDs tensor

        Raises:
            ValueError: If any token ID is outside the valid range [0, vocab_size)
        """
        max_id = input_ids.max().item()
        min_id = input_ids.min().item()
        if max_id >= self.vocab_size or min_id < 0:
            invalid_mask = (input_ids >= self.vocab_size) | (input_ids < 0)
            invalid_ids = input_ids[invalid_mask].tolist()
            raise ValueError(
                f"Token IDs out of range [0, {self.vocab_size}). "
                f"Found min={min_id}, max={max_id}. Invalid IDs: {invalid_ids}"
            )

    def _compute_swa_num_blocks(self, sliding_window: int, block_size: int) -> int:
        """Minimum number of KV blocks covering the sliding window, padded up
        to the attention kernel's P_MAX tile divisibility."""
        # Trimmed SWA kv block counts must produce S_ctx divisible by P_MAX.
        P_MAX = 128
        min_blocks = sliding_window // block_size + 1
        blocks_per_pmax = P_MAX // block_size
        return (min_blocks + blocks_per_pmax - 1) // blocks_per_pmax * blocks_per_pmax

    def _compute_swa_decode_tensors(
        self,
        blk_table,
        padded_num_reqs: int,
        sliding_window: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """For a SWA decode step, compute:
        - the block_table_tensor trimmed to the window-relevant blocks, so
          the attention kernel reads only ~window/block_size blocks from HBM.
        - the per-seq `swa_kv_pos_offset` tensor (= start_block * block_size),
          which the model subtracts from pos_ids so the causal mask is
          computed in the trimmed frame.
        """
        num_swa_blocks = self._compute_swa_num_blocks(sliding_window, block_size)
        ctx_lens = torch.from_numpy(
            self.input_batch.num_computed_tokens_cpu[:padded_num_reqs]
        )
        start_blocks = torch.clamp(ctx_lens // block_size - num_swa_blocks + 1, min=0)
        gather_idx = start_blocks.unsqueeze(1) + torch.arange(
            num_swa_blocks, dtype=start_blocks.dtype
        ).unsqueeze(0)
        gpu_bt = blk_table.get_device_tensor(padded_num_reqs)
        trimmed_bt = gpu_bt.gather(1, gather_idx.long().to(self.device))
        swa_kv_pos_offset = (start_blocks * block_size).to(
            dtype=torch.int32, device=self.device
        )
        return trimmed_bt, swa_kv_pos_offset

    def _resolve_decode_ctx_blocks(
        self,
        block_size: int,
        padded_num_reqs: int,
        max_num_draft_tokens: int,
    ) -> int:
        """Resolve max_blocks_per_seq for a non-SWA decode group.

        SWA groups are handled separately via `_compute_swa_decode_tensors`
        (the SWA window is always the tighter, more correct bound for those
        layers — `decode_context_length_buckets` does not apply). Callers
        must route SWA specs to the SWA path before invoking this helper.

        Args:
            block_size: KV-cache block size.
            padded_num_reqs: Padded batch size for this decode step.
            max_num_draft_tokens: 0 for non-spec decode; equals
                num_speculative_tokens during target-with-spec verify.

        Returns:
            max_blocks_per_seq for this group's block_table_tensor.

        Example:
            >>> runner._resolve_decode_ctx_blocks(32, 4, 0)
            512  # 16384 / 32 when no decode_context_length_buckets are set
        """
        from vllm_neuron.utils.bucket_utils import get_bucket_for_count

        # With DCP, each rank stores only ctx_for_blocks / dcp_size tokens.
        # Match the divisor used by `_build_warmup_attention_metadata` so the
        # runtime block_table shape matches the compiled NEFF shape.
        dcp_block_size = block_size * max(self._dcp_size, 1)
        full_blocks = (self.max_model_len + dcp_block_size - 1) // dcp_block_size

        if self.neuron_config.decode_context_length_buckets is None:
            return full_blocks

        max_ctx = int(self.input_batch.num_computed_tokens_cpu[:padded_num_reqs].max())
        needed_seq = max_ctx + 1 + max_num_draft_tokens
        ctx_bucket = get_bucket_for_count(
            needed_seq,
            self.neuron_config.decode_context_length_buckets + [self.max_model_len],
        )
        return (ctx_bucket + dcp_block_size - 1) // dcp_block_size

    def _build_attention_metadata(
        self,
        padded_num_reqs: int,
        total_num_scheduled_tokens: int,
        max_query_len: int,
        max_num_draft_tokens: int,
        cached_seq_len: int = 0,
    ) -> AttentionMetadata | None:
        """
        Build attention metadata for KV cache and attention computation.

        Args:
            padded_num_reqs: Padded number of requests
            total_num_scheduled_tokens: Total number of tokens in the batch
            max_query_len: Max size of each batch line
            max_num_draft_tokens: Max number of draft tokens for spec decode
            cached_seq_len: Actual number of valid cached tokens

        Returns:
            AttentionMetadata object or None for simplified cases
        """
        attn_metadata = {}
        decode_token_threshold = 1 + max_num_draft_tokens
        is_decode = max_query_len <= decode_token_threshold

        for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups
        ):
            spec = kv_cache_group_spec.kv_cache_spec
            block_size = spec.block_size
            blk_table = self.input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor(padded_num_reqs)
            slot_mapping = blk_table.slot_mapping.gpu[:total_num_scheduled_tokens]
            # DCP Gather-Q prefill: model receives only S/DCP owned tokens
            # after the SP gather + interleave slice. Extract slot_mapping
            # entries at owned positions using the same interleave pattern.
            # Size is always S/DCP (deterministic, no recompile).
            # Slice on CPU (Neuron tensors don't support .contiguous()) then move back.
            if (
                not is_decode
                and self.neuron_config.apply_prefill_dcp
                and self.cp_world_size > 1
            ):
                I = self.cp_kv_cache_interleave_size
                W = self.cp_world_size
                R = self.cp_rank
                S = total_num_scheduled_tokens
                slot_mapping = (
                    slot_mapping.cpu()
                    .view(S // (W * I), W, I)[:, R, :]
                    .contiguous()
                    .reshape(-1)
                    .to(self.device)
                )
            cached_seq_len_tensor = torch.tensor(
                [[cached_seq_len]], dtype=torch.int32, device=self.device
            )

            # Note there are more data available in self.input_batch you can
            # use in attention_metadata. Now we only use the ones below.
            kv_segment_size = 0
            if self.neuron_config.kv_segment_size_buckets is not None:
                kv_segment_size = self.neuron_config.kv_segment_size_buckets[0]

            swa_kv_pos_offset = None

            # Only include the raw (pre-transform) block table when running
            # the synthetic test model — it needs the scheduler's original
            # allocation (0-padded, no sentinel remap, no SWA trim).
            raw_blk_table_tensor = (
                blk_table.get_cpu_tensor()[:padded_num_reqs].to(self.device)
                if self._is_synthetic_model
                else None
            )

            # For SWA decode: slice the block table to only the window-relevant
            # blocks so the attention kernel reads the minimum necessary blocks
            # from HBM. Prefill uses the full table (writes via slot_mapping
            # span every scheduled slot, and the prefill NEFF is compiled for
            # the full shape).
            if is_decode and isinstance(spec, SlidingWindowSpec):
                num_swa_blocks = self._compute_swa_num_blocks(
                    spec.sliding_window, block_size
                )
                if num_swa_blocks < blk_table_tensor.shape[1]:
                    blk_table_tensor, swa_kv_pos_offset = (
                        self._compute_swa_decode_tensors(
                            blk_table,
                            padded_num_reqs,
                            spec.sliding_window,
                            block_size,
                        )
                    )
                else:
                    # Short sequence: no trimming needed, but always pass
                    # swa_kv_pos_offset to keep the FX graph consistent
                    # with warmup (avoids torch._dynamo recompilation).
                    swa_kv_pos_offset = torch.zeros(
                        padded_num_reqs, dtype=torch.int32, device=self.device
                    )

            elif (
                is_decode
                and self.neuron_config.decode_context_length_buckets is not None
            ):
                bucket_blocks = self._resolve_decode_ctx_blocks(
                    block_size=block_size,
                    padded_num_reqs=padded_num_reqs,
                    max_num_draft_tokens=max_num_draft_tokens,
                )
                if bucket_blocks < blk_table_tensor.shape[1]:
                    # index_select rather than a slice: dynamo guards check
                    # the literal stride of the input, and a slice
                    # (`blk_table_tensor[:, :bucket_blocks]`) only changes
                    # shape — the stride still reflects the parent's full
                    # `max_blocks_per_seq`. A `.contiguous()` or `.clone()`
                    # would normally rebuild the stride, but at B=1
                    # PyTorch treats any tensor with a size-1 dim as
                    # contiguous and skips the copy, leaving the wrong
                    # stride in place. index_select always allocates a
                    # fresh row-major tensor whose stride matches what
                    # warmup compiled, so guards hit instead of recompile.
                    # The extra device-side copy is asynchronous and
                    # overlaps with the prior decode forward, so it costs
                    # no extra wall time.
                    bucket_idx = torch.arange(bucket_blocks, device=self.device)
                    blk_table_tensor = torch.index_select(
                        blk_table_tensor, 1, bucket_idx
                    )
                swa_kv_pos_offset = torch.zeros(
                    padded_num_reqs, dtype=torch.int32, device=self.device
                )

            attn_metadata_i = {
                "block_table_tensor": blk_table_tensor,
                "slot_mapping": slot_mapping,
                "max_query_len": max_query_len,
                "block_size": block_size,
                "max_blocks_per_seq": blk_table_tensor.shape[1],
                "decode_token_threshold": decode_token_threshold,
                "cached_seq_len": cached_seq_len_tensor,
                "kv_segment_size": kv_segment_size,
            }
            if raw_blk_table_tensor is not None:
                attn_metadata_i["raw_block_table_tensor"] = raw_blk_table_tensor
            if swa_kv_pos_offset is not None:
                attn_metadata_i["swa_kv_pos_offset"] = swa_kv_pos_offset

            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata

    def _build_warmup_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        cached_seq_len: int = 0,
        decode_token_threshold: int | None = None,
        ctx_bucket: int | None = None,
    ) -> dict:
        """
        Build attention metadata for warmup without using InputBatch.

        Creates synthetic block_table_tensor and slot_mapping tensors
        that match the shapes used by InputBatch during actual inference.

        Args:
            num_tokens: Total number of tokens in the batch
            num_reqs: Number of requests in the batch
            cached_seq_len: Actual number of valid cached tokens
            decode_token_threshold: Threshold to distinguish prefill from decode.
                Defaults to 1 + num_speculative_tokens when drafter is present,
                otherwise 1.
            ctx_bucket: Decode sequence bucket size. When None, sized at
                ``max_model_len`` (today's behavior). Otherwise sized at
                ``ceil(ctx_bucket / block_size)`` blocks. Only meaningful
                for non-SWA groups; SWA groups always trim to the window.

        Returns:
            dict mapping layer names to attention metadata dicts
        """
        attn_metadata = {}

        if decode_token_threshold is None:
            num_spec_tokens = 0
            if self.drafter is not None:
                num_spec_tokens = self.drafter.num_speculative_tokens
            decode_token_threshold = 1 + num_spec_tokens

        is_prefill = (num_tokens // num_reqs) > decode_token_threshold

        for _, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            spec = kv_cache_group_spec.kv_cache_spec
            block_size = spec.block_size
            # Block-table seq dim. Defaults to max_model_len; overridden by
            # ctx_bucket for decode_context_length_buckets warmup. SWA path below
            # may further reduce this for SWA groups.
            # With DCP, each rank stores only ctx_for_blocks / dcp_size tokens.
            # vLLM's BlockTableManager uses total_cp_world_size = dcp_size (no PCP).
            ctx_for_blocks = (
                ctx_bucket if ctx_bucket is not None else self.max_model_len
            )
            dcp_block_size = block_size * max(self._dcp_size, 1)
            max_num_blocks_per_req = (
                ctx_for_blocks + dcp_block_size - 1
            ) // dcp_block_size

            swa_kv_pos_offset = None
            # For SWA decode, warmup must match the trimmed shape that
            # `_build_attention_metadata` produces at runtime so the compiled
            # NEFF receives the correct block_table dim.
            if not is_prefill and isinstance(spec, SlidingWindowSpec):
                num_swa_blocks = self._compute_swa_num_blocks(
                    spec.sliding_window, block_size
                )
                if num_swa_blocks < max_num_blocks_per_req:
                    max_num_blocks_per_req = num_swa_blocks
                swa_kv_pos_offset = torch.zeros(
                    num_reqs, dtype=torch.int32, device=self.device
                )

            # Dummy block_table_tensor: [num_reqs, max_num_blocks_per_req]
            # Use sequential block IDs starting from 0
            block_table_tensor = (
                torch.arange(max_num_blocks_per_req, dtype=torch.int32)
                .unsqueeze(0)
                .expand(num_reqs, -1)
                .contiguous()
                .to(self.device)
            )

            # Dummy slot_mapping: [num_tokens] or [num_tokens/DCP] for DCP prefill
            if (
                is_prefill
                and self.neuron_config.apply_prefill_dcp
                and self.cp_world_size > 1
            ):
                I = self.cp_kv_cache_interleave_size
                W = self.cp_world_size
                R = self.cp_rank
                slot_mapping = (
                    torch.arange(num_tokens, dtype=torch.int64)
                    .view(num_tokens // (W * I), W, I)[:, R, :]
                    .contiguous()
                    .reshape(-1)
                    .to(self.device)
                )
            else:
                slot_mapping = torch.arange(num_tokens, dtype=torch.int64).to(
                    self.device
                )

            # max_query_len: for prefill = bucket_size, for decode = 1
            max_query_len = num_tokens // num_reqs

            kv_segment_size = 0
            if self.neuron_config.kv_segment_size_buckets is not None:
                kv_segment_size = self.neuron_config.kv_segment_size_buckets[0]

            attn_metadata_i = {
                "block_table_tensor": block_table_tensor,
                "slot_mapping": slot_mapping,
                "max_query_len": max_query_len,
                "block_size": block_size,
                "max_blocks_per_seq": max_num_blocks_per_req,
                "decode_token_threshold": decode_token_threshold,
                "cached_seq_len": torch.tensor(
                    [[cached_seq_len]], dtype=torch.int32, device=self.device
                ),
                "kv_segment_size": kv_segment_size,
            }
            if swa_kv_pos_offset is not None:
                attn_metadata_i["swa_kv_pos_offset"] = swa_kv_pos_offset

            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata

    def _build_prefill_synthetic_inputs(
        self, bucket_size: int, kv_segment_size: int
    ) -> dict:
        """Build synthetic inputs for prefill warmup.

        Args:
            bucket_size: Number of active tokens (prefill bucket size)
            kv_segment_size: Size of each KV segment bucket

        Returns:
            dict: Keyword arguments for self.model() forward call
        """
        cached_seq_len = 0
        num_reqs = 1
        num_tokens = bucket_size

        # Create synthetic inputs: 1 request with bucket_size tokens
        input_ids = torch.arange(1, num_tokens + 1, dtype=torch.int32).to(self.device)
        # Positions start from cached_seq_len (simulating continuation after cached prefix)
        positions = torch.arange(
            cached_seq_len, cached_seq_len + num_tokens, dtype=torch.long
        ).to(self.device)
        # Sample from last token only
        logits_indices = torch.tensor([num_tokens - 1], dtype=torch.long).to(
            self.device
        )

        attn_metadata = self._build_warmup_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            cached_seq_len=cached_seq_len,
            # Prefill has no draft tokens, so decode_token_threshold=1
            # (matches _build_attention_metadata where max_num_draft_tokens=0).
            decode_token_threshold=1,
        )

        # Create dummy sampling params for warmup
        dummy_sampling_params = torch.tensor(
            [[0, 1.0, 1.0]],  # [top_k, top_p, temperature]
            dtype=torch.float32,
            device=self.device,
        )

        # Create a dummy logit mask.
        # We always pass logit_mask to ensure torch.compile traces a single code path
        dummy_logit_mask = torch.ones(
            num_reqs, self.vocab_size, dtype=torch.bool, device=self.device
        )

        warmup_kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=logits_indices,
            sampling_params=dummy_sampling_params,
            spec_decode_metadata=None,
            rank=self.rank_tensor,
            logit_mask=dummy_logit_mask,
        )
        if self.enable_prompt_embeds:
            warmup_kwargs["inputs_embeds"] = torch.zeros(
                num_tokens,
                self.inputs_embeds_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            warmup_kwargs["is_token_ids"] = torch.ones(
                num_tokens,
                dtype=torch.bool,
                device=self.device,
            )
        return warmup_kwargs

    def extract_prefill_graphs(self, bucket_size: int, kv_segment_size: int) -> None:
        """
        Extract the prefill HLO graphs with a specific bucket size and KV segment size.
        This uses the neuron graph capture backend to get all the HLO Graphs and throws
        CaptureComplete exception to dynamo which is suppressed here as it marks
        successful graph capture.

        Args:
            bucket_size: The number of active tokens to warm up (prefill bucket size)
            kv_segment_size: Size of each KV segment bucket for warmup
        """
        if self.capture_backend_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        logger.info(
            "Capturing prefill graphs for: bucket_size=%s, kv_segment_size=%s",
            bucket_size,
            kv_segment_size,
        )
        kwargs = self._build_prefill_synthetic_inputs(bucket_size, kv_segment_size)
        try:
            _ = self.capture_backend_model(**kwargs)
        except CaptureComplete:
            logger.debug(
                "Graph capture for prefill completed: bucket_size=%s, kv_segment_size=%s",
                bucket_size,
                kv_segment_size,
            )

        # === Draft model graph capture ===
        if self.drafter is not None:
            logger.info(
                "Capturing EAGLE3 prefill graphs for bucket size: %d", bucket_size
            )
            self.drafter.graph_extract(
                num_tokens=bucket_size,
                num_reqs=1,
                attn_metadata=kwargs["attn_metadata"],
            )

    def warmup_prefill(self, bucket_size: int, kv_segment_size: int) -> None:
        """
        Warm up the model for prefill with a specific bucket size and KV segment size.

        Creates synthetic inputs and calls self.model directly to trigger
        compilation without modifying any internal state. Also warms up the
        draft model's initial pass if speculative decoding is enabled.

        Args:
            bucket_size: The number of active tokens to warm up (prefill bucket size)
            kv_segment_size: Size of each KV segment bucket for warmup
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.kv_cache_config is None:
            raise RuntimeError(
                "KV cache not initialized. Call initialize_kv_cache() first."
            )
        logger.info(
            "Warming up prefill: bucket_size=%s, kv_segment_size=%s",
            bucket_size,
            kv_segment_size,
        )
        warmup_kwargs = self._build_prefill_synthetic_inputs(
            bucket_size, kv_segment_size
        )
        compile_start = time.perf_counter()
        if self._tensor_replacer is not None:
            set_active_context(
                self._tensor_replacer.warmup_context(bucket_size, self.device)
            )
        _ = self.model(**warmup_kwargs)
        if self._tensor_replacer is not None:
            set_active_context(None)
        compile_elapsed = time.perf_counter() - compile_start
        bucket_name = f"prefill_s{bucket_size}"
        if kv_segment_size > 0:
            bucket_name += f"_kv{kv_segment_size}"
        COMPILATION_TIME.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=bucket_name,
        ).set(compile_elapsed)
        logger.debug(
            "Prefill warmup completed: bucket_size=%s, kv_segment_size=%s",
            bucket_size,
            kv_segment_size,
        )

        # === Draft model warmup ===
        if self.drafter is not None:
            logger.info("Warming up EAGLE3 for bucket size: %d", bucket_size)
            self.drafter.warmup(
                num_tokens=bucket_size,
                num_reqs=1,
                attn_metadata=warmup_kwargs["attn_metadata"],
            )

    def _create_warmup_spec_decode_metadata(
        self,
        batch_size: int,
        num_spec_tokens: int,
    ) -> SpecDecodeMetadata:
        """
        Create synthetic spec_decode_metadata for warmup.

        Creates metadata with consistent tensor shapes matching inference,
        so torch.dynamo caches the correct compiled graph.

        Args:
            batch_size: Number of requests
            num_spec_tokens: Number of speculative tokens per request

        Returns:
            SpecDecodeMetadata with synthetic data
        """
        # Each request has num_spec_tokens draft tokens
        num_draft_tokens = [num_spec_tokens] * batch_size
        total_draft_tokens = batch_size * num_spec_tokens

        # Dummy draft token IDs
        draft_token_ids = torch.ones(total_draft_tokens, dtype=torch.int32)

        # Cumulative sums
        cu_num_draft = np.cumsum(num_draft_tokens, dtype=np.int64)
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft.copy())

        num_sampled = [num_spec_tokens + 1] * batch_size
        cu_num_sampled = np.cumsum(num_sampled, dtype=np.int64)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled.copy())

        # Build indices matching _build_spec_decode_metadata logic
        # target_logits_indices: indices for draft token verification
        target_indices = []
        for i in range(batch_size):
            base = i * (num_spec_tokens + 1)
            for j in range(num_spec_tokens):
                target_indices.append(base + j)
        target_logits_indices = torch.tensor(target_indices, dtype=torch.int64)

        # bonus_logits_indices: last position of each request
        bonus_logits_indices = torch.tensor(
            [cu_num_sampled[i] - 1 for i in range(batch_size)], dtype=torch.int64
        )

        # logits_indices: all sampled positions (total = batch_size * (num_spec_tokens + 1))
        logits_indices = torch.arange(
            batch_size * (num_spec_tokens + 1), dtype=torch.int64
        )

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens,
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

    def _build_decode_synthetic_inputs(
        self,
        batch_size: int,
        context_len: int = 256,
        spec_decode_enabled=False,
        decode_token_threshold: int | None = None,
        ctx_bucket: int | None = None,
    ) -> dict:
        """Build synthetic inputs for decode warmup.

        Args:
            batch_size: Number of concurrent decode requests
            context_len: Simulated context length for KV cache sizing
            ctx_bucket: If set, sizes the block_table columns to this sequence
                length instead of max_model_len (enables per-seq-bucket NEFFs).

        Returns:
            dict: Keyword arguments for self.model() forward call
        """
        # With spec decode: target model verifies batch_size * (1 + num_spec_tokens) tokens
        # Without spec decode: target model decodes batch_size tokens (1 per request)
        if spec_decode_enabled:
            num_spec_tokens = self.speculative_config.num_speculative_tokens
            num_tokens = batch_size * (1 + num_spec_tokens)
            num_reqs = batch_size

            # Create synthetic spec_decode_metadata to match inference kwargs
            spec_decode_metadata = self._create_warmup_spec_decode_metadata(
                batch_size=batch_size,
                num_spec_tokens=num_spec_tokens,
            )
            # Move spec_decode_metadata tensors to device (matching execute_model path)
            spec_decode_metadata.cu_num_draft_tokens = (
                spec_decode_metadata.cu_num_draft_tokens.to(self.device)
            )
            spec_decode_metadata.draft_token_ids = (
                spec_decode_metadata.draft_token_ids.to(self.device)
            )
        else:
            num_reqs = batch_size
            num_tokens = batch_size  # 1 token per request
            spec_decode_metadata = None

        # Create synthetic inputs
        input_ids = torch.ones(num_tokens, dtype=torch.int32).to(self.device)
        positions = torch.full((num_tokens,), context_len - 1, dtype=torch.long).to(
            self.device
        )
        logits_indices = torch.arange(num_tokens, dtype=torch.long).to(self.device)

        attn_metadata = self._build_warmup_attention_metadata(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            decode_token_threshold=decode_token_threshold,
            ctx_bucket=ctx_bucket,
        )

        # Create dummy sampling params for warmup
        dummy_sampling_params = torch.tensor(
            [[0, 1.0, 1.0]] * batch_size,  # [top_k, top_p, temperature] per request
            dtype=torch.float32,
            device=self.device,
        )

        # Replicate dummy sampling params for spec-decode verify step
        # (no-op when speculative decoding is not active).
        dummy_sampling_params = self._maybe_replicate_for_spec_decode(
            dummy_sampling_params, spec_decode_metadata
        )

        dummy_logit_mask = torch.ones(
            num_tokens, self.vocab_size, dtype=torch.bool, device=self.device
        )

        decode_warmup_kwargs = dict(
            input_ids=input_ids,
            positions=positions,
            attn_metadata=attn_metadata,
            sampling_positions=logits_indices,
            sampling_params=dummy_sampling_params,
            spec_decode_metadata=spec_decode_metadata,
            rank=self.rank_tensor,
            logit_mask=dummy_logit_mask,
        )
        if self.enable_prompt_embeds:
            decode_warmup_kwargs["inputs_embeds"] = torch.zeros(
                num_tokens,
                self.inputs_embeds_size,
                dtype=torch.bfloat16,
                device=self.device,
            )
            decode_warmup_kwargs["is_token_ids"] = torch.ones(
                num_tokens,
                dtype=torch.bool,
                device=self.device,
            )

        return decode_warmup_kwargs

    def extract_decode_graphs(
        self,
        batch_size: int,
        context_len: int = 256,
        ctx_bucket: int | None = None,
    ) -> None:
        """
        Extract the decode HLO graphs with a specific bucket size and KV segment size.
        This uses the neuron graph capture backend to get all the HLO Graphs and throws
        CaptureComplete exception to dynamo which is suppressed here as it marks
        successful graph capture.

        Args:
            batch_size: Number of concurrent decode requests
            context_len: Simulated context length for KV cache sizing (default 256)
            ctx_bucket: If set, sizes the block_table columns to this sequence
                length for per-seq-bucket NEFF compilation.
        """
        if self.capture_backend_model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        logger.info("Capturing decode graphs for batch size: %d", batch_size)
        spec_decode_enabled = (
            self.drafter is not None and self.speculative_config is not None
        )
        kwargs = self._build_decode_synthetic_inputs(
            batch_size,
            context_len,
            spec_decode_enabled=spec_decode_enabled,
            ctx_bucket=ctx_bucket,
        )
        try:
            _ = self.capture_backend_model(**kwargs)
        except CaptureComplete:
            logger.debug(
                "Graph capture for decode completed: batch size=%s", batch_size
            )

        if spec_decode_enabled:
            # === Target model decode WITHOUT spec decode ===
            logger.debug(
                "Target model decode graph capture WITH spec decode completed: batch size=%s",
                batch_size,
            )
            kwargs = self._build_decode_synthetic_inputs(
                batch_size,
                context_len,
                spec_decode_enabled=False,
                decode_token_threshold=1,
                ctx_bucket=ctx_bucket,
            )
            try:
                _ = self.capture_backend_model(**kwargs)
            except CaptureComplete:
                logger.debug(
                    "Graph capture for target model decode completed: batch size=%s",
                    batch_size,
                )
            logger.debug(
                "Target model decode graph capture WITHOUT spec decode completed: batch size=%s",
                batch_size,
            )

            # === Draft model decode graph capture ===
            logger.info("Capturing EAGLE3 decode graphs for batch size: %d", batch_size)
            num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_num_tokens = batch_size * (1 + num_spec_tokens)

            draft_attn_metadata = self._build_warmup_attention_metadata(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                ctx_bucket=ctx_bucket,
            )

            self.drafter.graph_extract(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                attn_metadata=draft_attn_metadata,
            )

    def warmup_decode(
        self,
        batch_size: int,
        context_len: int = 256,
        ctx_bucket: int | None = None,
    ) -> None:
        """
        Warm up the model for decode with a specific batch size.

        Creates synthetic inputs for batch_size concurrent decode requests,
        each generating 1 token, and calls self.model directly to trigger
        compilation without modifying any internal state. Also warms up the
        draft model's recurrent pass if speculative decoding is enabled.

        Args:
            batch_size: Number of concurrent decode requests
            context_len: Simulated context length for KV cache sizing (default 256)
            ctx_bucket: If set, sizes the block_table columns to this sequence
                length for per-seq-bucket NEFF compilation.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if self.kv_cache_config is None:
            raise RuntimeError(
                "KV cache not initialized. Call initialize_kv_cache() first."
            )
        logger.info("Warming up decode for batch size: %d", batch_size)
        spec_decode_enabled = (
            self.drafter is not None and self.speculative_config is not None
        )
        kwargs = self._build_decode_synthetic_inputs(
            batch_size,
            context_len,
            spec_decode_enabled=spec_decode_enabled,
            ctx_bucket=ctx_bucket,
        )
        compile_start = time.perf_counter()
        if self._tensor_replacer is not None:
            set_active_context(
                self._tensor_replacer.warmup_context(
                    kwargs["input_ids"].shape[0], self.device
                )
            )
        _ = self.model(**kwargs)
        if self._tensor_replacer is not None:
            set_active_context(None)
        compile_elapsed = time.perf_counter() - compile_start
        bucket_name = f"decode_b{batch_size}"
        if kwargs["spec_decode_metadata"] is not None:
            bucket_name += f"_s{1 + self.speculative_config.num_speculative_tokens}"
        # Include ctx_bucket in the label so per-pair compile time
        # is observable in the COMPILATION_TIME metric.
        if ctx_bucket is not None:
            bucket_name += f"_ctx{ctx_bucket}"
        COMPILATION_TIME.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=bucket_name,
        ).set(compile_elapsed)
        logger.debug(
            "Target model decode warmup completed: %d tokens",
            kwargs["input_ids"].shape[0],
        )

        if spec_decode_enabled:
            # === Target model decode warmup WITHOUT spec decode ===
            # When spec decode is enabled, also compile the non-spec path.
            # This NEFF is used when:
            # 1. Propose is skipped near max_model_len (max_position > limit)
            # 2. DI (disaggregated inference): prefill aux_hidden_states aren't
            #    transferred to decode, so the first decode re-runs without spec.
            kwargs = self._build_decode_synthetic_inputs(
                batch_size,
                context_len,
                spec_decode_enabled=False,
                decode_token_threshold=1,
                ctx_bucket=ctx_bucket,
            )
            compile_start = time.perf_counter()
            if self._tensor_replacer is not None:
                set_active_context(
                    self._tensor_replacer.warmup_context(
                        kwargs["input_ids"].shape[0], self.device
                    )
                )
            _ = self.model(**kwargs)
            if self._tensor_replacer is not None:
                set_active_context(None)
            compile_elapsed = time.perf_counter() - compile_start
            no_spec_bucket_name = f"decode_b{batch_size}_s1"
            if ctx_bucket is not None:
                no_spec_bucket_name += f"_ctx{ctx_bucket}"
            COMPILATION_TIME.labels(
                model_name=self.vllm_config.model_config.model,
                bucket_name=no_spec_bucket_name,
            ).set(compile_elapsed)
            logger.debug(
                "Target model decode warmup WITHOUT spec decode completed: %d tokens",
                kwargs["input_ids"].shape[0],
            )

            # === Draft model warmup ===
            logger.info("Warming up EAGLE3 for batch size: %d", batch_size)
            num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_num_tokens = batch_size * (1 + num_spec_tokens)

            draft_attn_metadata = self._build_warmup_attention_metadata(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                ctx_bucket=ctx_bucket,
            )

            self.drafter.warmup(
                num_tokens=draft_num_tokens,
                num_reqs=batch_size,
                attn_metadata=draft_attn_metadata,
            )

    def parallel_compile(self):
        """Deterministic parallel compile — requires a barrier before and after.

        Assumes all ranks in the TP group have finished graph extraction and
        been synchronized via a TP-scoped barrier, so the cache directory
        contains the complete set of HLOs for this model replica. Only
        TP-rank 0 performs compilation.
        """
        logger.info("Initiating parallel compile for captured graphs")
        from vllm.distributed.parallel_state import get_tp_group

        tp_group = get_tp_group()
        rank = tp_group.rank_in_group if tp_group.world_size >= 1 else None
        world_size = tp_group.world_size if tp_group.world_size >= 1 else None

        from vllm_neuron.compile.parallel_compile import (
            parallel_compile as neuron_parallel_compile,
        )

        neuron_parallel_compile(
            options=self.compile_options,
            rank=rank,
            world_size=world_size,
            remote_cache_dir=envs.VLLM_NEURON_REMOTE_CACHE,
        )

    # ------------------------------------------------------------------------
    # MODEL EXECUTION
    # ------------------------------------------------------------------------
    def _materialize_pending_async_output(self, reason: str) -> bool:
        """Materialize the previous async step if it has not been consumed.

        Args:
            reason: Debug context explaining why async flow is being broken.

        Returns:
            True if a pending async output was materialized, False otherwise.
        """
        if not self.use_async_scheduling:
            return False

        async_output = self.async_execution_buffer.get("async_output")
        if async_output is None:
            return False

        logger.debug("Materializing pending async output: %s", reason)
        async_output.get_output()
        return True

    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        kv_caches: list[torch.Tensor] | None = None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        """
        Execute model forward pass and store state for sampling.

        For async execution, this returns None and stores state in execute_model_state.
        For sync execution, this calls sample_tokens() internally and returns output.

        Args:
            scheduler_output: Output from vLLM scheduler

        Returns:
            - None for async execution (call sample_tokens() separately)
            - ModelRunnerOutput for sync execution
            - AsyncModelRunnerOutput if use_async_scheduling is on
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        logger.debug("=== Starting NeuronModelRunner.execute_model ===")
        logger.debug("scheduler_output=%s", scheduler_output)
        logger.debug(
            "Processing scheduler output: %s new, %s cached requests",
            len(getattr(scheduler_output, "scheduled_new_reqs", [])),
            len(getattr(scheduler_output.scheduled_cached_reqs, "req_ids", [])),
        )

        # Materialize pending async output BEFORE _update_states when the
        # batch composition is about to change.
        #
        # With async scheduling, the previous step's sampled tokens may still
        # be on-device futures (not yet written to token_ids_cpu or
        # req_state.output_token_ids). When _update_states calls
        # input_batch.add_request() for resumed/new requests, it copies from
        # req_state.output_token_ids into token_ids_cpu. If we haven't
        # materialized, output_token_ids is missing the latest sampled token
        # and the copied data would be stale.
        #
        # We detect batch composition changes by comparing the current step's
        # scheduled request IDs against the previous step's. This is simpler
        # and more robust than checking individual signals (finished_req_ids,
        # scheduled_new_reqs, etc.) because it catches ALL cases including
        # the unscheduled gap (max_tokens pipelining) in a single comparison.
        if self.use_async_scheduling:
            # Skip on first iteration — no pending async output to materialize.
            if not self.async_execution_buffer:
                self._batch_composition_changed = False
            else:
                curr_req_ids = frozenset(scheduler_output.num_scheduled_tokens.keys())
                prev_req_ids_ordered = self.async_execution_buffer.get(
                    "prev_req_ids_ordered"
                )
                prev_req_ids = (
                    frozenset(prev_req_ids_ordered)
                    if prev_req_ids_ordered is not None
                    else None
                )
                composition_changed = (
                    prev_req_ids is None or curr_req_ids != prev_req_ids
                )
                if composition_changed:
                    logger.debug(
                        "Breaking async flow: prev_req_ids=%s, curr_req_ids=%s.",
                        sorted(prev_req_ids) if prev_req_ids else None,
                        sorted(curr_req_ids),
                    )
                    self._materialize_pending_async_output("batch composition changed")
                    self._batch_composition_changed = True
                else:
                    self._batch_composition_changed = False

        # Update cached states (includes sequence ID management)
        self._update_states(scheduler_output)

        # Check if there's work to do
        total_scheduled_tokens = getattr(
            scheduler_output, "total_num_scheduled_tokens", 0
        )
        logger.debug("Total scheduled tokens: %s", total_scheduled_tokens)

        if total_scheduled_tokens == 0:
            if not has_kv_transfer_group():
                # Return empty ModelRunnerOutput if there's no work to do.
                logger.debug("No scheduled tokens, returning empty output")
                return EMPTY_MODEL_RUNNER_OUTPUT

            return self.kv_connector_no_forward(scheduler_output, self.vllm_config)

        # Prepare model inputs using vectorized operations
        input_ids, positions, logits_indices, attn_metadata, spec_decode_metadata = (
            self._prepare_model_input(scheduler_output)
        )

        # Async scheduling: replace CPU-built input_ids with the device future
        # from the previous step (steady-state), or keep CPU-built input_ids
        # as-is (broken flow / prefill / first iteration).
        if self.use_async_scheduling:
            if (
                self.async_execution_buffer
                and self._is_decode()
                and not self._batch_composition_changed
            ):
                future = self.async_execution_buffer["futures_sampled_token_ids"]
                # Reuse the sampled-token future only when it matches the
                # padded decode input expected by the compiled graph.
                if (
                    future.shape[0] == input_ids.shape[0]
                    and future.dtype == torch.uint32
                    and input_ids.dtype == torch.int32
                ):
                    # NKI argmax may return UInt32 token IDs. Token IDs fit in
                    # Int32, so reinterpret the bits instead of doing a device
                    # cast that would materialize or fail on Neuron.
                    future = future.view(torch.int32)

                if (
                    future.shape[0] == input_ids.shape[0]
                    and future.dtype == input_ids.dtype
                ):
                    input_ids, broke_async = self._maybe_remap_async_future(
                        future, input_ids
                    )
                    if broke_async:
                        self._sync_fallback_steps += 1
                    else:
                        self._async_steps += 1
                else:
                    materialized = self._materialize_pending_async_output(
                        "async future shape/dtype mismatch"
                    )
                    if materialized:
                        (
                            input_ids,
                            positions,
                            logits_indices,
                            attn_metadata,
                            spec_decode_metadata,
                        ) = self._prepare_model_input(scheduler_output)
                    logger.debug(
                        "Async future mismatch: future=(shape=%s, dtype=%s), "
                        "input_ids=(shape=%s, dtype=%s). "
                        "Using CPU-built input_ids%s.",
                        future.shape[0],
                        future.dtype,
                        input_ids.shape[0],
                        input_ids.dtype,
                        " after materializing pending async output"
                        if materialized
                        else "",
                    )
                    self._sync_fallback_steps += 1
            else:
                self._sync_fallback_steps += 1

        # num_valid_tokens tracks the pre-DP-padding count so we can strip
        # padding from logits later.  DP padding is applied inside
        # _prepare_model_input (before attn_metadata is built) so that
        # block_table, slot_mapping, and input_ids all share the same shape.
        num_valid_tokens = len(self.input_batch.req_ids)

        logger.debug(
            "Executing model with %s real requests, %s tokens (after DP+bucket padding)",
            num_valid_tokens,
            input_ids.shape[0],
        )

        # Get grammar bitmask for structured outputs.
        # The application point differs based on sampling mode:
        #   - On-device sampling: mask passed INTO model.forward() for device-side masked_fill
        #   - CPU sampling: mask applied to logits AFTER they return to CPU, before _sample()
        grammar_bitmask = self._get_grammar_bitmask(scheduler_output)
        cpu_grammar_bitmask = None

        # logit_mask must match lm_head output shape.
        # After DP padding, logits_indices may have grown, so use its length.
        num_logit_rows = (
            input_ids.shape[0]
            if spec_decode_metadata is not None
            else len(logits_indices)
        )
        all_true_mask = torch.ones(
            num_logit_rows, self.vocab_size, dtype=torch.bool, device=self.device
        )

        if self.on_device_sampling:
            # ON-DEVICE PATH: Pass grammar mask (or all-True no-op) into compiled model.
            # The "always pass, never None" pattern avoids torch.compile recompilation
            # (torch.compile traces different graphs for None vs tensor, causing NEFF recompile).
            logit_mask = (
                grammar_bitmask if grammar_bitmask is not None else all_true_mask
            )

            # Pad logit_mask to match decode batch padding.
            logit_mask = self._pad_logit_mask_for_decode(logit_mask, input_ids.shape[0])

            # TODO: Structured output + async scheduling is supported in vLLM 0.16
            # on GPU. Integrate this for Neuron so the guard can be removed.
            # Guard: Structured outputs with async scheduling is not supported
            if grammar_bitmask is not None and self.use_async_scheduling:
                raise AssertionError(
                    "Structured outputs with async scheduling is not supported. "
                    "Disable async scheduling with: --no-async-scheduling"
                )
        else:
            # CPU SAMPLING PATH: Grammar mask will be applied to logits on CPU
            # after the forward pass, before sampling.
            # The model still receives an all-True mask for torch.compile graph
            # consistency (warmup always passes logit_mask, so the compiled
            # graph expects it).
            cpu_grammar_bitmask = grammar_bitmask
            logit_mask = all_true_mask

            # Pad logit_mask for CPU path too (same reason as on-device path).
            logit_mask = self._pad_logit_mask_for_decode(logit_mask, input_ids.shape[0])

            # Guard: SO with async scheduling requires synchronous bitmask application
            if cpu_grammar_bitmask is not None and self.use_async_scheduling:
                raise AssertionError(
                    "Structured outputs with CPU sampling requires synchronous scheduling. "
                    "Grammar bitmask must be applied to logits before sampling. "
                    "Disable async scheduling with: --no-async-scheduling"
                )

        # Wrap with set_forward_context for DI support
        with (
            set_forward_context(
                None,
                self.vllm_config,
            ),
            self.maybe_get_kv_connector_output(scheduler_output) as kv_connector_output,
            self._kv_cache_dump_context(),
        ):
            # Execute model with padded inputs
            model_output_tensor, aux_hidden_states = self._execute_model_forward(
                input_ids,
                positions,
                logits_indices,
                attn_metadata,
                spec_decode_metadata,
                logit_mask=logit_mask,
            )

        # CPU SAMPLING PATH: Apply grammar bitmask to logits on CPU.
        # At this point, model_output_tensor contains logits on CPU (padding removed).
        # masked_fill runs in eager PyTorch — no torch.compile graph sensitivity.
        if cpu_grammar_bitmask is not None:
            model_output_tensor = self._apply_grammar_bitmask_cpu(
                model_output_tensor, cpu_grammar_bitmask
            )

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            model_output_tensor,
            spec_decode_metadata,
            positions,
            logits_indices,
            aux_hidden_states,
            input_ids,
            attn_metadata,
        )

        self.kv_connector_output = kv_connector_output
        return None

    def sample_tokens(
        self,
        grammar_output: GrammarOutput | None,
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        """
        Sample tokens from model output and generate final output.

        This is the second phase of the execute_model() -> sample_tokens() flow.
        It retrieves the state stored by execute_model() and performs sampling.

        Args:
            grammar_output: Grammar output for structured generation (unused for now)

        Returns:
            ModelRunnerOutput with sampled tokens
        """
        kv_connector_output = self.kv_connector_output
        self.kv_connector_output = None

        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            if not kv_connector_output:
                return None  # type: ignore[return-value]

            # In case of PP with kv transfer, we need to pass through the
            # kv_connector_output
            if kv_connector_output.is_empty():
                return EMPTY_MODEL_RUNNER_OUTPUT

            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            output.kv_connector_output = kv_connector_output
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            model_output_tensor,
            spec_decode_metadata,
            positions,
            logits_indices,
            aux_hidden_states,
            input_ids,
            attn_metadata,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Minimal bookkeeping logic to retain request id information for
        # vLLM async scheduling.
        # Note _bookkeeping_sync is not introducing new data structures to InputBatch, but
        # rather keeping a snapshot of the states of some fields in InputBatch before they
        # are overwritten by vllm scheduler to schedule next step asynchronously. We use
        # the snapshots for preparing model input for the current step.
        # This is similar to `_bookkeeping_sync` in gpu_model_runner.
        (
            req_ids_output_copy,
            req_id_to_index_output_copy,
        ) = self._bookkeeping_sync()

        # Sample
        # run sampler with torch compile calls disabled, and ran eagerly
        # ex. https://github.com/vllm-project/vllm/blob/main/vllm/v1/sample/ops/logprobs.py
        with torch.compiler.set_stance("force_eager"):
            sampler_output = self._sample(model_output_tensor, spec_decode_metadata)

        # Capture raw rejection sampler tensor for draft model's on-device
        # token extraction. Only available with on-device sampling + spec decode.
        raw_sampled_token_ids = (
            model_output_tensor
            if self.on_device_sampling and spec_decode_metadata is not None
            else None
        )

        # Debug: write raw logits to shared memory for test validation.
        # Written AFTER sampling so we can filter to accepted tokens for spec decode.
        if self._debug_logits_dir:
            logits_to_write = (
                self._on_device_logits
                if self._on_device_logits is not None
                else model_output_tensor
            )
            if spec_decode_metadata is not None:
                # Write logits for all accepted token positions (including
                # the correction position). The correction position's logit
                # was computed with the rejected draft token as input, but
                # the correct logit will be recomputed in the next verify
                # step. The collector uses the last entry per position_id,
                # so the correct logit overwrites the wrong one.
                accepted_ids = sampler_output.sampled_token_ids
                accepted_indices = []
                accepted_req_ids = []
                offset = 0
                # accepted_ids may be padded to padded_num_reqs for spec decode;
                # only write logits for real requests.
                real_num_reqs = self.input_batch.num_reqs
                for req_i, req_tokens in enumerate(accepted_ids):
                    if req_i >= real_num_reqs:
                        break
                    n_sampled = 1 + spec_decode_metadata.num_draft_tokens[req_i]
                    n_accepted = len(req_tokens)
                    for j in range(n_accepted):
                        idx = offset + j
                        if idx < logits_to_write.shape[0]:
                            accepted_indices.append(idx)
                            accepted_req_ids.append(req_ids_output_copy[req_i])
                    offset += n_sampled
                if accepted_indices:
                    indices_t = torch.tensor(accepted_indices, dtype=torch.long)
                    self._debug_write_logits(
                        logits_to_write[indices_t],
                        accepted_req_ids,
                        positions,
                        logits_indices[indices_t],
                    )
            else:
                self._debug_write_logits(
                    logits_to_write,
                    req_ids_output_copy,
                    positions,
                    logits_indices,
                )

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )

        # If spec decode enabled, log acceptance stats and propose draft tokens.
        # aux_hidden_states is None when eagle3 is not active.
        if self.is_eagle3_spec and aux_hidden_states is not None:
            if spec_decode_metadata is not None:
                accepted = sampler_output.sampled_token_ids
                for req_i, tokens in enumerate(accepted):
                    n_draft = spec_decode_metadata.num_draft_tokens[req_i]
                    logger.debug(
                        "Spec decode: req=%d, drafted=%d, accepted=%d/%d",
                        req_i,
                        n_draft,
                        len(tokens) - 1,
                        n_draft,
                    )

            max_position = positions.max().item() if positions.numel() > 0 else 0
            num_spec_tokens = self.drafter.num_speculative_tokens
            # Stop proposing early enough that the scheduler never trims
            # draft tokens. After this step's verification with full acceptance,
            # num_computed_tokens becomes max_position + 1. The scheduler trims
            # when num_computed_tokens >= max_model_len - 1 - num_spec_tokens.
            # So we skip when max_position + 1 >= max_model_len - 1 - num_spec_tokens,
            # i.e., max_position >= max_model_len - num_spec_tokens - 2.
            spec_decode_limit = self.max_model_len - num_spec_tokens - 2

            # Skip draft proposal in two cases to avoid recompilation:
            # 1. Near max_model_len: proposed tokens would exceed the
            #    sequence length limit.
            # 2. Previous step had no spec decode (spec_decode_metadata is
            #    None in a decode step): the target ran with 1 token per
            #    request, so input_ids.shape[0] is a small value (e.g. 1
            #    for single-request batch). Calling propose() would hit
            #    draft_prefill_s{small} — a bucket that was never warmed.
            skip_propose = max_position >= spec_decode_limit or (
                spec_decode_metadata is None
                and input_ids.shape[0] == self.input_batch.num_reqs
            )

            if skip_propose:
                logger.debug(
                    "Skipping draft proposal: max_position=%d, limit=%d, "
                    "spec_decode_metadata=%s, num_tokens=%d",
                    max_position,
                    spec_decode_limit,
                    "present" if spec_decode_metadata is not None else "None",
                    input_ids.shape[0],
                )
                self._draft_token_ids = None
            else:
                self._draft_token_ids = self._propose_draft_token_ids(
                    scheduler_output=scheduler_output,
                    sampled_token_ids=sampler_output.sampled_token_ids,
                    aux_hidden_states=aux_hidden_states,
                    spec_decode_metadata=spec_decode_metadata,
                    input_ids=input_ids,
                    positions=positions,
                    attn_metadata=attn_metadata,
                    logits_indices=logits_indices,
                    raw_sampled_token_ids=raw_sampled_token_ids,
                )

        # Trim sampler output back to real num_reqs before output generation.
        # With spec decode batch padding, the sampler runs at padded_num_reqs;
        # output/bookkeeping expect only real requests.
        #
        # Skip when async scheduling is active: the device tensor future must
        # retain its padded shape so the next step's swap logic
        # (future.shape[0] == input_ids.shape[0]) matches the
        # decode-batch-padded input_ids. Trimming to [real_num_reqs] while
        # input_ids stays at [padded_num_reqs] would permanently disable the
        # async device-to-device path. The async materialization path in
        # get_output() already indexes by snapshot_req_ids, so only real
        # requests get their tokens written back.
        if not self.use_async_scheduling:
            real_num_reqs = self.input_batch.num_reqs
            sids = sampler_output.sampled_token_ids
            if isinstance(sids, list) and len(sids) > real_num_reqs:
                sampler_output.sampled_token_ids = sids[:real_num_reqs]
            elif isinstance(sids, torch.Tensor) and sids.shape[0] > real_num_reqs:
                sampler_output.sampled_token_ids = sids[:real_num_reqs]

        # Generate output
        output = self._generate_model_runner_output(
            sampler_output,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            kv_connector_output,
            scheduler_output,
        )

        logger.debug("=== Completed NeuronModelRunner.execute_model ===")

        if not self.use_async_scheduling:
            logger.debug(f"Model runner output: {output}")
            return output

        async_output = AsyncNeuronModelRunnerOutput(
            model_runner_output=output,
            model_runner=self,
            partial_prefill_req_ids=getattr(
                self, "_async_partial_prefill_req_ids", set()
            ),
        )
        self.async_execution_buffer["async_output"] = async_output
        return async_output

    def _is_decode(self) -> bool:
        """Check if current batch is decode-only (no prefill requests).

        A request is in decode phase when all its prompt tokens have been
        computed, i.e. ``num_computed_tokens >= num_prompt_tokens``.
        Prefix-caching may set ``num_computed_tokens`` to a non-zero value
        before the first forward pass, so we cannot simply check for zero.

        This is used by the async scheduling path in ``execute_model()`` to
        decide whether the previous step's sampled-token future can be reused
        as ``input_ids``. Feeding a stale 1-token future into a new prefill
        request causes a shape mismatch in the NKI attention kernel.

        Returns:
            True if all requests are in decode phase, False otherwise
        """
        num_reqs = self.input_batch.num_reqs
        if num_reqs == 0:
            return False

        for i in range(num_reqs):
            req_id = self.input_batch.req_ids[i]
            req_state = self.requests.get(req_id)
            if req_state is None:
                # Unknown request — be conservative, treat as prefill
                return False
            num_computed = self.input_batch.num_computed_tokens_cpu[i]
            num_prompt = req_state.num_prompt_tokens
            if num_computed < num_prompt:
                return False

        return True

    def _maybe_remap_async_future(
        self,
        future: torch.Tensor,
        cpu_input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, bool]:
        """Remap the device future if condense() reordered the batch.

        When async scheduling reuses the previous step's sampled-token tensor
        as ``input_ids``, the tokens are in the slot ordering from when the
        future was produced. If ``condense()`` has since shifted requests to
        fill gaps left by finished requests, the slot ordering no longer
        matches and each request would receive another request's token.

        This method compares the current batch ordering against the stored
        ``prev_req_ids_ordered`` and, if they differ, builds a permutation
        index to gather the future's tokens into the correct slots on-device.

        Args:
            future: The device tensor of sampled token IDs from the previous
                step, shaped ``[padded_batch_size]``.
            cpu_input_ids: CPU-built input IDs for the current step. Used as
                fallback if the previous future cannot be remapped safely.

        Returns:
            A tuple of ``(input_ids, broke_async)``. ``broke_async`` is False
            only when the original future can be reused directly. If remapping
            is needed, the clone/gather materializes the future on the worker
            main thread, so ``broke_async`` is True. If a request in the
            current batch wasn't present in the previous step (shouldn't happen
            when composition_changed is False, but handled defensively), the
            CPU-built input IDs are returned and ``broke_async`` is True.
        """
        prev_req_ids_ordered = self.async_execution_buffer.get("prev_req_ids_ordered")
        num_reqs = self.input_batch.num_reqs
        if prev_req_ids_ordered is None or num_reqs == 0:
            return future, False

        # Build a mapping from req_id -> old slot index.
        prev_id_to_idx = {
            req_id: idx for idx, req_id in enumerate(prev_req_ids_ordered)
        }

        needs_remap = False
        perm: list[int] = []
        for cur_idx in range(num_reqs):
            req_id = self.input_batch.req_ids[cur_idx]
            old_idx = prev_id_to_idx.get(req_id)
            if old_idx is None:
                # Request wasn't in the previous step. This shouldn't happen
                # when composition_changed is False, but if it does, signal
                # the caller to fall back to the CPU-built input_ids.
                logger.debug("Async future remap failed, using CPU-built input_ids.")
                return cpu_input_ids, True
            perm.append(old_idx)
            if old_idx != cur_idx:
                needs_remap = True

        if not needs_remap:
            return future, False

        # Gather tokens from old positions into new positions on-device.
        # Only remap the real request slots; padding slots stay as-is.
        # NOTE: clone() forces the future and therefore breaks the steady-state
        # async path for accounting purposes, even though the remapped tensor
        # remains the correct input for this step.
        perm_tensor = torch.tensor(perm, dtype=torch.long, device=future.device)
        remapped = future.clone()
        remapped[:num_reqs] = future[perm_tensor]
        logger.debug("Remapped device future: perm=%s", perm)
        return remapped, True

    def _build_prompt_embeds_tensors(
        self,
        input_ids: torch.Tensor,
        token_indices_tensor: torch.Tensor,
        req_ids: list[str],
        scheduler_output: Any,
        padding_map: dict[str, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build batch-aligned `inputs_embeds` and `is_token_ids` tensors.

        vLLM flattens scheduled requests into one per-step token layout.
        `is_token_ids` is a per-position mask used by `torch.where`:
        True keeps `embed_tokens(input_ids)`, False uses `inputs_embeds`.

        Example for one request with 5 prompt tokens padded to 8:
          - token ids: [10, 20, 30, 40, 50, 0, 0, 0]
          - user embeddings are provided for positions 1..3

          inputs_embeds:
            [[0,...], [e1], [e2], [e3], [0,...], [0,...], [0,...], [0,...]]
          is_token_ids:
            [True, False, False, False, True, True, True, True]

        In the model backbone:
          torch.where(is_token_ids, embed_tokens_output, inputs_embeds)
        keeps `embed_tokens_output` where True, and swaps in user embeddings
        where False.
        """
        num_tokens = input_ids.shape[0]
        dtype = torch.bfloat16

        inputs_embeds = torch.zeros(num_tokens, self.inputs_embeds_size, dtype=dtype)
        is_token_ids = torch.ones(num_tokens, dtype=torch.bool)

        if not self.enable_prompt_embeds:
            return inputs_embeds, is_token_ids

        if (
            not hasattr(self.input_batch, "req_prompt_embeds")
            or not self.input_batch.req_prompt_embeds
        ):
            return inputs_embeds, is_token_ids

        if hasattr(self.input_batch, "is_token_ids_tensor"):
            is_token_ids_flat = self.input_batch.is_token_ids_tensor.flatten()
            # src_idx walks scheduled (unpadded) tokens; dst_idx walks padded output slots.
            src_idx = 0
            dst_idx = 0
            flat_max_idx = is_token_ids_flat.shape[0] - 1

            for req_id in req_ids:
                num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                padding = padding_map.get(req_id, 0)
                total_for_req = num_sched + padding

                if num_sched > 0:
                    req_token_indices = token_indices_tensor[
                        src_idx : src_idx + num_sched
                    ]
                    req_token_indices = req_token_indices.clamp(0, flat_max_idx)
                    req_is_token_ids = torch.index_select(
                        is_token_ids_flat, 0, req_token_indices
                    )
                    is_token_ids[dst_idx : dst_idx + num_sched] = req_is_token_ids

                src_idx += num_sched
                dst_idx += total_for_req

        output_idx = 0
        for i, req_id in enumerate(req_ids):
            req_idx = self.input_batch.req_id_to_index.get(req_id)
            if req_idx is None:
                num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
                padding = padding_map.get(req_id, 0)
                output_idx += num_sched + padding
                continue

            num_sched = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            padding = padding_map.get(req_id, 0)
            total_for_req = num_sched + padding

            if req_idx not in self.input_batch.req_prompt_embeds:
                output_idx += total_for_req
                continue

            req_embeds = self.input_batch.req_prompt_embeds[req_idx]
            start_pos = self.input_batch.num_computed_tokens_cpu[req_idx]
            actual_num_sched = min(num_sched, req_embeds.shape[0] - start_pos)
            if actual_num_sched > 0:
                embed_slice = req_embeds[start_pos : start_pos + actual_num_sched]
                inputs_embeds[output_idx : output_idx + actual_num_sched] = (
                    embed_slice.to(dtype)
                )

            output_idx += total_for_req

        logger.debug(
            "Built prompt_embeds tensors: inputs_embeds=%s, is_token_ids=%s, "
            "num_embed_tokens=%d",
            inputs_embeds.shape,
            is_token_ids.shape,
            int((~is_token_ids).sum().item()),
        )

        return inputs_embeds, is_token_ids

    def _execute_model_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
        logit_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Execute the model forward pass.

        Args:
            input_ids: Input token IDs
            positions: Position IDs
            attn_metadata: Attention metadata
            spec_decode_metadata: Spec decode metadata for rejection sampling
            kv_caches: Key-value caches

        Returns:
            - model_output_tensor: torch.Tensor
                Can be logits (for CPU sampling)
                or sampled token ids (already sampled on device)
        """
        # TODO: avoid f-string in production logging as f-strings are evaluated even if
        # the log is not printed
        if not self.use_async_scheduling:
            # we need to avoid actualizing the input_ids tensor futures here.
            logger.debug(
                "Starting model forward pass with input_ids=%s, positions=%s",
                input_ids,
                positions,
            )

        # Move inputs to Neuron device
        input_ids = input_ids.to(self.device)
        positions = positions.to(self.device)
        logits_indices = logits_indices.to(self.device)
        if spec_decode_metadata is not None:
            spec_decode_metadata.cu_num_draft_tokens = (
                spec_decode_metadata.cu_num_draft_tokens.to(self.device)
            )
            spec_decode_metadata.draft_token_ids = (
                spec_decode_metadata.draft_token_ids.to(self.device)
            )
        # Prepare sampling params tensor for on-device sampling
        sampling_params_tensor = None
        if self.on_device_sampling:
            sampling_metadata = self.input_batch.sampling_metadata
            num_reqs = self.input_batch.num_reqs

            sampling_params_tensor = build_sampling_params_tensor(
                sampling_metadata, num_reqs, self.device
            )

            # Pad sampling_params to match the decode batch bucket size.
            # For decode, block_table rows == padded_num_reqs (1 token per
            # request), but num_reqs may be smaller before padding.
            # For prefill, sampling_params is already [num_reqs, 3] = [1, 3]
            # which matches warmup — no padding needed.
            if attn_metadata:
                first_meta = next(iter(attn_metadata.values()))
                bt_rows = first_meta["block_table_tensor"].shape[0]
                if sampling_params_tensor.shape[0] < bt_rows:
                    pad_rows = bt_rows - sampling_params_tensor.shape[0]
                    sampling_params_tensor = torch.nn.functional.pad(
                        sampling_params_tensor, (0, 0, 0, pad_rows), value=0
                    )

            # Replicate sampling params for spec-decode verify step
            # (no-op when speculative decoding is not active).
            sampling_params_tensor = self._maybe_replicate_for_spec_decode(
                sampling_params_tensor, spec_decode_metadata
            )

            sampling_params_tensor = sampling_params_tensor.to(self.device)

        # Execute model on Neuron device (compiled with vllm_neuron backend)
        model_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "attn_metadata": attn_metadata,
            "sampling_positions": logits_indices,
            "sampling_params": sampling_params_tensor,
            "spec_decode_metadata": spec_decode_metadata,
            "rank": self.rank_tensor,
        }
        model_kwargs["logit_mask"] = logit_mask

        # Keep forward signature stable for torch.compile by always passing these
        # kwargs when prompt-embeds is enabled. Use dummy tensors when this step
        # has no real prompt_embeds.
        if self.enable_prompt_embeds:
            if self._current_inputs_embeds is not None:
                model_kwargs["inputs_embeds"] = self._current_inputs_embeds.to(
                    self.device
                )
                model_kwargs["is_token_ids"] = self._current_is_token_ids.to(
                    self.device
                )
            else:
                num_tokens = input_ids.shape[0]
                model_kwargs["inputs_embeds"] = torch.zeros(
                    num_tokens,
                    self.inputs_embeds_size,
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                model_kwargs["is_token_ids"] = torch.ones(
                    num_tokens,
                    dtype=torch.bool,
                    device=self.device,
                )

        # Execute model on Neuron device (compiled with vllm_neuron backend)
        if self._tensor_replacer is not None:
            first_attn_meta = (
                next(iter(attn_metadata.values())) if attn_metadata else None
            )
            if first_attn_meta is not None:
                is_prefill = (
                    first_attn_meta["max_query_len"]
                    > first_attn_meta["decode_token_threshold"]
                )
            else:
                is_prefill = len(positions) > len(self.input_batch.req_ids)
            set_active_context(
                self._tensor_replacer.build_context(
                    req_ids=list(self.input_batch.req_ids),
                    positions=positions.cpu().tolist(),
                    is_prefill=is_prefill,
                    device=self.device,
                )
            )
        with model_forward_context(self.vllm_config):
            model_output = self.model(**model_kwargs)
        if self._tensor_replacer is not None:
            set_active_context(None)

        # Determine bucket name for NEFF execution count metric.
        first_attn_metadata = (
            next(iter(attn_metadata.values())) if attn_metadata else None
        )
        is_prefill = (
            first_attn_metadata is not None
            and first_attn_metadata["max_query_len"]
            > first_attn_metadata["decode_token_threshold"]
        )
        if is_prefill:
            neff_bucket_name = f"prefill_s{input_ids.shape[0]}"
            kv_segment_size = first_attn_metadata.get("kv_segment_size", 0)
            if kv_segment_size > 0:
                neff_bucket_name += f"_kv{kv_segment_size}"
        else:
            num_reqs = self.input_batch.num_reqs or input_ids.shape[0]
            tokens_per_req = input_ids.shape[0] // num_reqs
            neff_bucket_name = f"decode_b{num_reqs}"
            if self.drafter is not None:
                neff_bucket_name += f"_s{tokens_per_req}"

        NEFF_EXECUTION_COUNT.labels(
            model_name=self.vllm_config.model_config.model,
            bucket_name=neff_bucket_name,
        ).inc()

        # Strip and save captured tensors if capture is active
        if self._capture_model is not None:
            model_output, captures = self._capture_registry.extract(
                model_output, self._capture_model.original_output_count
            )
            if (
                self._capture_registry is not None
                and self._capture_registry.enabled
                and captures
            ):
                self._capture_registry.write(
                    captures=captures,
                    capture_names=self._capture_model.capture_names,
                    req_ids=list(self.input_batch.req_ids),
                    positions=positions,
                    is_prefill=is_prefill,
                )

        # Parse model output based on on-device sampling and spec decode configuration
        self._on_device_logits = None

        if self.on_device_sampling:
            if self.is_eagle3_spec:
                # Eagle3: model returns (sampled_tokens, aux_hidden_states, gathered_logits)
                (
                    model_output_tensor,
                    aux_hidden_states,
                    self._on_device_logits,
                ) = model_output
            else:
                if isinstance(model_output, tuple):
                    # TODO: Require all ODS models to return (sampled_tokens, gathered_logits)
                    # and remove this isinstance check once gpt-oss and other models are updated.
                    model_output_tensor, self._on_device_logits = model_output
                else:
                    model_output_tensor = model_output
                aux_hidden_states = None
        elif self.is_eagle3_spec:
            # Eagle3 without ODS: model returns (logits, aux_hidden_states)
            model_output_tensor, aux_hidden_states = model_output
        else:
            model_output_tensor = model_output
            aux_hidden_states = None
            # Store logits for debug logits writing in non-ODS mode
            if self._debug_logits_dir:
                self._on_device_logits = model_output_tensor

        # Move model_output_tensor (logits or sampled token ids) back to CPU
        if not self.use_async_scheduling:
            model_output_tensor = model_output_tensor.to("cpu")

        return model_output_tensor, aux_hidden_states

    def _has_component_dp(self) -> bool:
        """Return True if any component DP size > 1 (attention, embedding, mlp, lm_head)."""
        nc = self.neuron_config
        return (
            nc.attention_dp_size > 1
            or nc.embedding_dp_size > 1
            or nc.mlp_dp_size > 1
            or nc.lm_head_dp_size > 1
        )

    def _get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        """
        Determine padding needed to synchronize batch sizes across DP ranks.

        **Critical: Padding prevents collective operation deadlocks**

        In DP deployments with MoE/expert layers, collectives (all-to-all, all-gather)
        operate across DP ranks. These collectives require identical tensor shapes from
        all participants. If ranks have different batch sizes:
          - Collectives expect uniform shapes but receive mismatched shapes
          - Operations hang indefinitely waiting for data that will never arrive
          - System deadlocks - requires manual intervention to recover

        To prevent this:
        1. All ranks coordinate via all_gather to find maximum token count
        2. Each rank pads its batch to match the maximum
        3. Collectives proceed with uniform batch sizes across all ranks
        4. Padding tokens are removed before sampling (only real tokens are sampled)

        Example preventing deadlock:
          4 DP ranks with token counts: [100, 150, 80, 120]
          Without padding → Collective HANGS (mismatched sizes)
          With padding → All pad to 150 → Collective succeeds

        Args:
            num_tokens: Number of tokens on this rank (before padding)

        Returns:
            tuple of:
            - num_pad_tokens: Number of padding tokens to add on this rank
            - num_tokens_after_padding: Tensor with padded token count for all ranks
        """
        # TODO: Leverage vLLM's forward_context DPMetadata for DP coordination
        # Why: vLLM has built-in DP coordination via forward_context.py that could replace/enhance this logic
        # See vllm/forward_context.py:
        # - DPMetadata.num_tokens_across_dp() for token count synchronization via all_reduce
        # - DPMetadata.should_ubatch_across_dp() for coordinated microbatching decisions for PP cases
        # - Automatic integration via set_forward_context() when parallel_config.data_parallel_size > 1
        logger.debug("Getting DP padding info")

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank

        if dp_size == 1:
            # No DP, no padding needed
            return 0, None

        # Synchronize batch sizes across DP ranks when cross-DP collectives
        # are present.  EP introduces all-to-all/all-gather in MoE layers;
        # component DP (attention_dp, embedding_dp, mlp_dp, lm_head_dp)
        # introduces all-gather/all-reduce on TP supergroups that span DP.
        # Both require uniform tensor shapes across DP ranks.
        ep_enabled = self.vllm_config.parallel_config.enable_expert_parallel
        if not ep_enabled and not self._has_component_dp():
            return 0, None

        from vllm.distributed.parallel_state import get_dp_group

        group = get_dp_group().cpu_group
        device = "cpu"
        num_tokens_across_dp = [0] * dp_size
        num_tokens_across_dp[dp_rank] = num_tokens
        num_tokens_tensor = torch.tensor(
            num_tokens_across_dp, device=device, dtype=torch.int32
        )
        torch.distributed.all_reduce(num_tokens_tensor, group=group)

        # Find maximum token count across all ranks
        max_tokens_across_dp = torch.max(num_tokens_tensor).item()

        # Create tensor showing final token count for each rank (all padded to max)
        num_tokens_after_padding = torch.tensor(
            [max_tokens_across_dp] * dp_size, device="cpu", dtype=torch.int32
        )

        # Calculate padding needed for this rank
        num_pad = max_tokens_across_dp - num_tokens

        logger.debug(
            "DP padding: rank=%s, local_tokens=%s, max_across_dp=%s, padding=%s",
            dp_rank,
            num_tokens,
            max_tokens_across_dp,
            num_pad,
        )

        return num_pad, num_tokens_after_padding

    def _add_padding_to_inputs(
        self, input_ids: torch.Tensor, positions: torch.Tensor, num_pad: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add padding tokens to input tensors for DP synchronization.

        Args:
            input_ids: Original input token IDs
            positions: Original position IDs
            num_pad: Number of padding tokens to add

        Returns:
            tuple of (padded_input_ids, padded_positions)
        """
        if num_pad == 0:
            return input_ids, positions

        # Create padding tokens (use token_id=0, position=0)
        pad_ids = torch.zeros(num_pad, dtype=torch.int32)
        pad_pos = torch.zeros(num_pad, dtype=torch.long)

        # Concatenate original + padding
        # Tensors to torch.compile functions must be contiguous
        # (Neuron runtime requires contiguous input tensors for compiled graphs)
        # NOTE: This works for async execution as inputs are dummy inputs and there is
        # no data dependency from previous forward output.
        padded_input_ids = torch.cat([input_ids, pad_ids], dim=0).contiguous()
        padded_positions = torch.cat([positions, pad_pos], dim=0).contiguous()

        logger.debug(
            "Added %s padding tokens: input_ids %s → %s",
            num_pad,
            input_ids.shape,
            padded_input_ids.shape,
        )

        return padded_input_ids, padded_positions

    def _remove_padding_from_logits(
        self, logits: torch.Tensor, num_valid_tokens: int
    ) -> torch.Tensor:
        """
        Remove padding from logits tensor to get only valid outputs.

        Args:
            logits: Logits tensor that may contain padding
            num_valid_tokens: Number of valid (non-padding) tokens

        Returns:
            Logits tensor with padding removed
        """
        if logits.shape[0] == num_valid_tokens:
            # No padding to remove
            return logits

        # Slice to remove padding positions
        valid_logits = logits[:num_valid_tokens]

        logger.debug(
            "Removed padding from logits: %s → %s", logits.shape, valid_logits.shape
        )

        return valid_logits

    def _debug_write_logits(
        self,
        logits: torch.Tensor,
        req_ids: list[str],
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
    ) -> None:
        """Write raw logits to shared memory for test infrastructure validation.

        Enabled by setting debug_logits_dir in additional_config.neuron_config.
        Only TP rank 0 writes to avoid duplicates within DP groups.
        Each DP rank writes its own file with _dp{rank} suffix.

        Binary format:
            Header (16 bytes): num_items (int64), vocab_size (int64)
            Per item: batch_idx (int64), position_id (int64), logits (float32[vocab_size])

        File naming: step_{N}_dp{rank}.bin

        The position_id allows the reader to consolidate logits across prefill/decode
        steps regardless of batch size variations. Each logit corresponds to exactly
        one sequence position where a token is generated.
        """
        # Only TP rank 0 writes (avoids duplicates within DP group)
        tp_group = get_tp_group()
        if tp_group and tp_group.rank_in_group != 0:
            return

        dp_rank = self.vllm_config.parallel_config.data_parallel_rank
        logits_dir = self._debug_logits_dir
        os.makedirs(logits_dir, exist_ok=True)

        step = self._debug_logits_step_counter
        tmp_path = os.path.join(logits_dir, f"step_{step}_dp{dp_rank}.tmp")
        final_path = os.path.join(logits_dir, f"step_{step}_dp{dp_rank}.bin")

        # Write logits for all sampling positions (handles spec decode
        # verification where one step produces logits for multiple tokens).
        logits_cpu = logits.cpu().to(torch.float32).numpy()
        num_logit_rows = logits_cpu.shape[0]
        vocab_size = logits_cpu.shape[-1]

        items = []
        for row in range(num_logit_rows):
            try:
                batch_idx = int(req_ids[row].split("-")[-2])
            except (ValueError, IndexError):
                batch_idx = row
            position_id = int(positions[logits_indices[row]].item())
            items.append((batch_idx, position_id, logits_cpu[row]))

        with open(tmp_path, "wb") as f:
            np.array([len(items), vocab_size], dtype=np.int64).tofile(f)
            for batch_idx, position_id, logit_row in items:
                np.array([batch_idx], dtype=np.int64).tofile(f)
                np.array([position_id], dtype=np.int64).tofile(f)
                logit_row.tofile(f)

        # Atomic rename signals completion to reader
        os.rename(tmp_path, final_path)
        self._debug_logits_step_counter = step + 1

    def execute_dummy_batch(self) -> None:
        """
        Execute a dummy batch for Data Parallel synchronization.

        When using Data Parallel (DP), if this rank has no real requests but
        other DP ranks do, this method runs a minimal dummy forward pass to
        keep all ranks synchronized during collective operations.

        This is called automatically by the vLLM engine when DP workload
        imbalance is detected (see vllm/v1/engine/llm_engine.py).

        All tensor shapes must exactly match what execute_model produces for
        a real decode batch of the same size, otherwise torch.compile traces
        a different graph and the collectives across DP ranks deadlock.
        """
        logger.debug("Executing dummy batch for DP synchronization")

        ep_enabled = self.vllm_config.parallel_config.enable_expert_parallel
        if not ep_enabled and not self._has_component_dp():
            return

        # Use the smallest decode bucket as the dummy token count.  This rank
        # has zero real requests; the only purpose is to participate in the
        # cross-DP EP collectives so that busy ranks don't deadlock.
        # _get_dp_padding will pad us up to match whatever the busy rank
        # actually has, so starting at the smallest bucket avoids forcing the
        # busy rank to pad beyond its natural bucket size.
        num_tokens = (
            self.neuron_config.num_seqs_buckets[0]
            if self.neuron_config.num_seqs_buckets
            else self.max_num_reqs
        )
        dummy_input_ids = torch.zeros(num_tokens, dtype=torch.int32)
        dummy_positions = torch.zeros(num_tokens, dtype=torch.long)

        # Coordinate with other DP ranks via padding logic
        num_pad, num_tokens_across_dp = self._get_dp_padding(num_tokens)

        if num_pad > 0:
            dummy_input_ids, dummy_positions = self._add_padding_to_inputs(
                dummy_input_ids, dummy_positions, num_pad
            )
            num_tokens = dummy_input_ids.shape[0]

        # All shapes below must exactly match what execute_model produces
        # for a real decode batch of size num_tokens.

        # logits_indices: [num_tokens] — one per decode "request"
        dummy_logits_indices = torch.arange(num_tokens, dtype=torch.long)

        # logit_mask: [num_tokens, vocab_size] — all-True (no constraints)
        dummy_logit_mask = torch.ones(
            num_tokens, self.vocab_size, dtype=torch.bool, device=self.device
        )

        # Attention metadata: build directly (self.input_batch has no requests
        # so _build_attention_metadata would read stale / wrong shapes).
        # Must mirror _build_warmup_attention_metadata for decode: SWA layers
        # use trimmed block tables, full-attention layers use full tables.
        dummy_attn_metadata = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
            self.kv_cache_config.kv_cache_groups
        ):
            spec = kv_cache_group_spec.kv_cache_spec
            block_size = spec.block_size
            max_num_blocks = (
                self.max_model_len + block_size * max(self._dcp_size, 1) - 1
            ) // (block_size * max(self._dcp_size, 1))

            swa_kv_pos_offset = None
            if isinstance(spec, SlidingWindowSpec):
                num_swa_blocks = self._compute_swa_num_blocks(
                    spec.sliding_window, block_size
                )
                if num_swa_blocks < max_num_blocks:
                    max_num_blocks = num_swa_blocks
                swa_kv_pos_offset = torch.zeros(
                    num_tokens, dtype=torch.int32, device=self.device
                )

            blk_table_tensor = torch.full(
                (num_tokens, max_num_blocks),
                PAD_SLOT_ID,
                dtype=torch.int32,
                device=self.device,
            )
            slot_mapping = torch.full(
                (num_tokens,), PAD_SLOT_ID, dtype=torch.int64, device=self.device
            )
            cached_seq_len_tensor = torch.tensor(
                [[0]], dtype=torch.int32, device=self.device
            )
            attn_metadata_i = {
                "block_table_tensor": blk_table_tensor,
                "slot_mapping": slot_mapping,
                "max_query_len": 1,
                "block_size": block_size,
                "max_blocks_per_seq": max_num_blocks,
                "decode_token_threshold": 1,
                "cached_seq_len": cached_seq_len_tensor,
                "kv_segment_size": 0,
            }
            if swa_kv_pos_offset is not None:
                attn_metadata_i["swa_kv_pos_offset"] = swa_kv_pos_offset
            for layer_name in kv_cache_group_spec.layer_names:
                dummy_attn_metadata[layer_name] = attn_metadata_i

        # Execute model forward pass
        # All DP ranks will have the same number of tokens after padding
        prev_inputs_embeds = self._current_inputs_embeds
        prev_is_token_ids = self._current_is_token_ids
        self._current_inputs_embeds = None
        self._current_is_token_ids = None
        try:
            # Capture output to keep NrtaFuture reference alive, preventing GC
            # from freeing NRT tensor slices while async execution is in flight.
            dummy_output_tensor, aux_hidden_states = self._execute_model_forward(
                input_ids=dummy_input_ids,
                positions=dummy_positions,
                logits_indices=dummy_logits_indices,
                attn_metadata=dummy_attn_metadata,
                logit_mask=dummy_logit_mask,
            )
            if self.use_async_scheduling:
                dummy_output_tensor.cpu()
                if aux_hidden_states is not None:
                    # aux_hidden_states is a list of tensors (one per Eagle3 capture layer)
                    for hidden_state in aux_hidden_states:
                        hidden_state.cpu()
        finally:
            self._current_inputs_embeds = prev_inputs_embeds
            self._current_is_token_ids = prev_is_token_ids

        logger.debug("Dummy batch execution completed")

    def _generate_model_runner_output(
        self,
        sampler_output: SamplerOutput,
        req_ids_output_copy: list[str],
        req_id_to_index_output_copy: dict[str, int],
        kv_connector_output: KVConnectorOutput | None = None,
        scheduler_output: "SchedulerOutput | None" = None,
    ) -> ModelRunnerOutput:
        """
        Generate ModelRunnerOutput from model logits using vLLM Sampler.

        Args:
            sampler_output: sampler output
            req_ids_output_copy: snapshot of req_ids for async scheduling
            req_id_to_index_output_copy: snapshot of req_id_to_index
            kv_connector_output: kv connector output
            scheduler_output: scheduler output, used to detect partial prefills

        Returns:
            ModelRunnerOutput with correctly ordered results
        """
        if self.use_async_scheduling:
            self._async_partial_prefill_req_ids = self._get_partial_prefill_req_ids(
                scheduler_output,
                req_ids_output_copy,
            )
            model_runner_output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=sampler_output.sampled_token_ids,
                logprobs=(
                    sampler_output.logprobs_tensors.tolists()
                    if sampler_output.logprobs_tensors
                    else None
                ),
                prompt_logprobs_dict={},
                pooler_output=[None] * len(self.input_batch.req_ids),
                kv_connector_output=kv_connector_output,
            )
            self.async_execution_buffer["futures_sampled_token_ids"] = (
                sampler_output.sampled_token_ids
            )
            # Record the ordered request IDs at the time the future was
            # produced. The next step uses this both for composition change
            # detection (set comparison) and for remapping the device future
            # when condense() reorders the batch (index lookup).
            self.async_execution_buffer["prev_req_ids_ordered"] = list(
                req_ids_output_copy
            )
            return model_runner_output

        # Not async scheduling, tensor to list conversion
        sampled_token_ids_tensor = sampler_output.sampled_token_ids

        # Spec decode: rejection sampler returns [B, max_spec_len+1] padded with -1.
        # Strip placeholders before updating state / returning output.
        if isinstance(sampled_token_ids_tensor, torch.Tensor):
            if sampled_token_ids_tensor.ndim == 1:
                sampled_token_ids_tensor = sampled_token_ids_tensor.unsqueeze(1)
            if (
                sampled_token_ids_tensor.ndim == 2
                and sampled_token_ids_tensor.shape[1] > 1
            ):
                sampled_token_ids = self._parse_rejection_sampling_output(
                    sampled_token_ids_tensor,
                    self.input_batch.vocab_size,
                )
            else:
                sampled_token_ids = sampled_token_ids_tensor.cpu().tolist()
        else:
            # Already a list for on-device sampling
            sampled_token_ids = sampled_token_ids_tensor

        # Discard sampled tokens from partial (chunked) prefill requests.
        # During chunked prefill, the model produces logits for the last
        # position of each chunk, but those are NOT valid output tokens
        # until the entire prompt has been processed.
        partial_prefill_req_ids = self._get_partial_prefill_req_ids(
            scheduler_output,
            self.input_batch.req_ids,
        )
        for req_idx, req_id in enumerate(self.input_batch.req_ids):
            if req_id not in partial_prefill_req_ids:
                continue
            logger.debug(
                "Discarding sampled token for partial prefill request %s",
                req_id,
            )
            if req_idx < len(sampled_token_ids):
                sampled_token_ids[req_idx] = []

        # Convert logprobs efficiently if present
        logprobs_lists = (
            sampler_output.logprobs_tensors.tolists()
            if sampler_output.logprobs_tensors
            else None
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(self.input_batch.req_ids),
            kv_connector_output=kv_connector_output,
        )

        # Update batch state with sampled tokens
        self._update_batch_state_with_samples(model_runner_output.sampled_token_ids)

        logger.debug(
            "Generated output for %s requests", len(model_runner_output.req_ids)
        )

        return model_runner_output

    def _get_partial_prefill_req_ids(
        self,
        scheduler_output: "SchedulerOutput | None",
        req_ids: list[str],
    ) -> set[str]:
        """Return request IDs whose current chunked-prefill step is not final."""
        if scheduler_output is None:
            return set()

        partial_prefill_req_ids = set()
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id in req_ids:
            if req_id not in self.requests:
                continue
            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                continue
            req_state = self.requests[req_id]
            num_computed = self.input_batch.num_computed_tokens_cpu[req_index]
            num_scheduled = num_scheduled_tokens.get(req_id, 0)
            if num_computed + num_scheduled < req_state.num_prompt_tokens:
                partial_prefill_req_ids.add(req_id)

        return partial_prefill_req_ids

    def _sample(
        self,
        model_output_tensor: torch.Tensor,
        spec_decode_metadata: SpecDecodeMetadata | None = None,
    ) -> SamplerOutput:
        # Use InputBatch's automatically maintained sampling metadata
        # The metadata is created and updated by InputBatch.refresh_metadata() in _update_cached_states()
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata

        if self.use_async_scheduling:
            if spec_decode_metadata is not None:
                raise NotImplementedError("Spec decode not supported in async mode")
            return SamplerOutput(model_output_tensor, None)

        if spec_decode_metadata is None:
            if not self.on_device_sampling:
                logger.info("Using vLLM Sampler")
                sampler_output = self.sampler(
                    logits=model_output_tensor,
                    sampling_metadata=sampling_metadata,
                )
            else:
                output_token_ids = [[x] for x in model_output_tensor.tolist()]
                # Use gathered logits to compute logprobs via vLLM Sampler
                logprobs = None
                if (
                    self._on_device_logits is not None
                    and (sampling_metadata.max_num_logprobs or 0) > 0
                ):
                    logits_cpu = self._on_device_logits.to("cpu")
                    logprobs_output = self.sampler(
                        logits=logits_cpu,
                        sampling_metadata=sampling_metadata,
                    )
                    logprobs = logprobs_output.logprobs_tensors
                sampler_output = SamplerOutput(output_token_ids, logprobs)
            return sampler_output

        # Spec decode
        else:
            # On-device rejection sampling
            if self.on_device_sampling:
                # Model already performed rejection sampling on-device
                # model_output_tensor: [batch_size, max_spec_len+1] with -1 padding
                # Parse to list[list[int]] format expected by vLLM
                output_token_ids = self._parse_rejection_sampling_output(
                    model_output_tensor,
                    self.input_batch.vocab_size,
                )
                sampler_output = SamplerOutput(output_token_ids, None)
                return sampler_output

            # CPU rejection sampling
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            logger.info("Using vLLM Sampler")
            bonus_logits = model_output_tensor[
                spec_decode_metadata.bonus_logits_indices
            ]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = model_output_tensor[
                spec_decode_metadata.target_logits_indices
            ]

            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )

            sampler_output.sampled_token_ids = output_token_ids

        return sampler_output

    def _parse_rejection_sampling_output(
        self,
        output_token_ids: torch.Tensor,
        vocab_size: int,
    ) -> list[list[int]]:
        """Parse the output of the rejection sampler.

        Args:
            output_token_ids: The sampled token IDs in shape
                [batch_size, max_spec_len + 1]. The rejected tokens are
                replaced with `PLACEHOLDER_TOKEN_ID`(=-1) by the rejection sampler
                and will be filtered out in this function.
            vocab_size: The size of the vocabulary.

        Returns:
            A list of lists of token IDs.
        """
        output_token_ids_np = output_token_ids.cpu().numpy()
        # Create mask for valid tokens.
        valid_mask = (output_token_ids_np != -1) & (output_token_ids_np < vocab_size)
        outputs = [
            row[valid_mask[i]].tolist() for i, row in enumerate(output_token_ids_np)
        ]
        return outputs

    def _update_states_after_model_execute(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: list[list[int]],
    ) -> None:
        """Update the cached states after model execution.

        On GPU this handles MTP/EAGLE for hybrid models (linear attention
        state shifting). Neuron does not support hybrid models yet, so this
        is a no-op.
        """
        pass

    def _update_batch_state_with_samples(
        self,
        sampled_token_ids: list[list[int]],
        snapshot_req_ids: list[str] | None = None,
    ) -> None:
        """
        Update persistent batch state with newly sampled tokens.

        This is critical for maintaining token history across iterations.

        Args:
            sampled_token_ids: List of sampled token lists for each request,
                indexed by the batch order at the time of sampling.
            snapshot_req_ids: Optional snapshot of req_ids from the time of
                sampling. When provided, tokens are mapped to the current
                batch index via req_id lookup, which is safe across
                condense() reordering. When None, positional indexing is
                used (only safe when the batch hasn't been reordered).
        """
        for sample_idx, tokens in enumerate(sampled_token_ids):
            if not tokens:
                continue

            # Resolve the current batch index for this request.
            if snapshot_req_ids is not None:
                if sample_idx >= len(snapshot_req_ids):
                    continue
                req_id = snapshot_req_ids[sample_idx]
                req_idx = self.input_batch.req_id_to_index.get(req_id)
                if req_idx is None:
                    # Request was removed from the batch (finished).
                    continue
            else:
                req_idx = sample_idx
                if req_idx >= len(self.input_batch.req_ids):
                    continue
                # Resolve req_id eagerly so debug logging and the post-loop
                # state update both see it. Sync mode relied on a late
                # binding previously.
                req_id = self.input_batch.req_ids[req_idx]

            # Update token counts using vLLM's numpy arrays
            if req_idx < len(self.input_batch.num_tokens_no_spec):
                start_idx = self.input_batch.num_tokens_no_spec[req_idx]
                end_idx = start_idx + len(tokens)

                # Update token_ids_cpu using vLLM's token_ids_cpu tensor
                if (
                    req_idx < self.input_batch.token_ids_cpu_tensor.shape[0]
                    and end_idx <= self.input_batch.token_ids_cpu_tensor.shape[1]
                ):
                    # Convert tokens to tensor and update the CPU tensor
                    token_tensor = torch.tensor(tokens, dtype=torch.int32)
                    self.input_batch.token_ids_cpu_tensor[
                        req_idx, start_idx:end_idx
                    ] = token_tensor

                # Update counters in numpy arrays
                self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            # Update request state
            if req_id in self.requests:
                self.requests[req_id].output_token_ids.extend(tokens)

        logger.debug(
            "Updated batch state with %s total tokens",
            len([t for tokens in sampled_token_ids for t in tokens]),
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        if isinstance(self._draft_token_ids, torch.Tensor):
            draft_token_ids = self._draft_token_ids.tolist()
        else:
            draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def _propose_draft_token_ids(
        self,
        scheduler_output: Any,
        sampled_token_ids: torch.Tensor | list[list[int]],
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: dict,
        logits_indices: torch.Tensor | None = None,
        raw_sampled_token_ids: torch.Tensor | None = None,
    ) -> list[list[int]] | torch.Tensor:
        # TODO: now only supports EAGLE speculative decoding
        # Add more when needed
        assert isinstance(self.drafter, EagleProposer)

        num_reqs = self.input_batch.num_reqs

        if num_reqs == 0:
            logger.info("No requests to process, returning empty tensor")
            return torch.empty(
                (0, self.drafter.num_speculative_tokens), dtype=torch.int32
            )

        # Compute last_token_indices: one index per request pointing to the
        # last real token in the padded input tensor.
        if spec_decode_metadata is not None:
            last_token_indices_cpu = spec_decode_metadata.bonus_logits_indices.cpu().to(
                torch.long
            )
        elif logits_indices is not None:
            last_token_indices_cpu = logits_indices.cpu().to(torch.long)
        else:
            req_ids = self.input_batch.req_ids
            num_scheduled_tokens_per_req = [
                scheduler_output.num_scheduled_tokens[req_id] for req_id in req_ids
            ]
            cu_num_tokens = list(
                torch.tensor(num_scheduled_tokens_per_req).cumsum(0).tolist()
            )
            last_token_indices_cpu = torch.tensor(
                [c - 1 for c in cu_num_tokens], dtype=torch.long
            )

        logger.debug(
            "Draft model indices: last_token_indices=%s",
            last_token_indices_cpu.tolist(),
        )

        # Build raw_sampled_token_ids tensor for on-device extraction.
        # For on-device sampling + spec decode, raw_sampled_token_ids is
        # the [bs, max_spec_len+1] tensor passed from sample_tokens().
        # For prefill (no spec decode), build a [bs, 1] tensor from the list.
        if raw_sampled_token_ids is None:
            # Prefill case or CPU sampling: build from list
            next_token_ids_cpu, _ = extract_next_token_ids(
                sampled_token_ids, self.vocab_size
            )
            raw_sampled_token_ids = next_token_ids_cpu.unsqueeze(1)

        # EAGLE3 feeds concatenated aux hidden states into the draft model.
        assert aux_hidden_states is not None
        target_hidden_states = torch.cat(aux_hidden_states, dim=-1)

        draft_token_ids = self.drafter.propose(
            target_token_ids=input_ids,
            target_positions=positions,
            target_hidden_states=target_hidden_states,
            last_token_indices=last_token_indices_cpu,
            attn_metadata=attn_metadata,
            raw_sampled_token_ids=raw_sampled_token_ids,
        )
        draft_token_ids = draft_token_ids.cpu()

        return draft_token_ids

    # ------------------------------------------------------------------------
    # KV CACHE INTERFACE
    # ------------------------------------------------------------------------
    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initialize KV cache with the given configuration.

        Args:
            kv_cache_config: KV cache configuration object
        """
        logger.debug("Initializing KV cache in NeuronModelRunner")

        logger.debug("kv_cache_config = %s", kv_cache_config)

        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config

        # TODO do we need to initialize attn_backend?

        # Metadata Builder Initialization: It sets up attention metadata builders
        # for each attention group. These builders are responsible for creating
        # the attention metadata that kernels need during forward passes.

        # Extract block sizes for each KV cache group (one per group for InputBatch)
        block_sizes = [
            group.kv_cache_spec.block_size for group in kv_cache_config.kv_cache_groups
        ]

        # Initialize InputBatch with block_sizes list matching the number of KV cache groups
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_batched_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.vocab_size,
            block_sizes=block_sizes,
            kernel_block_sizes=block_sizes,
            logitsprocs=None,
            logitsprocs_need_output_token_ids=False,
            is_spec_decode=self.speculative_config is not None,
            # Neuron doesn't support pooling models yet
            is_pooling_model=False,
        )

        # Initialize the KV Cache tensors
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for tensor in kv_cache_config.kv_cache_tensors:
            raw_tensor = torch.zeros(tensor.size, dtype=torch.int8, device=self.device)
            # Case where the KV cache is shared across layers
            for layer_name in tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = raw_tensor

        # Build KV caches per group
        kv_caches = {}
        for group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = group.kv_cache_spec

            # This is the case that all layers have the same kv_hidden_size.
            if isinstance(kv_cache_spec, (FullAttentionSpec, SlidingWindowSpec)):
                for layer_name in group.layer_names:
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0

                    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes

                    block_size = kv_cache_spec.block_size
                    num_kv_heads = kv_cache_spec.num_kv_heads
                    head_size = kv_cache_spec.head_size
                    from vllm_neuron.envs import is_native_backend

                    if is_native_backend():
                        cache_shape = (num_blocks, num_kv_heads, block_size, head_size)
                        k_cache = torch.zeros(
                            cache_shape,
                            dtype=kv_cache_spec.dtype,
                            device=raw_tensor.device,
                        )
                        v_cache = torch.zeros(
                            cache_shape,
                            dtype=kv_cache_spec.dtype,
                            device=raw_tensor.device,
                        )
                        kv_caches[layer_name] = [k_cache, v_cache]
                    else:
                        kv_shape = (2, num_blocks, num_kv_heads, block_size, head_size)
                        typed_tensor = raw_tensor.view(kv_cache_spec.dtype).view(
                            kv_shape
                        )
                        kv_caches[layer_name] = [typed_tensor[0], typed_tensor[1]]
                        # Store the full (2, ...) tensor for DI connector registration
                        # typed_tensor[0] and typed_tensor[1] are views into this tensor
                        self._kv_cache_full_tensors[layer_name] = typed_tensor

            # Spec decoding specifically use this because hidden_size of different layers
            # (draft and target model) are different.
            # https://github.com/vllm-project/vllm/pull/25101
            elif isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                for layer_name in group.layer_names:
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    kv_cache_layer_spec = kv_cache_spec.kv_cache_specs[layer_name]
                    assert raw_tensor.numel() % kv_cache_layer_spec.page_size_bytes == 0

                    num_blocks = (
                        raw_tensor.numel() // kv_cache_layer_spec.page_size_bytes
                    )

                    block_size = kv_cache_layer_spec.block_size
                    num_kv_heads = kv_cache_layer_spec.num_kv_heads
                    head_size = kv_cache_layer_spec.head_size

                    if isinstance(
                        kv_cache_layer_spec, (FullAttentionSpec, SlidingWindowSpec)
                    ):
                        kv_shape = (2, num_blocks, num_kv_heads, block_size, head_size)
                    else:
                        raise NotImplementedError(
                            f"Unsupported Attention spec type: {type(kv_cache_layer_spec)}"
                        )

                    typed_tensor = raw_tensor.view(kv_cache_layer_spec.dtype).view(
                        kv_shape
                    )
                    kv_caches[layer_name] = [typed_tensor[0], typed_tensor[1]]  # [k, v]
                    self._kv_cache_full_tensors[layer_name] = typed_tensor

            else:
                raise NotImplementedError(
                    f"Unsupported Attention spec type: {type(kv_cache_spec)}"
                )

        # This binds the cache tensors to the model
        self.model.bind_kv_cache(kv_caches)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # This binds the cache tensors to the draft model
            self.drafter.model.bind_kv_cache(kv_caches)
            # validate all draft model layers belong to the same kv cache group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            kv_caches_view = self.get_kv_cache_view_for_connector_registration(
                kv_caches
            )
            get_kv_transfer_group().register_kv_caches(kv_caches_view)

        return kv_caches

    def get_kv_cache_view_for_connector_registration(self, kv_caches):
        """
        Get KV cache view for connector registration.

        Returns the full (2, num_blocks, num_kv_heads, block_size, head_size)
        tensors that share memory with the [k_cache, v_cache] slices used by
        the model.

        For the native backend (which allocates separate K/V tensors), falls
        back to returning kv_caches as-is.
        """
        if self._kv_cache_full_tensors:
            return self._kv_cache_full_tensors
        return kv_caches

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Get KV cache specifications for all model layers.

        Returns:
            Dictionary mapping layer names to their KV cache specifications
        """
        all_kv_cache_specs = {}
        block_size = self.vllm_config.cache_config.block_size
        kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            self.vllm_config.cache_config.cache_dtype, self.vllm_config.model_config
        )

        target_kv_spec = self.model.get_kv_spec()
        for layer in target_kv_spec.layers:
            layer_name = layer.name
            # Use SlidingWindowSpec for SWA layers so HMA can create separate
            # KV cache groups. When --no-disable-hybrid-kv-cache-manager is set,
            # this enables block clipping in the NiXL connector.
            if layer.sliding_window_size is None:
                spec = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                    sliding_window=layer.sliding_window_size,
                    attention_chunk_size=layer.chunk_size,
                )
            else:
                spec = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                    sliding_window=layer.sliding_window_size,
                )
            all_kv_cache_specs[layer_name] = spec

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)

            drafter_kv_spec = self.drafter.model.get_kv_spec()
            for layer in drafter_kv_spec.layers:
                layer_name = layer.name
                all_kv_cache_specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=layer.num_kv_heads,
                    head_size=layer.head_size,
                    dtype=kv_cache_dtype,
                    sliding_window=layer.sliding_window_size,
                    attention_chunk_size=layer.chunk_size,
                )

        return all_kv_cache_specs

    def _bookkeeping_sync(
        self,
    ) -> tuple[
        list[str],
        dict[str, int],
    ]:
        """
        Bookkeeping a snapshot of the states of some fields in InputBatch before they
        are overwritten by vllm scheduler to schedule future step asynchronously. We use
        the snapshots for preparing model input for the current step.

        This is similar to `_bookkeeping_sync` in gpu_model_runner.
        """
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        return req_ids_output_copy, req_id_to_index_output_copy

    # ------------------------------------------------------------------------
    # KV CACHE DEBUG UTILITIES
    # ------------------------------------------------------------------------
    def get_kv_caches(self) -> dict[str, bytes]:
        """Extract KV cache tensors from model layers, serialized as bytes.

        Example:
            >>> kv = model_runner.get_kv_caches()
        """
        import io

        if self.model is None:
            return {}

        kv_caches = {}
        # Unwrap torch.compile to reach actual model
        model = self.model
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        layers = getattr(getattr(model, "model", model), "layers", [])

        for i, layer in enumerate(layers):
            layer_name = f"layers.{i}.self_attn"
            attn = getattr(layer, "self_attn", None)
            if attn and getattr(attn, "k_cache", None) is not None:
                # Serialize tensors to bytes to avoid pickle issues
                k_cpu = attn.k_cache.detach().cpu()
                v_cpu = attn.v_cache.detach().cpu()
                buf = io.BytesIO()
                torch.save({"k": k_cpu, "v": v_cpu}, buf)
                kv_caches[layer_name] = buf.getvalue()

        return kv_caches

    def get_block_table_info(self) -> dict:
        """Extract block table info for KV cache reconstruction.

        Returns per-group block tables and layer-to-group mapping.
        Non-hybrid models have a single group containing all layers.

        Example:
            >>> info = model_runner.get_block_table_info()
        """
        if self.input_batch is None:
            return {}

        groups = []
        for bt in self.input_batch.block_table.block_tables:
            groups.append(
                {
                    "block_table": bt.block_table.cpu.numpy().tolist(),
                    "num_blocks_per_row": bt.num_blocks_per_row.tolist(),
                    "block_size": bt.block_size,
                }
            )

        layer_to_group = {}
        if self.kv_cache_config:
            for group_idx, group in enumerate(self.kv_cache_config.kv_cache_groups):
                for layer_name in group.layer_names:
                    layer_to_group[layer_name] = group_idx

        return {
            "groups": groups,
            "layer_to_group": layer_to_group,
            "seq_lens": self.input_batch.num_tokens_no_spec.tolist(),
        }

    def get_kv_cache_config(self) -> dict:
        """Return KVCacheConfig info as serializable dict for KV reconstruction.

        Returns:
            Dict with:
            - groups: List of per-group dicts, each containing the group's
              layer_names and all fields from its KVCacheSpec (block_size,
              num_kv_heads, head_size, dtype, sliding_window, etc.)
              plus 'spec_type' indicating the spec class name.
            - tp_size: Tensor parallel size
            - total_num_kv_heads: Total KV heads across all TP ranks
            - dcp_size: Decode context parallel size (must be 1 for now)
            - pcp_size: Prefill context parallel size (must be 1 for now)

        Example:
            >>> cfg = model_runner.get_kv_cache_config()
        """
        from dataclasses import fields as dc_fields

        if self.kv_cache_config is None:
            return {}

        groups = []
        for g in self.kv_cache_config.kv_cache_groups:
            spec = g.kv_cache_spec
            group_info = {
                "layer_names": g.layer_names,
                "spec_type": type(spec).__name__,
            }
            for f in dc_fields(spec):
                val = getattr(spec, f.name)
                if isinstance(val, torch.dtype):
                    val = str(val)
                group_info[f.name] = val
            groups.append(group_info)

        parallel_config = self.vllm_config.parallel_config
        model_config = self.vllm_config.model_config

        return {
            "num_blocks": self.kv_cache_config.num_blocks,
            "groups": groups,
            "tp_size": parallel_config.tensor_parallel_size,
            "total_num_kv_heads": model_config.get_total_num_kv_heads(),
            "dcp_size": parallel_config.decode_context_parallel_size,
            "pcp_size": parallel_config.prefill_context_parallel_size,
        }

    def get_block_tables(self) -> list[bytes]:
        """Return block tables for all KV cache groups as serialized bytes.

        Returns the current block table if requests are active, otherwise
        returns the last snapshot taken during the most recent forward pass.

        On first call, enables block table snapshotting for subsequent
        forward passes (zero overhead until first call).

        Example:
            >>> tables = model_runner.get_block_tables()
        """
        import io

        # Enable snapshotting for future forward passes
        self._kv_snapshot_enabled = True

        if self.input_batch is None:
            if self._block_table_snapshot is None:
                return []
            result = []
            for bt_cpu in self._block_table_snapshot:
                buf = io.BytesIO()
                torch.save(bt_cpu, buf)
                result.append(buf.getvalue())
            return result

        # Check if any requests are still active
        has_active = False
        for bt in self.input_batch.block_table.block_tables:
            if hasattr(bt, "num_blocks_per_row") and any(
                n > 0 for n in bt.num_blocks_per_row
            ):
                has_active = True
                break

        if not has_active:
            # Blocks freed — use snapshot from last forward pass
            if self._block_table_snapshot is not None:
                result = []
                for bt_cpu in self._block_table_snapshot:
                    buf = io.BytesIO()
                    torch.save(bt_cpu, buf)
                    result.append(buf.getvalue())
                return result

        result = []
        for bt in self.input_batch.block_table.block_tables:
            buf = io.BytesIO()
            torch.save(_remap_null_block_to_sentinel(bt.block_table.cpu), buf)
            result.append(buf.getvalue())
        return result

    def clear_kv_snapshot(self) -> None:
        """Release block table snapshot memory and disable snapshotting.

        Call after KV cache analysis is complete to free memory and
        restore zero-overhead production behavior.

        Example:
            >>> model_runner.clear_kv_snapshot()
        """
        self._kv_snapshot_enabled = False
        self._block_table_snapshot = None

    @contextmanager
    def _kv_cache_dump_context(self):
        """Context manager that dumps KV cache before and after a block.

        Enabled by setting the environment variable VLLM_NEURON_DUMP_KV_CACHE=1.
        When disabled this is a zero-cost no-op.
        """
        enabled = os.environ.get("VLLM_NEURON_DUMP_KV_CACHE", "0") == "1"
        if not enabled:
            yield
            return

        self._dump_kv_cache_impl("before_forward")
        try:
            yield
        finally:
            self._dump_kv_cache_impl("after_forward")

    def _dump_kv_cache_impl(self, tag: str) -> None:
        """Save KV cache tensors to disk with timestamp for debugging.

        Args:
            tag: A label for the save point (e.g. 'begin', 'end').
        """
        import datetime
        import io
        import pathlib

        rank = self.rank_tensor.item()
        if rank != 0:
            return

        save_dir = pathlib.Path("/tmp/kv_cache_dumps")
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        kv_caches = self.get_kv_caches()
        if not kv_caches:
            logger.debug("dump_kv_cache_impl(%s): no KV caches to save", tag)
            return

        save_path = save_dir / f"kv_cache_rank{rank}_{ts}_{tag}.pt"

        tensors = {}
        for layer_name, raw_bytes in kv_caches.items():
            buf = io.BytesIO(raw_bytes)
            tensors[layer_name] = torch.load(buf, weights_only=True)

        torch.save(tensors, str(save_path))

    # ------------------------------------------------------------------------
    # TENSOR CAPTURE
    # ------------------------------------------------------------------------
    def enable_capture(self) -> None:
        """Enable tensor capture writing to disk."""
        if self._capture_registry is not None:
            self._capture_registry.enable()

    def disable_capture(self) -> None:
        """Disable tensor capture writing to disk."""
        if self._capture_registry is not None:
            self._capture_registry.disable()

    # ------------------------------------------------------------------------
    # OTHER VLLM INTERFACE METHODS
    # ------------------------------------------------------------------------
    def profile_run(self) -> None:
        """Profile the model (no-op for now)."""
        logger.debug("Profile run called - no-op implementation")

    def capture_model(self) -> None:
        """Capture the model for optimization (no-op for now)."""
        logger.debug("Capture model called - no-op implementation")

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        """
        Get the tasks supported by this model runner.

        Returns:
            Tuple of supported tasks
        """
        return ("generate",)
