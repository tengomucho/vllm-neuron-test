# SPDX-License-Identifier: Apache-2.0
import contextlib
import logging
import time

import torch
import torch.nn as nn
from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group
from vllm.v1.kv_cache_interface import KVCacheConfig

from vllm_neuron import envs
from vllm_neuron.compile.backend import model_forward_context
from vllm_neuron.compile.capture_backend import CaptureComplete
from vllm_neuron.metrics import (
    COMPILATION_TIME,
    MODEL_LOAD_SIZE,
    MODEL_LOAD_TIME,
    NEFF_EXECUTION_COUNT,
)
from vllm_neuron.model.neuron_config import (
    NeuronConfig,
    OnDeviceSamplingConfig,
)
from vllm_neuron.model.registry import get_models

logger = logging.getLogger(__name__)


class EagleProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        on_device_sampling: bool = True,
    ):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        assert self.speculative_config is not None

        self.draft_model_config = self.speculative_config.draft_model_config
        self.method = self.speculative_config.method
        assert self.method == "eagle3"

        self.device = device
        self.on_device_sampling = on_device_sampling
        self.num_speculative_tokens = self.speculative_config.num_speculative_tokens

        self.attn_layer_names: list[str] = []
        self.capture_backend_model = None

        # We pass rank as a tensor input to avoid it becoming a constant
        # TODO: Update dist.get_rank() to avoid becoming a constant during lowering, and remove this
        world_group = get_world_group()
        world_rank = world_group.rank if world_group else 0
        self.rank_tensor = torch.tensor(
            world_rank, dtype=torch.int32, device=self.device
        )

    def load_model(self, target_hidden_size: int) -> None:
        # Load draft model
        target_num_layers = self.vllm_config.model_config.hf_config.num_hidden_layers
        self.model = self.compile_and_load_draft_model(
            start_layer_idx=target_num_layers,
            target_hidden_size=target_hidden_size,
        )
        logger.info("Completed compilation for draft model.")

        # Set draft model attn layer names
        draft_num_layers = self.draft_model_config.hf_config.num_hidden_layers
        self.attn_layer_names = [
            f"layers.{i}.self_attn"
            for i in range(target_num_layers, target_num_layers + draft_num_layers)
        ]

        # TODO some weights can be shared across target and draft models. e.g., embedding, lm_heads in certain cases

    def _build_synthetic_inputs(
        self,
        num_tokens: int,
        num_reqs: int,
    ) -> dict:
        """
        Build synthetic inputs for the draft model. Shared by ``warmup`` and
        ``graph_extract`` so both trace the same code path.

        Args:
            num_tokens: Total number of tokens
            num_reqs: Number of requests in the batch

        Returns:
            dict: Keyword arguments for ``propose``-style forward calls
                  (excluding ``attn_metadata`` which the caller owns).
        """
        assert self.model is not None
        hidden_size = self.model.config.hidden_size

        # Create synthetic inputs
        target_token_ids = torch.ones(num_tokens, dtype=torch.int32).to(self.device)

        # Uses safe position values (zeros) to avoid KV cache overflow during recurrent passes.
        target_positions = torch.zeros(num_tokens, dtype=torch.long).to(self.device)

        target_hidden_states = torch.ones(
            num_tokens, hidden_size * 3, dtype=torch.bfloat16
        ).to(self.device)

        # last_token_indices: index of last token for each request
        tokens_per_req = num_tokens // num_reqs
        last_token_indices = torch.tensor(
            [(i + 1) * tokens_per_req - 1 for i in range(num_reqs)], dtype=torch.long
        ).to(self.device)

        # raw_sampled_token_ids must be passed during
        # warmup so torch.compile traces the same code path as runtime.
        # Shape must match runtime: [num_reqs, 1] for prefill, [num_reqs, num_spec+1] for decode.
        if num_tokens == num_reqs * (1 + self.num_speculative_tokens):
            raw_sampled_cols = self.num_speculative_tokens + 1
        else:
            raw_sampled_cols = 1
        raw_sampled_token_ids = torch.ones(
            num_reqs, raw_sampled_cols, dtype=torch.int32
        ).to(self.device)

        return dict(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            last_token_indices=last_token_indices,
            raw_sampled_token_ids=raw_sampled_token_ids,
        )

    def warmup(
        self,
        num_tokens: int,
        num_reqs: int,
        attn_metadata: AttentionMetadata,
    ) -> None:
        """
        Warm up the draft model by calling propose() with synthetic inputs.

        Uses safe position values (zeros) to avoid KV cache overflow
        during recurrent passes.

        Args:
            num_tokens: Total number of tokens
            num_reqs: Number of requests in the batch
            attn_metadata: Full attention metadata
        """
        assert self.model is not None

        logger.info("EAGLE3 warmup: num_tokens=%d, num_reqs=%d", num_tokens, num_reqs)

        kwargs = self._build_synthetic_inputs(num_tokens, num_reqs)

        _ = self.propose(
            attn_metadata=attn_metadata,
            is_warmup=True,
            **kwargs,
        )

    def graph_extract(
        self,
        num_tokens: int,
        num_reqs: int,
        attn_metadata: AttentionMetadata,
    ) -> None:
        """
        Extract the draft model's HLO graph using the
        ``vllm_neuron_graph_capture`` backend. Mirrors
        ``NeuronModelRunner.extract_prefill_graphs`` and swallows the
        ``CaptureComplete`` exception raised by the capture backend after
        a successful trace.

        Args:
            num_tokens: Total number of tokens
            num_reqs: Number of requests in the batch
            attn_metadata: Full attention metadata
        """
        assert self.capture_backend_model is not None

        logger.info(
            "EAGLE3 graph capture: num_tokens=%d, num_reqs=%d", num_tokens, num_reqs
        )

        kwargs = self._build_synthetic_inputs(num_tokens, num_reqs)

        try:
            _ = self.propose(
                attn_metadata=attn_metadata,
                is_warmup=True,
                model_override=self.capture_backend_model,
                **kwargs,
            )
        except CaptureComplete:
            logger.debug(
                "Graph capture for draft model completed: num_tokens=%d, num_reqs=%d",
                num_tokens,
                num_reqs,
            )

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Validate that all eagle layers belong to the same KVCacheGroup.
        Need this assumption to ensure all eagle layers can use the same AttentionMetadata.
        TODO May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len({kv_cache_groups[layer_name] for layer_name in self.attn_layer_names})
            == 1
        ), "All eagle layers should belong to the same kv cache group"

    # TODO this should be refactored to a centralized model loader
    def compile_and_load_draft_model(
        self,
        start_layer_idx: int,
        target_hidden_size: int,
    ) -> nn.Module:
        # TODO registry should examine and make sure EAGLE3 is prepended.
        draft_model_arch = self.draft_model_config.architecture
        if self.method == "eagle3" and not draft_model_arch.startswith("Eagle3"):
            draft_model_arch = f"Eagle3{draft_model_arch}"

        # Use vLLM Neuron's model registry to get the vLLM Neuron-specific model implementation
        vllm_neuron_models = dict(get_models())
        if draft_model_arch not in vllm_neuron_models:
            raise ValueError(
                f"Draft model architecture '{draft_model_arch}' not found in vLLM Neuron model registry. "
                f"Supported architectures: {list(vllm_neuron_models.keys())}"
            )
        model_cls = vllm_neuron_models[draft_model_arch]

        draft_hf_config = self.draft_model_config.hf_config

        # Set padding config if target has different (padded) hidden_size
        # This allows draft model to work with padded hidden states from target model
        draft_hf_config.unpadded_hidden_size = draft_hf_config.hidden_size
        if target_hidden_size != draft_hf_config.hidden_size:
            draft_hf_config.hidden_size = target_hidden_size

        # Override to greedy sampling for draft model.
        on_device_sampling_config = None
        if self.on_device_sampling:
            on_device_sampling_config = OnDeviceSamplingConfig(all_greedy=True)

        neuron_config = NeuronConfig(
            on_device_sampling_config=on_device_sampling_config,
        )

        cpu_compile = envs.VLLM_NEURON_CPU_COMPILE

        with torch.device("meta"):
            self.model = model_cls.from_configs(
                config=draft_hf_config,
                start_layer_idx=start_layer_idx,
                neuron_config=neuron_config,
            )
        self.model.num_speculative_tokens = self.num_speculative_tokens
        load_start = time.perf_counter()
        if not cpu_compile:
            self.model.load_weights(
                self.speculative_config.model,
                self.device,
                self.vllm_config.load_config.download_dir,
            )

            logger.info("Moving draft model to device: %s", self.device)
            self.model = self.model.to(self.device)

            draft_model_name = self.speculative_config.model
            MODEL_LOAD_TIME.labels(model_name=draft_model_name).set(
                time.perf_counter() - load_start
            )
            MODEL_LOAD_SIZE.labels(model_name=draft_model_name).set(
                sum(p.nbytes for p in self.model.parameters())
            )
        else:
            if hasattr(self.model, "load_weights_lite"):
                self.model.load_weights_lite(
                    self.speculative_config.model,
                    torch.device("cpu"),  # cpu needed getting compile time constants
                    self.vllm_config.load_config.download_dir,
                )
                logger.info(
                    "Light weight loading complete for draft model CPU compilation."
                )
            logger.info(
                "CPU Compilation is enabled. Skipping full weight loading for draft model."
            )
            # Explicit call to force all buffers to use meta.
            self.model = self.model.to("meta")

        # Check for debug mode environment variable
        # Set VLLM_NEURON_DEBUG_MODE=1 to disable fullgraph for debugging (allows print statements)
        debug_mode = envs.VLLM_NEURON_DEBUG_MODE
        fullgraph_enabled = not debug_mode

        logger.info("Compiling draft model with vllm_neuron backend")
        compile_backend = envs.get_compile_backend_name()

        cpu_mode = envs.VLLM_NEURON_CPU_MODE
        eager_mode = self.vllm_config.model_config.enforce_eager
        skip_graph_capture_backend = envs.VLLM_NEURON_DISABLE_GRAPH_CAPTURE_BACKEND
        if eager_mode or cpu_mode or debug_mode or skip_graph_capture_backend:
            logger.debug(
                "Draft model graph capture backend disabled "
                "(eager/cpu/debug/skip_graph_capture_backend)."
            )
            self.capture_backend_model = None
        else:
            self.capture_backend_model = torch.compile(
                self.model,
                backend="vllm_neuron_graph_capture",
                fullgraph=fullgraph_enabled,
            )

        return torch.compile(
            self.model, backend=compile_backend, fullgraph=fullgraph_enabled
        )

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        last_token_indices: torch.Tensor,
        attn_metadata: AttentionMetadata,
        raw_sampled_token_ids: torch.Tensor,
        is_warmup: bool = False,
        model_override: nn.Module | None = None,
    ) -> torch.Tensor:
        assert self.model is not None
        model = model_override if model_override is not None else self.model

        batch_size = last_token_indices.shape[0]
        num_tokens = target_token_ids.shape[0]
        if num_tokens == 0:
            return torch.empty(
                (batch_size, self.num_speculative_tokens),
                device=target_token_ids.device,
                dtype=torch.int32,
            )

        # Shift input ids by one token.
        # next_token patching is handled on-device by _extract_accepted_tokens.
        if num_tokens > 1:
            shifted = target_token_ids.narrow(0, 1, num_tokens - 1)
            last_token = target_token_ids.narrow(0, num_tokens - 1, 1)
            input_ids = torch.cat([shifted, last_token])
        else:
            input_ids = target_token_ids.clone()

        input_ids = input_ids.to(self.device)
        target_positions = target_positions.to(self.device)
        last_token_indices = last_token_indices.to(self.device)
        raw_sampled_token_ids = raw_sampled_token_ids.to(self.device)

        # Filter attn_metadata to draft model layers only.
        first_attn_metadata = {ln: attn_metadata[ln] for ln in self.attn_layer_names}

        # Single fused forward: KV cache update + unrolled recurrent decode loop.
        draft_model_name = self.speculative_config.model
        model_forward_start = time.perf_counter()
        with (
            contextlib.nullcontext()
            if is_warmup
            else model_forward_context(self.vllm_config)
        ):
            draft_token_ids, _, _ = model(
                input_ids=input_ids,
                positions=target_positions,
                initial_target_hidden_states=target_hidden_states,
                attn_metadata=first_attn_metadata,
                sampling_positions=last_token_indices,
                rank=self.rank_tensor,
                raw_sampled_token_ids=raw_sampled_token_ids,
            )
        model_forward_elapsed = time.perf_counter() - model_forward_start
        bucket_name = f"draft_fused_s{num_tokens}"
        if is_warmup:
            COMPILATION_TIME.labels(
                model_name=draft_model_name, bucket_name=bucket_name
            ).set(model_forward_elapsed)
        else:
            NEFF_EXECUTION_COUNT.labels(
                model_name=draft_model_name, bucket_name=bucket_name
            ).inc()

        return draft_token_ids
