# SPDX-License-Identifier: Apache-2.0
import math
from dataclasses import dataclass, field
from typing import Any, List


# TODO: expand support of device sampling configs
class OnDeviceSamplingConfig:
    def __init__(self, **kwargs: Any):
        all_greedy_val = kwargs.pop("all_greedy", False)
        # Handle string "true"/"false" from config
        if isinstance(all_greedy_val, str):
            self.all_greedy = all_greedy_val.lower() in ("true", "1", "yes")
        else:
            self.all_greedy = bool(all_greedy_val)

        self.max_top_k = int(kwargs.pop("max_top_k", 256))

        deterministic_val = kwargs.pop("deterministic", False)
        if isinstance(deterministic_val, str):
            self.deterministic = deterministic_val.lower() in ("true", "1", "yes")
        else:
            self.deterministic = bool(deterministic_val)

        # Data-parallel sampling degree (1 = disabled, >1 = batch sharding)
        self.sampling_dp_degree = int(kwargs.pop("sampling_dp_degree", 1))


@dataclass
class TensorCaptureConfig:
    """Configuration for tensor capture debugging.

    Args:
        modules: List of module paths for hook-based capture
            (e.g., ["model.layers.0", "lm_head"])
        capture_dir: Directory to save captured tensors
        capture_filter: Optional list of tensor names for write-time
            filtering. When set, only captured tensors whose names match
            this list are written to disk; all others are silently
            skipped.
    """

    modules: List[str] = field(default_factory=list)
    capture_dir: str = "/tmp/vllm_neuron_tensor_captures"
    capture_filter: List[str] | None = None


@dataclass
class TensorReplacementConfig:
    """Configuration for tensor replacement.

    Args:
        tensors: Per-prompt reference tensors as a list of
            {module_name: [step_tensors]} dicts, where list index
            is the prompt index, step 0 is prefill, steps 1+ are decode.
        tensors_path: Path to a torch.save'd file containing
            tensors. Used for online serving where tensors cannot
            be passed in-memory via JSON config.
        prompt_token_ids: Per-prompt input token IDs used to match
            incoming requests to the correct prompt captures via
            prefix hashing. Required for multi-prompt validation.
        prompt_token_ids_path: Path to a torch.save'd file containing
            prompt_token_ids. Used for online serving.
    """

    tensors: list = field(default_factory=list, repr=False)
    tensors_path: str | None = None
    prompt_token_ids: list | None = field(default=None, repr=False)
    prompt_token_ids_path: str | None = None

    def resolve(self) -> tuple[list, list | None]:
        """Validate config and load tensors/prompt_token_ids from paths if needed."""
        import torch

        tensors = self.tensors
        if tensors and self.tensors_path:
            raise ValueError(
                "tensor_replacement: specify either 'tensors' or "
                "'tensors_path', not both."
            )
        if not tensors and self.tensors_path:
            tensors = torch.load(self.tensors_path, weights_only=False)
        if not tensors:
            raise ValueError(
                "tensor_replacement: must specify either 'tensors' or "
                "'tensors_path' with valid data."
            )

        prompt_token_ids = self.prompt_token_ids
        if not prompt_token_ids and self.prompt_token_ids_path:
            prompt_token_ids = torch.load(
                self.prompt_token_ids_path, weights_only=False
            )

        return tensors, prompt_token_ids

    def __repr__(self):
        if self.tensors_path:
            return f"TensorReplacementConfig(path={self.tensors_path!r})"
        n_prompts = len(self.tensors)
        n_modules = len(self.tensors[0]) if n_prompts > 0 else 0
        return f"TensorReplacementConfig(prompts={n_prompts}, modules={n_modules})"


@dataclass
class NeuronConfig:
    # TODO: Revisit sharding config consistency between vLLM Neuron and vLLM.
    # In the case of EP, we define an ep_degree argument in NeuronConfig because vLLM doesn't
    # support configuring EP degree.
    ep_degree: int = 1  # Expert parallelism degree
    attention_dp_size: int = (
        1  # KV data parallelism size (decode-only Q/O sharding across DP)
    )
    embedding_dp_size: int = 1  # Shard Embedding across TP * this many DP ranks
    lm_head_dp_size: int = 1  # Shard LM Head across TP * this many DP ranks
    mlp_dp_size: int = 1  # Shard dense MLP across TP * this many DP ranks
    apply_prefill_dcp: bool = (
        False  # Shard attention across DCP sub-group (prefill DI server)
    )
    on_device_sampling_config: OnDeviceSamplingConfig | None = field(
        default_factory=OnDeviceSamplingConfig
    )
    # TODO: Remove max_logprobs from NeuronConfig; use vllm_config directly in the model instead
    max_logprobs: int = 0  # 0 = logprobs disabled; -1 = gather all logits for logprobs;
    prefill_sequence_buckets: list[int] | None = None
    decode_batch_buckets: list[int] | None = None
    tensor_capture: TensorCaptureConfig | None = None
    tensor_replacement: TensorReplacementConfig | None = None
    kv_segment_size_buckets: list[int] | None = None
    # Alternative to max_logprobs for test validation. Avoids the serialization
    # inefficiency of vLLM's logprobs pipeline by writing raw logits directly to
    # /dev/shm/{debug_logits_dir}/ with position metadata.
    # Used by e2e logit tests to compare Neuron outputs against golden baselines.
    debug_logits_dir: str | None = None
    # Model quantization type (e.g., "bf16", "mxfp4")
    quantization: str | None = None
    # Bucket configurations for scheduling
    num_batched_tokens_buckets: list[int] | None = None
    num_seqs_buckets: list[int] | None = None
    # Decode context-length bucketing. When set, decode is compiled per
    # (batch_bucket, ctx_bucket) pair plus a max_model_len fallback NEFF.
    # See doc/vllm_neuron/source/design/vllm/decode-context-length-bucketing.rst.
    decode_context_length_buckets: list[int] | None = None

    @classmethod
    def from_dict(cls, config_dict: dict) -> "NeuronConfig":
        """Build NeuronConfig from a dictionary (e.g., from additional_config['neuron_config']).

        Args:
            config_dict: Dictionary with neuron configuration options.

        Returns:
            NeuronConfig instance with parsed values.
        """
        # Parse on_device_sampling_config
        on_device_sampling_dict = config_dict.get("on_device_sampling_config")
        if on_device_sampling_dict is not None:
            on_device_sampling_config = OnDeviceSamplingConfig(
                **on_device_sampling_dict
            )
        elif "on_device_sampling_config" in config_dict:
            # Explicitly set to None → disable ODS
            on_device_sampling_config = None
        else:
            # Key absent → use default (ODS enabled)
            on_device_sampling_config = OnDeviceSamplingConfig()

        # Parse tensor_capture config
        tensor_capture = None
        tensor_capture_dict = config_dict.get("tensor_capture")
        if tensor_capture_dict is not None:
            tensor_capture = TensorCaptureConfig(
                modules=tensor_capture_dict.get("modules", []),
                capture_dir=tensor_capture_dict.get(
                    "capture_dir", "/tmp/vllm_neuron_tensor_captures"
                ),
                capture_filter=tensor_capture_dict.get("capture_filter"),
            )

        # Parse tensor_replacement config
        tensor_replacement = None
        tensor_replacement_value = config_dict.get("tensor_replacement")
        if tensor_replacement_value is not None:
            if isinstance(tensor_replacement_value, TensorReplacementConfig):
                tensor_replacement = tensor_replacement_value
            else:
                tensor_replacement = TensorReplacementConfig(
                    tensors=tensor_replacement_value.get("tensors", []),
                    tensors_path=tensor_replacement_value.get("tensors_path"),
                    prompt_token_ids=tensor_replacement_value.get("prompt_token_ids"),
                    prompt_token_ids_path=tensor_replacement_value.get(
                        "prompt_token_ids_path"
                    ),
                )

        return cls(
            ep_degree=config_dict.get("ep_degree", 1),
            attention_dp_size=config_dict.get("attention_dp_size", 1),
            embedding_dp_size=config_dict.get("embedding_dp_size", 1),
            lm_head_dp_size=config_dict.get("lm_head_dp_size", 1),
            mlp_dp_size=config_dict.get("mlp_dp_size", 1),
            apply_prefill_dcp=config_dict.get("apply_prefill_dcp", False),
            on_device_sampling_config=on_device_sampling_config,
            max_logprobs=config_dict.get("max_logprobs", 0),
            prefill_sequence_buckets=config_dict.get("prefill_sequence_buckets"),
            decode_batch_buckets=config_dict.get("decode_batch_buckets"),
            tensor_capture=tensor_capture,
            tensor_replacement=tensor_replacement,
            kv_segment_size_buckets=config_dict.get("kv_segment_size_buckets"),
            debug_logits_dir=config_dict.get("debug_logits_dir"),
            quantization=config_dict.get("quantization"),
            num_batched_tokens_buckets=config_dict.get("num_batched_tokens_buckets"),
            num_seqs_buckets=config_dict.get("num_seqs_buckets"),
            decode_context_length_buckets=config_dict.get(
                "decode_context_length_buckets"
            ),
        )

    def __post_init__(self):
        """Validate configuration."""
        if not isinstance(self.ep_degree, int) or self.ep_degree < 1:
            raise ValueError(
                f"ep_degree must be a positive integer, got {self.ep_degree}"
            )
        if not isinstance(self.attention_dp_size, int) or self.attention_dp_size < 1:
            raise ValueError(
                f"attention_dp_size must be a positive integer, got {self.attention_dp_size}"
            )
        if not isinstance(self.embedding_dp_size, int) or self.embedding_dp_size < 1:
            raise ValueError(
                f"embedding_dp_size must be a positive integer, got {self.embedding_dp_size}"
            )
        if not isinstance(self.lm_head_dp_size, int) or self.lm_head_dp_size < 1:
            raise ValueError(
                f"lm_head_dp_size must be a positive integer, got {self.lm_head_dp_size}"
            )
        if not isinstance(self.mlp_dp_size, int) or self.mlp_dp_size < 1:
            raise ValueError(
                f"mlp_dp_size must be a positive integer, got {self.mlp_dp_size}"
            )


@dataclass
class VisionNeuronConfig:
    """Neuron-specific configuration for vision encoders.

    Args:
        num_vision_tokens_buckets: Bucket sizes for vision token sequences.
            Each bucket defines a compiled graph for that token count.
        vision_attention_block_size: Block size for chunked vision attention.
            Vision sequences are split into blocks of this size for attention.
        tp_size: Tensor parallelism degree for the vision encoder.
            -1 means "same as text model TP" (default).
            When set explicitly, the vision encoder uses its own TP/DP split
            independent of the text model. dp_size is derived as
            world_size / tp_size.
    """

    num_vision_tokens_buckets: list[int] | None = None
    vision_attention_block_size: int = 2048
    tp_size: int = -1  # -1 = same as text model TP; >0 = explicit vision TP degree

    # Derived fields — set in __post_init__
    num_total_vision_attention_blocks: list[int] | None = None

    @property
    def dp_size(self) -> int:
        """Derive vision DP size from world_size and tp_size.

        Returns -1 if tp_size is not explicitly set (defers to runtime).
        At runtime, dp_size = world_size / tp_size.
        """
        return -1 if self.tp_size == -1 else -1  # resolved at runtime

    def resolve_tp_dp(self, world_size: int, text_tp_size: int) -> tuple[int, int]:
        """Resolve the actual vision TP and DP sizes at runtime.

        Args:
            world_size: Total number of ranks.
            text_tp_size: The text model's TP size (used when tp_size == -1).

        Returns:
            (vision_tp_size, vision_dp_size) tuple.

        Raises:
            ValueError: If world_size is not divisible by vision tp_size.
        """
        effective_tp = text_tp_size if self.tp_size == -1 else self.tp_size
        if world_size % effective_tp != 0:
            raise ValueError(
                f"world_size ({world_size}) must be divisible by "
                f"vision tp_size ({effective_tp})"
            )
        effective_dp = world_size // effective_tp
        return effective_tp, effective_dp

    def __post_init__(self):
        """Derive num_total_vision_attention_blocks from buckets and block size."""
        if self.num_vision_tokens_buckets is not None:
            self.num_total_vision_attention_blocks = [
                math.ceil(bucket / self.vision_attention_block_size)
                for bucket in self.num_vision_tokens_buckets
            ]
        if self.tp_size != -1 and self.tp_size < 1:
            raise ValueError(
                f"vision tp_size must be -1 (auto) or a positive integer, got {self.tp_size}"
            )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "VisionNeuronConfig":
        """Build VisionNeuronConfig from a dictionary.

        Args:
            config_dict: Dictionary with vision neuron configuration options.

        Returns:
            VisionNeuronConfig instance with parsed values.
        """
        return cls(
            num_vision_tokens_buckets=config_dict.get("num_vision_tokens_buckets"),
            vision_attention_block_size=config_dict.get(
                "vision_attention_block_size", 2048
            ),
            tp_size=config_dict.get("tp_size", -1),
        )
