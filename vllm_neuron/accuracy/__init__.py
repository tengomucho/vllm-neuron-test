# SPDX-License-Identifier: Apache-2.0
"""Accuracy validation toolkit for vLLM Neuron.

Public API organized by domain:

**Logit validation** (``vllm_neuron.accuracy.logit_validation``):
    E2E logit comparison between vLLM Neuron and HuggingFace reference.

    - :func:`logit_validation` — single-prompt logit comparison
    - :func:`multi_prompt_logit_validation` — multi-prompt with aggregation

**Tensor capture** (``vllm_neuron.accuracy.tensor_capture``):
    Instrument model forward passes to capture intermediate tensors.

    - :class:`TensorCaptureModel` — wraps a model for tensor capture
    - :func:`capture_tensor` — decorator to mark tensors for capture

**Tensor comparison** (``vllm_neuron.accuracy.tensor_compare``):
    Compare captured tensors between implementations.

    - :func:`compare_tensors` — two-way tensor comparison
    - :func:`compare_tensors_three_way` — three-way tensor comparison
    - :func:`compare_capture_dirs` — compare full capture directories

**KV cache analysis** (``vllm_neuron.accuracy.kv_cache_analysis``):
    Extract and compare KV caches between HF and vLLM.

    - :func:`compare_kv_caches` — compare KV caches with per-head metrics
    - :func:`extract_hf_kv_caches` — extract KV caches from HF model
    - :func:`extract_vllm_kv_caches` — extract KV caches from vLLM

**Tensor I/O** (``vllm_neuron.accuracy.tensor_io``):
    Read/write captured tensors to disk.

    - :func:`tensor_io_read` — read captured forward pass from disk
    - :func:`tensor_io_write` — write captured forward pass to disk

**Visualization** (``vllm_neuron.accuracy.plotting``,
``vllm_neuron.accuracy.logit_visualization``,
``vllm_neuron.accuracy.kv_cache_visualize``):
    Plotting and HTML report generation for debugging accuracy issues.

**Testing assertions** (``vllm_neuron.accuracy.testing``):
    Two-way and three-way numerical comparison for module-level tests.

    - :func:`assert_close` — assert target ≈ expected (two-way)
    - :func:`assert_close_three_way` — assert target ≈ expected with
      baseline precision floor (three-way)
"""

# ── Logit validation ─────────────────────────────────────────────────────
from vllm_neuron.accuracy.logit_validation import (
    logit_validation,
    multi_prompt_logit_validation,
)

# ── Tensor capture ───────────────────────────────────────────────────────
from vllm_neuron.accuracy.tensor_capture import (
    TensorCaptureModel,
    capture_tensor,
    expand_patterns,
)

# ── Tensor comparison ────────────────────────────────────────────────────
from vllm_neuron.accuracy.tensor_compare import (
    compare_tensors,
    compare_tensors_three_way,
    compare_capture_dirs,
    compare_capture_dirs_three_way,
    print_comparison_report,
    print_three_way_report,
    compute_aggregate_metrics,
    align_shapes,
)

# ── KV cache analysis ───────────────────────────────────────────────────
from vllm_neuron.accuracy.kv_cache_analysis import (
    compare_kv_caches,
    extract_hf_kv_caches,
    extract_hf_kv_caches_teacher_forced,
    extract_vllm_kv_caches,
    extract_vllm_kv_cache_config,
    extract_vllm_block_tables,
    cleanup_kv_snapshot,
    reconstruct_contiguous_kv,
    reconstruct_contiguous_kv_batch,
    aggregate_kv_bc_across_prompts,
    compute_combined_bc_per_layer,
    generate_hf_logits_and_kv,
    print_kv_report,
)

# ── Tensor I/O ───────────────────────────────────────────────────────────
from vllm_neuron.accuracy.tensor_io import (
    read as tensor_io_read,
    write as tensor_io_write,
)

# ── Tensor replacement ───────────────────────────────────────────────────
from vllm_neuron.accuracy.tensor_replacement import (
    TensorReplacer,
    set_active_context,
    get_replacement_tensor,
)

# ── Visualization ────────────────────────────────────────────────────────
from vllm_neuron.accuracy.plotting import (
    plot_error_distributions,
    plot_error_qqplot,
    plot_scatter,
    plot_three_way,
)

# ── Testing assertions ───────────────────────────────────────────────────
from vllm_neuron.accuracy.testing import (
    assert_close,
    assert_close_three_way,
)

__all__ = [
    # Logit validation
    "logit_validation",
    "multi_prompt_logit_validation",
    # Tensor capture
    "TensorCaptureModel",
    "capture_tensor",
    "expand_patterns",
    # Tensor comparison
    "compare_tensors",
    "compare_tensors_three_way",
    "compare_capture_dirs",
    "compare_capture_dirs_three_way",
    "print_comparison_report",
    "print_three_way_report",
    "compute_aggregate_metrics",
    "align_shapes",
    # KV cache analysis
    "compare_kv_caches",
    "extract_hf_kv_caches",
    "extract_hf_kv_caches_teacher_forced",
    "extract_vllm_kv_caches",
    "extract_vllm_kv_cache_config",
    "extract_vllm_block_tables",
    "cleanup_kv_snapshot",
    "reconstruct_contiguous_kv",
    "reconstruct_contiguous_kv_batch",
    "aggregate_kv_bc_across_prompts",
    "compute_combined_bc_per_layer",
    "generate_hf_logits_and_kv",
    "print_kv_report",
    # Tensor I/O
    "tensor_io_read",
    "tensor_io_write",
    # Tensor replacement
    "TensorReplacer",
    "set_active_context",
    "get_replacement_tensor",
    # Visualization
    "plot_error_distributions",
    "plot_error_qqplot",
    "plot_scatter",
    "plot_three_way",
    # Testing assertions
    "assert_close",
    "assert_close_three_way",
]
