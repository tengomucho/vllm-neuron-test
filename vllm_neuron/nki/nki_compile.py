# SPDX-License-Identifier: Apache-2.0
"""NKI kernel compilation bridge.

Provides compile_nki() which compiles NKI kernels via CompileKernel
and returns the fields the HOP infrastructure needs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from nki.language.buffers import shared_hbm
from nki.language.tensor import NkiTensor
from nki.framework.compiled import CompileKernel

from ..compile.platform import get_platform_target
from .nki_dtype import nki_dtype_to_torch, torch_to_nki_dtype


@dataclass
class NKICompileResult:
    """Compilation output consumed by the HOP dispatch implementations."""

    dumped_config: str
    return_types: tuple[tuple[torch.dtype, tuple[int, ...]], ...]
    operand_output_aliases: dict[int, int]


def compile_nki(
    func: Callable,
    args: dict[str, Any],
    grid: tuple[int, ...],
) -> NKICompileResult:
    """Compile an NKI kernel via CompileKernel.

    Args:
        func: Raw NKI kernel function.
        args: Parameter name → value (torch.Tensor or scalar).
        grid: LNC grid, e.g. (2,).

    Returns:
        NKICompileResult with everything the HOP needs.
    """
    lnc = grid[0] if grid else 1
    # Explicit target avoids neuron-ls subprocess in _detect_target() (NKI-1976)
    kernel = CompileKernel(func=func, lnc=lnc, target=get_platform_target())
    inputs = {name: _convert_input(v, name) for name, v in args.items()}

    compile_opts = kernel._compile_opts()
    nir = kernel._cached_compile_to_bir(
        frontend=kernel._frontend_cls(
            enable_backend_opt=kernel._enable_backend_opt,
        ),
        inputs=inputs,
        compile_opts=compile_opts,
    )

    config = nir.build_config()

    return NKICompileResult(
        dumped_config=config.backend_config_b64.decode("ascii"),
        return_types=tuple(
            (nki_dtype_to_torch(s.dtype), tuple(s.shape)) for s in config.output_specs
        ),
        operand_output_aliases=config.operand_output_aliases,
    )


def _convert_input(x: Any, name: str) -> Any:
    if isinstance(x, torch.Tensor):
        # convert to NkiTensor
        return NkiTensor(
            name=name,
            shape=x.shape,
            dtype=torch_to_nki_dtype(x.dtype),
            storage=None,
            buffer=shared_hbm,
        )

    return x
