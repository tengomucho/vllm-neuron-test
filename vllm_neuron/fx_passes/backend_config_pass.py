# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Dict, Tuple

from .base import FXPass

from vllm_neuron.nki.nki_hop import (
    NKIKernelWrapper,
    _get_kernel_config,
)


class NkiKernelWriteBackendConfigPass(FXPass):
    """Transforms NKI kernel calls in FX graph to include the backend config string,
    which is used to detect changes to the kernel call for the purposes of caching.

    Example transformation:
        Before: call_function[target=torch.ops.higher_order.nki_kernel_wrapper](args = (), kwargs = {kernel_idx: 0, grid: [], backend_config: 0
        After:  call_function[target=torch.ops.higher_order.nki_kernel_wrapper](args = (), kwargs = {kernel_idx: 0, grid: [], backend_config: eyJrZXJuZWxfd...

    The backend config is unique to a traced kernel and the compile-time arguments it receives
    """

    def __init__(self):
        pass

    @property
    def name(self) -> str:
        """Return the pass name."""
        return "nki_kernel_backend_config_pass"

    def run(
        self, gm: torch.fx.GraphModule, **kwargs
    ) -> Tuple[torch.fx.GraphModule, Dict]:
        """Execute the NkiKernelWriteBackendConfigPass.

        Args:
            gm: The PyTorch FX GraphModule to transform
            **kwargs: Additional arguments (target_device, etc.)

        Returns:
            Tuple containing the transformed GraphModule and metadata

        Raises:
            RuntimeError: If the transformation fails
        """
        try:
            kernel_call_count = 0
            for node in gm.graph.nodes:
                if node.op != "call_function":
                    continue

                if type(node.target) != NKIKernelWrapper:
                    continue

                node_kwargs = dict(node.kwargs.copy())
                kernel_idx = node_kwargs["kernel_idx"]
                grid = node_kwargs["grid"]
                args = tuple(
                    [
                        arg_node.meta["example_value"]
                        if isinstance(arg_node, torch.fx.Node)
                        else arg_node
                        for arg_node in node_kwargs["args"]
                    ]
                )
                arg_names = node_kwargs["arg_names"]
                constant_args_key = node_kwargs["constant_args_key"]
                result = _get_kernel_config(
                    kernel_idx, grid, args, arg_names, constant_args_key
                )

                node_kwargs["backend_config"] = result.dumped_config
                # operand_output_aliases from NKI uses tensor-only input indices (removes
                # scalar/None args). We need to remap those to use the index within
                # the full arg list
                tensor_positions = [
                    i
                    for i, _arg in enumerate(node_kwargs["args"])
                    if isinstance(_arg, torch.fx.Node)
                ]
                remapped_aliases = {
                    tensor_positions[input_idx]: output_idx
                    for input_idx, output_idx in result.operand_output_aliases.items()
                }
                node_kwargs["operand_output_aliases"] = remapped_aliases

                node.kwargs = node_kwargs
                kernel_call_count += 1

            # Recompile and return
            gm.recompile()
            metadata = {"kernel_call_count": kernel_call_count}

            return gm, metadata

        except Exception as e:
            raise RuntimeError(
                f"Writing backend config to NKI kernel calls failed: {str(e)}"
            ) from e
