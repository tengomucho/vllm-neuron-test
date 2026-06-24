# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger

from vllm_neuron import envs
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)

import vllm_neuron.functional as NF
from vllm_neuron.parallel.neuron_parallel_state import (
    get_neuron_ep_group,
)

logger = init_logger(__name__)


class NeuronAll2AllManager(All2AllManagerBase):
    """
    All2All communication based on NKI EP kernels, for 2D Torus and NeuronSwitch topologies.

    2D Torus: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn2-arch.html
    NeuronSwitch: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trn3-arch.html
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

        self.is_neuron_switch = envs.VLLM_NEURON_SWITCH_CC
        self._dispatch_func = None
        self._combine_func = (
            self._combine_switch if self.is_neuron_switch else self._combine_torus
        )

        self.all2all_group = get_neuron_ep_group()

        self.dispatch_recv_counts = None
        self.num_tokens = None
        self.experts_per_token = None

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        """Neuron implementation of EP dispatch with All2All-v."""
        raise NotImplementedError(
            "NeuronAll2AllManager does not yet support dispatch()"
        )

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False):
        """Neuron implementation of EP combine with All2All-v."""

        assert is_sequence_parallel, (
            f"NeuronAll2AllManager.combine() requires is_sequence_parallel=True but got {is_sequence_parallel=}"
        )

        combine_metadata = NF.build_all2all_combine_metadata(
            send_counts=self.dispatch_recv_counts,
            recv_displs=None,
        )
        combine_recv, _ = self._combine_func(hidden_states, combine_metadata)
        output_reduced = NF.topk_reduce(
            combine_recv,
            self.num_tokens,
            self.experts_per_token,
            is_sequence_parallel=is_sequence_parallel,
        )
        return output_reduced

    def _combine_torus(self, data, metadata):
        raise NotImplementedError(
            "All2All communication not yet supported for 2D Torus topology."
        )

    def _combine_switch(self, data, metadata):
        # FIXME[NKI-1927]: This is a hack to get around the constraint that I/O shapes must match
        output = torch.zeros_like(data)
        # FIXME[P432515522]: Initialize final 2 columns of output to -1 in int32 with hex, since FX/HLO/compiler do not support .view()
        output[:, -2:] = 0xFFFFFFFF
        output_recv, output_metadata = NF.all_to_all_v(
            input=data,
            output=output,
            group=self.all2all_group,
            metadata=metadata,
            recv_counts_known=True,
            has_rdispls=False,
        )
        return output_recv, output_metadata
