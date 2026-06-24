# SPDX-License-Identifier: Apache-2.0
"""
NeuronNixlConnector — standalone NIXL connector for Neuron with DCP support.

Replaces the monkey-patch approach. Users specify this connector directly:

    --kv-transfer-config '{
        "kv_connector": "NeuronNixlConnector",
        "kv_connector_module_path": "vllm_neuron.vllm.kv_connector.neuron_nixl_connector",
        ...
    }'

Supports all DCP DI topologies:
  - DCP prefill → TP decode
  - DCP prefill → DCP decode (same or different DCP degrees)
  - TP prefill → DCP decode
  - Standard TP → TP (passthrough)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
    NixlAgentMetadata,
    NixlConnector,
    NixlConnectorWorker,
    NixlHandshakePayload,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorRole,
    )
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


@dataclass
class NeuronNixlAgentMetadata(NixlAgentMetadata):
    """Extended metadata with Neuron DCP info.

    The prefill advertises its DCP degree so the decode can compute
    correct remote rank mappings without user config.
    """

    dcp_size: int = 0


class NeuronNixlConnectorWorker(NixlConnectorWorker):
    """NIXL connector worker with DCP-aware KV transfer.

    Handles:
    - DCP prefill: block interleaving across multiple remote cp_ranks
    - DCP decode: filtering incoming blocks to local dcp_rank's subset
    - Head splitting/merging when P and D have different heads-per-block
    - NeuronNixlAgentMetadata for prefill DCP advertisement
    """

    def __init__(self, *args, **kwargs):
        # The upstream decoder uses the module-global NixlAgentMetadata class
        # for deserialization. We swap it for our extended version so the
        # dcp_size field is decoded from the prefill's handshake payload.
        import vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector as _nixl_mod

        if _nixl_mod.NixlAgentMetadata is not NeuronNixlAgentMetadata:
            _nixl_mod.NixlAgentMetadata = NeuronNixlAgentMetadata

        super().__init__(*args, **kwargs)
        self._dcp_prefill_engines: set = set()
        self._dcp_prefill_head_ratio: int = 1
        self._dcp_prefill_inverse_head_ratio: int = 1
        self._local_dcp_size: int = 1
        self._local_dcp_rank: int = 0
        self._remote_prefill_dcp: int = 1
        self._inverse_handle_cache: dict[str, list] = {}

    # ── DCP Detection ────────────────────────────────────────────────────

    def _detect_dcp_prefill(self, nixl_agent_meta, remote_tp_size: int):
        """Detect DCP prefill from TP mismatch with compatible block_lens."""
        eid = nixl_agent_meta.engine_id
        if eid in self._dcp_prefill_engines:
            return
        local_tp = self.kv_topo.tp_size
        if local_tp == remote_tp_size:
            return
        if not hasattr(self, "block_len_per_layer") or not self.block_len_per_layer:
            return
        remote_block_len = nixl_agent_meta.block_lens[0]
        local_block_len = self.block_len_per_layer[0]
        larger = max(remote_block_len, local_block_len)
        smaller = min(remote_block_len, local_block_len)
        if smaller <= 0 or larger % smaller != 0:
            return

        head_ratio = max(1, remote_block_len // local_block_len)
        inverse_head_ratio = max(1, local_block_len // remote_block_len)

        self._remote_prefill_dcp = nixl_agent_meta.dcp_size
        self._dcp_prefill_engines.add(eid)
        self._dcp_prefill_head_ratio = head_ratio
        self._dcp_prefill_inverse_head_ratio = inverse_head_ratio

        logger.info(
            "DCP NIXL: detected DCP prefill on %s "
            "(remote_block_len=%d, local_block_len=%d, head_ratio=%d, "
            "inverse_head_ratio=%d, P_TP=%d, D_TP=%d, prefill_dcp=%d)",
            eid,
            remote_block_len,
            local_block_len,
            head_ratio,
            inverse_head_ratio,
            remote_tp_size,
            local_tp,
            self._remote_prefill_dcp,
        )

    def _is_dcp_prefill_engine(self, engine_id: str) -> bool:
        return engine_id in self._dcp_prefill_engines

    # ── Override: register_kv_caches ─────────────────────────────────────

    def register_kv_caches(self, kv_caches):
        import msgspec

        super().register_kv_caches(kv_caches)
        neuron_cfg = self.vllm_config.additional_config.get("neuron_config", {})

        if neuron_cfg.get("apply_prefill_dcp", False):
            # Prefill: re-encode handshake metadata with DCP degree.
            dcp = self.vllm_config.parallel_config.decode_context_parallel_size
            agent_meta = NeuronNixlAgentMetadata(
                engine_id=self.engine_id,
                agent_metadata=self.nixl_wrapper.get_agent_metadata(),
                device_id=self.device_id,
                kv_caches_base_addr=self.kv_caches_base_addr[self.engine_id][
                    self.tp_rank
                ],
                num_blocks=self.num_blocks,
                block_lens=self.block_len_per_layer,
                kv_cache_layout=self.kv_cache_layout
                if not self.use_host_buffer
                else self.host_buffer_kv_cache_layout,
                block_size=self.block_size,
                ssm_sizes=self._mamba_ssm_size,
                dcp_size=dcp,
            )
            encoder = msgspec.msgpack.Encoder()
            self.xfer_handshake_metadata = NixlHandshakePayload(
                compatibility_hash=self.compat_hash,
                agent_metadata_bytes=encoder.encode(agent_meta),
            )
            logger.info("DCP NIXL: prefill metadata tagged with dcp_size=%d", dcp)
        else:
            # Decode: store local DCP rank for block filtering.
            from vllm.distributed.parallel_state import get_dcp_group

            g = get_dcp_group()
            self._local_dcp_size = g.world_size
            self._local_dcp_rank = g.rank_in_group

        # Patch tp_ratio so upstream's add_remote_agent (used for match case)
        # doesn't apply head splitting for DCP engines.
        if self.kv_topo is not None:
            orig_tp_ratio = self.kv_topo.tp_ratio

            def _patched_tp_ratio(remote_tp_size, _orig=orig_tp_ratio):
                for eid in self._dcp_prefill_engines:
                    if self.kv_topo.remote_tp_size.get(eid) == remote_tp_size:
                        return 1
                return _orig(remote_tp_size)

            self.kv_topo.tp_ratio = _patched_tp_ratio

        if self._local_dcp_size > 1:
            logger.info(
                "DCP NIXL: local decode dcp_size=%d, dcp_rank=%d, tp_rank=%d",
                self._local_dcp_size,
                self._local_dcp_rank,
                self.tp_rank,
            )

    # ── Override: _validate_remote_agent_handshake ────────────────────────

    def _validate_remote_agent_handshake(self, nixl_agent_meta, remote_tp_size):
        """Skip strict validation for DCP prefill engines."""
        assert self.kv_topo is not None
        local_tp = self.kv_topo.tp_size
        if local_tp != remote_tp_size:
            remote_block_len = nixl_agent_meta.block_lens[0]
            local_block_len = self.block_len_per_layer[0]
            larger = max(remote_block_len, local_block_len)
            smaller = min(remote_block_len, local_block_len)
            if smaller > 0 and larger % smaller == 0:
                self._detect_dcp_prefill(nixl_agent_meta, remote_tp_size)
                assert len(nixl_agent_meta.kv_caches_base_addr) == len(
                    self.block_len_per_layer
                ), "Layer count mismatch between P and D"
                return
        super()._validate_remote_agent_handshake(nixl_agent_meta, remote_tp_size)

    # ── Override: add_remote_agent ────────────────────────────────────────

    def add_remote_agent(self, nixl_agent_meta, remote_tp_rank=0, remote_tp_size=1):
        """Handle head offset (split), inverse registration (merge), or passthrough."""
        eid = nixl_agent_meta.engine_id
        self._detect_dcp_prefill(nixl_agent_meta, remote_tp_size)

        if not self._is_dcp_prefill_engine(eid):
            return super().add_remote_agent(
                nixl_agent_meta, remote_tp_rank, remote_tp_size
            )

        head_ratio = self._dcp_prefill_head_ratio
        inverse_head_ratio = self._dcp_prefill_inverse_head_ratio

        if head_ratio > 1:
            return self._add_remote_agent_split(
                nixl_agent_meta, remote_tp_rank, remote_tp_size, head_ratio
            )
        if inverse_head_ratio > 1:
            return self._add_remote_agent_merge(
                nixl_agent_meta, remote_tp_rank, remote_tp_size
            )
        return super().add_remote_agent(nixl_agent_meta, remote_tp_rank, remote_tp_size)

    def _add_remote_agent_split(
        self, nixl_agent_meta, remote_tp_rank, remote_tp_size, head_ratio
    ):
        """DCP split: register remote descriptors at head_offset.

        Handles full registration independently (no super() call) because
        upstream's add_remote_agent asserts block_len proportionality that
        doesn't hold for DCP split topology.
        """
        eid = nixl_agent_meta.engine_id
        head_idx = self.tp_rank % head_ratio
        local_block_len = self.block_len_per_layer[0]
        head_offset = head_idx * local_block_len

        if eid not in self._tp_size:
            self._tp_size[eid] = remote_tp_size
        if eid not in self.dst_num_blocks:
            self.dst_num_blocks[eid] = nixl_agent_meta.num_blocks
        if eid not in self._block_size:
            self._block_size[eid] = nixl_agent_meta.block_size

        self.kv_caches_base_addr[eid][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        self._remote_agents[eid][remote_tp_rank] = remote_agent_name

        kv_addrs = nixl_agent_meta.kv_caches_base_addr
        blocks_data = []
        for layer_idx, base_addr in enumerate(kv_addrs):
            page_size = nixl_agent_meta.block_lens[layer_idx]
            for block_id in range(nixl_agent_meta.num_blocks):
                addr = base_addr + block_id * page_size + head_offset
                blocks_data.append((addr, local_block_len, nixl_agent_meta.device_id))
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        handle = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        self.dst_xfer_side_handles[eid][remote_tp_rank] = handle
        return remote_agent_name

    def _add_remote_agent_merge(self, nixl_agent_meta, remote_tp_rank, remote_tp_size):
        """DCP merge: register remote descriptors with smaller block size."""
        eid = nixl_agent_meta.engine_id
        remote_block_len = nixl_agent_meta.block_lens[0]

        if eid not in self._tp_size:
            self._tp_size[eid] = remote_tp_size
        if eid not in self.dst_num_blocks:
            self.dst_num_blocks[eid] = nixl_agent_meta.num_blocks
        if eid not in self._block_size:
            self._block_size[eid] = nixl_agent_meta.block_size

        self.kv_caches_base_addr[eid][remote_tp_rank] = (
            nixl_agent_meta.kv_caches_base_addr
        )

        remote_agent_name = self.nixl_wrapper.add_remote_agent(
            nixl_agent_meta.agent_metadata
        )
        self._remote_agents[eid][remote_tp_rank] = remote_agent_name

        kv_addrs = nixl_agent_meta.kv_caches_base_addr
        blocks_data = []
        for layer_idx, base_addr in enumerate(kv_addrs):
            page_size = nixl_agent_meta.block_lens[layer_idx]
            for block_id in range(nixl_agent_meta.num_blocks):
                addr = base_addr + block_id * page_size
                blocks_data.append((addr, remote_block_len, nixl_agent_meta.device_id))
        descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
        handle = self.nixl_wrapper.prep_xfer_dlist(remote_agent_name, descs)
        self.dst_xfer_side_handles[eid][remote_tp_rank] = handle
        return remote_agent_name

    # ── Override: _nixl_handshake ─────────────────────────────────────────

    def _nixl_handshake(self, host, port, remote_tp_size, expected_engine_id):
        """Connect to all remote ranks when TP differs (DCP prefill)."""
        local_tp = self.kv_topo.tp_size if self.kv_topo else remote_tp_size
        if remote_tp_size != local_tp:
            self.kv_topo.get_target_remote_ranks = lambda rtp: list(range(rtp))
            try:
                result = super()._nixl_handshake(
                    host, port, remote_tp_size, expected_engine_id
                )
            finally:
                if "get_target_remote_ranks" in self.kv_topo.__dict__:
                    del self.kv_topo.__dict__["get_target_remote_ranks"]
            return result
        return super()._nixl_handshake(host, port, remote_tp_size, expected_engine_id)

    # ── Override: _read_blocks_for_req ────────────────────────────────────

    def _read_blocks_for_req(self, req_id, meta):
        """DCP-aware block reading with interleaving and filtering."""
        assert meta.remote is not None and self.kv_topo is not None
        eid = meta.remote.engine_id
        is_dcp_prefill = self._is_dcp_prefill_engine(eid)

        if not is_dcp_prefill and self._local_dcp_size > 1:
            self._read_blocks_dcp_decode(req_id, meta)
            return

        if not is_dcp_prefill:
            super()._read_blocks_for_req(req_id, meta)
            return

        self._read_blocks_dcp_prefill(req_id, meta)

    def _read_blocks_dcp_decode(self, req_id, meta):
        """DCP decode: filter remote blocks to this rank's interleaved subset."""
        filtered_remote = []
        for group_blocks in meta.remote.block_ids:
            filtered_remote.append(
                [
                    b
                    for j, b in enumerate(group_blocks)
                    if j % self._local_dcp_size == self._local_dcp_rank
                ]
            )
        trimmed_local = []
        for i, group_blocks in enumerate(meta.local_physical_block_ids):
            n_remote = len(filtered_remote[i])
            trimmed_local.append(group_blocks[:n_remote])

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "DCP decode read: dcp_rank=%d/%d, tp_rank=%d, "
                "remote_blocks=%s, filtered=%s, local=%s",
                self._local_dcp_rank,
                self._local_dcp_size,
                self.tp_rank,
                [list(g)[:4] for g in meta.remote.block_ids],
                [list(g)[:4] for g in filtered_remote],
                [list(g)[:4] for g in trimmed_local],
            )

        meta.remote.block_ids = filtered_remote
        meta.local_physical_block_ids = trimmed_local
        super()._read_blocks_for_req(req_id, meta)

    def _read_blocks_dcp_prefill(self, req_id, meta):
        """DCP prefill: read from multiple remote ranks with block interleaving."""
        eid = meta.remote.engine_id
        inverse_head_ratio = self._dcp_prefill_inverse_head_ratio
        prefill_dcp = self._remote_prefill_dcp

        remote_block_size = self.kv_topo.remote_block_size[eid]
        remote_tp_size = self._tp_size[eid]
        local_tp = self.kv_topo.tp_size
        local_dcp = self._local_dcp_size

        # Total number of prefill ranks this decode rank reads from.
        # For split: tp_ratio * head_ratio (multiple head groups × cp_ranks).
        # For match: tp_ratio (just cp_ranks, head_ratio=1).
        # For merge: tp_ratio (cp_ranks × head_positions handled below).
        tp_ratio = max(1, remote_tp_size // local_tp)
        head_ratio = self._dcp_prefill_head_ratio
        num_target_ranks = tp_ratio * head_ratio

        if inverse_head_ratio > 1:
            # Merge: read from multiple head positions × cp_ranks.
            p_dcp = num_target_ranks // inverse_head_ratio
            tp_pair_size = remote_tp_size // p_dcp
            effective_pos_start = self.tp_rank * inverse_head_ratio
            remote_ranks = []
            for hp in range(inverse_head_ratio):
                ep = effective_pos_start + hp
                for cp in range(p_dcp):
                    remote_ranks.append(ep + cp * tp_pair_size)
            num_token_groups = p_dcp
        else:
            # Split or match: read from one head position × all cp_ranks.
            num_head_groups = remote_tp_size // num_target_ranks
            effective_pos = self.tp_rank * num_head_groups // local_tp
            stride = remote_tp_size // num_target_ranks
            remote_ranks = [effective_pos + i * stride for i in range(num_target_ranks)]
            num_token_groups = num_target_ranks

        if local_dcp > 1:
            remote_ranks = [
                rank
                for idx, rank in enumerate(remote_ranks)
                if idx % local_dcp == self._local_dcp_rank
            ]
            num_token_groups = len(remote_ranks)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "DCP prefill read: req=%s, tp_rank=%d, dcp_rank=%d/%d, "
                "remote_ranks=%s, num_token_groups=%d, "
                "local_blocks=%s, remote_blocks=%s",
                req_id,
                self.tp_rank,
                self._local_dcp_rank,
                self._local_dcp_size,
                remote_ranks,
                num_token_groups,
                [list(g)[:4] for g in meta.local_physical_block_ids],
                [list(g)[:4] for g in meta.remote.block_ids],
            )

        if inverse_head_ratio > 1:
            split_local_handles = self._get_or_create_inverse_handles(
                eid, inverse_head_ratio
            )
            local_xfer_side_handle = None
        else:
            local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                remote_block_size
            ]
            split_local_handles = None

        # Precompute block partitions by token group (avoids recomputing per rank).
        partitions = [
            [
                [b for j, b in enumerate(group) if j % num_token_groups == tg]
                for group in meta.local_physical_block_ids
            ]
            for tg in range(num_token_groups)
        ]

        for i, remote_rank in enumerate(remote_ranks):
            remote_xfer_side_handle = self.dst_xfer_side_handles[eid][remote_rank]
            split_local = partitions[i % num_token_groups]

            if not any(len(g) > 0 for g in split_local):
                self._read_blocks(
                    request_id=req_id,
                    dst_engine_id=eid,
                    remote_request_id=meta.remote.request_id,
                    local_block_ids=[],
                    remote_block_ids=meta.remote.block_ids,
                    remote_rank=remote_rank,
                    local_xfer_side_handle=local_xfer_side_handle
                    if split_local_handles is None
                    else split_local_handles[0],
                    remote_xfer_side_handle=remote_xfer_side_handle,
                )
                continue

            if split_local_handles is not None:
                head_group_idx = i // num_token_groups
                cur_local_handle = split_local_handles[head_group_idx]
            else:
                cur_local_handle = local_xfer_side_handle

            self._read_blocks(
                request_id=req_id,
                dst_engine_id=eid,
                remote_request_id=meta.remote.request_id,
                local_block_ids=split_local,
                remote_block_ids=meta.remote.block_ids,
                remote_rank=remote_rank,
                local_xfer_side_handle=cur_local_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
            )

        # Ensure get_finished() can mark this request as done even if all
        # remote ranks had empty blocks (defaultdict creates the empty list).
        if req_id not in self._recving_transfers:
            self._recving_transfers[req_id]

    def _get_or_create_inverse_handles(self, eid, inverse_head_ratio):
        """Create local split handles for merge (inverse_head_ratio > 1)."""
        if eid in self._inverse_handle_cache:
            return self._inverse_handle_cache[eid]

        remote_block_len = self.block_len_per_layer[0] // inverse_head_ratio
        local_base_addrs = self.kv_caches_base_addr[self.engine_id][self.tp_rank]
        split_handles = []
        for hp in range(inverse_head_ratio):
            blocks_data = []
            for layer_idx, base_addr in enumerate(local_base_addrs):
                local_page_size = self.block_len_per_layer[layer_idx]
                for block_id in range(self.num_blocks):
                    addr = (
                        base_addr + block_id * local_page_size + hp * remote_block_len
                    )
                    blocks_data.append((addr, remote_block_len, self.device_id))
            descs = self.nixl_wrapper.get_xfer_descs(blocks_data, self.nixl_memory_type)
            handle = self.nixl_wrapper.prep_xfer_dlist("NIXL_INIT_AGENT", descs)
            split_handles.append(handle)
        self._inverse_handle_cache[eid] = split_handles
        return split_handles


class NeuronNixlConnector(NixlConnector):
    """Neuron NIXL connector with DCP support.

    Drop-in replacement for NixlConnector that handles DCP prefill and
    DCP decode topologies. Uses NeuronNixlConnectorWorker on the worker
    side and NeuronNixlAgentMetadata for handshake.

    Usage in --kv-transfer-config:
        "kv_connector": "NeuronNixlConnector"

    For standard (non-DCP) transfers, behaves identically to NixlConnector.
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: "KVConnectorRole",
        kv_cache_config: "KVCacheConfig",
    ):
        # Bypass NixlConnector.__init__ to use our worker class.
        # Call grandparent init directly.
        from vllm.distributed.kv_transfer.kv_connector.v1.base import (
            KVConnectorBase_V1,
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
            KVConnectorRole,
            NixlConnectorScheduler,
        )

        KVConnectorBase_V1.__init__(self, vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.kv_cache_config = kv_cache_config
        self.engine_id = vllm_config.kv_transfer_config.engine_id
        self.kv_transfer_config = vllm_config.kv_transfer_config

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler = NixlConnectorScheduler(
                vllm_config, self.engine_id, kv_cache_config
            )
            self.connector_worker = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = NeuronNixlConnectorWorker(
                vllm_config, self.engine_id, kv_cache_config
            )
