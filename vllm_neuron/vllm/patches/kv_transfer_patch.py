# SPDX-License-Identifier: Apache-2.0
"""
Monkey-patch for bi-directional KV cache transfer (D→P) on NixlConnectorScheduler.

Based on https://github.com/vllm-project/vllm/pull/32553 by Sunita Nadampalli.

This patch adds three capabilities to the existing NixlConnectorScheduler:

1. get_num_new_matched_tokens: When a P-node request arrives with
   remote_block_ids from D (multi-turn), report those as external tokens
   so the scheduler skips recomputing them.

2. update_state_after_alloc: When P has remote_block_ids and external
   tokens, populate _reqs_need_recv so the worker pulls KV from D.

3. request_finished: When a D-node decode request finishes, return
   kv_transfer_params with remote_block_ids so the proxy can forward
   them to the next turn's prefill request.

Toggle: set VLLM_NEURON_BIDIRECTIONAL_KV=1 to enable (default: disabled).
"""

import os
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

ENVVAR = "VLLM_NEURON_BIDIRECTIONAL_KV"


def is_enabled() -> bool:
    return os.environ.get(ENVVAR, "0") == "1"


def apply_bidirectional_kv_patch():
    """Monkey-patch NixlConnectorScheduler with bi-directional KV transfer."""
    if not is_enabled():
        logger.info("Bi-directional KV patch disabled (set %s=1)", ENVVAR)
        return

    if getattr(apply_bidirectional_kv_patch, "_applied", False):
        return
    apply_bidirectional_kv_patch._applied = True

    logger.info("Applying bi-directional KV transfer patch...")

    from vllm.distributed.kv_transfer.kv_connector.v1.nixl_connector import (
        NixlConnectorScheduler,
    )

    _orig_get_num = NixlConnectorScheduler.get_num_new_matched_tokens
    _orig_update_state = NixlConnectorScheduler.update_state_after_alloc
    _orig_request_finished = NixlConnectorScheduler.request_finished

    def _get_remote_pull_threshold(self) -> int:
        """Read remote_pull_threshold from kv_connector_extra_config."""
        if not hasattr(self, "_remote_pull_threshold"):
            self._remote_pull_threshold = int(
                self.vllm_config.kv_transfer_config.get_from_extra_config(
                    "remote_pull_threshold", 0
                )
            )
        return self._remote_pull_threshold

    # -- 1. get_num_new_matched_tokens ------------------------------------
    def _patched_get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        count, is_async = _orig_get_num(self, request, num_computed_tokens)
        if count > 0:
            return count, is_async

        params = request.kv_transfer_params
        if (
            params is not None
            and params.get("do_remote_decode")
            and params.get("remote_block_ids")
            and all(
                p in params for p in ("remote_engine_id", "remote_host", "remote_port")
            )
        ):
            max_external = len(request.prompt_token_ids or [])
            remote_num_tokens = params.get("remote_num_tokens") or 0
            remote_block_ids = params.get("remote_block_ids") or []

            if remote_block_ids and isinstance(remote_block_ids[0], list):
                max_blocks = max(len(g) for g in remote_block_ids)
            else:
                max_blocks = len(remote_block_ids)

            if remote_num_tokens > max_blocks * self.block_size:
                logger.warning(
                    "remote_num_tokens (%d) exceeds max_blocks * block_size "
                    "(%d * %d = %d) for %s, skipping remote pull",
                    remote_num_tokens,
                    max_blocks,
                    self.block_size,
                    max_blocks * self.block_size,
                    request.request_id,
                )
                return 0, False
            ext_count = min(remote_num_tokens, max_external) - num_computed_tokens
            logger.debug(
                "Bi-directional KV get_num_new_matched_tokens: "
                "req=%s, num_computed=%d, remote_num_tokens=%d, "
                "max_external=%d, ext_count=%d",
                request.request_id,
                num_computed_tokens,
                remote_num_tokens,
                max_external,
                ext_count,
            )
            if ext_count > 0:
                threshold = _get_remote_pull_threshold(self)
                if threshold > 0 and ext_count < threshold:
                    logger.info(
                        "Skipping remote pull for %s: %d tokens < threshold %d",
                        request.request_id,
                        ext_count,
                        threshold,
                    )
                    return 0, False
                return ext_count, True

        return 0, False

    # -- 2. update_state_after_alloc --------------------------------------
    def _patched_update_state_after_alloc(
        self, request, blocks, num_external_tokens: int
    ):
        _orig_update_state(self, request, blocks, num_external_tokens)

        params = request.kv_transfer_params
        if not params:
            return
        # Guard: only process remote blocks once per request.
        # After recv completes and the request is rescheduled,
        # update_state_after_alloc is called again — this flag
        # prevents a duplicate entry in _reqs_need_recv.
        if params.get("_remote_blocks_processed"):
            return
        if (
            params.get("do_remote_decode")
            and num_external_tokens > 0
            and params.get("remote_block_ids")
            and all(
                p in params for p in ("remote_engine_id", "remote_host", "remote_port")
            )
        ):
            threshold = _get_remote_pull_threshold(self)
            if threshold > 0 and num_external_tokens < threshold:
                logger.debug(
                    "Skipping remote pull in update_state_after_alloc "
                    "for %s: %d external tokens < threshold %d",
                    request.request_id,
                    num_external_tokens,
                    threshold,
                )
                return

            local_block_ids = self.get_sw_clipped_blocks(
                blocks.get_unhashed_block_ids_all_groups()
            )
            self._reqs_need_recv[request.request_id] = (
                request,
                local_block_ids,
            )
            params["_remote_blocks_processed"] = True

    # -- 3. request_finished ----------------------------------------------
    def _patched_request_finished(
        self, request, block_ids
    ) -> tuple[bool, dict[str, Any] | None]:
        import vllm.envs as envs
        from vllm.v1.request import RequestStatus

        params = request.kv_transfer_params
        if not params:
            return False, None

        if params.get("do_remote_prefill"):
            return _orig_request_finished(self, request, block_ids)

        if params.get("do_remote_decode"):
            # P-node finished: hold blocks for D to read.
            # The original method only allows FINISHED_LENGTH_CAPPED here,
            # but FINISHED_STOPPED (EOS) is equally valid — P's KV cache
            # is still useful for D regardless of why P stopped.
            if request.status not in (
                RequestStatus.FINISHED_LENGTH_CAPPED,
                RequestStatus.FINISHED_STOPPED,
            ):
                self._reqs_not_processed.add(request.request_id)
                self._reqs_need_save.pop(request.request_id, None)
                return False, None

            delay_free = any(len(g) > 0 for g in block_ids)
            if delay_free:
                self._reqs_need_send[request.request_id] = (
                    time.perf_counter() + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
                )

            block_ids = self.get_sw_clipped_blocks(block_ids)
            return delay_free, dict(
                do_remote_prefill=True,
                do_remote_decode=False,
                remote_block_ids=block_ids,
                remote_engine_id=self.engine_id,
                remote_request_id=request.request_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                remote_num_tokens=request.num_computed_tokens,
            )

        # D-node decode finished — return block info for next turn.
        # NOTE: We only handle FINISHED_STOPPED and FINISHED_LENGTH_CAPPED.
        # FINISHED_ABORTED requests intentionally don't return block IDs —
        # an aborted request's KV cache shouldn't be reused in the next turn
        # since the generation was incomplete/invalid.
        if request.status in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH_CAPPED,
        ):
            delay_free = any(len(g) > 0 for g in block_ids)
            if delay_free:
                self._reqs_in_batch.add(request.request_id)
                self._reqs_need_send[request.request_id] = (
                    time.perf_counter() + envs.VLLM_NIXL_ABORT_REQUEST_TIMEOUT
                )

            block_ids = self.get_sw_clipped_blocks(block_ids)
            # Use num_computed_tokens, not num_tokens — the latter
            # includes the last sampled token which has no KV in cache.
            remote_num_tokens = request.num_computed_tokens

            return delay_free, dict(
                do_remote_prefill=False,
                do_remote_decode=True,
                remote_block_ids=block_ids,
                remote_engine_id=self.engine_id,
                remote_request_id=request.request_id,
                remote_host=self.side_channel_host,
                remote_port=self.side_channel_port,
                tp_size=self.vllm_config.parallel_config.tensor_parallel_size,
                remote_num_tokens=remote_num_tokens,
            )

        return False, None

    # -- Apply patches ----------------------------------------------------
    NixlConnectorScheduler.get_num_new_matched_tokens = (
        _patched_get_num_new_matched_tokens
    )
    NixlConnectorScheduler.update_state_after_alloc = _patched_update_state_after_alloc
    NixlConnectorScheduler.request_finished = _patched_request_finished

    logger.info("Bi-directional KV transfer patch applied (%s=1)", ENVVAR)


_streaming_patch_applied = False


def apply_streaming_kv_patch():
    """Patch the OpenAI chat/completion streaming to include kv_transfer_params
    in the final SSE chunk.
    """
    global _streaming_patch_applied
    if _streaming_patch_applied:
        return
    if not is_enabled():
        return

    try:
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
        from vllm.entrypoints.openai.completion.serving import (
            OpenAIServingCompletion,
        )
    except ImportError:
        return

    _streaming_patch_applied = True

    import json as _json

    logger.info("Applying streaming kv_transfer_params patch...")

    # -- Chat completions streaming patch ---------------------------------
    # NOTE: This SSE interception approach is fragile — it parses raw SSE
    # text lines and injects kv_transfer_params into the penultimate chunk
    # (the one before "data: [DONE]"). If vLLM changes its SSE format
    # (extra lines, different ordering, etc.), this will break silently.
    # This is a temporary workaround until Sunita's upstream PR #32553
    # merges, which adds kv_transfer_params natively to the streaming
    # protocol. Once merged, this entire streaming patch can be removed.
    _orig_chat_stream = OpenAIServingChat.chat_completion_stream_generator

    async def _patched_chat_stream(self, request, result_generator, *args, **kwargs):
        captured = {}

        async def _capturing_generator(gen):
            async for res in gen:
                if hasattr(res, "kv_transfer_params") and res.kv_transfer_params:
                    captured["kv_transfer_params"] = res.kv_transfer_params
                yield res

        prev_line = None
        prev_prev_line = None
        async for sse_line in _orig_chat_stream(
            self, request, _capturing_generator(result_generator), *args, **kwargs
        ):
            if prev_prev_line is not None:
                yield prev_prev_line
            prev_prev_line = prev_line
            prev_line = sse_line

        if prev_prev_line is not None and captured.get("kv_transfer_params"):
            if (
                prev_prev_line.startswith("data: ")
                and prev_prev_line.strip() != "data: [DONE]"
            ):
                try:
                    chunk_json = _json.loads(prev_prev_line[6:])
                    chunk_json["kv_transfer_params"] = captured["kv_transfer_params"]
                    prev_prev_line = f"data: {_json.dumps(chunk_json)}\n\n"
                except (ValueError, KeyError):
                    pass
        if prev_prev_line is not None:
            yield prev_prev_line
        if prev_line is not None:
            yield prev_line

    OpenAIServingChat.chat_completion_stream_generator = _patched_chat_stream

    # -- Completions streaming patch --------------------------------------
    _orig_completion_stream = OpenAIServingCompletion.completion_stream_generator

    async def _patched_completion_stream(
        self, request, result_generator, *args, **kwargs
    ):
        captured = {}

        async def _capturing_generator(gen):
            async for res in gen:
                if hasattr(res, "kv_transfer_params") and res.kv_transfer_params:
                    captured["kv_transfer_params"] = res.kv_transfer_params
                yield res

        prev_line = None
        prev_prev_line = None
        async for sse_line in _orig_completion_stream(
            self, request, _capturing_generator(result_generator), *args, **kwargs
        ):
            if prev_prev_line is not None:
                yield prev_prev_line
            prev_prev_line = prev_line
            prev_line = sse_line

        if prev_prev_line is not None and captured.get("kv_transfer_params"):
            if (
                prev_prev_line.startswith("data: ")
                and prev_prev_line.strip() != "data: [DONE]"
            ):
                try:
                    chunk_json = _json.loads(prev_prev_line[6:])
                    chunk_json["kv_transfer_params"] = captured["kv_transfer_params"]
                    prev_prev_line = f"data: {_json.dumps(chunk_json)}\n\n"
                except (ValueError, KeyError):
                    pass
        if prev_prev_line is not None:
            yield prev_prev_line
        if prev_line is not None:
            yield prev_line

    OpenAIServingCompletion.completion_stream_generator = _patched_completion_stream

    logger.info("Streaming kv_transfer_params patch applied")
