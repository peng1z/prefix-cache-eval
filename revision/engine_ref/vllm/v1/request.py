"""Minimal Request for the engine comparison: only the attributes the vendored
prefix-cache core reads. Block hashing follows vLLM's update_block_hashes contract."""
from __future__ import annotations
import enum
from collections.abc import Callable


class RequestStatus(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    PREEMPTED = 2
    FINISHED_STOPPED = 3


class Request:
    def __init__(self, request_id: str, prompt_token_ids: list[int], block_hasher: Callable | None = None):
        self.request_id = request_id
        self.prompt_token_ids = list(prompt_token_ids)
        self.num_prompt_tokens = len(self.prompt_token_ids)
        self._all_token_ids = list(self.prompt_token_ids)
        self._output_token_ids: list[int] = []
        self.num_computed_tokens = 0
        self.num_preemptions = 0
        self.num_in_flight_tokens = 0
        self.status = RequestStatus.WAITING
        self.lora_request = None
        self.mm_features: list = []
        self.cache_salt = None
        self.prompt_embeds = None
        self.session_id = None
        self.shared_prefix_boundary = 0
        self.skip_reading_prefix_cache = False
        self._prompt_embeds_per_block_hashes: dict = {}
        self.block_hashes: list = []
        self._block_hasher = block_hasher
        self.update_block_hashes()

    @property
    def all_token_ids(self):
        return self._all_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self._all_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)

    def append_output_token_ids(self, token_ids):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        self._output_token_ids.extend(token_ids)
        self._all_token_ids.extend(token_ids)
        self.update_block_hashes()

    def update_block_hashes(self) -> None:
        if self._block_hasher is not None:
            self.block_hashes.extend(self._block_hasher(self))
