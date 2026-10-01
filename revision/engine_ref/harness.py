"""Engine reference: drive vLLM's own V1 prefix-cache core (vendored at commit
8f816a3f…, pure Python, no GPU) with the same request streams the simulator sees,
and record per-request prefix-cache hits.

What is real vLLM code: KVCacheManager, KVCacheCoordinator, FullAttentionManager,
BlockPool, FreeKVCacheBlockQueue, block hashing (get_request_block_hasher /
hash_block_tokens). What is ours: the hash function (sha256 over pickle), the
minimal Request object, and the scheduler loop below (serial, one request at a
time, decode in block-sized steps). Only LRU exists in vLLM.

Semantics that differ from the simulator by construction and are reported, not
hidden: vLLM orders eviction by *release time* (blocks enter the free queue when
their last user frees them, tail blocks first) and any released block is a
candidate; a block completed by decode is hashed on append but only enters the
cache map at the next allocate_slots (or an explicit cache_blocks call).
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch  # noqa: E402  (CPU only; needed by kv_cache_interface dtype sizes)

from vllm.v1.core.kv_cache_manager import KVCacheManager  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor)
from vllm.v1.request import Request, RequestStatus  # noqa: E402

VENDORED_COMMIT = "8f816a3f665489d7f0d222115d4f72ebab01076b"


def sha256_hash_fn(obj) -> bytes:
    return hashlib.sha256(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)).digest()


class VLLMRef:
    """One KVCacheManager with a single full-attention group, `capacity_tokens` of
    usable KV (vLLM keeps block 0 as the null block, so num_blocks = usable + 1)."""

    def __init__(self, capacity_tokens: int | None, B: int, max_model_len: int = 1 << 20,
                 cache_on_complete: bool = True):
        init_none_hash(sha256_hash_fn)
        self.B = B
        usable = (capacity_tokens // B) if capacity_tokens is not None else (1 << 18)  # 'infinite' for tests: 4M tokens at B=16
        spec = FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=1, dtype=torch.float16)
        cfg = KVCacheConfig(
            num_blocks=usable + 1,
            kv_cache_tensors=[KVCacheTensor(size=(usable + 1) * spec.page_size_bytes, layers=["l0"],
                                            layer_stride=0, block_stride=spec.page_size_bytes)],
            kv_cache_groups=[KVCacheGroupSpec(layer_names=["l0"], kv_cache_spec=spec)],
        )
        self.mgr = KVCacheManager(cfg, max_model_len=max_model_len, scheduler_block_size=B,
                                  hash_block_size=B, enable_caching=True)
        self.hasher = get_request_block_hasher(B, sha256_hash_fn)
        self.cache_on_complete = cache_on_complete
        self.n_infeasible = 0

    # ---- one request = prefill, decode steps, completion (same event shape as the simulator)
    def prefill(self, rid: str, prompt: list[int]):
        req = Request(rid, prompt, block_hasher=self.hasher)
        blocks, n_hit, _ = self.mgr.get_computed_blocks(req)
        new = self.mgr.allocate_slots(req, num_new_tokens=len(prompt) - n_hit,
                                      num_new_computed_tokens=n_hit, new_computed_blocks=blocks)
        if new is None:
            self.n_infeasible += 1
            return None, 0
        req.num_computed_tokens = len(prompt)
        req.status = RequestStatus.RUNNING
        return req, n_hit

    def decode_step(self, req: Request, tokens: list[int]) -> bool:
        """Real order: allocate space for the tokens about to be produced, then append them."""
        new = self.mgr.allocate_slots(req, num_new_tokens=len(tokens))
        if new is None:
            return False
        req.append_output_token_ids(tokens)
        req.num_computed_tokens += len(tokens)
        return True

    def complete(self, req: Request) -> None:
        if self.cache_on_complete:
            # make the blocks completed by the last decode tokens cacheable, as the
            # scheduler's next step would; reported as a harness choice
            self.mgr.cache_blocks(req, req.num_computed_tokens)
        self.mgr.free(req)

    @property
    def usage(self) -> float:
        return self.mgr.usage


def run_serial(groups, capacity_tokens: int | None, B: int, cache_on_complete: bool = True):
    """Serial (K_exec = 1) replay of groups of sim Requests: returns per-request hit
    tokens, totals, and the number of requests vLLM could not admit."""
    ref = VLLMRef(capacity_tokens, B, cache_on_complete=cache_on_complete)
    hits = {}
    total = missed = 0
    for g in groups:
        for r in g:
            if len(r.prompt) == 0:
                continue
            req, n_hit = ref.prefill(r.request_id, r.prompt)
            if req is None:
                continue
            hits[r.request_id] = n_hit
            total += len(r.prompt)
            missed += len(r.prompt) - n_hit
            pos = 0
            dec = r.decode
            while pos < len(dec):
                boundary = ((len(r.prompt) + pos) // B + 1) * B
                step = min(boundary - (len(r.prompt) + pos), len(dec) - pos)
                if not ref.decode_step(req, dec[pos:pos + step]):
                    break
                pos += step
            ref.complete(req)
    return {"per_request_hit": hits, "total": total, "missed": missed,
            "reuse": (1 - missed / total) if total else None, "n_infeasible": ref.n_infeasible}
