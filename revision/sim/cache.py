"""Block cache with ledger, pins, leaf-only candidates and three policies
(EVENT_MODEL R-C1..R-C3, R-C7, R-C8).

Eviction candidates live in a lazy heap keyed by the policy priority. A block
is *structurally eligible* when it is unpinned and (under leaf-only) has no
resident child; it is pushed when it becomes eligible and its entry is
invalidated (version bump) when it becomes ineligible or its priority changes.
Per-admission protected sets are small and handled by deferring popped
entries. `sim/tests/test_cache_fast.py` checks the victim against a
brute-force scan on random caches.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass
class Block:
    key: bytes
    parent: bytes | None
    ntok: int
    insert_step: int
    last_access: int
    count: int = 1
    pins: int = 0
    version: int = 0
    inserter: dict = field(default_factory=dict)  # attribution metadata (R-A1)


class BlockCache:
    def __init__(self, capacity: int | None, policy: str, B: int, leaf_only: bool = True,
                 ignore_pins: bool = False):
        """ignore_pins: historical diagnostic (legacy_no_pin): pinned blocks stay eviction
        candidates, as in the submitted simulator. Never used in the main tables."""
        if policy not in ("LRU", "LFU", "FIFO"):
            raise ValueError(policy)
        self.ignore_pins = ignore_pins
        self.capacity = capacity  # None = infinite
        self.policy = policy
        self.B = B
        self.leaf_only = leaf_only
        self.blocks: dict[bytes, Block] = {}
        self.children_of: dict[bytes, set[bytes]] = {}   # resident children, kept even if parent evicted
        self.pinned_keys: set[bytes] = set()
        self._heap: list[tuple[tuple, bytes, int]] = []
        self._ver = 0   # global version counter: unique across a key's lifetimes (fifth review)
        # ledger (R-C2)
        self.resident = 0
        self.partial_slots = 0
        self.unused_reservations = 0
        self.n_evictions = 0

    # ------------------------------------------------------------ ledger
    @property
    def ledger(self) -> int:
        return self.resident + self.partial_slots + self.unused_reservations

    @property
    def free(self) -> int:
        return (10**18 if self.capacity is None else self.capacity) - self.ledger

    def pinned_tokens(self) -> int:
        return sum(self.blocks[k].ntok for k in self.pinned_keys)

    # -------------------------------------------------------- residency
    def is_resident(self, key: bytes) -> bool:
        return key in self.blocks

    def n_children(self, key: bytes) -> int:
        return len(self.children_of.get(key, ()))

    def _prio(self, b: Block) -> tuple:
        if self.policy == "LRU":
            return (b.last_access, b.insert_step, b.key)
        if self.policy == "LFU":
            return (b.count, b.insert_step, b.key)
        return (b.insert_step, b.key)

    def _eligible(self, k: bytes, b: Block) -> bool:
        return (b.pins == 0 or self.ignore_pins) and (not self.leaf_only or self.n_children(k) == 0)

    def _refresh(self, k: bytes) -> None:
        """Invalidate any heap entry for k and push a fresh one if k is eligible."""
        b = self.blocks.get(k)
        if b is None:
            return
        self._ver += 1
        b.version = self._ver
        if self.capacity is not None and self._eligible(k, b):
            heapq.heappush(self._heap, (self._prio(b), k, b.version))

    def insert(self, key: bytes, parent: bytes | None, step: int, inserter: dict | None = None,
               pinned: bool = True, count: int = 1) -> Block:
        """Insert a non-resident full block; caller has already made room (R-C4 commit)."""
        assert key not in self.blocks
        b = Block(key=key, parent=parent, ntok=self.B, insert_step=step, last_access=step,
                  count=count, pins=1 if pinned else 0, inserter=inserter or {})
        self.blocks[key] = b
        self.resident += self.B
        if pinned:
            self.pinned_keys.add(key)
        if parent is not None:
            kids = self.children_of.setdefault(parent, set())
            kids.add(key)
            if len(kids) == 1:
                self._refresh(parent)   # parent may have stopped being a leaf
        self._refresh(key)
        return b

    def preload(self, keys: list[bytes], count: int = 1, step: int = 0) -> None:
        """Test seam and warm-up: insert a chain of completed, unpinned blocks."""
        parent = None
        for i, k in enumerate(keys):
            if k not in self.blocks:
                self.insert(k, parent, step=step + i, pinned=False, count=count)
            parent = k

    def touch(self, key: bytes, step: int) -> None:
        """Policy metadata update, once per event per block (R-C8)."""
        b = self.blocks[key]
        if self.policy == "LRU":
            b.last_access = step
        elif self.policy == "LFU":
            b.count += 1
        else:
            return  # FIFO: nothing changes
        self._refresh(key)

    def pin(self, key: bytes) -> None:
        b = self.blocks[key]
        b.pins += 1
        if b.pins == 1:
            self.pinned_keys.add(key)
            if not self.ignore_pins:
                self._ver += 1
                b.version = self._ver   # entry (if any) becomes invalid; no push while pinned

    def unpin(self, key: bytes) -> None:
        b = self.blocks.get(key)
        if b is None:
            # only reachable under ignore_pins (legacy_no_pin): the block was evicted while pinned
            assert self.ignore_pins, "unpin of a non-resident block outside the legacy path"
            self.pinned_keys.discard(key)
            return
        assert b.pins > 0
        b.pins -= 1
        if b.pins == 0:
            self.pinned_keys.discard(key)
            if not self.ignore_pins:
                self._refresh(key)

    # --------------------------------------------------------- eviction
    def candidates(self, protected: set[bytes] | None = None) -> set[bytes]:
        """Brute-force candidate set (tests and diagnostics only)."""
        protected = protected or set()
        return {k for k, b in self.blocks.items() if self._eligible(k, b) and k not in protected}

    def peek_victim(self, protected: set[bytes]) -> bytes | None:
        deferred = []
        victim = None
        while self._heap:
            prio, k, ver = self._heap[0]
            b = self.blocks.get(k)
            if b is None or ver != b.version or not self._eligible(k, b):
                heapq.heappop(self._heap)
                continue
            if k in protected:
                deferred.append(heapq.heappop(self._heap))
                continue
            victim = k
            break
        for e in deferred:
            heapq.heappush(self._heap, e)
        return victim

    def evict_one(self, protected: set[bytes]) -> bool:
        k = self.peek_victim(protected)
        if k is None:
            return False
        b = self.blocks.pop(k)
        self.resident -= b.ntok
        if b.parent is not None:
            kids = self.children_of.get(b.parent)
            if kids is not None:
                kids.discard(k)
                if not kids:
                    self._refresh(b.parent)   # parent became a leaf
        self.n_evictions += 1
        return True

    def make_room(self, need: int, protected: set[bytes]) -> bool:
        while self.free < need:
            if not self.evict_one(protected):
                return False
        return True
