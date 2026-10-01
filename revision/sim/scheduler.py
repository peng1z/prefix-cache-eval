"""Event-level scheduler with slots, admission transaction, decode steps,
infeasible requests and coverage (EVENT_MODEL R-E1..R-E4, R-X1..R-X7, R-C4..R-C6).

Phase 1: no preemption; full-footprint reservation at admission.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

import numpy as np

from .attribution import BUCKETS, credit_bucket, sys_len_for
from .cache import BlockCache
from .events import Request, chain_key_one, chain_keys


@dataclass
class Config:
    B: int
    capacity: int | None            # None = infinite
    policy: str = "LRU"
    K_exec: int = 1
    seed: int = 0
    leaf_only: bool = True
    full_hit_correction: bool = False
    selection: str = "uniform"      # "uniform" | "round_robin" (D-SEL)
    # historical diagnostic switches (EVENT_MODEL §10); never on in main tables
    legacy_no_pin: bool = False            # per-block insertion with eviction that ignores pins (self-eviction)
    legacy_lfu_double_count: bool = False  # hit blocks incremented at lookup and again at insert
    legacy_per_block_alloc: bool = False   # old allocation mode (evict per block on demand) but with the
                                           # current request's hit and inserted blocks protected (P split)


@dataclass
class Running:
    req: Request
    full: list[int]                 # prompt + decode tokens (built once at admission)
    keys: list[bytes]               # chain keys of completed blocks so far (prompt blocks first)
    pos: int                        # current array end
    pinned: list[bytes]
    reserved: int                   # unused reservation blocks
    has_partial_slot: bool


@dataclass
class Result:
    total: int = 0
    missed: int = 0
    n_full_hit_corrections: int = 0
    n_empty_prefill: int = 0
    n_infeasible_requests: int = 0
    infeasible_prefill_tokens: int = 0
    n_decode_events: int = 0
    decode_tokens_written: int = 0
    n_evictions: int = 0
    max_ledger: int = 0
    per_request_hit: dict[str, int] = field(default_factory=dict)
    executable_request_ids: set[str] = field(default_factory=set)
    strata: dict[str, dict[str, int]] = field(default_factory=lambda: {
        "regeneration": {"n": 0, "total": 0, "hit": 0, "missed": 0},
        "non_regeneration": {"n": 0, "total": 0, "hit": 0, "missed": 0},
    })
    attribution: dict[str, int] = field(default_factory=lambda: {b: 0 for b in BUCKETS})
    log: list[tuple[int, str, str]] = field(default_factory=list)  # (step, kind, request_id)
    events: list[tuple] = field(default_factory=list)  # canonical stream: P / Db / C (R-X8)

    @property
    def reuse(self) -> float | None:
        return None if self.total == 0 else 1.0 - self.missed / self.total

    @property
    def coverage(self) -> float:
        d = self.total + self.infeasible_prefill_tokens
        return 1.0 if d == 0 else self.total / d

    def comparable_with(self, other: "Result") -> bool:
        return self.executable_request_ids == other.executable_request_ids


class _Slot:
    def __init__(self, group: list[Request]):
        self.group = group
        self.next_idx = 0
        self.running: Running | None = None
        self.blocked = False        # admission refused; re-probe only after some request completes (R-X4)
        self.last_need = 0          # need at the last refused probe (cheap pre-check on unblock)


def _footprint_blocks(r: Request, B: int) -> int:
    return ceil((len(r.prompt) + len(r.decode)) / B)


def run_schedule(groups: list[list[Request]], cfg: Config, cache: BlockCache | None = None,
                 observer=None, system_prefixes: set | None = None) -> Result:
    """groups: frame-ordered list of groups, each a depth-first-ordered list of requests."""
    B = cfg.B
    cache = cache or BlockCache(cfg.capacity, cfg.policy, B, cfg.leaf_only, ignore_pins=cfg.legacy_no_pin)
    if cfg.legacy_no_pin:
        cache.ignore_pins = True
    rng = np.random.Generator(np.random.PCG64(cfg.seed))
    res = Result()
    queue = list(groups)
    slots: list[_Slot] = []
    dead: set[str] = set()          # infeasible requests and their descendants
    step = 0
    rr = 0

    def fill_slots():
        while queue and len(slots) < max(1, cfg.K_exec):
            slots.append(_Slot(queue.pop(0)))

    def is_dead(r: Request) -> bool:
        p = r.parent
        while p is not None:
            if p in dead:
                return True
            p = by_id[p].parent
        return r.request_id in dead

    by_id = {r.request_id: r for g in groups for r in g}
    fill_slots()

    def try_admit(slot: _Slot) -> str:
        """Returns 'admitted' | 'wait' | 'infeasible' | 'skipped' (dead) | 'empty'."""
        r = slot.group[slot.next_idx]
        if is_dead(r):
            # descendant of an infeasible request: excluded and counted (R-C6)
            dead.add(r.request_id)
            res.n_infeasible_requests += 1
            res.infeasible_prefill_tokens += len(r.prompt)
            res.log.append((step, "infeasible_descendant", r.request_id))
            slot.next_idx += 1
            return "skipped"
        L = len(r.prompt)
        keys = chain_keys(r.prompt, B)
        # 1. lookup
        H: list[bytes] = []
        for k in keys:
            if cache.is_resident(k):
                H.append(k)
            else:
                break
        # 2. reserve footprint
        n_prompt_full = len(keys)
        partial = 1 if L % B else 0
        total_slots = _footprint_blocks(r, B)
        reserved = total_slots - ceil(L / B) if L > 0 else total_slots
        # resident prompt blocks beyond the hit prefix are orphans (any_unpinned only):
        # they need no new space but must be kept through make_room, and are never hits
        orphans = [keys[i] for i in range(len(H), n_prompt_full) if cache.is_resident(keys[i])]
        n_new_prompt = (n_prompt_full - len(H)) - len(orphans)
        need = B * (n_new_prompt + partial + reserved)
        if cfg.capacity is not None:
            if B * total_slots > cfg.capacity:
                # 3b. infeasible (R-C6)
                dead.add(r.request_id)
                res.n_infeasible_requests += 1
                res.infeasible_prefill_tokens += L
                res.log.append((step, "infeasible", r.request_id))
                slot.next_idx += 1
                return "infeasible"
            # 3. feasibility probe: protected = pins of others ∪ H ∪ orphans, counted as a set
            # union without materialising it (every block is B tokens; pinned_keys is maintained)
            pinned = set() if cfg.legacy_no_pin else cache.pinned_keys
            n_protected = 0 if cfg.legacy_no_pin else (len(pinned) + sum(1 for k in H if k not in pinned) + sum(1 for k in orphans if k not in pinned))
            avail = (cfg.capacity - B * n_protected - cache.partial_slots - cache.unused_reservations)
            if need > avail:
                slot.last_need = need
                return "wait"
            protected = set() if cfg.legacy_no_pin else (pinned | set(H) | set(orphans))
        # attribution of hit tokens (R-A2), credited before this request touches anything
        sl = sys_len_for(r, system_prefixes)
        n_hit_blocks = len(H)
        if cfg.full_hit_correction and L > 0 and B * len(H) == L:
            n_hit_blocks -= 1  # the corrected last block is not a hit
        for i in range(n_hit_blocks):
            bucket = credit_bucket(r, i * B, (i + 1) * B, sl, cache.blocks[H[i]].inserter)
            res.attribution[bucket] += B
        # 4. commit
        for k in H:
            cache.pin(k)
            cache.touch(k, step)
        if cfg.legacy_lfu_double_count and cfg.policy == "LFU":
            for k in H:
                if cache.is_resident(k):
                    cache.touch(k, step)   # old insert() re-touched every hit block
        if cfg.capacity is not None and not cfg.legacy_no_pin and not cfg.legacy_per_block_alloc:
            ok = cache.make_room(need, protected)
            assert ok, "make_room failed after a successful probe"
        parent = H[-1] if H else (None)
        meta = {"group": r.group, "path": r.path, "kind": "prefill", "request_id": r.request_id}
        pinned = list(H)
        for i in range(len(H), n_prompt_full):
            k = keys[i]
            if cfg.legacy_no_pin and cfg.capacity is not None and not cache.is_resident(k):
                # old simulator: evict just enough for this one block, any block eligible
                ok = cache.make_room(B, set())
                assert ok or cache.free >= B, "legacy eviction could not free a block"
            elif cfg.legacy_per_block_alloc and cfg.capacity is not None and not cache.is_resident(k):
                # old allocation mode, but the request's hits and already-inserted blocks are
                # pinned and therefore never candidates: isolates protection from allocation
                ok = cache.make_room(B, protected)
                assert ok or cache.free >= B, "per-block allocation could not free a block"
            if cache.is_resident(k):
                # orphan under any_unpinned (R-C7 alternative): not matchable, but
                # already resident; re-attach by pinning instead of re-inserting
                cache.pin(k)
                cache.touch(k, step)
            else:
                cache.insert(k, keys[i - 1] if i > 0 else None, step, inserter=meta, pinned=True)
            pinned.append(k)
        cache.partial_slots += B * partial
        cache.unused_reservations += B * reserved
        # accounting (R-E1)
        hit = B * len(H)
        if L == 0:
            res.n_empty_prefill += 1
        else:
            if cfg.full_hit_correction and hit == L:
                hit -= B
                res.n_full_hit_corrections += 1
            res.total += L
            res.missed += L - hit
            s = res.strata["regeneration" if r.is_regeneration else "non_regeneration"]
            s["n"] += 1; s["total"] += L; s["hit"] += hit; s["missed"] += L - hit
        res.per_request_hit[r.request_id] = hit
        res.executable_request_ids.add(r.request_id)
        res.log.append((step, "admit", r.request_id))
        res.events.append(("P", r.request_id))
        slot.running = Running(req=r, full=r.prompt + r.decode, keys=list(keys), pos=L, pinned=pinned,
                               reserved=reserved, has_partial_slot=bool(partial))
        slot.next_idx += 1
        if not r.decode:
            complete(slot)
        return "admitted"

    def complete(slot: _Slot):
        run = slot.running
        for k in run.pinned:
            cache.unpin(k)
        if run.has_partial_slot:
            cache.partial_slots -= B
        cache.unused_reservations -= B * run.reserved
        res.log.append((step, "complete", run.req.request_id))
        res.events.append(("C", run.req.request_id))
        slot.running = None
        # capacity was released: blocked slots whose last need could now fit re-probe (R-X4).
        # A slot's true need can only be smaller than last_need if more of its prompt became
        # resident, and larger if hits were evicted; the full probe re-computes it either way,
        # and a slot that stays blocked is re-checked at the next completion.
        quick_avail = (cfg.capacity - B * len(cache.pinned_keys) - cache.partial_slots
                       - cache.unused_reservations) if cfg.capacity is not None else 10**18
        for o in slots:
            if o.blocked and o.last_need <= quick_avail + B * 8:
                o.blocked = False

    def decode_step(slot: _Slot):
        run = slot.running
        r = run.req
        end = len(r.prompt) + len(r.decode)
        start = run.pos
        boundary = (start // B + 1) * B
        new_pos = min(boundary, end)
        if start % B == 0 and new_pos > start:
            # opening a new block: reservation -> partial slot
            assert run.reserved > 0
            run.reserved -= 1
            cache.unused_reservations -= B
            cache.partial_slots += B
            run.has_partial_slot = True
        run.pos = new_pos
        res.decode_tokens_written += new_pos - start
        if new_pos % B == 0:
            # block completed: partial slot converts (R-E2, R-C5); one hash, chained on the previous key
            idx = new_pos // B - 1
            k = chain_key_one(run.keys[idx - 1] if idx > 0 else None, run.full[idx * B:(idx + 1) * B])
            run.keys.append(k)
            keys = run.keys
            cache.partial_slots -= B
            run.has_partial_slot = False
            if cache.is_resident(k):
                pass  # already resident: reservation/slot simply released
            else:
                meta = {"group": r.group, "path": r.decode_path or r.path, "kind": "decode",
                        "request_id": r.request_id}
                cache.insert(k, keys[idx - 1] if idx > 0 else None, step, inserter=meta, pinned=False)
            cache.pin(k)
            run.pinned.append(k)
            res.events.append(("Db", r.request_id, idx))
        if new_pos >= end:
            res.n_decode_events += 1
            complete(slot)

    while slots:
        step += 1
        # advanceable slots: running, or able to admit / skip / declare infeasible
        advanceable = []
        for i, s in enumerate(slots):
            if s.running is not None or (s.next_idx < len(s.group) and not s.blocked):
                advanceable.append(i)
        if not advanceable:
            if any(s.blocked for s in slots) and not any(s.running is not None for s in slots):
                raise RuntimeError("deadlock: nothing running and every slot blocked")
            slots = [s for s in slots if s.running is not None or s.next_idx < len(s.group)]
            fill_slots()
            if not slots:
                break
            continue
        if cfg.selection == "round_robin":
            i = advanceable[rr % len(advanceable)]
            rr += 1
        elif len(advanceable) == 1:
            i = advanceable[0]
        else:
            i = advanceable[int(rng.integers(len(advanceable)))]
        s = slots[i]
        if s.running is None:
            outcome = try_admit(s)
            if outcome == "wait":
                s.blocked = True
                # progress guarantee (R-X7): some other slot must be running
                if not any(o.running is not None for o in slots):
                    raise RuntimeError("deadlock: nothing running and admission refused")
        else:
            decode_step(s)
        res.max_ledger = max(res.max_ledger, cache.ledger)
        if cfg.capacity is not None:
            assert cache.ledger <= cfg.capacity, "ledger invariant violated"
        if observer is not None:
            observer(step, None)
        # retire exhausted slots
        keep = []
        for o in slots:
            if o.running is None and o.next_idx >= len(o.group):
                continue
            keep.append(o)
        slots = keep
        fill_slots()

    res.n_evictions = cache.n_evictions
    return res
