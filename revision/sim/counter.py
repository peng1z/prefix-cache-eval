"""Infinite-capacity abstract counter (THEORY_DRAFT A-1..A-6; EVENT_MODEL R-E1, R-E2)."""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import Request, decode_keys, prompt_keys


@dataclass
class CountResult:
    total: int = 0
    missed: int = 0
    n_full_hit_corrections: int = 0
    n_empty_prefill: int = 0
    per_request_hit: dict[str, int] = field(default_factory=dict)

    @property
    def reuse(self) -> float | None:
        return None if self.total == 0 else 1.0 - self.missed / self.total


def count_infinite(
    requests: list[Request],
    events: list[tuple],
    B: int,
    full_hit_correction: bool = False,
) -> CountResult:
    """Run `events` against an empty infinite cache.

    Event forms:
      ("P", rid)          prefill
      ("D", rid)          atomic decode: inserts every block the decode completes
      ("Db", rid, idx)    block-granular decode: inserts the block with array index idx
      ("C", rid)          completion marker (no cache effect)
    A ("D", rid) is equivalent to its ("Db", ...) events in order followed by ("C", rid).
    """
    by_id = {r.request_id: r for r in requests}
    resident: set[bytes] = set()
    res = CountResult()
    for ev in events:
        kind, rid = ev[0], ev[1]
        r = by_id[rid]
        if kind == "P":
            L = len(r.prompt)
            if L == 0:
                res.n_empty_prefill += 1
                res.per_request_hit[rid] = 0
                continue
            keys = prompt_keys(r, B)
            hit_blocks = 0
            for k in keys:  # lookup on the resident set BEFORE this request inserts (R-E1)
                if k in resident:
                    hit_blocks += 1
                else:
                    break
            hit = hit_blocks * B
            if full_hit_correction and hit == L:
                hit -= B
                res.n_full_hit_corrections += 1
            resident.update(keys)
            res.total += L
            res.missed += L - hit
            res.per_request_hit[rid] = hit
        elif kind == "D":
            resident.update(decode_keys(r, B))
        elif kind == "Db":
            idx = ev[2]
            n_prompt_blocks = len(r.prompt) // B
            resident.add(decode_keys(r, B)[idx - n_prompt_blocks])
        elif kind == "C":
            pass
        else:
            raise ValueError(kind)
    return res
