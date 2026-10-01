"""Causal partial order and legality checker (EVENT_MODEL R-X6, R-X8; THEORY A-4).

Canonical (fine-grained) event stream per request r:
  ("P", r)  then  ("Db", r, idx) for each block its decode completes, in order,
  then ("C", r) completion.
Coarse form ("D", r) expands to its Db events followed by ("C", r); a request
with an empty decode has ("C", r) implied immediately after ("P", r) when the
coarse form is used.

Partial order `≺`:
  (P, r) ≺ (Db, r, ·) ≺ (C, r)           in index order within a request
  (C, parent(r)) ≺ (P, r)                a request follows completion of the
                                         request that produced its parent node
Sibling branches and different groups are unordered.
"""
from __future__ import annotations

from .events import Request, decode_keys


class IllegalSchedule(ValueError):
    pass


def ancestors(rid: str, by_id: dict[str, Request]) -> list[str]:
    out = []
    cur = by_id[rid].parent
    while cur is not None:
        out.append(cur)
        cur = by_id[cur].parent
    return out


def normalize(requests: list[Request], events: list[tuple], B: int) -> list[tuple]:
    """Expand coarse events into the canonical fine-grained stream."""
    by_id = {r.request_id: r for r in requests}
    out: list[tuple] = []
    coarse = any(ev[0] == "D" for ev in events) or not any(ev[0] == "C" for ev in events)
    for ev in events:
        kind, rid = ev[0], ev[1]
        r = by_id[rid]
        if kind == "P":
            out.append(("P", rid))
            if coarse and not r.decode:
                out.append(("C", rid))
        elif kind == "D":
            n_prompt = len(r.prompt) // B
            for j in range(len(decode_keys(r, B))):
                out.append(("Db", rid, n_prompt + j))
            out.append(("C", rid))
        else:
            out.append(ev)
    return out


def check_legal(requests: list[Request], events: list[tuple], B: int = 4) -> None:
    """Raise IllegalSchedule unless `events` is a linear extension of ≺ containing
    every canonical event exactly once."""
    by_id = {r.request_id: r for r in requests}
    events = normalize(requests, events, B)
    expected: set[tuple] = set()
    for r in requests:
        expected.add(("P", r.request_id))
        n_prompt = len(r.prompt) // B
        for j in range(len(decode_keys(r, B))):
            expected.add(("Db", r.request_id, n_prompt + j))
        expected.add(("C", r.request_id))
    seen: set[tuple] = set()
    last_db: dict[str, int] = {}
    for ev in events:
        if ev not in expected:
            raise IllegalSchedule(f"unexpected event {ev}")
        if ev in seen:
            raise IllegalSchedule(f"duplicate event {ev}")
        kind, rid = ev[0], ev[1]
        r = by_id[rid]
        if kind == "P":
            if r.parent is not None and ("C", r.parent) not in seen:
                raise IllegalSchedule(f"{ev} before completion of parent {r.parent}")
        elif kind == "Db":
            if ("P", rid) not in seen:
                raise IllegalSchedule(f"{ev} before its prefill")
            prev = last_db.get(rid, len(r.prompt) // B - 1)
            if ev[2] != prev + 1:
                raise IllegalSchedule(f"{ev} out of block order (expected {prev + 1})")
            last_db[rid] = ev[2]
        elif kind == "C":
            if ("P", rid) not in seen:
                raise IllegalSchedule(f"{ev} before its prefill")
            n_expected = len(decode_keys(r, B))
            n_done = last_db.get(rid, len(r.prompt) // B - 1) - (len(r.prompt) // B - 1)
            if n_done != n_expected:
                raise IllegalSchedule(f"{ev} before all its decode blocks ({n_done}/{n_expected})")
        seen.add(ev)
    missing = expected - seen
    if missing:
        raise IllegalSchedule(f"missing events {sorted(missing)}")
