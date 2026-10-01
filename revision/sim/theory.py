"""Key classes and order-invariant bounds (THEORY_DRAFT v0.2 §1-§4).

V_p    keys demanded by some prefill
V_d    keys completed by some decode (by content, schedule-independent)
V_safe keys every demanding prefill of which is causally supplied: some decode
       completing the key precedes that prefill in the causal partial order,
       i.e. the producing request is a proper ancestor of the demanding request.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .causal import ancestors
from .events import Request, decode_keys, prompt_keys


@dataclass
class KeyClasses:
    V_p: set[bytes]
    V_d: set[bytes]
    V_safe: set[bytes]

    @property
    def n_always_missed(self) -> int:
        return len(self.V_p - self.V_d)

    @property
    def n_always_hit(self) -> int:
        return len(self.V_safe)

    @property
    def n_undetermined(self) -> int:
        return len((self.V_p & self.V_d) - self.V_safe)


def key_classes(requests: list[Request], B: int) -> KeyClasses:
    by_id = {r.request_id: r for r in requests}
    demanders: dict[bytes, set[str]] = defaultdict(set)
    producers: dict[bytes, set[str]] = defaultdict(set)
    for r in requests:
        for k in prompt_keys(r, B):
            demanders[k].add(r.request_id)
        for k in decode_keys(r, B):
            producers[k].add(r.request_id)
    V_p = set(demanders)
    V_d = set(producers)
    V_safe: set[bytes] = set()
    for k in V_p & V_d:
        if all(producers[k] & set(ancestors(p, by_id)) for p in demanders[k]):
            V_safe.add(k)
    return KeyClasses(V_p, V_d, V_safe)


@dataclass
class Bounds:
    T: int
    Q: int
    raw_lower: float
    tightened_lower: float
    upper: float


def bounds(requests: list[Request], B: int) -> Bounds:
    cls = key_classes(requests, B)
    T = sum(len(r.prompt) for r in requests)
    Q = sum(len(r.prompt) % B for r in requests)
    if T == 0:
        raise ValueError("no prefill tokens")
    return Bounds(
        T=T,
        Q=Q,
        raw_lower=1 - (Q + B * len(cls.V_p)) / T,
        tightened_lower=1 - (Q + B * len(cls.V_p - cls.V_safe)) / T,
        upper=1 - (Q + B * len(cls.V_p - cls.V_d)) / T,
    )
