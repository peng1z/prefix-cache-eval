"""Hit-token attribution (EVENT_MODEL R-A1..R-A5).

register_system_prefixes: R-A3 registration over groups.
sys_len_for: per-request longest registered prefix the prompt actually begins with.
credit_bucket: R-A2 bucket for one hit block given inserter metadata.
"""
from __future__ import annotations

from collections import Counter

from .events import Request

BUCKETS = (
    "system_prefix",
    "own_path_decode",
    "own_path_prefill",
    "branch_sibling",
    "branch_out_of_order",
    "cross_group_prefill",
    "cross_group_decode",
)


def register_system_prefixes(groups: list[list[Request]], min_groups: int = 50, min_len: int = 16) -> set[tuple[int, ...]]:
    """Leading `min_len`-token heads shared by >= min_groups groups, extended to the
    longest common continuation among the groups sharing that head."""
    heads: Counter = Counter()
    first_prompt: dict[str, list[int]] = {}
    for g in groups:
        r0 = g[0]
        if len(r0.prompt) >= min_len:
            first_prompt[r0.group] = r0.prompt
            heads[tuple(r0.prompt[:min_len])] += 1
    registered: set[tuple[int, ...]] = set()
    for head, n in heads.items():
        if n < min_groups:
            continue
        members = [p for p in first_prompt.values() if tuple(p[:min_len]) == head]
        # longest common prefix among members
        lcp = list(members[0])
        for p in members[1:]:
            m = 0
            while m < len(lcp) and m < len(p) and lcp[m] == p[m]:
                m += 1
            lcp = lcp[:m]
        registered.add(tuple(lcp))
    return registered


def sys_len_for(r: Request, registered: set[tuple[int, ...]] | None) -> int:
    if r.sys_len:
        return min(r.sys_len, len(r.prompt))
    best = 0
    if not registered:
        return 0
    for pref in registered:
        n = len(pref)
        if n > best and len(r.prompt) >= n and tuple(r.prompt[:n]) == pref:
            best = n
    return best


def credit_bucket(r: Request, block_start: int, block_end: int, sys_len: int, inserter: dict) -> str:
    if sys_len and block_end <= sys_len:
        return "system_prefix"
    if inserter.get("group") == r.group:
        ipath = tuple(inserter.get("path") or ())
        if len(ipath) <= len(r.path) and r.path[: len(ipath)] == ipath:
            return "own_path_decode" if inserter.get("kind") == "decode" else "own_path_prefill"
        if len(ipath) > len(r.path) and ipath[: len(r.path)] == r.path:
            return "branch_out_of_order"
        return "branch_sibling"
    return "cross_group_decode" if inserter.get("kind") == "decode" else "cross_group_prefill"
