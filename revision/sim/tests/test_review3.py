"""Tests added after the third review (FINDINGS F2 conditions, F5, F6 and the
two implementation gaps): full-hit order dependence, coverage of dead
descendants, orphan re-insertion under any_unpinned, decode_path attribution,
per-row OpenOrca sys_len, OASST path reading, exact dedup, leading-assistant
counters and exclusion."""
import pytest

from sim.cache import BlockCache
from sim.counter import count_infinite
from sim.events import Request, chain_keys
from sim.ingest import build_groups, exact_dedup, render
from sim.scheduler import Config, run_schedule
from sim.tests.test_ingest import FakeTok, rec_linear, rec_tree

B = 4


def blk(c):
    return [100 * c + i for i in range(B)]


def req(rid, group, prompt, decode=(), parent=None, path=None, regen=False, decode_path=None):
    kw = {}
    if decode_path is not None:
        kw["decode_path"] = decode_path
    return Request(rid, group=group, path=path or (rid,), prompt=list(prompt), decode=list(decode),
                   parent=parent, is_regeneration=regen, **kw)


def cfg(**kw):
    base = dict(B=B, capacity=None, policy="LRU", K_exec=1, seed=0, leaf_only=True,
                full_hit_correction=False, selection="round_robin")
    base.update(kw)
    return Config(**base)


# ---------------------------------------------------------------- F2 condition

def test_T_FULLHIT_ORDER_correction_breaks_order_invariance():
    a = req("a", "ga", blk(1))
    ab = req("ab", "gab", blk(1) + blk(2))
    off1 = count_infinite([a, ab], [("P", "a"), ("P", "ab")], B=B)
    off2 = count_infinite([a, ab], [("P", "ab"), ("P", "a")], B=B)
    assert off1.reuse == off2.reuse == pytest.approx(1 / 3)
    on1 = count_infinite([a, ab], [("P", "a"), ("P", "ab")], B=B, full_hit_correction=True)
    on2 = count_infinite([a, ab], [("P", "ab"), ("P", "a")], B=B, full_hit_correction=True)
    assert on1.reuse == pytest.approx(1 / 3) and on2.reuse == 0.0


# ------------------------------------------------------- implementation gaps

def test_T_COVERAGE_DESC_dead_descendants_count_as_excluded():
    r1 = req("r1", "g", blk(1))                                     # 4, feasible
    r2 = req("r2", "g", blk(1) + blk(2) + blk(3), parent="r1")      # 12, infeasible at C=8
    r3 = req("r3", "g", blk(1) + blk(2) + blk(3) + blk(4), parent="r2")  # 16, dead descendant
    res = run_schedule([[r1, r2, r3]], cfg(capacity=8))
    assert res.total == 4
    assert res.infeasible_prefill_tokens == 28
    assert res.n_infeasible_requests == 2
    assert res.coverage == pytest.approx(4 / 32)


def test_T_ORPHAN_any_unpinned_reinsertion_of_evicted_ancestor_does_not_crash():
    c = BlockCache(capacity=12, policy="LFU", B=B, leaf_only=False)
    AB = chain_keys(blk(1) + blk(2), B)
    c.preload(AB[:1], count=1)
    c.preload(AB, count=5)          # B gets count 5, A stays 1
    x = req("x", "gx", blk(8))
    y = req("y", "gy", blk(9))      # forces eviction of A (count 1, oldest) -> B orphaned
    z = req("z", "gz", blk(1) + blk(2))  # re-requests A B
    res = run_schedule([[x], [y], [z]], cfg(capacity=12, policy="LFU", leaf_only=False), cache=c)
    assert res.per_request_hit["z"] == 0     # orphan B is not matchable without A
    assert all(k in c.blocks for k in AB)
    assert c.blocks[AB[0]].pins == 0 and c.blocks[AB[1]].pins == 0


# ---------------------------------------------------------- decode_path (R-A1)

def test_T_A5_REGEN_SIBLING_decode_blocks_credited_to_their_branch():
    u1, x, u3 = blk(1), blk(2), [301, 302, 303]
    a1 = x                      # first reply: exactly one block
    a2 = x + [999]              # regeneration: same first block, then more
    r1 = req("r1", "t", u1, decode=a1, path=("u1",), decode_path=("u1", "a1"))
    r3 = req("r3", "t", u1, decode=a2, path=("u1",), regen=True, decode_path=("u1", "a2"))
    r4 = req("r4", "t", u1 + a2 + u3, parent="r3", path=("u1", "a2", "u3"))
    res = run_schedule([[r1, r3, r4]], cfg())
    # r4 hits K1 (own-path prefill, inserted by r1's prefill) and K2 (inserted by r1's DECODE on
    # branch a1, reused on branch a2 -> sibling), not own_path_decode
    assert res.per_request_hit["r4"] == 8
    assert res.attribution["own_path_prefill"] >= 4
    assert res.attribution["branch_sibling"] == 4
    assert res.attribution["own_path_decode"] == 0


def test_ingest_sets_decode_path_for_linear_and_tree():
    tok = FakeTok()
    (g,), _ = build_groups([rec_linear("c", [("user", "q"), ("assistant", "a")])], tok)
    assert g[0].decode_path == g[0].path + ("c::a0",)
    rec = rec_tree("t", [("u1", "None", "user", "q"), ("a1", "u1", "assistant", "r1"),
                         ("a2", "u1", "assistant", "r2")])
    (g,), _ = build_groups([rec], tok)
    assert [r.decode_path for r in g] == [("u1", "a1"), ("u1", "a2")]


# ------------------------------------------------------- per-row sys_len (R-A3)

def test_T_SYSLEN_ROW_openorca_system_span_from_row_field():
    class Tok16:
        def encode(self, s):
            return [hash(s) % 1000 + i for i in range(16)]  # every piece is one 16-token block
    tok = Tok16()
    rows = [{"conv_id": f"o{i}", "kind": "flat", "timestamp": None,
             "messages": [{"role": "system", "text": "same system"}, {"role": "user", "text": f"q{i}"}]}
            for i in range(2)]
    groups, _ = build_groups(rows, tok)
    assert groups[0][0].sys_len == 16
    res = run_schedule(groups, cfg(B=16))
    assert res.attribution["system_prefix"] == 16 and res.attribution["cross_group_prefill"] == 0


# ------------------------------------------------------------ path reading

def test_T_PATH_READING_splits_tree_into_root_to_leaf_groups():
    tok = FakeTok()
    rec = rec_tree("t", [("u1", "None", "user", "q"), ("a1", "u1", "assistant", "r1"),
                         ("a2", "u1", "assistant", "r2"), ("u2", "a1", "user", "q2")])
    groups, st = build_groups([rec], tok, tree_reading="path")
    assert len(groups) == 2
    assert len({g[0].group for g in groups}) == 2
    lens = sorted(len(g) for g in groups)
    assert lens == [1, 2]
    two = next(g for g in groups if len(g) == 2)
    assert two[1].parent == two[0].request_id and two[0].group == two[1].group
    rids = [r.request_id for g in groups for r in g]
    assert len(set(rids)) == 3


# ----------------------------------------------------------------- dedup (F5)

def test_T_DEDUP_exact_on_canonical_bytes_independent_of_id():
    a = rec_linear("id1", [("user", "q"), ("assistant", "a")])
    b = rec_linear("id2", [("user", "q"), ("assistant", "a")])      # same text, other id
    c = rec_linear("id1", [("user", "z"), ("assistant", "y")])      # same id, other text
    kept, man = exact_dedup([a, b, c])
    assert [r["conv_id"] for r in kept] == ["id1", "id1"]
    assert man["n_in"] == 3 and man["n_out"] == 2 and man["n_dropped"] == 1
    assert man["rule"].startswith("sha256(canonical")
    assert len(man["kept_ids_sha256"]) == 64


# ------------------------------------------------- leading assistant (F6)

def test_T_LEADING_ASSISTANT_counters_and_exclusion():
    tok = FakeTok()
    recs = [
        rec_linear("lead", [("assistant", "hello"), ("user", "q"), ("assistant", "a")]),
        rec_linear("consec", [("user", "q"), ("assistant", "a"), ("assistant", "b"), ("user", "q2")]),
        rec_linear("sys", [("system", "s"), ("assistant", "hi"), ("user", "q")]),
        rec_linear("ok", [("user", "q"), ("assistant", "a")]),
    ]
    groups, st = build_groups(recs, tok)
    assert st["n_groups_leading_assistant"] == 1
    assert st["n_consecutive_assistant"] == 1
    assert st["n_unpaired_assistant_after_system"] == 1
    assert st["n_unpaired_assistant"] == 3     # trigger count, kept for comparison
    groups2, st2 = build_groups(recs, tok, leading_assistant="exclude")
    assert [g[0].group for g in groups2] == ["consec", "ok"]
    assert st2["n_groups_excluded_leading_assistant"] == 2


def test_linear_next_turn_path_extends_decode_path_so_decode_blocks_are_own_path():
    tok = FakeTok()
    (g,), _ = build_groups([rec_linear("c", [("user", "q"), ("assistant", "a"), ("user", "q2"), ("assistant", "b")])], tok)
    r0, r1 = g
    assert r1.path[: len(r0.decode_path)] == r0.decode_path
    # B must divide the pieces so that a decode block exists: use a tokenizer-free check via scheduler
    from sim.events import Request as R
    u1, a1, u2 = blk(1), blk(2), blk(3)
    x0 = R("x0", group="x", path=("u0",), prompt=u1, decode=a1, parent=None, decode_path=("u0", "a0"))
    x1 = R("x1", group="x", path=("u0", "a0", "u1"), prompt=u1 + a1 + u2, decode=[], parent="x0")
    res = run_schedule([[x0, x1]], cfg())
    assert res.attribution["own_path_decode"] == 4 and res.attribution["branch_sibling"] == 0


# ------------------------------------------------ fourth review: orphan capacity

def test_T_ORPHAN_CAP_protected_orphan_is_not_evicted_during_make_room():
    """B=4, C=12, FIFO, any_unpinned. Resident: orphan B (parent A absent), X, Y.
    Request A B: only A is new (need 4); B must be protected from the eviction
    that makes room, otherwise A and B are both inserted and the ledger hits 16."""
    c = BlockCache(capacity=12, policy="FIFO", B=B, leaf_only=False)
    AB = chain_keys(blk(1) + blk(2), B)
    c.preload(AB, step=0)                     # A (step 0), B (step 1)
    X, Y = chain_keys(blk(8), B), chain_keys(blk(9), B)
    c.preload(X, step=2); c.preload(Y, step=3)
    # evict A by hand to create the orphan (any_unpinned lets a request do this too)
    c.evict_one(protected={AB[1], X[0], Y[0]})
    assert AB[0] not in c.blocks and AB[1] in c.blocks
    z = req("z", "gz", blk(1) + blk(2))
    res = run_schedule([[z]], cfg(capacity=12, policy="FIFO", leaf_only=False), cache=c)
    assert res.max_ledger <= 12
    assert res.per_request_hit["z"] == 0     # orphan is protected but never a hit
    assert all(k in c.blocks for k in AB)
    assert X[0] not in c.blocks              # FIFO victim among candidates {X, Y}


# --------------------------------------- fourth review: path-reading regeneration label

def test_T_PATH_READING_first_generation_on_a_path_is_not_a_regeneration():
    tok = FakeTok()
    rec = rec_tree("t", [("u1", "None", "user", "q"), ("a1", "u1", "assistant", "r1"),
                         ("a2", "u1", "assistant", "r2"), ("u2", "a1", "user", "q2")])
    groups, _ = build_groups([rec], tok, tree_reading="path")
    assert all(not r.is_regeneration for g in groups for r in g)
    ranks = sorted(g[0].tree_sibling_rank for g in groups)
    assert ranks == [0, 1]                    # branch identity kept as a source field
    groups_t, _ = build_groups([rec], tok)   # regeneration reading unchanged
    assert [r.is_regeneration for r in groups_t[0]] == [False, False, True]
