"""Finite-capacity tests from TESTS.md: ledger, pins, admission transaction,
leaf-only eviction, policies, infeasible requests, termination, coverage."""
import pytest

from sim.cache import BlockCache
from sim.events import Request, chain_keys
from sim.scheduler import Config, run_schedule
from sim.theory import bounds

B = 4


def blk(c):
    return [100 * c + i for i in range(B)]


def req(rid, group, prompt, decode=(), parent=None, path=None, regen=False):
    return Request(rid, group=group, path=path or (rid,), prompt=list(prompt),
                   decode=list(decode), parent=parent, is_regeneration=regen)


def cfg(**kw):
    base = dict(B=B, capacity=12, policy="LRU", K_exec=1, seed=0,
                leaf_only=True, full_hit_correction=False, selection="round_robin")
    base.update(kw)
    return Config(**base)


# ---------------------------------------------------------------- cache unit

def test_T_LEAF_candidates():
    c = BlockCache(capacity=100, policy="LRU", B=B, leaf_only=True)
    keys = chain_keys(blk(1) + blk(2) + blk(3), B)
    c.preload(keys)
    assert c.candidates() == {keys[2]}
    c2 = BlockCache(capacity=100, policy="LRU", B=B, leaf_only=False)
    c2.preload(keys)
    assert c2.candidates() == set(keys)


def test_T_LFU_COUNT_one_increment_per_event():
    c = BlockCache(capacity=100, policy="LFU", B=B, leaf_only=True)
    keys = chain_keys(blk(1) + blk(2), B)
    c.preload(keys)  # counts start at 1
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    res = run_schedule([[r]], cfg(policy="LFU", capacity=100), cache=c)
    assert res.per_request_hit["r"] == 8
    assert c.blocks[keys[0]].count == 2 and c.blocks[keys[1]].count == 2
    k3 = chain_keys(r.prompt, B)[2]
    assert c.blocks[k3].count == 1


# ------------------------------------------------------- infeasible / ledger

def test_T_E4_infeasible_request_is_excluded_and_counted():
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    res = run_schedule([[r]], cfg(capacity=8))
    assert res.total == 0 and res.missed == 0
    assert res.n_infeasible_requests == 1 and res.infeasible_prefill_tokens == 12
    assert res.coverage == 0.0


def test_T_LEDGER_reservations_count_against_capacity():
    r1 = req("r1", "X", blk(1), decode=blk(2) + blk(3))     # footprint 12
    r2 = req("r2", "Y", blk(9))                             # footprint 4
    res = run_schedule([[r1], [r2]], cfg(capacity=12, K_exec=2))
    assert res.max_ledger <= 12
    admit = [e for e in res.log if e[1] == "admit"]
    complete = [e for e in res.log if e[1] == "complete"]
    step_r2_admit = next(s for s, k, rid in admit if rid == "r2")
    step_r1_done = next(s for s, k, rid in complete if rid == "r1")
    assert step_r2_admit > step_r1_done
    assert (res.total, res.missed) == (8, 8)


def test_T_PROTECT_UNION_shared_hit_block_counted_once():
    c = BlockCache(capacity=16, policy="LRU", B=B, leaf_only=True)
    A = chain_keys(blk(1), B)
    c.preload(A)
    r1 = req("r1", "X", blk(1) + blk(2), decode=blk(3))      # hits A, inserts B, reserves 1
    r2 = req("r2", "Y", blk(1) + blk(4))                      # hits A, needs 4
    res = run_schedule([[r1], [r2]], cfg(capacity=16, K_exec=2), cache=c)
    step_r2_admit = next(s for s, k, rid in res.log if k == "admit" and rid == "r2")
    step_r1_done = next(s for s, k, rid in res.log if k == "complete" and rid == "r1")
    assert step_r2_admit < step_r1_done


def test_T_PROBE_NOMETA_failed_probe_touches_nothing():
    c = BlockCache(capacity=12, policy="LFU", B=B, leaf_only=True)
    A = chain_keys(blk(1), B)
    Bk = chain_keys(blk(2), B)
    c.preload(A, count=3)
    c.preload(Bk, count=1)
    r1 = req("r1", "X", blk(2), decode=blk(5))   # pins B, reserves 4 -> ledger 12
    r2 = req("r2", "Y", blk(1), decode=blk(6))   # hits A, needs 4 -> must wait
    seen = {}

    def observer(step, sched):
        if step == 2:  # r2 probed and refused at step 2 (round robin)
            seen["count_A"] = c.blocks[A[0]].count
            seen["B_resident"] = Bk[0] in c.blocks
    res = run_schedule([[r1], [r2]], cfg(capacity=12, K_exec=2, policy="LFU"), cache=c, observer=observer)
    assert seen == {"count_A": 3, "B_resident": True}
    assert c.blocks[A[0]].count == 4  # incremented exactly once, at r2's real admission


# ----------------------------------------------------------- self-eviction

def test_T_E5_LFU_no_self_eviction():
    c = BlockCache(capacity=12, policy="LFU", B=B, leaf_only=True)
    H1, H2 = chain_keys(blk(7), B), chain_keys(blk(8), B)
    c.preload(H1, count=5)
    c.preload(H2, count=5)
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    run_schedule([[r]], cfg(capacity=12, policy="LFU"), cache=c)
    assert set(c.blocks) == set(chain_keys(r.prompt, B))


def test_T_E5_FIFO_hit_prefix_is_pinned():
    c = BlockCache(capacity=12, policy="FIFO", B=B, leaf_only=True)
    P1 = chain_keys(blk(1), B)
    X, Y = chain_keys(blk(8), B), chain_keys(blk(9), B)
    c.preload(P1); c.preload(X); c.preload(Y)  # insertion order P1, X, Y
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    res = run_schedule([[r]], cfg(capacity=12, policy="FIFO"), cache=c)
    assert res.per_request_hit["r"] == 4
    assert set(c.blocks) == set(chain_keys(r.prompt, B))


def test_T_E5_LRU_2SLOT_pins_survive_concurrent_request():
    c = BlockCache(capacity=12, policy="LRU", B=B, leaf_only=True)
    AB = chain_keys(blk(1) + blk(2), B)
    c.preload(AB)
    r1 = req("r1", "X", blk(1) + blk(2), decode=blk(3))
    r2 = req("r2", "Y", blk(9))
    resident_ok = []

    def observer(step, sched):
        resident_ok.append(all(k in c.blocks for k in AB))
    res = run_schedule([[r1], [r2]], cfg(capacity=12, K_exec=2), cache=c, observer=observer)
    assert all(resident_ok)
    assert res.per_request_hit["r1"] == 8
    assert (res.total, res.missed) == (12, 4)
    D = chain_keys(blk(1) + blk(2) + blk(3), B)[2]
    assert D not in c.blocks  # the only leaf, evicted for r2


# ----------------------------------------------------------- partial slots

def test_T_PARTIAL_fill_converts_without_allocation():
    r = req("r", "g", [1, 2, 3, 4, 5, 6], decode=[7, 8])  # footprint 8
    res = run_schedule([[r]], cfg(capacity=8))
    assert res.n_infeasible_requests == 0 and res.max_ledger == 8
    assert (res.total, res.missed) == (6, 6)


def test_T_PARTIAL_third_block_needs_reservation():
    r = req("r", "g", [1, 2, 3, 4, 5, 6], decode=[7, 8, 9])  # footprint 12
    assert run_schedule([[r]], cfg(capacity=8)).n_infeasible_requests == 1
    res = run_schedule([[r]], cfg(capacity=12))
    assert res.n_infeasible_requests == 0 and res.max_ledger == 12 and res.n_evictions == 0


# ------------------------------------------------------------- termination

def test_T_TERM_infeasible_middle_group_and_waiting_group():
    g0 = [req("a", "g0", blk(1) + blk(2))]
    g1 = [req("b", "g1", blk(3) + blk(4) + blk(5))]
    g2 = [req("c", "g2", blk(6) + blk(7))]
    res = run_schedule([g0, g1, g2], cfg(capacity=8, K_exec=2))
    assert res.n_infeasible_requests == 1 and (res.total, res.missed) == (16, 16)
    h0 = [req("a", "g0", [1, 2, 3, 4, 5, 6], decode=[7, 8])]
    h1 = [req("b", "g1", [11, 12, 13, 14, 15, 16])]
    res = run_schedule([h0, h1], cfg(capacity=8, K_exec=2))
    assert res.n_infeasible_requests == 0 and (res.total, res.missed) == (12, 12)


# ----------------------------------------------------------- regeneration

def _tree():
    u1, a1, u2, a2 = blk(1), blk(2), blk(3), blk(4)
    r1 = req("r1", "t", u1, decode=a1, path=("u1",))
    r2 = req("r2", "t", u1 + a1 + u2, parent="r1", path=("u1", "a1", "u2"))
    r3 = req("r3", "t", u1, decode=a2, path=("u1",), regen=True)
    return r1, r2, r3


def test_T_REGEN_INFEASIBLE_sibling_still_runs():
    r1, r2, r3 = _tree()
    res = run_schedule([[r1, r2, r3]], cfg(capacity=8))
    assert res.n_infeasible_requests == 1 and res.infeasible_prefill_tokens == 12
    assert res.coverage == pytest.approx(8 / 20)
    assert res.per_request_hit["r3"] == 4
    assert (res.total, res.missed) == (8, 4)
    assert res.strata["regeneration"] == {"n": 1, "total": 4, "hit": 4, "missed": 0}


def test_T_REGEN_REUSE_leaf_only_keeps_ancestor():
    r1, r2, _ = _tree()
    w = req("w", "w", blk(8) + blk(9))
    r3 = req("r3", "t2", blk(1), decode=blk(4), regen=True)
    leaf = run_schedule([[r1, r2], [w], [r3]], cfg(capacity=16, leaf_only=True))
    anyb = run_schedule([[r1, r2], [w], [r3]], cfg(capacity=16, leaf_only=False))
    assert leaf.per_request_hit["r3"] == 4
    assert anyb.per_request_hit["r3"] == 0


# ------------------------------------------------------------- K_exec = 1

def test_T_K1_uniform_equals_round_robin_equals_dfs():
    groups = [[req("a1", "A", blk(1), decode=blk(2)), req("a2", "A", blk(1) + blk(2) + blk(3), parent="a1")],
              [req("b1", "B", blk(1) + blk(5))]]
    u = run_schedule(groups, cfg(capacity=10**9, selection="uniform"))
    r = run_schedule(groups, cfg(capacity=10**9, selection="round_robin"))
    assert u.log == r.log
    assert [rid for _, k, rid in u.log if k == "admit"] == ["a1", "a2", "b1"]


# --------------------------------------------------------------- coverage

def test_T_COVERAGE_executable_set_and_comparability():
    g0 = [req("a", "g0", blk(1) + blk(2))]
    g1 = [req("b", "g1", blk(3) + blk(4) + blk(5))]
    small = run_schedule([g0, g1], cfg(capacity=8))
    big = run_schedule([g0, g1], cfg(capacity=16))
    assert small.coverage == pytest.approx(8 / 20) and big.coverage == 1.0
    assert small.executable_request_ids == {"a"} and big.executable_request_ids == {"a", "b"}
    assert not small.comparable_with(big)
    b = bounds([r for g in [g0, g1] for r in g if r.request_id in small.executable_request_ids], B=B)
    assert b.upper == 0.0  # the executable set alone: two unique blocks, all missed
