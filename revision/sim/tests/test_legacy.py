"""Historical diagnostic switches (EVENT_MODEL §10, plan item F). They reproduce the
submitted simulator's behaviour inside the frozen core and are used only in the
diagnostic path."""
from sim.cache import BlockCache
from sim.events import Request, chain_keys
from sim.ingest import build_groups, render
from sim.scheduler import Config, run_schedule
from sim.tests.test_ingest import FakeTok, rec_linear, rec_tree

B = 4


def blk(c):
    return [100 * c + i for i in range(B)]


def req(rid, group, prompt):
    return Request(rid, group=group, path=(rid,), prompt=list(prompt), decode=[], parent=None)


def cfg(**kw):
    base = dict(B=B, capacity=12, policy="LFU", K_exec=1, seed=0, leaf_only=False,
                full_hit_correction=False, selection="round_robin")
    base.update(kw)
    return Config(**base)


def test_legacy_no_pin_reproduces_LFU_self_eviction():
    """Old code: cache full of count-5 blocks; inserting N1 N2 N3 one block at a time
    evicts N1 then N2 (frequency 1, oldest) -> cache ends {H1, H2, N3}."""
    c = BlockCache(capacity=12, policy="LFU", B=B, leaf_only=False)
    H1, H2 = chain_keys(blk(7), B), chain_keys(blk(8), B)
    c.preload(H1, count=5); c.preload(H2, count=5)
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    res = run_schedule([[r]], cfg(legacy_no_pin=True), cache=c)
    keys = chain_keys(r.prompt, B)
    assert set(c.blocks) == {H1[0], H2[0], keys[2]}
    assert res.per_request_hit["r"] == 0


def test_legacy_no_pin_FIFO_evicts_own_hit_prefix():
    c = BlockCache(capacity=12, policy="FIFO", B=B, leaf_only=False)
    P1 = chain_keys(blk(1), B); X, Y = chain_keys(blk(8), B), chain_keys(blk(9), B)
    c.preload(P1, step=0); c.preload(X, step=1); c.preload(Y, step=2)
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    res = run_schedule([[r]], cfg(policy="FIFO", legacy_no_pin=True), cache=c)
    assert res.per_request_hit["r"] == 4          # hit counted before eviction
    assert P1[0] not in c.blocks                   # then evicted by its own tail insertion


def test_legacy_lfu_double_count():
    c = BlockCache(capacity=100, policy="LFU", B=B, leaf_only=False)
    keys = chain_keys(blk(1) + blk(2), B)
    c.preload(keys)
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    run_schedule([[r]], cfg(capacity=100, legacy_lfu_double_count=True), cache=c)
    assert c.blocks[keys[0]].count == 3 and c.blocks[keys[1]].count == 3   # 1 + 2 increments


def test_legacy_assistant_prefill_request_construction():
    tok = FakeTok()
    rec = rec_linear("c", [("user", "q"), ("assistant", "a"), ("user", "q2"), ("assistant", "b")])
    (g,), st = build_groups([rec], tok, legacy_assistant_prefill=True)
    assert [len(r.decode) for r in g] == [0, 0, 0, 0]
    U1, A1, U2, A2 = (tok.encode(render(r, t)) for r, t in [("user", "q"), ("assistant", "a"), ("user", "q2"), ("assistant", "b")])
    assert [r.prompt for r in g] == [U1, U1 + A1, U1 + A1 + U2, U1 + A1 + U2 + A2]
    assert [r.parent for r in g] == [None, "c::0", "c::1", "c::2"]
    tree = rec_tree("t", [("u1", "None", "user", "q"), ("a1", "u1", "assistant", "r1"), ("u2", "a1", "user", "q2")])
    (gt,), _ = build_groups([tree], tok, legacy_assistant_prefill=True)
    assert len(gt) == 3 and all(not r.decode for r in gt)   # one prefill per node, as the old walk


def test_legacy_switches_are_off_by_default_and_flagged_in_config():
    c = cfg()
    assert c.legacy_no_pin is False and c.legacy_lfu_double_count is False


def test_legacy_per_block_alloc_with_protection_keeps_own_blocks():
    """Seventh review: separate the two things legacy_no_pin changes. Per-block insertion
    with eviction on demand (old allocation mode) but with the current request's hit and
    inserted blocks protected: the LFU self-eviction must not occur."""
    c = BlockCache(capacity=12, policy="LFU", B=B, leaf_only=False)
    H1, H2 = chain_keys(blk(7), B), chain_keys(blk(8), B)
    c.preload(H1, count=5); c.preload(H2, count=5)
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    res = run_schedule([[r]], cfg(legacy_per_block_alloc=True), cache=c)
    assert set(c.blocks) == set(chain_keys(r.prompt, B))      # H1, H2 evicted; own blocks kept
    assert res.per_request_hit["r"] == 0


def test_legacy_per_block_alloc_FIFO_keeps_hit_prefix():
    c = BlockCache(capacity=12, policy="FIFO", B=B, leaf_only=False)
    P1 = chain_keys(blk(1), B); X, Y = chain_keys(blk(8), B), chain_keys(blk(9), B)
    c.preload(P1, step=0); c.preload(X, step=1); c.preload(Y, step=2)
    r = req("r", "g", blk(1) + blk(2) + blk(3))
    res = run_schedule([[r]], cfg(policy="FIFO", legacy_per_block_alloc=True), cache=c)
    assert res.per_request_hit["r"] == 4 and P1[0] in c.blocks
