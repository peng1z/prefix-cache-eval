"""Hand-computed tests from revision/TESTS.md (counting section).

Token ids are small ints; distinct letters in TESTS.md are distinct blocks of
B ints. B=4 throughout so that examples are hand-checkable.
"""
from sim.events import Request
from sim.counter import count_infinite

B = 4


def blk(c: int) -> list[int]:
    """A distinct full block: B ints starting at 100*c."""
    return [100 * c + i for i in range(B)]


A, Bb, C = blk(1), blk(2), blk(3)


def test_T_E1_single_path_decode_supplied_block():
    r1 = Request("r1", group="g", path=("n1",), prompt=A, decode=Bb, parent=None)
    r2 = Request("r2", group="g", path=("n1", "n2"), prompt=A + Bb + C, decode=[], parent="r1")
    res = count_infinite([r1, r2], [("P", "r1"), ("D", "r1"), ("P", "r2")], B=B)
    assert res.total == 4 * B
    assert res.missed == 2 * B
    assert res.reuse == 0.5


def _e2():
    x = Request("x", group="X", path=("x1",), prompt=A, decode=Bb, parent=None)
    z = Request("z", group="Z", path=("z1",), prompt=A + Bb + C, decode=[], parent=None)
    return [x, z]


def test_T_E2_x_first_gives_50pct():
    res = count_infinite(_e2(), [("P", "x"), ("D", "x"), ("P", "z")], B=B)
    assert (res.total, res.missed) == (4 * B, 2 * B)


def test_T_E2_z_first_gives_25pct():
    res = count_infinite(_e2(), [("P", "z"), ("P", "x"), ("D", "x")], B=B)
    assert (res.total, res.missed) == (4 * B, 3 * B)


def test_T_E3_decode_completes_partial_block():
    a = [1, 2, 3, 4, 5, 6]
    b = [11, 12, 13, 14, 15]
    u = [21, 22, 23]
    r1 = Request("r1", group="g", path=("n1",), prompt=a, decode=b, parent=None)
    r2 = Request("r2", group="g", path=("n1", "n2"), prompt=a + b + u, decode=[], parent="r1")
    res = count_infinite([r1, r2], [("P", "r1"), ("D", "r1"), ("P", "r2")], B=4)
    assert (res.total, res.missed) == (20, 12)
    assert res.per_request_hit == {"r1": 0, "r2": 8}


def e7_requests():
    u1 = [1, 2, 3, 4]
    a1 = [11, 12]
    u2 = [21, 22, 23]
    a2 = [11, 12, 21, 22, 31]
    r1 = Request("r1", group="t", path=("u1",), prompt=u1, decode=a1, parent=None)
    r2 = Request("r2", group="t", path=("u1", "a1", "u2"), prompt=u1 + a1 + u2, decode=[], parent="r1")
    r3 = Request("r3", group="t", path=("u1",), prompt=u1, decode=a2, parent=None, is_regeneration=True)
    return [r1, r2, r3]


E7_BRANCH1_FIRST = [("P", "r1"), ("D", "r1"), ("P", "r2"), ("P", "r3"), ("D", "r3")]
E7_BRANCH2_FIRST = [("P", "r1"), ("P", "r3"), ("D", "r3"), ("D", "r1"), ("P", "r2")]


def test_T_E7_branch1_first():
    res = count_infinite(e7_requests(), E7_BRANCH1_FIRST, B=4)
    assert (res.total, res.missed) == (17, 9)


def test_T_E7_branch2_first():
    res = count_infinite(e7_requests(), E7_BRANCH2_FIRST, B=4)
    assert (res.total, res.missed) == (17, 5)


def test_T_FULLHIT_E7_correction_adds_exactly_B_per_full_hit():
    for order, missed_off in ((E7_BRANCH1_FIRST, 9), (E7_BRANCH2_FIRST, 5)):
        off = count_infinite(e7_requests(), order, B=4, full_hit_correction=False)
        on = count_infinite(e7_requests(), order, B=4, full_hit_correction=True)
        assert off.missed == missed_off
        assert on.n_full_hit_corrections == 1
        assert on.missed == off.missed + 4 * on.n_full_hit_corrections


def test_T_EMPTY_prefill_contributes_nothing_and_is_never_corrected():
    r = Request("r", group="g", path=("n",), prompt=[], decode=A, parent=None)
    res = count_infinite([r], [("P", "r"), ("D", "r")], B=B, full_hit_correction=True)
    assert res.total == 0 and res.missed == 0
    assert res.n_empty_prefill == 1
    assert res.n_full_hit_corrections == 0
    assert res.per_request_hit["r"] == 0
    assert res.reuse is None


def test_T_FREEZE_hits_are_computed_before_own_insertion():
    r = Request("r", group="g", path=("n",), prompt=A + Bb, decode=[], parent=None)
    res = count_infinite([r], [("P", "r")], B=B)
    assert res.per_request_hit["r"] == 0


def test_fine_grained_decode_blocks_equal_atomic_decode_when_uninterleaved():
    reqs = e7_requests()
    coarse = count_infinite(reqs, E7_BRANCH1_FIRST, B=4)
    fine = [("P", "r1"), ("C", "r1"), ("P", "r2"), ("C", "r2"), ("P", "r3"), ("Db", "r3", 1), ("C", "r3")]
    res = count_infinite(reqs, fine, B=4)
    assert (res.total, res.missed) == (coarse.total, coarse.missed)


def test_fine_grained_block_completion_is_visible_before_completion_event():
    # Group X decodes two blocks; Z's prefill runs after X's first block completes
    # but before X's decode finishes. Z hits that first block.
    x = Request("x", group="X", path=("x",), prompt=A, decode=Bb + C, parent=None)
    z = Request("z", group="Z", path=("z",), prompt=A + Bb, decode=[], parent=None)
    fine = [("P", "x"), ("Db", "x", 1), ("P", "z"), ("Db", "x", 2), ("C", "x"), ("C", "z")]
    res = count_infinite([x, z], fine, B=B)
    assert res.per_request_hit["z"] == 2 * B
