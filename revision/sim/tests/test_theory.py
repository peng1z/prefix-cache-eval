"""V_p, V_d, V_safe, bounds and classes (THEORY_DRAFT §1-§4; TESTS T-CLASS)."""
from sim.events import Request
from sim.theory import key_classes, bounds
from sim.tests.test_counting import A, Bb, C, B, _e2, e7_requests


def _e1():
    r1 = Request("r1", group="g", path=("n1",), prompt=A, decode=Bb, parent=None)
    r2 = Request("r2", group="g", path=("n1", "n2"), prompt=A + Bb + C, decode=[], parent="r1")
    return [r1, r2]


def test_T_CLASS_E1():
    cls = key_classes(_e1(), B=B)
    assert (cls.n_always_missed, cls.n_always_hit, cls.n_undetermined) == (2, 1, 0)


def test_T_CLASS_E2():
    cls = key_classes(_e2(), B=B)
    assert (cls.n_always_missed, cls.n_always_hit, cls.n_undetermined) == (2, 0, 1)


def test_T_CLASS_E7():
    cls = key_classes(e7_requests(), B=4)
    assert (cls.n_always_missed, cls.n_always_hit, cls.n_undetermined) == (1, 0, 1)


def test_bounds_E1_raw_and_tightened():
    b = bounds(_e1(), B=B)
    assert b.T == 4 * B and b.Q == 0
    assert (b.raw_lower, b.upper) == (0.25, 0.5)
    assert (b.tightened_lower, b.upper) == (0.5, 0.5)


def test_bounds_E2():
    b = bounds(_e2(), B=B)
    assert (b.raw_lower, b.tightened_lower, b.upper) == (0.25, 0.25, 0.5)


def test_bounds_E7():
    b = bounds(e7_requests(), B=4)
    assert b.T == 17 and b.Q == 1
    assert abs(b.raw_lower - (1 - 9 / 17)) < 1e-12
    assert abs(b.tightened_lower - (1 - 9 / 17)) < 1e-12
    assert abs(b.upper - (1 - 5 / 17)) < 1e-12
