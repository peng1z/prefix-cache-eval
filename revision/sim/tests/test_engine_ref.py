"""Engine comparison (vendored vLLM V1 prefix-cache core, commit 8f816a3f…): the
real KVCacheManager must agree with the simulator on hand-computed cases and on
random synthetic groups at infinite capacity under the serial schedule."""
import random

from engine_ref.harness import run_serial
from sim.events import Request
from sim.scheduler import Config, run_schedule
from sim.tests.test_invariants import make_groups

B = 4


def blk(c):
    return [100 * c + i for i in range(B)]


def _sim(groups, cap):
    return run_schedule(groups, Config(B=B, capacity=cap, K_exec=1, policy="LRU", full_hit_correction=True))


def test_E1_and_E3_match_vllm_core():
    A, Bb, C = blk(1), blk(2), blk(3)
    r1 = Request("r1", group="g", path=("n1",), prompt=A, decode=Bb, parent=None)
    r2 = Request("r2", group="g", path=("n1", "n2"), prompt=A + Bb + C, decode=[], parent="r1")
    ref, sim = run_serial([[r1, r2]], None, B), _sim([[r1, r2]], None)
    assert ref["per_request_hit"] == sim.per_request_hit and (ref["total"], ref["missed"]) == (sim.total, sim.missed)
    a, b, u = [1, 2, 3, 4, 5, 6], [11, 12, 13, 14, 15], [21, 22, 23]
    s1 = Request("s1", group="h", path=("m1",), prompt=a, decode=b, parent=None)
    s2 = Request("s2", group="h", path=("m1", "m2"), prompt=a + b + u, decode=[], parent="s1")
    ref, sim = run_serial([[s1, s2]], None, B), _sim([[s1, s2]], None)
    assert ref["per_request_hit"] == sim.per_request_hit and (ref["total"], ref["missed"]) == (sim.total, sim.missed)


def test_full_hit_rule_matches_vllm_core():
    A = blk(1)
    t1 = Request("t1", group="x", path=("x1",), prompt=A, decode=[], parent=None)
    t2 = Request("t2", group="y", path=("y1",), prompt=A, decode=[], parent=None)
    ref, sim = run_serial([[t1], [t2]], None, B), _sim([[t1], [t2]], None)
    assert ref["per_request_hit"] == sim.per_request_hit == {"t1": 0, "t2": 0}


def test_random_groups_infinite_capacity_serial_match_vllm_core():
    rng = random.Random(5)
    n = 0
    for trial in range(40):
        reqs = make_groups(rng, n_groups=5)
        groups = {}
        for r in reqs:
            groups.setdefault(r.group, []).append(r)
        # vLLM cannot take an empty prompt (the D-LEAD `decode` condition writes such a
        # request's decode to the cache); compare only groups without one
        glist = [g for g in groups.values() if all(len(r.prompt) > 0 for r in g)]
        reqs = [r for g in glist for r in g]
        ref, sim = run_serial(glist, None, B), _sim(glist, None)
        nonempty = {r.request_id for r in reqs if len(r.prompt) > 0}
        assert {k: v for k, v in sim.per_request_hit.items() if k in nonempty} == ref["per_request_hit"], trial
        assert (ref["total"], ref["missed"]) == (sim.total, sim.missed), trial
        n += len(nonempty)
    assert n > 300
