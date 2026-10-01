"""Glue: the scheduler at infinite capacity must agree, field by field, with the
abstract counter fed the scheduler's own emitted event order; and that order
must pass the legality checker. Random synthetic groups, both selection rules,
several K_exec."""
import random

from sim.causal import check_legal
from sim.counter import count_infinite
from sim.scheduler import Config, run_schedule
from sim.tests.test_invariants import make_groups


def test_scheduler_infinite_matches_abstract_counter_on_its_own_order():
    rng = random.Random(7)
    n = 0
    for trial in range(120):
        reqs = make_groups(rng, n_groups=5)
        groups = {}
        for r in reqs:
            groups.setdefault(r.group, []).append(r)
        glist = list(groups.values())
        by_id = {r.request_id: r for r in reqs}
        for K in (1, 2, 4):
            for sel in ("uniform", "round_robin"):
                for fh in (False, True):
                    cfg = Config(B=4, capacity=None, K_exec=K, seed=trial, selection=sel,
                                 full_hit_correction=fh)
                    res = run_schedule(glist, cfg)
                    evs = res.events
                    check_legal(reqs, evs, B=4)
                    ref = count_infinite(reqs, evs, B=4, full_hit_correction=fh)
                    assert (res.total, res.missed) == (ref.total, ref.missed), (trial, K, sel, fh)
                    assert res.n_full_hit_corrections == ref.n_full_hit_corrections
                    assert res.n_empty_prefill == ref.n_empty_prefill
                    assert res.per_request_hit == ref.per_request_hit
                    n += 1
    assert n == 120 * 12
