"""Randomised refutation test of THEORY Prop. 1 / Cor. 2 / Prop. 3 on synthetic
groups (linear and tree), over random legal schedules. A failure is either an
implementation bug or a counterexample to the theory; either stops phase 1."""
import random

from sim.causal import check_legal
from sim.counter import count_infinite
from sim.events import Request
from sim.theory import bounds


def _rand_tokens(rng, n, vocab):
    return [rng.randrange(vocab) for _ in range(n)]


def make_groups(rng, n_groups=6, vocab=3, B=4):
    """Small vocab so that blocks collide across groups (cross-group sharing)."""
    reqs = []
    for g in range(n_groups):
        # root prompt
        root = Request(f"g{g}r0", group=f"g{g}", path=(f"g{g}n0",),
                       prompt=_rand_tokens(rng, rng.randrange(0, 9), vocab),
                       decode=_rand_tokens(rng, rng.randrange(0, 9), vocab), parent=None)
        reqs.append(root)
        frontier = [root]
        for i in range(1, rng.randrange(1, 5)):
            par = rng.choice(frontier)
            if par.decode and rng.random() < 0.3:
                # regeneration at the same prefill point
                r = Request(f"g{g}r{i}", group=f"g{g}", path=par.path,
                            prompt=list(par.prompt), decode=_rand_tokens(rng, rng.randrange(0, 9), vocab),
                            parent=par.parent, is_regeneration=True)
            else:
                # continuation of par's branch
                r = Request(f"g{g}r{i}", group=f"g{g}", path=par.path + (f"g{g}n{i}",),
                            prompt=list(par.prompt) + list(par.decode) + _rand_tokens(rng, rng.randrange(0, 9), vocab),
                            decode=_rand_tokens(rng, rng.randrange(0, 9), vocab), parent=par.request_id)
            reqs.append(r)
            frontier.append(r)
    return reqs


def random_legal_schedule(rng, reqs):
    by_id = {r.request_id: r for r in reqs}
    pending = set()
    for r in reqs:
        pending.add(("P", r.request_id))
        if r.decode:
            pending.add(("D", r.request_id))
    done = set()
    out = []
    while pending:
        ready = []
        for ev in pending:
            kind, rid = ev
            if kind == "D":
                ok = ("P", rid) in done
            else:
                p = by_id[rid].parent
                ok = p is None or (("D", p) if by_id[p].decode else ("P", p)) in done
            if ok:
                ready.append(ev)
        ev = rng.choice(sorted(ready))
        pending.remove(ev)
        done.add(ev)
        out.append(ev)
    return out


def test_random_schedules_lie_within_both_bound_pairs():
    rng = random.Random(20260905)
    n_checked = 0
    for trial in range(300):
        reqs = make_groups(rng)
        if sum(len(r.prompt) for r in reqs) == 0:
            continue
        b = bounds(reqs, B=4)
        for _ in range(5):
            sched = random_legal_schedule(rng, reqs)
            check_legal(reqs, sched, B=4)
            res = count_infinite(reqs, sched, B=4)
            R = res.reuse
            assert b.raw_lower - 1e-12 <= R <= b.upper + 1e-12, (trial, R, b)
            assert b.tightened_lower - 1e-12 <= R <= b.upper + 1e-12, (trial, R, b)
            assert b.raw_lower <= b.tightened_lower <= b.upper
            n_checked += 1
    assert n_checked > 1000
