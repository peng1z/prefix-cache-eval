"""R-A2 attribution buckets, R-A3 per-request system-prefix registration, R-A5
separation of buckets from key classes (TESTS T-SYS, T-A5)."""
from sim.events import Request
from sim.attribution import register_system_prefixes, sys_len_for
from sim.scheduler import Config, run_schedule
from sim.theory import key_classes
from sim.tests.test_counting import A, Bb, C, B, _e2

def _cfg(**kw):
    base = dict(B=B, capacity=None, K_exec=1, selection="round_robin")
    base.update(kw)
    return Config(**base)


def test_T_SYS_registration_and_per_request_span():
    head20 = list(range(500, 520))
    g = []
    for i in range(3):
        g.append([Request(f"s{i}", group=f"s{i}", path=(f"s{i}",), prompt=head20 + [i, i + 1], decode=[], parent=None)])
    g.append([Request("t", group="t", path=("t",), prompt=head20[:12] + [77, 78, 79, 80, 81], decode=[], parent=None)])
    reg = register_system_prefixes(g, min_groups=3, min_len=16)
    assert reg == {tuple(head20)}
    assert sys_len_for(g[0][0], reg) == 20
    assert sys_len_for(g[3][0], reg) == 0   # shares only 12 tokens (< 16)


def test_T_SYS_partial_match_of_a_registered_prefix_does_not_count():
    head = list(range(500, 516))
    g = [[Request(f"s{i}", group=f"s{i}", path=(f"s{i}",), prompt=head + [i], decode=[], parent=None)] for i in range(3)]
    reg = register_system_prefixes(g, min_groups=3, min_len=16)
    other = Request("o", group="o", path=("o",), prompt=head[:15] + [999, 1000], decode=[], parent=None)
    assert sys_len_for(other, reg) == 0


def test_T_A5_bucket_own_path_decode_in_E1():
    r1 = Request("r1", group="g", path=("n1",), prompt=A, decode=Bb, parent=None)
    r2 = Request("r2", group="g", path=("n1", "n2"), prompt=A + Bb + C, decode=[], parent="r1")
    res = run_schedule([[r1, r2]], _cfg())
    assert res.attribution["own_path_decode"] == B     # block B, produced by r1's decode
    assert res.attribution["own_path_prefill"] == B    # block A, inserted by r1's prefill
    assert res.attribution["cross_group_decode"] == 0
    cls = key_classes([r1, r2], B)
    assert cls.n_always_hit == 1                        # theory class, reported separately


def test_T_A5_bucket_cross_group_decode_in_E2_x_first():
    x, z = _e2()
    res = run_schedule([[x], [z]], _cfg())
    assert res.attribution["cross_group_prefill"] == B  # A inserted by X's prefill
    assert res.attribution["cross_group_decode"] == B   # B inserted by X's decode
    assert res.attribution["own_path_decode"] == 0
    cls = key_classes([x, z], B)
    assert cls.n_undetermined == 1                      # class != bucket


def test_T_A5_system_prefix_bucket_takes_precedence():
    head = list(range(500, 516))  # one full block of 16 at B=16
    groups = [[Request(f"s{i}", group=f"s{i}", path=(f"s{i}",), prompt=head + [i, i + 1], decode=[], parent=None)] for i in range(3)]
    res = run_schedule(groups, _cfg(B=16), system_prefixes=register_system_prefixes(groups, 3, 16))
    assert res.attribution["system_prefix"] == 32       # groups 2 and 3 hit the head block
    assert res.attribution["cross_group_prefill"] == 0


def test_attribution_sums_to_hit_tokens():
    x, z = _e2()
    res = run_schedule([[x], [z]], _cfg())
    assert sum(res.attribution.values()) == res.total - res.missed
