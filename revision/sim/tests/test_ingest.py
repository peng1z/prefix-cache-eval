"""Request construction rules R-S2..R-S5 on hand-built records.

A fake tokenizer maps each rendered piece to a distinct run of ids so that
boundaries are checkable; the exact ids do not matter.
"""
from sim.ingest import build_groups, render, MAX_TRANSCRIPT_TOKENS


class FakeTok:
    """Each distinct rendered string gets its own id range; length = len(text)//3+1."""

    def __init__(self):
        self.table = {}

    def encode(self, s):
        if s not in self.table:
            base = 1000 * (len(self.table) + 1)
            self.table[s] = [base + i for i in range(len(s) // 3 + 1)]
        return self.table[s]


def rec_linear(cid, roles_texts):
    return {"conv_id": cid, "kind": "linear", "timestamp": None,
            "messages": [{"role": r, "text": t} for r, t in roles_texts]}


def test_R_S2_user_assistant_pairs_become_prefill_decode():
    tok = FakeTok()
    rec = rec_linear("c", [("user", "hi there"), ("assistant", "hello!"), ("user", "more"), ("assistant", "ok")])
    (g,), st = build_groups([rec], tok)
    assert [r.request_id for r in g] == ["c::0", "c::1"]
    u1, a1, u2, a2 = (tok.encode(render(r, t)) for r, t in
                      [("user", "hi there"), ("assistant", "hello!"), ("user", "more"), ("assistant", "ok")])
    assert g[0].prompt == u1 and g[0].decode == a1 and g[0].parent is None
    assert g[1].prompt == u1 + a1 + u2 and g[1].decode == a2 and g[1].parent == "c::0"
    assert st["n_requests"] == 2 and st.get("n_unpaired_assistant", 0) == 0


def test_R_S2_leading_system_folds_into_first_prefill():
    tok = FakeTok()
    rec = rec_linear("c", [("system", "be nice"), ("user", "q"), ("assistant", "a")])
    (g,), _ = build_groups([rec], tok)
    assert g[0].prompt == tok.encode(render("system", "be nice")) + tok.encode(render("user", "q"))
    assert len(g) == 1


def test_R_S2_consecutive_users_give_empty_decode_then_next_request():
    tok = FakeTok()
    rec = rec_linear("c", [("user", "q1"), ("user", "q2"), ("assistant", "a")])
    (g,), _ = build_groups([rec], tok)
    assert len(g) == 2 and g[0].decode == [] and g[1].decode == tok.encode(render("assistant", "a"))
    assert g[1].prompt == tok.encode(render("user", "q1")) + tok.encode(render("user", "q2"))


def test_R_S2_leading_assistant_is_empty_prefill_request():
    tok = FakeTok()
    rec = rec_linear("c", [("assistant", "greeting"), ("user", "q"), ("assistant", "a")])
    (g,), st = build_groups([rec], tok)
    assert g[0].prompt == [] and g[0].decode == tok.encode(render("assistant", "greeting"))
    assert st["n_groups_leading_assistant"] == 1
    assert g[1].prompt == g[0].decode + tok.encode(render("user", "q"))


def test_R_S5_truncation_ends_the_path():
    class BigTok:
        def encode(self, s):
            return list(range(MAX_TRANSCRIPT_TOKENS // 2 + 10))
    rec = rec_linear("c", [("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")])
    (g,), st = build_groups([rec], BigTok())
    assert len(g) == 1  # second turn dropped: array already at the cap
    assert len(g[0].prompt) + len(g[0].decode) == MAX_TRANSCRIPT_TOKENS
    assert st["n_truncated_paths"] == 1


def rec_tree(cid, nodes):
    return {"conv_id": cid, "kind": "tree", "timestamp": None,
            "nodes": [{"id": i, "parent": p, "role": r, "text": t} for i, p, r, t in nodes]}


def test_R_S3_regeneration_reading_E6():
    tok = FakeTok()
    rec = rec_tree("t", [("u1", "None", "user", "q"), ("a1", "u1", "assistant", "r1"),
                         ("a2", "u1", "assistant", "r2"), ("u2", "a1", "user", "q2")])
    (g,), st = build_groups([rec], tok)
    by = {r.request_id: r for r in g}
    # DFS: request for u1 with decode a1, then u2 under a1, then regeneration with decode a2
    assert [r.request_id for r in g] == ["t::1", "t::2", "t::3"]
    r1, r2, r3 = g
    U1 = tok.encode(render("user", "q")); A1 = tok.encode(render("assistant", "r1"))
    assert r1.prompt == U1 and r1.decode == A1 and not r1.is_regeneration and r1.parent is None
    assert r2.prompt == U1 + A1 + tok.encode(render("user", "q2")) and r2.parent == "t::1"
    assert r3.prompt == U1 and r3.decode == tok.encode(render("assistant", "r2"))
    assert r3.is_regeneration and r3.parent is None
    assert r1.path == ("u1",) and r3.path == ("u1",) and r2.path == ("u1", "a1", "u2")


def test_R_S3_user_under_regenerated_branch_has_regeneration_as_parent():
    tok = FakeTok()
    rec = rec_tree("t", [("u1", "None", "user", "q"), ("a1", "u1", "assistant", "r1"),
                         ("a2", "u1", "assistant", "r2"), ("u3", "a2", "user", "q3")])
    (g,), _ = build_groups([rec], tok)
    ids = [r.request_id for r in g]
    regen = next(r for r in g if r.is_regeneration)
    child = next(r for r in g if r.path == ("u1", "a2", "u3"))
    assert child.parent == regen.request_id
    assert child.prompt == regen.prompt + regen.decode + tok.encode(render("user", "q3"))


def test_R_S4_flat_without_response_has_empty_decode_and_is_counted():
    tok = FakeTok()
    rec = {"conv_id": "o", "kind": "flat", "timestamp": None,
           "messages": [{"role": "system", "text": "sys"}, {"role": "user", "text": "q"}]}
    (g,), st = build_groups([rec], tok)
    assert g[0].prompt == tok.encode(render("system", "sys")) + tok.encode(render("user", "q"))
    assert g[0].decode == [] and st["n_flat_without_response"] == 1
    rec["response"] = "ans"
    (g2,), st2 = build_groups([rec], tok)
    assert g2[0].decode == tok.encode(render("assistant", "ans")) and "n_flat_without_response" not in st2


def test_R_S1_duplicate_conv_ids_get_unique_group_ids_and_are_counted():
    tok = FakeTok()
    a = rec_linear("dup", [("user", "q"), ("assistant", "a")])
    b = rec_linear("dup", [("user", "q"), ("assistant", "a")])
    c = rec_linear("other", [("user", "z"), ("assistant", "y")])
    groups, st = build_groups([a, b, c], tok)
    gids = [g[0].group for g in groups]
    assert len(set(gids)) == 3
    assert st["n_duplicate_conv_ids"] == 1
    rids = [r.request_id for g in groups for r in g]
    assert len(set(rids)) == len(rids)
