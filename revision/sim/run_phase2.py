"""Phase 2 ablations at 8k-group scale (PHASE2_PLAN.md A–E).

Every cell is labelled with its family ("legal_event_schedule" or
"static_replay"), its conditions, and its coverage. Abstract model, pending
engine validation.

Usage:
  python3 -m sim.run_phase2 --data-dir <derived> --out results/phase2 --exp A [--corpora ...]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .attribution import register_system_prefixes
from .events import Request, clear_key_memo
from .ingest import Tok, build_groups, exact_dedup, load_records, subsample
from .scheduler import Config, run_schedule
from .theory import bounds
from .run_tinv import code_sha256, sha256_file

CAPS = (100_000, 1_000_000, 10_000_000)
POLICIES = ("LRU", "LFU", "FIFO")
KS = (1, 8, 64, 512)


def load(corpus, data_dir, n_groups, tok, **ingest_kw):
    src = Path(data_dir) / f"{corpus}.jsonl"
    recs = load_records(src)
    n_raw = len(recs)
    recs, dman = exact_dedup(recs)
    recs = subsample(recs, n_groups, salt="tinv")
    groups, stats = build_groups(recs, tok, **ingest_kw)
    manifest = {"corpus": corpus, "input_file": str(src), "input_sha256": sha256_file(src),
                "n_raw_records": n_raw, "exact_dedup": dman,
                "subsample": {"target": n_groups, "salt": "tinv", "seed": 20260826, "n_kept": len(recs)},
                "ingest_kw": {k: str(v) for k, v in ingest_kw.items()}, "ingest_stats": dict(stats),
                "code_sha256": code_sha256()}
    return groups, stats, manifest


def cell(groups, cfg, sysp, label, family, extra=None):
    t0 = time.time()
    res = run_schedule(groups, cfg, system_prefixes=sysp)
    out = {
        "label": label, "family": family, "B": cfg.B, "capacity": cfg.capacity, "policy": cfg.policy,
        "K_exec": cfg.K_exec, "seed": cfg.seed, "leaf_only": cfg.leaf_only,
        "full_hit_correction": cfg.full_hit_correction, "selection": cfg.selection,
        "reuse": res.reuse, "total": res.total, "missed": res.missed,
        "coverage": res.coverage, "coverage_flag": res.coverage < 1.0,
        "n_infeasible_requests": res.n_infeasible_requests,
        "n_full_hit_corrections": res.n_full_hit_corrections, "n_evictions": res.n_evictions,
        "max_ledger": res.max_ledger, "strata": res.strata, "attribution": res.attribution,
        "t_s": time.time() - t0,
    }
    if extra:
        out.update(extra)
    print(f"  {label}: R={res.reuse if res.reuse is not None else float('nan'):.4f} cov={res.coverage:.4f} "
          f"evict={res.n_evictions} {out['t_s']:.1f}s", flush=True)
    return out


def exp_A(groups, sysp, B, corpus, cond):
    """Finite-capacity grid, legal event schedules, correction on, leaf-only."""
    rows = []
    for cap in CAPS:
        for pol in POLICIES:
            for K in KS:
                cfg = Config(B=B, capacity=cap, policy=pol, K_exec=K, seed=0, leaf_only=True,
                             full_hit_correction=True, selection="uniform")
                rows.append(cell(groups, cfg, sysp, f"{corpus}/{cond}/A cap={cap} {pol} K={K}",
                                 "legal_event_schedule", {"corpus": corpus, "leading_assistant": cond}))
    return rows


def exp_B(groups, sysp, B, corpus, cond):
    rows = []
    for cap in (100_000, 1_000_000):
        for pol in ("LRU", "LFU"):
            for leaf in (True, False):
                cfg = Config(B=B, capacity=cap, policy=pol, K_exec=64, seed=0, leaf_only=leaf,
                             full_hit_correction=True, selection="uniform")
                rows.append(cell(groups, cfg, sysp, f"{corpus}/{cond}/B cap={cap} {pol} leaf_only={leaf}",
                                 "legal_event_schedule", {"corpus": corpus, "leading_assistant": cond}))
    return rows


def exp_C(data_dir, n_groups, tok, B):
    rows = []
    for reading in ("regeneration", "path"):
        groups, stats, _ = load("oasst", data_dir, n_groups, tok, tree_reading=reading)
        sysp = register_system_prefixes(groups, 50, 16)
        b = bounds([r for g in groups for r in g], B)
        for cap in (None, 1_000_000):
            cfg = Config(B=B, capacity=cap, policy="LRU", K_exec=64 if cap else 1, seed=0, leaf_only=True,
                         full_hit_correction=(cap is not None), selection="uniform")
            rows.append(cell(groups, cfg, sysp, f"oasst/C reading={reading} cap={cap}", "legal_event_schedule",
                             {"corpus": "oasst", "tree_reading": reading, "n_groups": len(groups),
                              "n_requests": sum(len(g) for g in groups),
                              "bounds_abstract": {"tightened_lower": b.tightened_lower, "upper": b.upper,
                                                  "raw_lower": b.raw_lower},
                              "ingest_stats": dict(stats)}))
    return rows


def static_requests(groups):
    """Static prefill-only family (R-P1): strip decodes; each request its own group."""
    out = []
    for g in groups:
        for r in g:
            out.append(Request(r.request_id, group=r.group, path=r.path, prompt=r.prompt, decode=[],
                               parent=None, is_regeneration=r.is_regeneration, decode_path=r.decode_path,
                               sys_len=r.sys_len, tree_sibling_rank=r.tree_sibling_rank))
    return out


def within_group_permutation(reqs, seed):
    """R-P2: keep the group-position sequence, permute which request of the group fills each slot."""
    rng = np.random.Generator(np.random.PCG64(seed))
    by_group: dict[str, list[Request]] = {}
    for r in reqs:
        by_group.setdefault(r.group, []).append(r)
    pools = {g: list(rng.permutation(len(lst))) for g, lst in by_group.items()}
    out = []
    idx = {g: 0 for g in by_group}
    for r in reqs:
        lst = by_group[r.group]
        out.append(lst[pools[r.group][idx[r.group]]])
        idx[r.group] += 1
    return out


def exp_D(groups, sysp, B, corpus, cond):
    reqs = static_requests(groups)
    serial = [[r] for r in reqs]
    rows = []
    for cap in (100_000, 1_000_000):
        for pol in POLICIES:
            cfg = Config(B=B, capacity=cap, policy=pol, K_exec=1, seed=0, leaf_only=True,
                         full_hit_correction=True, selection="round_robin")
            rows.append(cell(serial, cfg, sysp, f"{corpus}/{cond}/D static serial cap={cap} {pol}", "static_replay",
                             {"corpus": corpus, "leading_assistant": cond, "order": "serial"}))
            for seed in range(5):
                perm = [[r] for r in within_group_permutation(reqs, seed)]
                rows.append(cell(perm, cfg, sysp, f"{corpus}/{cond}/D static within-group-perm seed={seed} cap={cap} {pol}",
                                 "static_replay", {"corpus": corpus, "leading_assistant": cond,
                                                   "order": "within_group_permutation", "perm_seed": seed}))
    return rows


def interleaved_positions(groups, K, seed):
    """Prefill order of a legal K_exec schedule on the static (decode-stripped) requests,
    grouped as in the corpus. Gives a group-position sequence with real interleaving."""
    static_groups = []
    for g in groups:
        static_groups.append([Request(r.request_id, group=r.group, path=r.path, prompt=r.prompt, decode=[],
                                      parent=r.parent, is_regeneration=r.is_regeneration,
                                      decode_path=r.decode_path, sys_len=r.sys_len,
                                      tree_sibling_rank=r.tree_sibling_rank) for r in g])
    res = run_schedule(static_groups, Config(B=16, capacity=None, K_exec=K, seed=seed, selection="uniform"))
    by_id = {r.request_id: r for g in static_groups for r in g}
    order = [by_id[rid] for kind, rid in ((e[0], e[1]) for e in res.events) if kind == "P"]
    # detach parents so that each request can be its own group in any order
    return [Request(r.request_id, group=r.group, path=r.path, prompt=r.prompt, decode=[], parent=None,
                    is_regeneration=r.is_regeneration, decode_path=r.decode_path, sys_len=r.sys_len,
                    tree_sibling_rank=r.tree_sibling_rank) for r in order]


def exp_D2(groups, sysp, B, corpus, cond):
    """Paired static replay with an interleaved (K_exec=64) group-position sequence:
    within-group order intact vs permuted at the same positions (R-P2)."""
    reqs = interleaved_positions(groups, 64, 0)
    intact = [[r] for r in reqs]
    rows = []
    for cap in (100_000, 1_000_000):
        for pol in POLICIES:
            cfg = Config(B=B, capacity=cap, policy=pol, K_exec=1, seed=0, leaf_only=True,
                         full_hit_correction=True, selection="round_robin")
            rows.append(cell(intact, cfg, sysp, f"{corpus}/{cond}/D2 static K64-positions intact cap={cap} {pol}",
                             "static_replay", {"corpus": corpus, "leading_assistant": cond,
                                               "order": "interleaved_K64_intact"}))
            for seed in range(5):
                perm = [[r] for r in within_group_permutation(reqs, seed)]
                rows.append(cell(perm, cfg, sysp, f"{corpus}/{cond}/D2 static K64-positions perm seed={seed} cap={cap} {pol}",
                                 "static_replay", {"corpus": corpus, "leading_assistant": cond,
                                                   "order": "interleaved_K64_within_group_permutation",
                                                   "perm_seed": seed}))
    return rows


def exp_D3(groups, sysp, B, corpus, cond):
    """Historical static construction: uniform global shuffle of prefill-only requests
    (the submitted draft's seeded_shuffle), 3 seeds. Diagnostic; static family."""
    reqs = static_requests(groups)
    rows = []
    for cap in (100_000, 1_000_000):
        for pol in POLICIES:
            cfg = Config(B=B, capacity=cap, policy=pol, K_exec=1, seed=0, leaf_only=True,
                         full_hit_correction=True, selection="round_robin")
            for seed in range(3):
                rng = np.random.Generator(np.random.PCG64(seed))
                order = [reqs[int(i)] for i in rng.permutation(len(reqs))]
                rows.append(cell([[r] for r in order], cfg, sysp,
                                 f"{corpus}/{cond}/D3 static global-shuffle seed={seed} cap={cap} {pol}",
                                 "static_replay", {"corpus": corpus, "leading_assistant": cond,
                                                   "order": "global_uniform_shuffle", "perm_seed": seed}))
    return rows


def restore_within_group_order(reqs):
    """Keep each group's positions, fill them with that group's requests in causal order."""
    by_group: dict[str, list[Request]] = {}
    for r in reqs:
        by_group.setdefault(r.group, []).append(r)
    ordered = {g: sorted(lst, key=lambda r: len(r.prompt)) for g, lst in by_group.items()}
    idx = {g: 0 for g in by_group}
    out = []
    for r in reqs:
        out.append(ordered[r.group][idx[r.group]])
        idx[r.group] += 1
    return out


def exp_D4(groups, sysp, B, corpus, cond):
    """Paired global-shuffle: for each D3 seed, the same group-position sequence with
    (a) the shuffle's within-group order and (b) causal within-group order restored."""
    reqs = static_requests(groups)
    rows = []
    for cap in (100_000, 1_000_000):
        for pol in POLICIES:
            cfg = Config(B=B, capacity=cap, policy=pol, K_exec=1, seed=0, leaf_only=True,
                         full_hit_correction=True, selection="round_robin")
            for seed in range(3):
                rng = np.random.Generator(np.random.PCG64(seed))
                shuffled = [reqs[int(i)] for i in rng.permutation(len(reqs))]
                restored = restore_within_group_order(shuffled)
                for name, order in (("shuffle_order", shuffled), ("causal_order_same_positions", restored)):
                    rows.append(cell([[r] for r in order], cfg, sysp,
                                     f"{corpus}/{cond}/D4 {name} seed={seed} cap={cap} {pol}", "static_replay",
                                     {"corpus": corpus, "leading_assistant": cond, "order": name, "perm_seed": seed}))
    return rows


def exp_F(data_dir, n_groups, tok, B, corpus):
    """Historical implementation diagnosis (plan item F): full cross of
    R (request construction: legacy assistant-as-prefill vs user-only, both static / no decode)
    x P (legacy_no_pin vs pinned) x E (any_block vs leaf-only) [x C (LFU double count) for LFU],
    under serial order and the global shuffle (seed 0), capacities 1e5 and 1e6.
    Denominators differ across R levels; cells are never subtracted across R."""
    rows = []
    mans = []
    for R in ("legacy_assistant_prefill", "user_only"):
        groups, stats, man = load(corpus, data_dir, n_groups, tok,
                                  legacy_assistant_prefill=(R == "legacy_assistant_prefill"))
        man = dict(man, R=R, n_requests=sum(len(g) for g in groups),
                   prefill_tokens=sum(len(r.prompt) for g in groups for r in g))
        mans.append(man)
        reqs = static_requests(groups)
        sysp = register_system_prefixes(groups, 50, 16)
        rng = np.random.Generator(np.random.PCG64(0))
        shuffled = [reqs[int(i)] for i in rng.permutation(len(reqs))]
        for order_name, order in (("serial", reqs), ("global_shuffle_seed0", shuffled)):
            for cap in (100_000, 1_000_000):
                for pol in POLICIES:
                    for P in ("legacy_no_pin", "pinned"):
                        for E in ("any_block", "leaf_only"):
                            for C in (("legacy_double_count", "single_count") if pol == "LFU" else ("single_count",)):
                                cfg = Config(B=B, capacity=cap, policy=pol, K_exec=1, seed=0,
                                             leaf_only=(E == "leaf_only"), full_hit_correction=False,
                                             selection="round_robin", legacy_no_pin=(P == "legacy_no_pin"),
                                             legacy_lfu_double_count=(C == "legacy_double_count"))
                                rows.append(cell([[r] for r in order], cfg, sysp,
                                                 f"{corpus}/F R={R} P={P} E={E} C={C} order={order_name} cap={cap} {pol}",
                                                 "static_replay_diagnostic",
                                                 {"corpus": corpus, "R": R, "P": P, "E": E, "C": C, "order": order_name,
                                                  "n_requests": len(reqs), "manifest_ingest": man["ingest_kw"]}))
    return rows, mans


def exp_F2(data_dir, n_groups, tok, B, corpus):
    """P split (seventh review): serial order, any-block candidates (old rule), three policies,
    three allocation/protection modes: legacy_no_pin (old), per_block_protected (old allocation,
    blocks protected), atomic_pinned (current core)."""
    rows = []
    mans = []
    for R in ("legacy_assistant_prefill", "user_only"):
        groups, stats, man = load(corpus, data_dir, n_groups, tok,
                                  legacy_assistant_prefill=(R == "legacy_assistant_prefill"))
        man = dict(man, R=R, n_requests=sum(len(g) for g in groups),
                   prefill_tokens=sum(len(r.prompt) for g in groups for r in g))
        mans.append(man)
        reqs = static_requests(groups)
        sysp = register_system_prefixes(groups, 50, 16)
        for cap in (100_000, 1_000_000):
            for pol in POLICIES:
                for mode in ("legacy_no_pin", "per_block_protected", "atomic_pinned"):
                    cfg = Config(B=B, capacity=cap, policy=pol, K_exec=1, seed=0, leaf_only=False,
                                 full_hit_correction=False, selection="round_robin",
                                 legacy_no_pin=(mode == "legacy_no_pin"),
                                 legacy_per_block_alloc=(mode == "per_block_protected"))
                    rows.append(cell([[r] for r in reqs], cfg, sysp,
                                     f"{corpus}/F2 R={R} mode={mode} serial cap={cap} {pol}",
                                     "static_replay_diagnostic",
                                     {"corpus": corpus, "R": R, "mode": mode, "order": "serial",
                                      "n_requests": len(reqs)}))
    return rows, mans


def exp_S(data_dir, n_groups, tok, B, corpus):
    """Multi-seed validation of the key cells (validation list item 2): the 1e5 / K=512
    rank crossings (all three policies), the D-LEAD sign change (ShareGPT 1e6 / K=512 /
    LFU) and the cliff cell (WildChat 1e5 / K=64), seeds 1..5 (seed 0 is in A)."""
    rows, mans = [], []
    conds = ("decode", "exclude") if corpus in ("wildchat", "sharegpt") else ("decode",)
    for cond in conds:
        groups, stats, man = load(corpus, data_dir, n_groups, tok, leading_assistant=cond)
        mans.append(dict(man, cond=cond))
        sysp = register_system_prefixes(groups, 50, 16)
        cells = [(100_000, pol, 512) for pol in POLICIES]
        if corpus == "sharegpt":
            cells.append((1_000_000, "LFU", 512))
        if corpus == "wildchat":
            cells += [(100_000, "LRU", 64), (100_000, "LFU", 64)]
        for cap, pol, K in cells:
            for seed in range(1, 6):
                cfg = Config(B=B, capacity=cap, policy=pol, K_exec=K, seed=seed, leaf_only=True,
                             full_hit_correction=True, selection="uniform")
                rows.append(cell(groups, cfg, sysp, f"{corpus}/{cond}/S cap={cap} {pol} K={K} seed={seed}",
                                 "legal_event_schedule", {"corpus": corpus, "leading_assistant": cond}))
    return rows, mans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exp", required=True, choices=["A","B","C","D","D2","D3","D4","E","F","F2","S"])
    ap.add_argument("--corpora", nargs="*", default=["wildchat", "sharegpt", "oasst"])
    ap.add_argument("--n-groups", type=int, default=8000)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--tokenizer", default="cl100k_base")
    args = ap.parse_args()
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tok(args.tokenizer)
    B = args.B
    rows = []
    manifests = []
    if args.exp == "C":
        rows = exp_C(args.data_dir, args.n_groups, tok, B)
    elif args.exp in ("F", "F2", "S"):
        for corpus in args.corpora:
            clear_key_memo()
            fn = {"F": exp_F, "F2": exp_F2, "S": exp_S}[args.exp]
            new_rows, mans = fn(args.data_dir, args.n_groups, tok, B, corpus)
            rows += new_rows
            manifests += mans
            (out_dir / f"{args.exp}.json").write_text(json.dumps(rows, indent=1))
    else:
        corpora = ["openorca"] if args.exp == "E" else args.corpora
        for corpus in corpora:
            clear_key_memo()
            conds = ("decode", "exclude") if corpus in ("wildchat", "sharegpt") else ("decode",)
            for cond in conds:
                groups, stats, manifest = load(corpus, args.data_dir, args.n_groups, tok, leading_assistant=cond)
                manifests.append(manifest)
                sysp = register_system_prefixes(groups, 50, 16)
                print(f"== {corpus} {cond}: {len(groups)} groups ==", flush=True)
                fn = {"A": exp_A, "B": exp_B, "D": exp_D, "D2": exp_D2, "D3": exp_D3, "D4": exp_D4, "E": exp_A}[args.exp]
                rows += fn(groups, sysp, B, corpus, cond)
                (out_dir / f"{args.exp}.json").write_text(json.dumps(rows, indent=1))
    (out_dir / f"{args.exp}.json").write_text(json.dumps(rows, indent=1))
    (out_dir / "manifest.json").write_text(json.dumps({"code_sha256": code_sha256(), "inputs": manifests,
                                                        "exp": args.exp, "B": B, "tokenizer": args.tokenizer}, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
