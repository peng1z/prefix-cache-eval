"""T-INV on real corpora (TESTS.md), with provenance (FINDINGS F5).

Per corpus: load staged jsonl -> exact dedup on canonical bytes (id-independent)
-> seeded subsample (salt `tinv`) -> build groups -> key classes and bounds ->
serial + N random legal schedules at infinite capacity, full-hit correction OFF.
Fails (non-zero exit) if any schedule falls outside either bound pair.

Also runs the leading-assistant `exclude` condition (serial schedule only) so
the two F6 conditions can be compared, and writes a manifest with the input
file SHA-256, the dedup manifest, the subsample parameters and the SHA-256 of
the simulator sources.

Usage: python3 -m sim.run_tinv --data-dir <derived jsonl dir> --out revision/results/tinv_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from .attribution import register_system_prefixes
from .events import clear_key_memo
from .ingest import Tok, build_groups, exact_dedup, load_records, subsample
from .scheduler import Config, run_schedule
from .theory import bounds, key_classes

CORPORA = ("oasst", "openorca", "wildchat", "sharegpt")
HERE = Path(__file__).resolve().parent


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_sha256() -> str:
    h = hashlib.sha256()
    for p in sorted(HERE.glob("*.py")):
        h.update(p.name.encode() + p.read_bytes())
    return h.hexdigest()


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def summarize(res, b):
    R = res.reuse
    ok = (b.tightened_lower - 1e-12 <= R <= b.upper + 1e-12) and (b.raw_lower - 1e-12 <= R <= b.upper + 1e-12)
    return {
        "reuse": R, "total": res.total, "missed": res.missed, "within_bounds": ok,
        "strata": res.strata, "attribution": res.attribution, "n_empty_prefill": res.n_empty_prefill,
        "n_decode_events": res.n_decode_events, "decode_tokens_written": res.decode_tokens_written,
    }


def run_condition(groups, stats, B, K, n_sched, label):
    reqs = [r for g in groups for r in g]
    t0 = time.time()
    cls = key_classes(reqs, B)
    b = bounds(reqs, B)
    t_theory = time.time() - t0
    sysp = register_system_prefixes(groups, min_groups=50, min_len=16)
    out = {
        "label": label, "n_groups": len(groups), "n_requests": len(reqs),
        "n_regenerations": sum(r.is_regeneration for r in reqs),
        "prefill_tokens": b.T, "Q": b.Q, "ingest_stats": dict(stats),
        "classes": {"always_missed": cls.n_always_missed, "always_hit": cls.n_always_hit,
                    "undetermined": cls.n_undetermined, "V_p": len(cls.V_p), "V_d": len(cls.V_d)},
        "bounds": {"raw_lower": b.raw_lower, "tightened_lower": b.tightened_lower, "upper": b.upper},
        "system_prefixes_registered": sorted(len(p) for p in sysp),
        "schedules": [], "t_theory_s": t_theory,
    }
    cfgs = [("serial", Config(B=B, capacity=None, K_exec=1, seed=0))] + [
        (f"uniform_K{K}_seed{s}", Config(B=B, capacity=None, K_exec=K, seed=s, selection="uniform"))
        for s in range(n_sched)]
    for name, cfg in cfgs:
        t0 = time.time()
        res = run_schedule(groups, cfg, system_prefixes=sysp)
        rec = summarize(res, b)
        rec.update({"name": name, "t_s": time.time() - t0})
        out["schedules"].append(rec)
        print(f"  [{label}] {name}: R={rec['reuse']:.4f} bounds=[{b.tightened_lower:.4f},{b.upper:.4f}] "
              f"raw_lower={b.raw_lower:.4f} ok={rec['within_bounds']} {rec['t_s']:.1f}s", flush=True)
    out["all_within_bounds"] = all(s["within_bounds"] for s in out["schedules"])
    return out


def run_corpus(corpus, data_dir: Path, n_groups, B, K, n_sched, tok, tokenizer_name) -> dict:
    src = data_dir / f"{corpus}.jsonl"
    t0 = time.time()
    recs = load_records(src)
    n_raw = len(recs)
    recs, dedup_manifest = exact_dedup(recs)
    recs = subsample(recs, n_groups, salt="tinv")
    groups, stats = build_groups(recs, tok)
    t_ingest = time.time() - t0
    out = {
        "corpus": corpus, "B": B, "tokenizer": tokenizer_name,
        "manifest": {
            "input_file": str(src), "input_sha256": sha256_file(src), "n_raw_records": n_raw,
            "exact_dedup": dedup_manifest,
            "subsample": {"target": n_groups, "salt": "tinv", "seed": 20260826, "n_kept": len(recs)},
            "code_sha256": code_sha256(),
            "conditions": {"template": "fallback <|role|>\\n{text}\\n", "truncation": 16384,
                           "tree_reading": "regeneration", "full_hit_correction": False,
                           "capacity": "infinite"},
        },
        "t_ingest_s": t_ingest,
    }
    out["decode"] = run_condition(groups, stats, B, K, n_sched, "leading_assistant=decode")
    groups_x, stats_x = build_groups(recs, tok, leading_assistant="exclude")
    out["exclude"] = run_condition(groups_x, stats_x, B, K, 0, "leading_assistant=exclude")
    out["all_within_bounds"] = out["decode"]["all_within_bounds"] and out["exclude"]["all_within_bounds"]
    out["peak_rss_mb"] = rss_mb()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA))
    ap.add_argument("--n-groups", type=int, default=8000)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--n-sched", type=int, default=5)
    ap.add_argument("--tokenizer", default="cl100k_base")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = Tok(args.tokenizer)
    failed = []
    for corpus in args.corpora:
        clear_key_memo()
        print(f"== {corpus} ==", flush=True)
        res = run_corpus(corpus, Path(args.data_dir), args.n_groups, args.B, args.K, args.n_sched, tok, args.tokenizer)
        (out_dir / f"{corpus}.json").write_text(json.dumps(res, indent=2, sort_keys=True))
        print(f"  ingest {res['t_ingest_s']:.1f}s rss {res['peak_rss_mb']:.0f}MB "
              f"all_within_bounds={res['all_within_bounds']}", flush=True)
        if not res["all_within_bounds"]:
            failed.append(corpus)
    if failed:
        print(f"T-INV FAILED: out-of-bounds schedule on {failed}", flush=True)
        sys.exit(1)
    print("T-INV PASSED", flush=True)


if __name__ == "__main__":
    main()
