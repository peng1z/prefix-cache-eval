"""Validation item 3: does the zero-width causal interval survive (a) a different 8k
sample, (b) the raw stratum (near-duplicates kept), (c) the MinHash stratum?
Reports classes, bounds and the serial R for each (corpus, stratum, salt)."""
import argparse, json, time
from pathlib import Path
from .attribution import register_system_prefixes
from .events import clear_key_memo
from .ingest import Tok, build_groups, exact_dedup, load_records, minhash_dedup, subsample
from .run_tinv import code_sha256, sha256_file
from .scheduler import Config, run_schedule
from .theory import bounds, key_classes

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--corpora", nargs="*", default=["oasst", "wildchat", "sharegpt"])
ap.add_argument("--strata", nargs="*", default=["exact", "raw", "minhash"])
ap.add_argument("--salts", nargs="*", default=["tinv", "tinv2"])
ap.add_argument("--n-groups", type=int, default=8000); ap.add_argument("--B", type=int, default=16)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
tok = Tok("cl100k_base")
rows = []
for corpus in args.corpora:
    src = Path(args.data_dir) / f"{corpus}.jsonl"
    recs_all = load_records(src)
    for stratum in args.strata:
        t0 = time.time()
        if stratum == "exact":
            recs, dman = exact_dedup(recs_all)
        elif stratum == "minhash":
            recs, dman = minhash_dedup(recs_all)
        else:
            recs, dman = recs_all, {"rule": "raw", "n_in": len(recs_all), "n_out": len(recs_all), "n_dropped": 0}
        t_dedup = time.time() - t0
        for salt in (args.salts if stratum == "exact" else args.salts[:1]):
            clear_key_memo()
            sub = subsample(recs, args.n_groups, salt=salt)
            groups, stats = build_groups(sub, tok)
            reqs = [r for g in groups for r in g]
            cls = key_classes(reqs, args.B); b = bounds(reqs, args.B)
            sysp = register_system_prefixes(groups, 50, 16)
            res = run_schedule(groups, Config(B=args.B, capacity=None, K_exec=1), system_prefixes=sysp)
            row = {"corpus": corpus, "stratum": stratum, "salt": salt, "dedup": dman, "t_dedup_s": t_dedup,
                   "n_groups": len(groups), "n_requests": len(reqs), "n_duplicate_conv_ids": stats.get("n_duplicate_conv_ids", 0),
                   "classes": {"always_missed": cls.n_always_missed, "always_hit": cls.n_always_hit, "undetermined": cls.n_undetermined},
                   "bounds": {"raw_lower": b.raw_lower, "tightened_lower": b.tightened_lower, "upper": b.upper},
                   "width_pt": (b.upper - b.tightened_lower) * 100, "serial_R": res.reuse,
                   "attribution": res.attribution, "input_sha256": sha256_file(src), "code_sha256": code_sha256()}
            rows.append(row)
            print(f"{corpus:9s} {stratum:8s} salt={salt:6s} groups={len(groups)} dropped={dman['n_dropped']} "
                  f"undet={cls.n_undetermined} width={row['width_pt']:.3f}pt R={res.reuse:.4f} "
                  f"[{b.tightened_lower:.4f},{b.upper:.4f}] dedup {t_dedup:.0f}s", flush=True)
            (out / "strata.json").write_text(json.dumps(rows, indent=1))
print("done", flush=True)
