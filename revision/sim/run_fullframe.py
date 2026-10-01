"""Validation item 4: selective full-frame runs. Exact-dedup full frame of one corpus,
abstract bounds + serial infinite-capacity R, plus one finite cell (1e6, LRU, K_exec=64,
correction on). Logs peak RSS. Usage: python3 -m sim.run_fullframe --data-dir D --out O --corpus wildchat"""
import argparse, json, resource, time
from pathlib import Path
from .attribution import register_system_prefixes
from .events import clear_key_memo
from .ingest import Tok, build_groups, exact_dedup, load_records
from .run_tinv import code_sha256, sha256_file
from .scheduler import Config, run_schedule
from .theory import bounds, key_classes

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--corpus", required=True)
ap.add_argument("--B", type=int, default=16); ap.add_argument("--finite", action="store_true")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
tok = Tok("cl100k_base"); t0 = time.time()
src = Path(args.data_dir) / f"{args.corpus}.jsonl"
recs, dman = exact_dedup(load_records(src))
groups, stats = build_groups(recs, tok, leading_assistant="exclude")
reqs = [r for g in groups for r in g]
print(f"{args.corpus}: groups={len(groups)} requests={len(reqs)} ingest {time.time()-t0:.0f}s rss {rss():.1f}GB", flush=True)
t0 = time.time(); cls = key_classes(reqs, args.B); b = bounds(reqs, args.B)
print(f"theory {time.time()-t0:.0f}s: V_p={len(cls.V_p)} undet={cls.n_undetermined} [{b.tightened_lower:.4f},{b.upper:.4f}] rss {rss():.1f}GB", flush=True)
del cls
sysp = register_system_prefixes(groups, 50, 16)
t0 = time.time(); res = run_schedule(groups, Config(B=args.B, capacity=None, K_exec=1), system_prefixes=sysp)
print(f"serial inf {time.time()-t0:.0f}s: R={res.reuse:.4f} total={res.total} rss {rss():.1f}GB", flush=True)
row = {"corpus": args.corpus, "n_groups": len(groups), "n_requests": len(reqs), "prefill_tokens": b.T,
       "bounds": {"raw_lower": b.raw_lower, "tightened_lower": b.tightened_lower, "upper": b.upper},
       "serial_R_inf": res.reuse, "attribution_inf": res.attribution, "strata": res.strata, "ingest_stats": dict(stats),
       "dedup": dman, "input_sha256": sha256_file(src), "code_sha256": code_sha256(), "peak_rss_gb": rss()}
if args.finite:
    t0 = time.time(); fin = run_schedule(groups, Config(B=args.B, capacity=1_000_000, K_exec=64, policy="LRU", full_hit_correction=True), system_prefixes=sysp)
    print(f"1e6 LRU K64 {time.time()-t0:.0f}s: R={fin.reuse:.4f} cov={fin.coverage:.4f} rss {rss():.1f}GB", flush=True)
    row["R_1e6_LRU_K64"] = fin.reuse; row["coverage_1e6"] = fin.coverage
(out / f"{args.corpus}.json").write_text(json.dumps(row, indent=1)); print("done", flush=True)
