"""Validation item 3c: native chat template vs the fallback template on the same tokenizer,
same 8k exact-dedup sample (WildChat, ShareGPT): classes, bounds, serial R, prefix breaks."""
import argparse, json, time
from pathlib import Path
from .attribution import register_system_prefixes
from .events import clear_key_memo
from .ingest import Tok, build_groups, exact_dedup, load_records, subsample
from .native_template import NativeTok, build_groups_native
from .run_tinv import code_sha256
from .scheduler import Config, run_schedule
from .theory import bounds, key_classes

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tok-dir", required=True)
ap.add_argument("--tokenizers", nargs="*", default=["Qwen/Qwen2.5-0.5B", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"])
ap.add_argument("--corpora", nargs="*", default=["wildchat", "sharegpt"])
ap.add_argument("--n-groups", type=int, default=8000); ap.add_argument("--B", type=int, default=16)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
rows = []
for name in args.tokenizers:
    fb = Tok(name, Path(args.tok_dir)); nt = NativeTok(name, Path(args.tok_dir))
    for corpus in args.corpora:
        recs, _ = exact_dedup(load_records(Path(args.data_dir) / f"{corpus}.jsonl"))
        sub = subsample(recs, args.n_groups, salt="tinv")
        for tmpl in ("fallback", "native"):
            clear_key_memo(); t0 = time.time()
            if tmpl == "fallback":
                groups, stats = build_groups(sub, fb, leading_assistant="exclude")
            else:
                groups, stats = build_groups_native(sub, nt)
            reqs = [r for g in groups for r in g]
            cls = key_classes(reqs, args.B); b = bounds(reqs, args.B)
            sysp = register_system_prefixes(groups, 50, 16)
            res = run_schedule(groups, Config(B=args.B, capacity=None, K_exec=1), system_prefixes=sysp)
            fin = run_schedule(groups, Config(B=args.B, capacity=100_000, K_exec=64, policy="LRU", full_hit_correction=True), system_prefixes=sysp)
            row = {"tokenizer": name, "corpus": corpus, "template": tmpl, "n_groups": len(groups), "n_requests": len(reqs),
                   "prefill_tokens": b.T, "classes": {"always_missed": cls.n_always_missed, "always_hit": cls.n_always_hit, "undetermined": cls.n_undetermined},
                   "bounds": {"raw_lower": b.raw_lower, "tightened_lower": b.tightened_lower, "upper": b.upper},
                   "serial_R_inf": res.reuse, "R_1e5_LRU_K64": fin.reuse, "ingest_stats": dict(stats),
                   "attribution_inf": res.attribution, "code_sha256": code_sha256(), "t_s": time.time() - t0}
            rows.append(row)
            print(f"{name:32s} {corpus:9s} {tmpl:8s} groups={len(groups)} req={len(reqs)} T={b.T} undet={cls.n_undetermined} "
                  f"R_inf={res.reuse:.4f} [{b.tightened_lower:.4f},{b.upper:.4f}] R_1e5_K64={fin.reuse:.4f} "
                  f"breaks={stats.get('n_prefix_breaks',0)}/{stats.get('n_decode_prefix_breaks',0)} {time.time()-t0:.0f}s", flush=True)
            (out / "template.json").write_text(json.dumps(rows, indent=1))
print("done", flush=True)
