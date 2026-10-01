"""Engine comparison on the 8k exact-dedup samples: serial (K_exec = 1) replay of
the event-model requests through the real vLLM V1 prefix-cache core vs the
simulator (LRU, correction on), at infinite capacity, 1e5 and 1e6 tokens.
Reports per-request hit agreement and totals, for both harness choices of
caching the last decode-completed blocks at completion.

Usage: PYTHONPATH=. python3 engine_ref/run_compare.py --data-dir <derived> --out results/engine_ref
"""
import argparse, json, time
from pathlib import Path
from engine_ref.harness import run_serial, VENDORED_COMMIT
from sim.ingest import Tok
from sim.run_phase2 import load
from sim.run_tinv import code_sha256
from sim.scheduler import Config, run_schedule

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--corpora", nargs="*", default=["oasst", "wildchat", "sharegpt"])
ap.add_argument("--n-groups", type=int, default=8000); ap.add_argument("--B", type=int, default=16)
args = ap.parse_args()
out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
tok = Tok("cl100k_base")
B = args.B
for corpus in args.corpora:
    # leading_assistant='exclude': vLLM cannot take an empty prompt, so the `decode` condition's
    # empty-prefill requests are not representable in the engine (see engine_ref/README.md)
    groups, stats, man = load(corpus, args.data_dir, args.n_groups, tok, leading_assistant="exclude")
    res = {"corpus": corpus, "manifest": man, "vendored_commit": VENDORED_COMMIT, "code_sha256": code_sha256(), "cells": []}
    for cap in (10_000_000, 100_000, 1_000_000):   # 1e7 = no eviction on these samples (phase-2 grids)
        t0 = time.time()
        sim = run_schedule(groups, Config(B=B, capacity=cap, K_exec=1, policy="LRU", leaf_only=True, full_hit_correction=True))
        t_sim = time.time() - t0
        for coc in (True, False):
            t0 = time.time()
            ref = run_serial(groups, cap, B, cache_on_complete=coc)
            t_ref = time.time() - t0
            common = [k for k in sim.per_request_hit if k in ref["per_request_hit"]]
            n_eq = sum(sim.per_request_hit[k] == ref["per_request_hit"][k] for k in common)
            diffs = [(k, sim.per_request_hit[k], ref["per_request_hit"][k]) for k in common if sim.per_request_hit[k] != ref["per_request_hit"][k]]
            cell = {"capacity": cap, "cache_on_complete": coc,
                    "sim": {"reuse": sim.reuse, "total": sim.total, "missed": sim.missed, "n_infeasible": sim.n_infeasible_requests},
                    "vllm": {"reuse": ref["reuse"], "total": ref["total"], "missed": ref["missed"], "n_infeasible": ref["n_infeasible"]},
                    "n_requests_compared": len(common), "n_per_request_equal": n_eq,
                    "hit_token_diff_vllm_minus_sim": ref["total"] - ref["missed"] - (sim.total - sim.missed),
                    "example_diffs": diffs[:5], "t_sim_s": t_sim, "t_vllm_s": t_ref}
            res["cells"].append(cell)
            print(f"{corpus} cap={cap} coc={coc}: sim R={sim.reuse:.4f} vllm R={ref['reuse']:.4f} "
                  f"per-request equal {n_eq}/{len(common)} hit-token diff {cell['hit_token_diff_vllm_minus_sim']} "
                  f"infeasible sim/vllm {sim.n_infeasible_requests}/{ref['n_infeasible']} ({t_sim:.0f}s / {t_ref:.0f}s)", flush=True)
    (out_dir / f"{corpus}.json").write_text(json.dumps(res, indent=1))
print("done", flush=True)
