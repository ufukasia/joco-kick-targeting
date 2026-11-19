# -*- coding: utf-8 -*-
"""Post-protocol EXPLORATORY robustness sessions on holdout2 (registered in
results/kick_targeting/PROTOCOL_v3_exploratory.json before they were run):

  S1  holdout2_local_10s   classic local (segment) double-bridge vs targeted global kicks
  S2  holdout2_60s         long budget: random, excess proxy, GNN (5 seeds)
  S3  holdout2_rule_10s    pool-rule sensitivity of the excess proxy

Not part of the confirmatory analysis.  v2_targeting_experiment.py (hash-registered in
PROTOCOL_v2.json) is NOT modified; each arm goes through its task() exactly as in
v3_extra_experiment.py (raw length = scorer swap).

    py -3.13 v4_robustness_experiment.py --tag holdout2_local_10s --strategies ... [--seeds 10 --budget 10]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import v2_targeting_experiment as V
from kick_targeting_experiment import OUT_ROOT
from v3_extra_experiment import LengthProxy

_orig_excess = V.ExcessProxy
NOTE = "exploratory post-protocol robustness session (PROTOCOL_v3_exploratory.json)"


def task(args):
    suite, name, path, bks, seed, budget, strategies, out_dir = args
    out = Path(out_dir) / f"{name}__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    res_runs, meta = {}, None
    for s in strategies:                      # one sub-call per arm: same start tour, same kick seed
        src = V.parse_strategy(s)[0]
        with tempfile.TemporaryDirectory() as td:
            V.ExcessProxy = LengthProxy if src == "length" else _orig_excess
            s_run = s.replace("length_", "excess_", 1) if src == "length" else s
            try:
                V.task((suite, name, path, bks, seed, budget, [s_run], td))
            finally:
                V.ExcessProxy = _orig_excess
            d = json.loads((Path(td) / f"{name}__s{seed}.json").read_text(encoding="utf-8"))
        r = d["runs"][s_run]
        r["strategy"] = s
        res_runs[s] = r
        meta = d
    meta["runs"] = res_runs
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["note"] = NOTE
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(tmp, out)
    return f"{name} s{seed} " + " ".join(f"{k}={res_runs[k]['gap_pct']:.3f}" for k in strategies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--strategies", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    strategies = a.strategies.split(",")
    for s in strategies:
        src, geom, _, _ = V.parse_strategy(s)
        assert src in ("random", "gnn", "excess", "length") and geom in ("global", "local"), s
    insts = V.instances("holdout2")
    out_dir = OUT_ROOT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "suite": "holdout2", "seeds": a.seeds, "budget": a.budget, "strategies": strategies,
        "workers": a.workers, "confirmatory": False, "note": NOTE,
        "machine": os.environ.get("COMPUTERNAME", "?"), "started": time.strftime("%Y-%m-%d %H:%M:%S")},
        indent=1), encoding="utf-8")
    tasks = [("holdout2", nm, str(p), b, s, a.budget, strategies, str(out_dir))
             for nm, p, b in reversed(insts) for s in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=V._init) as pool:
        for i, msg in enumerate(pool.imap_unordered(task, tasks)):
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:7.0f}s] {msg}", flush=True)


if __name__ == "__main__":
    main()
