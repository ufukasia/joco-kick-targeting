# -*- coding: utf-8 -*-
"""Post-protocol EXPLORATORY arm requested in review round 1 (Ö4): raw edge length as the pool
signal (the penalty-free core of GLS), run on holdout2 together with random and the excess proxy
in the same session (paired, common random numbers), with the PROTOCOL_v2 settings
(10 s, 10 seeds, 4 workers).  Not part of the confirmatory analysis.

v2_targeting_experiment.py is hash-registered in PROTOCOL_v2.json and is therefore NOT modified;
this driver reuses its task() and adds one source by wrapping the scorer factory.

    py -3.13 v3_extra_experiment.py --tag holdout2_length_10s
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

import v2_targeting_experiment as V
from kick_targeting_experiment import OUT_ROOT


class LengthProxy:
    """edge_prob = -d(u,v): the m longest tour edges form the pool (GLS utility without penalties)."""

    def __init__(self, xs, ys, kind, knn):
        import fast_ils as F
        self.F, self.xs, self.ys, self.kind = F, xs, ys, kind

    def edge_prob(self, tour):
        u = np.ascontiguousarray(tour, dtype=np.int32)
        v = np.ascontiguousarray(np.roll(u, -1))
        return -self.F._edge_lengths(u, v, self.xs, self.ys, self.kind)


_orig_excess = V.ExcessProxy


def task(args):
    suite, name, path, bks, seed, budget, strategies, out_dir = args
    from pathlib import Path
    out = Path(out_dir) / f"{name}__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    # run the three arms through V.task in separate sub-calls so each source gets the right scorer
    res_runs, meta = {}, None
    import tempfile
    for s in strategies:
        src = V.parse_strategy(s)[0]
        with tempfile.TemporaryDirectory() as td:
            if src == "length":
                V.ExcessProxy = LengthProxy
                s_run = s.replace("length_", "excess_", 1)
            else:
                V.ExcessProxy = _orig_excess
                s_run = s
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
    meta["note"] = "exploratory post-protocol arm (review round 1, item Ö4)"
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    os.replace(tmp, out)
    return f"{name} s{seed} " + " ".join(f"{k}={res_runs[k]['gap_pct']:.3f}" for k in strategies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--strategies", default="random_global,excess_global@0.06:0.5,length_global@0.06:0.5")
    a = ap.parse_args()
    strategies = a.strategies.split(",")
    insts = V.instances("holdout2")
    out_dir = OUT_ROOT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "suite": "holdout2", "seeds": a.seeds, "budget": a.budget, "strategies": strategies,
        "workers": a.workers, "confirmatory": False, "note": "exploratory post-protocol (review Ö4)",
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
