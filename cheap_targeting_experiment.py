# -*- coding: utf-8 -*-
"""Exploratory campaign (not pre-registered): cheap learned scorer and large-instance scale.

Same engine (fast_ils.run_strategy, unchanged), same start tours, kick seeds, pool rule
and 10 % model-time cap as the primary campaign.  Adds two strategies that use the
cheap learned scorer (cheap_scorer.py) as the pool source:
    cheap_global / cheap_local  ==  gnn_global / gnn_local with CheapScorer in place of the GNN.

    py -3.13 cheap_targeting_experiment.py --suite vlsi_primary --tag vlsi_primary_cheap_10s \
        --seeds 5 --budget 10 --strategies random_global,knn_global,gnn_global,cheap_global
    py -3.13 cheap_targeting_experiment.py --suite vlsi_large --tag vlsi_large_30s \
        --seeds 3 --budget 30 --strategies random_global,knn_global,gnn_global,cheap_global,random_local,knn_local
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

from kick_targeting_experiment import HERE, OUT_ROOT, load_xy, suite_instances

LARGE = ["bgb4355", "xqd4966", "fea5557", "xsc6880", "lap7454", "dga9698", "xmc10150", "xrb14233",
         "pjh17845", "fnc19402", "lsb22777", "bbz25234", "fyg28534", "pbh30440", "bby34656", "ics39603"]

_W = None
_P = None
_PG = None


def instances(suite):
    if suite == "vlsi_large":
        import vlsi_datasets as v
        return [(k, HERE / "data_vlsi" / f"{k}.tsp", float(v.CATALOG[k][1])) for k in LARGE]
    return suite_instances(suite)


def _init():
    global _W, _P, _PG
    import fast_ils as F
    import gnn_trained as gt
    from cheap_scorer import GAIN_MODEL, load_params
    F.warmup()
    _W, _ = gt.get_weights(HERE / "models" / "gnn_trained.npz")
    _P = load_params()
    _PG = load_params(GAIN_MODEL) if GAIN_MODEL.exists() else None


def task(args):
    suite, name, path, bks, seed, budget, strategies, out_dir = args
    import fast_ils as F
    from cheap_scorer import CheapScorer
    out = Path(out_dir) / f"{name}__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    xs, ys, kind = load_xy(Path(path))
    n = len(xs)
    nbr = F.build_knn(xs, ys, kind, 10)
    knn24 = F.build_knn(xs, ys, kind, min(24, n - 1))
    gnn = F.GNNScorer(xs, ys, kind, _W)
    cheap = CheapScorer(xs, ys, kind, _P, knn=knn24)
    gain = CheapScorer(xs, ys, kind, _PG, knn=knn24) if _PG is not None else None
    init, c0, t_start = F.start_tour(xs, ys, kind, nbr, 7919 * (seed + 1) + n)
    kick_seed = 104729 + 1009 * seed
    order = list(strategies)
    r = seed % len(order)
    order = order[r:] + order[:r]
    runs = {}
    for s in order:
        if s.startswith(("cheap_", "gain_")):
            sc = cheap if s.startswith("cheap_") else gain
            res = F.run_strategy(xs, ys, kind, nbr, init, c0, "gnn_" + s.split("_", 1)[1], kick_seed, budget,
                                 gnn=sc, knn24=knn24)
            res["strategy"] = s
        else:
            res = F.run_strategy(xs, ys, kind, nbr, init, c0, s, kick_seed, budget, gnn=gnn, knn24=knn24)
        res["gap_pct"] = 100.0 * (res["cost"] - bks) / bks
        res["trace_gap_pct"] = {t: 100.0 * (c - bks) / bks for t, c in res["trace"].items()}
        runs[s] = res
    payload = {"suite": suite, "instance": name, "n": n, "bks": bks, "seed": seed, "budget": budget,
               "start_cost": c0, "start_gap_pct": 100.0 * (c0 - bks) / bks, "start_time": t_start,
               "runs": runs, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)
    return f"{name} s{seed} n={n} start={t_start:.1f}s " + " ".join(f"{k}={runs[k]['gap_pct']:.3f}" for k in strategies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--strategies", required=True)
    a = ap.parse_args()
    strategies = a.strategies.split(",")
    out_dir = OUT_ROOT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    insts = instances(a.suite)
    (out_dir / "config.json").write_text(json.dumps({"suite": a.suite, "seeds": a.seeds, "budget": a.budget,
                                                     "strategies": strategies, "instances": [r[0] for r in insts],
                                                     "workers": a.workers, "exploratory": True}, indent=1),
                                         encoding="utf-8")
    tasks = [(a.suite, nm, str(p), b, s, a.budget, strategies, str(out_dir))
             for nm, p, b in reversed(insts) for s in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as pool:
        for i, msg in enumerate(pool.imap_unordered(task, tasks)):
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:7.0f}s] {msg}", flush=True)


if __name__ == "__main__":
    main()
