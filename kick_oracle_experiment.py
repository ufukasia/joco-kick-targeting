# -*- coding: utf-8 -*-
"""Exploratory (not pre-registered): does better wrong-edge PREDICTION buy better kicks?

Adds an "oracle" pool: m tour edges that are absent from the LKH-3 reference
tour (results/kick_targeting/lkh_ref), re-sampled after every accepted kick at
no model cost.  Same engine, same pool rule (m, 0.75), same start tours and
kick seeds as the primary campaign (seeds 0..S-1).  knn_global is re-run in the
same session as a paired control.

    py -3.13 kick_oracle_experiment.py --seeds 3 --workers 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

import fast_ils as F
from kick_targeting_experiment import load_xy, suite_instances, OUT_ROOT


def run_oracle(xs, ys, kind, nbr, init_tour, init_cost, ref_succ, ref_pred, seed, budget,
               geometry=0, prob=0.75, window=50, batch_target_s=0.004):
    n = len(xs)
    tour = init_tour.copy()
    pos = np.empty(n, dtype=np.int32)
    pos[tour] = np.arange(n, dtype=np.int32)
    btour, bpos = tour.copy(), pos.copy()
    queue = np.empty(n, dtype=np.int32)
    inq = np.zeros(n, dtype=np.uint8)
    qht = np.zeros(3, dtype=np.int64)
    buf = np.empty(n, dtype=np.int32)
    stats = np.zeros(4, dtype=np.int64)
    F.seed_rng(seed)
    rng = np.random.default_rng(seed)
    m = F.pool_size(n)
    pool_u = np.zeros(m, dtype=np.int32)
    pool_v = np.zeros(m, dtype=np.int32)
    pool_m = 0
    cost = init_cost
    checkpoints = [t for t in F.CHECKPOINTS if t <= budget + 1e-9]
    trace = {}
    acc_at = -1
    model_time = 0.0
    batch = 8
    start = time.perf_counter()
    while True:
        el = time.perf_counter() - start
        while checkpoints and el >= checkpoints[0]:
            trace[str(checkpoints.pop(0))] = cost
        if el >= budget:
            break
        if stats[1] != acc_at:
            t1 = time.perf_counter()
            u = btour
            v = np.roll(btour, -1)
            wrong = np.nonzero(~((ref_succ[u] == v) | (ref_pred[u] == v)))[0]
            if len(wrong):
                idx = rng.choice(wrong, size=min(m, len(wrong)), replace=False)
                pool_m = len(idx)
                pool_u[:pool_m] = u[idx]
                pool_v[:pool_m] = v[idx]
            else:
                pool_m = 0
            acc_at = stats[1]
            model_time += time.perf_counter() - t1
        tb = time.perf_counter()
        cost = F.run_kicks(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, batch, geometry,
                           pool_m > 0, pool_u, pool_v, max(pool_m, 1) if pool_m else 0, prob, window,
                           queue, inq, qht, buf, stats)
        dt = time.perf_counter() - tb
        if dt < batch_target_s * 0.5:
            batch = min(batch * 2, 1 << 20)
        elif dt > batch_target_s * 2 and batch > 1:
            batch = max(1, batch // 2)
    for t in checkpoints:
        trace[str(t)] = cost
    exact = float(F.tour_cost(btour, xs, ys, kind))
    assert abs(exact - cost) <= 1e-6 * max(1.0, exact)
    return {"strategy": "oracle_global", "seed": seed, "cost": exact, "elapsed": el,
            "kicks": int(stats[0]), "accepted": int(stats[1]), "guided_cuts": int(stats[2]),
            "stale_cuts": int(stats[3]), "model_time": model_time, "trace": trace}


def task(args):
    name, path, bks, seed, budget, out_dir = args
    out = Path(out_dir) / f"{name}__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    xs, ys, kind = load_xy(Path(path))
    n = len(xs)
    ref = json.loads((OUT_ROOT / "lkh_ref" / f"{name}.json").read_text(encoding="utf-8"))["tour"]
    ref = np.asarray(ref)
    rs = np.empty(n, dtype=np.int64)
    rp = np.empty(n, dtype=np.int64)
    rs[ref] = np.roll(ref, -1)
    rp[ref] = np.roll(ref, 1)
    nbr = F.build_knn(xs, ys, kind, 10)
    knn24 = F.build_knn(xs, ys, kind, min(24, n - 1))
    init, c0, _ = F.start_tour(xs, ys, kind, nbr, 7919 * (seed + 1) + n)
    kseed = 104729 + 1009 * seed
    runs = {}
    order = ["oracle_global", "knn_global"] if seed % 2 == 0 else ["knn_global", "oracle_global"]
    for s in order:
        if s == "oracle_global":
            r = run_oracle(xs, ys, kind, nbr, init, c0, rs, rp, kseed, budget)
        else:
            r = F.run_strategy(xs, ys, kind, nbr, init, c0, s, kseed, budget, knn24=knn24)
        r["gap_pct"] = 100.0 * (r["cost"] - bks) / bks
        r["trace_gap_pct"] = {t: 100.0 * (c - bks) / bks for t, c in r["trace"].items()}
        runs[s] = r
    payload = {"suite": "vlsi_primary", "instance": name, "n": n, "bks": bks, "seed": seed,
               "budget": budget, "start_cost": c0, "start_gap_pct": 100.0 * (c0 - bks) / bks,
               "runs": runs}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)
    return f"{name} s{seed} oracle={runs['oracle_global']['gap_pct']:.3f} knn={runs['knn_global']['gap_pct']:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_dir = OUT_ROOT / "vlsi_primary_oracle_10s"
    out_dir.mkdir(parents=True, exist_ok=True)
    insts = suite_instances("vlsi_primary")
    tasks = [(nm, str(p), b, s, a.budget, str(out_dir)) for nm, p, b in reversed(insts) for s in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=F.warmup) as pool:
        for i, msg in enumerate(pool.imap_unordered(task, tasks)):
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:6.0f}s] {msg}", flush=True)


if __name__ == "__main__":
    main()
