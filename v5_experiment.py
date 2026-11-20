# -*- coding: utf-8 -*-
"""PROTOCOL_v4_exploratory driver: blocks B1 (equal kick count), B2 (published learned scorer,
NeuroLKH), B4 (pool-rule tuning for the learned scorers), B5 (improvement credit for online
learning) and B6 (randomised arm order, single worker).  B3 (LKH-3) is lkh_b3_experiment.py.

Nothing hash-registered is modified: the fast ILS (fast_ils.py), the PROTOCOL_v2 driver
(v2_targeting_experiment.py) and the online learner (online_targeting.py) are imported as they are.

New pool sources (all use the run_strategy `gnn=` interface: the m smallest edge_prob values are kicked)
    neurolkh      NeuroLKH SGN edge probabilities (authors' pretrained neurolkh_m.pt, neurolkh_infer.py);
                  the measured forward-pass time is deducted from the time budget
    neurolkhfree  the same scores with the forward pass treated as per-instance preparation (not charged),
                  like the neighbour lists of every other signal
Online credit variants (online_credit.py; warm start w0, otherwise identical to onlinewarm)
    onlinepos     improvement credit max(0, r)
    onlinesucc    success credit 1[accepted]

Modes
    --mode time   equal wall-clock budget (as in all earlier sessions)
    --mode kicks  equal number of kicks K; the pool is refreshed after every accepted kick and model time
                  is not charged (no 10 % cap, no stale cuts from deferred refreshes)
Arm order is a random permutation per (instance, seed) (crc32 of "name|seed"), not a rotation.

    py -3.13 v5_experiment.py --suite holdout2 --tag holdout2_v5_equal_10s --mode time --budget 10 \
        --seeds 10 --workers 4 --strategies random_global,excess_global@0.06:0.5,...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
import zlib
from pathlib import Path

import numpy as np

import v2_targeting_experiment as V
from kick_targeting_experiment import OUT_ROOT, load_xy

PROTO = OUT_ROOT / "PROTOCOL_v4_exploratory.json"
PROTO_SHA = OUT_ROOT / "PROTOCOL_v4_exploratory.sha256"
SCORES = OUT_ROOT / "neurolkh_scores"
CREDIT = {"onlinepos": 1, "onlinesucc": 2}
NEURO = ("neurolkh", "neurolkhfree")


def protocol_frozen() -> bool:
    if not (PROTO.exists() and PROTO_SHA.exists()):
        return False
    h = hashlib.sha256(PROTO.read_bytes()).hexdigest()
    return PROTO_SHA.read_text(encoding="utf-8").split()[0].strip().lower() == h


class NeuroLKHScorer:
    """Tour edge (u, v) -> max of the SGN probabilities of u->v and v->u (0 if v is not among u's 20
    nearest neighbours and u not among v's: such an edge is never a NeuroLKH candidate)."""

    def __init__(self, name, model="neurolkh_m"):
        z = np.load(SCORES / f"{name}.{model}.npz")
        self.ei = z["edge_index"].astype(np.int64)
        self.p = z["prob"].astype(np.float64)
        self.t_forward = float(z["t_forward_s"])

    def edge_prob(self, tour):
        u = np.asarray(tour, dtype=np.int64)
        v = np.roll(u, -1)
        pu = np.where(self.ei[u] == v[:, None], self.p[u], 0.0).max(1)
        pv = np.where(self.ei[v] == u[:, None], self.p[v], 0.0).max(1)
        return np.maximum(pu, pv)


def run_fixed_kicks(xs, ys, kind, nbr, init_tour, init_cost, strategy, seed, n_kicks, gnn=None,
                    knn24=None, prob=0.75, window=50):
    """fast_ils.run_strategy with a kick budget instead of a time budget; pool refreshed after every
    accepted kick, model time recorded but not charged."""
    import fast_ils as F
    geometry, source = F.STRATEGIES[strategy]
    n = len(xs)
    tour = init_tour.copy()
    pos = np.empty(n, dtype=np.int32)
    pos[tour] = np.arange(n, dtype=np.int32)
    btour = tour.copy()
    bpos = pos.copy()
    queue = np.empty(n, dtype=np.int32)
    inq = np.zeros(n, dtype=np.uint8)
    qht = np.zeros(3, dtype=np.int64)
    buf = np.empty(n, dtype=np.int32)
    stats = np.zeros(4, dtype=np.int64)
    F.seed_rng(seed)
    m = F.pool_size(n)
    pool_u = np.zeros(m, dtype=np.int32)
    pool_v = np.zeros(m, dtype=np.int32)
    pool_m = 0
    cost = init_cost
    model_time = 0.0
    refreshes = 0
    accepted_at_refresh = -1
    start = time.perf_counter()
    while stats[0] < n_kicks:
        if source is not None and stats[1] != accepted_at_refresh:
            t1 = time.perf_counter()
            if source == "gnn":
                p = gnn.edge_prob(btour)
                idx = np.argpartition(p, m - 1)[:m]
            else:
                s = F.knn_rank_scores(btour, knn24, xs, ys, kind)
                idx = np.argpartition(-s, m - 1)[:m]
            pool_u[:] = btour[idx]
            pool_v[:] = btour[(idx + 1) % n]
            pool_m = m
            refreshes += 1
            accepted_at_refresh = stats[1]
            model_time += time.perf_counter() - t1
        cost = F.run_kicks(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, 1,
                           geometry, source is not None, pool_u, pool_v, pool_m, prob,
                           window, queue, inq, qht, buf, stats)
    elapsed = time.perf_counter() - start
    exact = float(F.tour_cost(btour, xs, ys, kind))
    if abs(exact - cost) > 1e-6 * max(1.0, abs(exact)):
        raise RuntimeError(f"cost drift: tracked {cost} vs exact {exact}")
    if len(np.unique(btour)) != n:
        raise RuntimeError("invalid tour")
    return {"strategy": strategy, "seed": seed, "cost": exact, "initial_cost": init_cost, "elapsed": elapsed,
            "kicks": int(stats[0]), "accepted": int(stats[1]), "guided_cuts": int(stats[2]),
            "stale_cuts": int(stats[3]), "model_time": model_time, "refreshes": refreshes, "trace": {}}


def task(args):
    suite, name, path, bks, seed, mode, budget, n_kicks, strategies, out_dir = args
    import fast_ils as F
    from cheap_scorer import CheapScorer
    out = Path(out_dir) / f"{name}__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    xs, ys, kind = load_xy(Path(path))
    n = len(xs)
    nbr = F.build_knn(xs, ys, kind, 10)
    knn24 = F.build_knn(xs, ys, kind, min(24, n - 1))
    srcs = {V.parse_strategy(s)[0] for s in strategies}
    scorers = {}
    if "gnn" in srcs:
        scorers["gnn"] = F.GNNScorer(xs, ys, kind, V._W)
    if "gain" in srcs:
        scorers["gain"] = CheapScorer(xs, ys, kind, V._PG, knn=knn24)
    if "excess" in srcs:
        scorers["excess"] = V.ExcessProxy(xs, ys, kind, knn24)
    if srcs & set(NEURO):
        scorers["neurolkh"] = scorers["neurolkhfree"] = NeuroLKHScorer(name)
    w0 = None
    if srcs & (set(V.ONLINE) | set(CREDIT)):
        import online_targeting as OT
        w0 = OT.load_prior()
    init, c0, t_start = F.start_tour(xs, ys, kind, nbr, 7919 * (seed + 1) + n)
    kick_seed = 104729 + 1009 * seed
    rng = np.random.default_rng(zlib.crc32(f"{name}|{seed}".encode()))
    order = [strategies[i] for i in rng.permutation(len(strategies))]
    default_pool_size = F.pool_size
    runs = {}
    for s in order:
        src, geom, frac, prob = V.parse_strategy(s)
        F.pool_size = (lambda k, fr=frac: max(12, int(k * fr)))
        charged = 0.0
        try:
            if mode == "kicks":
                if src in ("random", "knn"):
                    res = run_fixed_kicks(xs, ys, kind, nbr, init, c0, f"{src}_{geom}", kick_seed, n_kicks,
                                          knn24=knn24, prob=prob)
                else:
                    res = run_fixed_kicks(xs, ys, kind, nbr, init, c0, f"gnn_{geom}", kick_seed, n_kicks,
                                          gnn=scorers[src], knn24=knn24, prob=prob)
            elif src in V.ONLINE:
                import online_targeting as OT
                res = OT.run_online(xs, ys, kind, nbr, init, c0, kick_seed, budget, knn24, V._P,
                                    prob=prob, **V.ONLINE[src](w0))
            elif src in CREDIT:
                import online_credit as OC
                res = OC.run_online_credit(xs, ys, kind, nbr, init, c0, kick_seed, budget, knn24, V._P,
                                           w0=w0, prob=prob, credit=CREDIT[src])
            elif src in ("random", "knn"):
                res = F.run_strategy(xs, ys, kind, nbr, init, c0, f"{src}_{geom}", kick_seed, budget,
                                     knn24=knn24, prob=prob)
            else:
                if src == "neurolkh":
                    charged = scorers[src].t_forward
                res = F.run_strategy(xs, ys, kind, nbr, init, c0, f"gnn_{geom}", kick_seed,
                                     max(budget - charged, 0.0), gnn=scorers[src], knn24=knn24, prob=prob)
        finally:
            F.pool_size = default_pool_size
        res.update({"strategy": s, "pool_frac": frac, "target_prob": prob, "charged_forward_s": charged})
        res["gap_pct"] = 100.0 * (res["cost"] - bks) / bks
        res["trace_gap_pct"] = {t: 100.0 * (c - bks) / bks for t, c in res["trace"].items()}
        runs[s] = res
    payload = {"suite": suite, "instance": name, "n": n, "bks": bks, "seed": seed, "mode": mode,
               "budget": budget, "n_kicks": n_kicks, "order": order, "start_cost": c0,
               "start_gap_pct": 100.0 * (c0 - bks) / bks, "start_time": t_start, "runs": runs,
               "pid": os.getpid(), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
               "note": "exploratory post-protocol session (PROTOCOL_v4_exploratory.json)"}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)
    return f"{name} s{seed} n={n} " + " ".join(f"{k}={runs[k]['gap_pct']:.3f}" for k in strategies)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=("time", "kicks"), default="time")
    ap.add_argument("--budget", type=float, default=10.0)
    ap.add_argument("--kicks", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--strategies", required=True)
    ap.add_argument("--smoke", action="store_true", help="development check only (no protocol lock)")
    a = ap.parse_args()
    if a.smoke and a.suite == "holdout2":
        raise SystemExit("smoke runs are for development suites only")
    if not a.smoke and not protocol_frozen():
        raise SystemExit("locked: freeze PROTOCOL_v4_exploratory.json (+ .sha256) before any run.")
    strategies = a.strategies.split(",")
    for s in strategies:
        V.parse_strategy(s)
    insts = V.instances(a.suite)
    if a.smoke:
        insts = insts[:1]
    out_dir = OUT_ROOT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = out_dir / "run_meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"suite": a.suite, "mode": a.mode, "budget": a.budget, "kicks": a.kicks,
                                    "seeds": list(range(a.first_seed, a.first_seed + a.seeds)),
                                    "strategies": strategies, "workers": a.workers,
                                    "instances": [r[0] for r in insts],
                                    "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    tasks = [(a.suite, nm, str(p), b, s, a.mode, a.budget, a.kicks, strategies, str(out_dir))
             for nm, p, b in reversed(insts) for s in range(a.first_seed, a.first_seed + a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=V._init) as pool:
        for i, msg in enumerate(pool.imap_unordered(task, tasks)):
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:6.0f}s] {msg}", flush=True)


if __name__ == "__main__":
    main()
