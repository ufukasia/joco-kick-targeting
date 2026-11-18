# -*- coding: utf-8 -*-
"""PROTOCOL_v2 campaign driver (development runs on old sets; confirmatory run on holdout2).

fast_ils.py and kick_targeting_experiment.py are hash-registered in PROTOCOL.json and are NOT
modified.  New pool sources are plugged in through run_strategy's `gnn=` object interface
(edge_prob(tour): the m SMALLEST values are kicked) and mapped onto the gnn_* strategies.

Pool sources
    random, knn, gnn       : as in the primary campaign
    cheap, gain            : learned O(1) scorers (cheap_scorer.py; membership / gain target)
    excess                 : training-free gain proxy -- edge length minus the mean of the two
                             endpoints' "expected optimal edge" length, approximated by the mean
                             distance to their 1st and 2nd nearest neighbours.  Untrained analogue
                             of the gain target; added because the paper's own criterion requires
                             testing any learned score against the simplest untrained proxy.
Pool-rule override (development / tuning only): a strategy may carry a suffix
    knn_global@0.04:0.9    -> pool fraction 0.04 (m = max(12, int(0.04 n))), target prob 0.9
implemented by rebinding fast_ils.pool_size inside the worker (the file itself is untouched).

holdout2 is locked: it runs only if results/kick_targeting/PROTOCOL_v2.json exists and its
SHA-256 equals the value in PROTOCOL_v2.sha256 (i.e. the protocol was frozen first).

    py -3.13 v2_targeting_experiment.py --suite vlsi_old --tag dev_excess_vlsi_old --seeds 5 \
        --strategies random_global,knn_global,gain_global,excess_global
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from kick_targeting_experiment import HERE, OUT_ROOT, load_xy, suite_instances

PROTO = OUT_ROOT / "PROTOCOL_v2.json"
PROTO_SHA = OUT_ROOT / "PROTOCOL_v2.sha256"

_W = _P = _PG = None

# online_targeting.run_online variants: name -> kwargs given the warm-start prior w0
ONLINE = {
    "online": lambda w0: {},                                  # cold start, learns from every kick
    "onlinewarm": lambda w0: {"w0": w0},                      # warm start from the gain prior, learns
    "linprior": lambda w0: {"w0": w0, "learn": False, "nu": 0.0},  # static linear gain prior, no learning
}


class ExcessProxy:
    """Training-free gain proxy.  edge_prob returns -excess so the most excessive edges are kicked."""

    def __init__(self, xs, ys, kind, knn):
        import fast_ils as F
        self.F, self.xs, self.ys, self.kind = F, xs, ys, kind
        idx = np.arange(len(xs), dtype=np.int32)
        d1 = F._edge_lengths(idx, np.ascontiguousarray(knn[:, 0], dtype=np.int32), xs, ys, kind)
        d2 = F._edge_lengths(idx, np.ascontiguousarray(knn[:, 1], dtype=np.int32), xs, ys, kind)
        self.dref = 0.5 * (d1 + d2)

    def edge_prob(self, tour):
        u = np.ascontiguousarray(tour, dtype=np.int32)
        v = np.ascontiguousarray(np.roll(u, -1))
        duv = self.F._edge_lengths(u, v, self.xs, self.ys, self.kind)
        return -(duv - 0.5 * (self.dref[u] + self.dref[v]))


def protocol_frozen() -> bool:
    if not (PROTO.exists() and PROTO_SHA.exists()):
        return False
    h = hashlib.sha256(PROTO.read_bytes()).hexdigest()
    return PROTO_SHA.read_text(encoding="utf-8").split()[0].strip().lower() == h


def instances(suite):
    if suite == "holdout2":
        if not protocol_frozen():
            raise SystemExit("holdout2 is locked: freeze PROTOCOL_v2.json (+ .sha256) before any run.")
        import holdout2_suite
        return holdout2_suite.instances()
    if suite == "vlsi_large":
        from cheap_targeting_experiment import instances as ci
        return ci(suite)
    return suite_instances(suite)


def parse_strategy(s):
    """'excess_global@0.04,0.9' -> ('excess', 'global', 0.04, 0.9); defaults 0.06 / 0.75."""
    base, _, rule = s.partition("@")
    src, geom = base.rsplit("_", 1)
    frac, prob = (0.06, 0.75)
    if rule:
        f, p = rule.replace(":", ",").split(",")   # ':' form is safe inside --strategies lists
        frac, prob = float(f), float(p)
    return src, geom, frac, prob


def _init():
    global _W, _P, _PG
    import fast_ils as F
    import gnn_trained as gt
    from cheap_scorer import GAIN_MODEL, MODEL, load_params
    F.warmup()
    _W, _ = gt.get_weights(HERE / "models" / "gnn_trained.npz")
    _P = load_params(MODEL) if MODEL.exists() else None
    _PG = load_params(GAIN_MODEL) if GAIN_MODEL.exists() else None
    import online_targeting as OT
    OT.warmup(_P)


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
    srcs = {parse_strategy(s)[0] for s in strategies}
    scorers = {}
    if "gnn" in srcs:
        scorers["gnn"] = F.GNNScorer(xs, ys, kind, _W)
    if "cheap" in srcs:
        scorers["cheap"] = CheapScorer(xs, ys, kind, _P, knn=knn24)
    if "gain" in srcs:
        scorers["gain"] = CheapScorer(xs, ys, kind, _PG, knn=knn24)
    if "excess" in srcs:
        scorers["excess"] = ExcessProxy(xs, ys, kind, knn24)
    if srcs & set(ONLINE):
        import online_targeting as OT
        w0 = OT.load_prior()
    init, c0, t_start = F.start_tour(xs, ys, kind, nbr, 7919 * (seed + 1) + n)
    kick_seed = 104729 + 1009 * seed
    order = list(strategies)
    r = seed % len(order)
    order = order[r:] + order[:r]
    default_pool_size = F.pool_size
    runs = {}
    for s in order:
        src, geom, frac, prob = parse_strategy(s)
        F.pool_size = (lambda k, fr=frac: max(12, int(k * fr)))
        try:
            if src in ONLINE:
                assert geom == "global", "online sources support the global geometry only"
                res = OT.run_online(xs, ys, kind, nbr, init, c0, kick_seed, budget, knn24, _P,
                                    prob=prob, **ONLINE[src](w0))
            elif src in ("random", "knn"):
                res = F.run_strategy(xs, ys, kind, nbr, init, c0, f"{src}_{geom}", kick_seed, budget,
                                     knn24=knn24, prob=prob)
            else:
                res = F.run_strategy(xs, ys, kind, nbr, init, c0, f"gnn_{geom}", kick_seed, budget,
                                     gnn=scorers[src], knn24=knn24, prob=prob)
        finally:
            F.pool_size = default_pool_size
        res.update({"strategy": s, "pool_frac": frac, "target_prob": prob})
        res["gap_pct"] = 100.0 * (res["cost"] - bks) / bks
        res["trace_gap_pct"] = {t: 100.0 * (c - bks) / bks for t, c in res["trace"].items()}
        runs[s] = res
    payload = {"suite": suite, "instance": name, "n": n, "bks": bks, "seed": seed, "budget": budget,
               "start_cost": c0, "start_gap_pct": 100.0 * (c0 - bks) / bks, "start_time": t_start,
               "runs": runs, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)
    return f"{name} s{seed} n={n} " + " ".join(f"{k}={runs[k]['gap_pct']:.3f}" for k in strategies)


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
    for s in strategies:
        src, geom, _, _ = parse_strategy(s)
        assert src in ("random", "knn", "gnn", "cheap", "gain", "excess", *ONLINE) and geom in ("global", "local"), s
    insts = instances(a.suite)
    out_dir = OUT_ROOT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "suite": a.suite, "seeds": a.seeds, "budget": a.budget, "strategies": strategies,
        "instances": [r[0] for r in insts], "workers": a.workers,
        "confirmatory": a.suite == "holdout2", "machine": os.environ.get("COMPUTERNAME", "?"),
        "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")
    tasks = [(a.suite, nm, str(p), b, s, a.budget, strategies, str(out_dir))
             for nm, p, b in reversed(insts) for s in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_init) as pool:
        for i, msg in enumerate(pool.imap_unordered(task, tasks)):
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:7.0f}s] {msg}", flush=True)


if __name__ == "__main__":
    main()
