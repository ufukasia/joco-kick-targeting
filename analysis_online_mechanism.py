# -*- coding: utf-8 -*-
"""Post-hoc mechanism analysis (exploratory, after the confirmatory test) for holdout2_10s.

1. Run-level counters per strategy: kicks/s, acceptance rate, model-time share, stale-cut share.
2. Online scorer drift: for each (instance, seed 0) online run, the pool chosen by the learned
   weights w_final vs the prior w0 on the instance's (seed-0) start tour: mean relative length
   of pool edges (d(u,v) / mean 5-NN distance), overlap with the excess-proxy pool, cosine(w0, w_final).
Writes results/kick_targeting/holdout2_10s_mechanism.json.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import fast_ils as F
import holdout2_suite
import online_targeting as OT
from cheap_scorer import MODEL, load_params, neighbour_dists
from kick_targeting_experiment import load_xy
from v2_targeting_experiment import ExcessProxy

R = Path("results/kick_targeting")
TAG = R / "holdout2_10s"


def counters():
    acc = defaultdict(lambda: defaultdict(list))
    for f in TAG.glob("*__s*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s, r in d["runs"].items():
            acc[s]["kicks_per_s"].append(r["kicks"] / r["elapsed"])
            acc[s]["accept_rate"].append(r["accepted"] / max(r["kicks"], 1))
            acc[s]["model_share"].append(r["model_time"] / r["elapsed"])
            acc[s]["stale_share"].append(r["stale_cuts"] / max(r["guided_cuts"], 1))
    return {s: {k: float(np.mean(v)) for k, v in m.items()} for s, m in acc.items()}


def drift():
    F.warmup()
    P = load_params(MODEL)
    OT.warmup(P)
    w0 = OT.load_prior()
    out = []
    for name, path, bks in holdout2_suite.instances():
        d = json.loads((TAG / f"{name}__s0.json").read_text(encoding="utf-8"))
        wf = np.array(d["runs"]["onlinewarm_global"]["w_final"])
        xs, ys, kind = load_xy(Path(path))
        n = len(xs)
        nbr = F.build_knn(xs, ys, kind, 10)
        knn24 = F.build_knn(xs, ys, kind, min(24, n - 1))
        tour, _, _ = F.start_tour(xs, ys, kind, nbr, 7919 * 1 + n)
        pol = OT.OnlinePolicy(xs, ys, kind, knn24, P, w0)
        X = pol.features(tour)
        m = max(12, int(0.06 * n))
        dk = neighbour_dists(knn24, xs, ys, kind)
        u = np.ascontiguousarray(tour, dtype=np.int32)
        v = np.ascontiguousarray(np.roll(u, -1))
        rel = F._edge_lengths(u, v, xs, ys, kind) / (0.5 * (dk[u, 1] + dk[v, 1]) + 1e-9)
        top = lambda s: set(np.argpartition(-s, m - 1)[:m].tolist())
        p0 = top(X[:, :-1] @ w0[:-1])
        pf = top(X[:, :-1] @ wf[:-1])
        pe = top(ExcessProxy(xs, ys, kind, knn24).edge_prob(tour) * -1.0)
        out.append({"instance": name, "n": n,
                    "rel_len_all": float(rel.mean()),
                    "rel_len_pool_prior": float(rel[list(p0)].mean()),
                    "rel_len_pool_learned": float(rel[list(pf)].mean()),
                    "rel_len_pool_excess": float(rel[list(pe)].mean()),
                    "overlap_prior_excess": len(p0 & pe) / m,
                    "overlap_learned_excess": len(pf & pe) / m,
                    "overlap_learned_prior": len(pf & p0) / m,
                    "cos_w0_wfinal": float(w0[:-1] @ wf[:-1] / (np.linalg.norm(w0[:-1]) * np.linalg.norm(wf[:-1]) + 1e-12))})
        print(name, {k: round(v, 3) for k, v in out[-1].items() if isinstance(v, float)}, flush=True)
    return out


if __name__ == "__main__":
    c = counters()
    for s, m in c.items():
        print(f"{s:28s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))
    dr = drift()
    summ = {k: float(np.mean([r[k] for r in dr])) for k in dr[0] if k not in ("instance", "n")}
    print("MEAN", {k: round(v, 3) for k, v in summ.items()})
    (R / "holdout2_10s_mechanism.json").write_text(json.dumps({"counters": c, "drift": dr, "drift_mean": summ},
                                                              indent=1), encoding="utf-8")
