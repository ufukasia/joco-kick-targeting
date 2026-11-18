# -*- coding: utf-8 -*-
"""What kind of edges does each signal put in the kick pool?  (exploratory diagnostic)

On the 58 VLSI ILS-2 s states of kick_mechanism_analysis.py: for the top-m pool of each
signal (GNN, cheap learned scorer, kNN proxy) and for a uniform sample of truly wrong edges
(the oracle pool), report precision, the mean mutual-kNN rank (min of the two directions),
the share of edges with no mutual 24-NN relation, the relative length, and -- the
quantity a kick can actually cash in -- the improvement of the best 2-opt move that
removes the pool edge together with any other tour edge whose endpoint is among the
10 nearest neighbours (the classical "one-sided" repair gain is 0 at a local optimum, so
we measure the LKH-reference repair instead: the length difference between the pool edge
and the reference-tour edges incident to its endpoints).

    py -3.13 pool_composition.py -> results/kick_targeting/pool_composition_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import fast_ils as F
import gnn_trained as gt
from cheap_scorer import CheapScorer, load_params, _ranks
from kick_mechanism_analysis import _ils_tour
from kick_targeting_experiment import load_xy, suite_instances

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "kick_targeting"


def main():
    F.warmup()
    w, _ = gt.get_weights(HERE / "models" / "gnn_trained.npz")
    P = load_params()
    rows = []
    rng = np.random.default_rng(0)
    for name, path, bks in suite_instances("vlsi"):
        ref_f = OUT / "lkh_ref" / f"{name}.json"
        if not ref_f.exists():
            continue
        ref = json.loads(ref_f.read_text(encoding="utf-8"))
        if not ref.get("ok"):
            continue
        xs, ys, kind = load_xy(Path(path))
        n = len(xs)
        rt = np.asarray(ref["tour"])
        rs = np.empty(n, dtype=np.int64)
        rp = np.empty(n, dtype=np.int64)
        rs[rt] = np.roll(rt, -1)
        rp[rt] = np.roll(rt, 1)
        nbr = F.build_knn(xs, ys, kind, 10)
        knn24 = F.build_knn(xs, ys, kind, min(24, n - 1))
        gnn = F.GNNScorer(xs, ys, kind, w)
        cheap = CheapScorer(xs, ys, kind, P, knn=knn24)
        t0, c0, _ = F.start_tour(xs, ys, kind, nbr, 7919 * 1 + n)
        tour = np.asarray(_ils_tour(xs, ys, kind, nbr, t0, c0, 2.0), dtype=np.int32)
        u = tour
        v = np.roll(tour, -1)
        wrong = ~((rs[u] == v) | (rp[u] == v))
        duv = F._edge_lengths(u, np.ascontiguousarray(v), xs, ys, kind)
        idx = np.arange(n, dtype=np.int32)
        d1 = F._edge_lengths(idx, np.ascontiguousarray(knn24[:, 0], dtype=np.int32), xs, ys, kind)
        rel = duv / np.maximum(0.5 * (d1[u] + d1[v]), 1e-9)
        rmin = np.minimum(_ranks(u, v, knn24), _ranks(v, u, knn24))
        # "excess length": edge length minus the mean length of the reference-tour edges at its endpoints
        dref_u = 0.5 * (F._edge_lengths(u, np.ascontiguousarray(rs[u].astype(np.int32)), xs, ys, kind)
                        + F._edge_lengths(u, np.ascontiguousarray(rp[u].astype(np.int32)), xs, ys, kind))
        dref_v = 0.5 * (F._edge_lengths(v, np.ascontiguousarray(rs[v].astype(np.int32)), xs, ys, kind)
                        + F._edge_lengths(v, np.ascontiguousarray(rp[v].astype(np.int32)), xs, ys, kind))
        excess = duv - 0.5 * (dref_u + dref_v)
        scale = np.median(duv)
        m = F.pool_size(n)
        pools = {
            "gnn": np.argpartition(gnn.edge_prob(tour), m - 1)[:m],
            "cheap": np.argpartition(cheap.logit(tour), m - 1)[:m],
            "knn": np.argpartition(-F.knn_rank_scores(tour, knn24, xs, ys, kind), m - 1)[:m],
            "oracle": rng.choice(np.nonzero(wrong)[0], size=min(m, int(wrong.sum())), replace=False),
        }
        row = {"instance": name, "n": n, "m": m}
        for k, p in pools.items():
            row[f"{k}_prec"] = float(wrong[p].mean())
            row[f"{k}_rank"] = float(rmin[p].mean())
            row[f"{k}_absent"] = float((rmin[p] >= 30).mean())
            row[f"{k}_rel"] = float(np.median(rel[p]))
            row[f"{k}_excess_wrong"] = float(np.mean(excess[p][wrong[p]]) / scale) if wrong[p].any() else 0.0
            row[f"{k}_excess_sum"] = float(np.sum(np.maximum(excess[p], 0)) / scale / m)
        row["overlap_cheap_knn"] = len(set(pools["cheap"]) & set(pools["knn"])) / m
        row["overlap_gnn_knn"] = len(set(pools["gnn"]) & set(pools["knn"])) / m
        rows.append(row)
        print(name, n, " ".join(f"{k}:prec={row[k + '_prec']:.2f},rank={row[k + '_rank']:.1f},abs={row[k + '_absent']:.2f},"
                                 f"exc={row[k + '_excess_wrong']:.2f}" for k in pools), flush=True)
    S = {"n_instances": len(rows)}
    for key in rows[0]:
        if key in ("instance", "n", "m"):
            continue
        S[key] = float(np.mean([r[key] for r in rows]))
    for a in ("gnn", "cheap", "oracle"):
        d = np.array([r[f"{a}_excess_wrong"] - r["knn_excess_wrong"] for r in rows])
        S[f"excess_wrong_{a}_minus_knn"] = {"mean": float(d.mean()), "a_higher": int((d > 0).sum()),
                                            "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
        d = np.array([r[f"{a}_excess_sum"] - r["knn_excess_sum"] for r in rows])
        S[f"excess_sum_{a}_minus_knn"] = {"mean": float(d.mean()), "a_higher": int((d > 0).sum()),
                                          "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
    (OUT / "pool_composition_summary.json").write_text(json.dumps({"summary": S, "rows": rows}, indent=1),
                                                       encoding="utf-8")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
