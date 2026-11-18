# -*- coding: utf-8 -*-
"""What does the LKH-GNN know that a kNN rank does not?

For every VLSI instance with an LKH reference tour (lkh_runner.py ref) we take
two kinds of local optima of the fast ILS testbed (the seed-0 start tour, and
the incumbent after 2 s of random_global ILS) and label each tour edge
"wrong" if it is absent from the LKH reference tour.  Each suspicion score is
evaluated as a detector of wrong edges: ROC-AUC, precision in the kick pool
(top m = max(12, 0.06 n) edges) against the base rate, and the rank
correlation between the GNN and the training-free proxies.

    py -3.13 kick_mechanism_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

import fast_ils as F
import gnn_trained as gt
from kick_targeting_experiment import load_xy, suite_instances

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "kick_targeting"


def auc(score, label):
    """P(score of a wrong edge > score of a right edge); ties count 1/2."""
    pos = label == 1
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    r = stats.rankdata(score)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def analyse_tour(tour, ref_edges, scores, m):
    n = len(tour)
    u = tour
    v = np.roll(tour, -1)
    label = np.array([0 if (min(a, b), max(a, b)) in ref_edges else 1 for a, b in zip(u, v)])
    row = {"base_rate_wrong": float(label.mean()), "n_wrong": int(label.sum())}
    for k, s in scores.items():
        top = np.argpartition(-s, m - 1)[:m]
        row[f"auc_{k}"] = auc(s, label)
        row[f"prec_{k}"] = float(label[top].mean())
    row["spearman_gnn_knn"] = float(stats.spearmanr(scores["gnn"], scores["knn"]).statistic)
    row["spearman_gnn_rellen"] = float(stats.spearmanr(scores["gnn"], scores["rellen"]).statistic)
    top_g = set(np.argpartition(-scores["gnn"], m - 1)[:m].tolist())
    top_k = set(np.argpartition(-scores["knn"], m - 1)[:m].tolist())
    row["pool_overlap_gnn_knn"] = len(top_g & top_k) / m
    return row


def main():
    F.warmup()
    w, _ = gt.get_weights(HERE / "models" / "gnn_trained.npz")
    rows = []
    for name, path, bks in suite_instances("vlsi"):
        ref_f = OUT / "lkh_ref" / f"{name}.json"
        if not ref_f.exists():
            continue
        ref = json.loads(ref_f.read_text(encoding="utf-8"))
        if not ref.get("ok"):
            continue
        xs, ys, kind = load_xy(Path(path))
        n = len(xs)
        rt = ref["tour"]
        ref_edges = {(min(a, b), max(a, b)) for a, b in zip(rt, rt[1:] + rt[:1])}
        nbr = F.build_knn(xs, ys, kind, 10)
        knn24 = F.build_knn(xs, ys, kind, min(24, n - 1))
        gnn = F.GNNScorer(xs, ys, kind, w)
        start_seed = 7919 * 1 + n
        t0, c0, _ = F.start_tour(xs, ys, kind, nbr, start_seed)
        t_ils = _ils_tour(xs, ys, kind, nbr, t0, c0, 2.0)
        m = F.pool_size(n)
        for state, tour in (("start_local_opt", t0), ("ils_2s", t_ils)):
            tour = np.asarray(tour, dtype=np.int32)
            u = tour
            v = np.roll(tour, -1)
            ln = F._edge_lengths(u, np.ascontiguousarray(v), xs, ys, kind)
            d1 = F._edge_lengths(np.arange(n, dtype=np.int32),
                                 np.ascontiguousarray(knn24[:, 0], dtype=np.int32), xs, ys, kind)
            scores = {
                "gnn": 1.0 - gnn.edge_prob(tour),
                "knn": F.knn_rank_scores(tour, knn24, xs, ys, kind),
                "rellen": ln / np.maximum(0.5 * (d1[u] + d1[v]), 1e-9),
                "length": ln,
            }
            row = analyse_tour(tour, ref_edges, scores, m)
            row.update({"instance": name, "n": n, "state": state,
                        "tour_gap_pct": 100.0 * (F.tour_cost(tour, xs, ys, kind) - bks) / bks,
                        "ref_gap_pct": ref["gap_pct"]})
            rows.append(row)
            print(f"{name:9s} n={n:5d} {state:16s} wrong={row['base_rate_wrong']:.3f} "
                  f"AUC gnn={row['auc_gnn']:.3f} knn={row['auc_knn']:.3f} rel={row['auc_rellen']:.3f} "
                  f"len={row['auc_length']:.3f} | prec gnn={row['prec_gnn']:.3f} knn={row['prec_knn']:.3f} "
                  f"| rho(g,k)={row['spearman_gnn_knn']:.2f} overlap={row['pool_overlap_gnn_knn']:.2f}", flush=True)
    summ = {}
    for state in ("start_local_opt", "ils_2s"):
        R = [r for r in rows if r["state"] == state]
        if not R:
            continue
        s = {"n_instances": len(R)}
        for k in ("base_rate_wrong", "auc_gnn", "auc_knn", "auc_rellen", "auc_length",
                  "prec_gnn", "prec_knn", "prec_rellen", "prec_length",
                  "spearman_gnn_knn", "spearman_gnn_rellen", "pool_overlap_gnn_knn"):
            s[k] = float(np.mean([r[k] for r in R]))
        dg = np.array([r["auc_gnn"] - r["auc_knn"] for r in R])
        s["auc_gnn_minus_knn"] = {"mean": float(dg.mean()), "gnn_higher": int((dg > 0).sum()),
                                  "wilcoxon_p": float(stats.wilcoxon(dg).pvalue)}
        dp = np.array([r["prec_gnn"] - r["prec_knn"] for r in R])
        s["prec_gnn_minus_knn"] = {"mean": float(dp.mean()), "gnn_higher": int((dp > 0).sum()),
                                   "wilcoxon_p": float(stats.wilcoxon(dp[np.abs(dp) > 0]).pvalue)
                                   if (np.abs(dp) > 0).sum() else 1.0}
        summ[state] = s
    (OUT / "mechanism_summary.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=1),
                                               encoding="utf-8")
    print(json.dumps(summ, indent=1))


def _ils_tour(xs, ys, kind, nbr, init, c0, budget):
    """Same as run_strategy('random_global') but returns the incumbent tour."""
    import time
    n = len(xs)
    tour = init.copy()
    pos = np.empty(n, dtype=np.int32)
    pos[tour] = np.arange(n, dtype=np.int32)
    bt, bp = tour.copy(), pos.copy()
    q = np.empty(n, dtype=np.int32)
    inq = np.zeros(n, dtype=np.uint8)
    qht = np.zeros(3, dtype=np.int64)
    buf = np.empty(n, dtype=np.int32)
    st = np.zeros(4, dtype=np.int64)
    F.seed_rng(104729)
    pu = np.zeros(1, dtype=np.int32)
    cost = c0
    t = time.perf_counter()
    while time.perf_counter() - t < budget:
        cost = F.run_kicks(tour, pos, bt, bp, cost, n, xs, ys, kind, nbr, 64, 0, False,
                           pu, pu, 0, 0.75, 50, q, inq, qht, buf, st)
    return bt


if __name__ == "__main__":
    main()
