# -*- coding: utf-8 -*-
"""Mechanism check for the cheap learned scorer: wrong-edge AUC / pool precision and
refresh cost on the same 58 VLSI states as kick_mechanism_analysis.py (ILS 2 s tours).

    py -3.13 cheap_mechanism.py   -> results/kick_targeting/cheap_mechanism_summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

import fast_ils as F
import gnn_trained as gt
from cheap_scorer import CheapScorer, load_params
from kick_mechanism_analysis import _ils_tour, auc
from kick_targeting_experiment import load_xy, suite_instances

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "kick_targeting"


def timeit(f, reps=5):
    f()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


def main():
    F.warmup()
    w, _ = gt.get_weights(HERE / "models" / "gnn_trained.npz")
    P = load_params()
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
        cheap = CheapScorer(xs, ys, kind, P, knn=knn24)
        t0, c0, _ = F.start_tour(xs, ys, kind, nbr, 7919 * 1 + n)
        tour = np.asarray(_ils_tour(xs, ys, kind, nbr, t0, c0, 2.0), dtype=np.int32)
        v = np.roll(tour, -1)
        label = np.array([0 if (min(a, b), max(a, b)) in ref_edges else 1 for a, b in zip(tour, v)])
        m = F.pool_size(n)
        scores = {"gnn": 1.0 - gnn.edge_prob(tour), "cheap": -cheap.logit(tour),
                  "knn": F.knn_rank_scores(tour, knn24, xs, ys, kind)}
        row = {"instance": name, "n": n, "base_rate_wrong": float(label.mean())}
        for k, s in scores.items():
            top = np.argpartition(-s, m - 1)[:m]
            row[f"auc_{k}"] = auc(s, label)
            row[f"prec_{k}"] = float(label[top].mean())
        row["ms_gnn"] = 1e3 * timeit(lambda: gnn.edge_prob(tour), 3)
        row["ms_cheap"] = 1e3 * timeit(lambda: cheap.edge_prob(tour))
        row["ms_knn"] = 1e3 * timeit(lambda: F.knn_rank_scores(tour, knn24, xs, ys, kind))
        rows.append(row)
        print(f"{name:9s} n={n:5d} AUC gnn={row['auc_gnn']:.3f} cheap={row['auc_cheap']:.3f} "
              f"knn={row['auc_knn']:.3f} | prec gnn={row['prec_gnn']:.3f} cheap={row['prec_cheap']:.3f} "
              f"knn={row['prec_knn']:.3f} | ms gnn={row['ms_gnn']:.1f} cheap={row['ms_cheap']:.2f} "
              f"knn={row['ms_knn']:.2f}", flush=True)
    s = {"n_instances": len(rows)}
    for k in ("base_rate_wrong", "auc_gnn", "auc_cheap", "auc_knn", "prec_gnn", "prec_cheap", "prec_knn",
              "ms_gnn", "ms_cheap", "ms_knn"):
        s[k] = float(np.mean([r[k] for r in rows]))
    for a, b in (("cheap", "knn"), ("gnn", "cheap")):
        for met in ("auc", "prec"):
            d = np.array([r[f"{met}_{a}"] - r[f"{met}_{b}"] for r in rows])
            nz = d[np.abs(d) > 0]
            s[f"{met}_{a}_minus_{b}"] = {"mean": float(d.mean()), "a_higher": int((d > 0).sum()),
                                         "wilcoxon_p": float(stats.wilcoxon(nz).pvalue) if len(nz) > 5 else 1.0}
    (OUT / "cheap_mechanism_summary.json").write_text(json.dumps({"summary": s, "rows": rows}, indent=1),
                                                      encoding="utf-8")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
