# -*- coding: utf-8 -*-
"""Exploratory analyses (not pre-registered): oracle targeting, fresh-GNN and 60 s budget.

    py -3.13 kick_oracle_analysis.py

Writes results/kick_targeting/exploratory_summary.json.  Unit of analysis is the instance
(seed means), paired Wilcoxon, bootstrap 95% CI -- same conventions as the primary analysis.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

KT = Path(__file__).resolve().parent / "results" / "kick_targeting"
CPS = ("0.1", "0.25", "0.5", "1.0", "2.0", "5.0", "10.0", "20.0", "30.0", "60.0")


def load(tag, seeds=None):
    """-> {instance: {strategy: [gap per seed]}}, plus anytime {cp: {strategy: {inst: [..]}}}."""
    d = KT / tag
    out, tr = {}, {}
    if not d.exists():
        return out, tr
    for f in d.glob("*__s*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        if seeds is not None and r["seed"] not in seeds:
            continue
        for s, run in r["runs"].items():
            out.setdefault(r["instance"], {}).setdefault(s, []).append(run["gap_pct"])
            for cp, g in run.get("trace_gap_pct", {}).items():
                tr.setdefault(str(float(cp)), {}).setdefault(s, {}).setdefault(r["instance"], []).append(g)
    return out, tr


def paired(A, B, rng=np.random.default_rng(0)):
    ks = sorted(set(A) & set(B))
    d = np.array([np.mean(A[k]) - np.mean(B[k]) for k in ks])
    nz = d[np.abs(d) > 1e-12]
    p = float(wilcoxon(nz).pvalue) if len(nz) >= 6 else float("nan")
    bs = [rng.choice(d, len(d)).mean() for _ in range(10000)]
    return {"n": len(ks), "mean_diff": float(d.mean()), "a_better": int((d < -1e-12).sum()),
            "ties": int((np.abs(d) <= 1e-12).sum()), "b_better": int((d > 1e-12).sum()),
            "wilcoxon_p": p, "bootstrap95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def strat(res, s):
    return {k: v[s] for k, v in res.items() if s in v}


def block(res, tr, pairs):
    o = {"mean_gap": {}, "pairs": {}, "anytime": {}}
    strategies = sorted({s for v in res.values() for s in v})
    for s in strategies:
        o["mean_gap"][s] = float(np.mean([np.mean(v) for v in strat(res, s).values()]))
    for a, b in pairs:
        if a in strategies and b in strategies:
            o["pairs"][f"{a}-{b}"] = paired(strat(res, a), strat(res, b))
    for cp in CPS:
        if cp in tr:
            o["anytime"][cp] = {s: float(np.mean([np.mean(v) for v in tr[cp][s].values()])) for s in tr[cp]}
            o["anytime"][cp]["pairs"] = {f"{a}-{b}": paired(tr[cp][a], tr[cp][b])
                                         for a, b in pairs if a in tr[cp] and b in tr[cp]}
    return o


def main():
    S = {}
    prim, prim_tr = load("vlsi_primary_10s")
    G = [("gnn_global", "random_global"), ("gnn_global", "knn_global"), ("knn_global", "random_global")]
    # fresh GNN (no overhead cap)
    fr, fr_tr = load("vlsi_primary_fresh_10s")
    if fr:
        S["fresh_10s"] = block(fr, fr_tr, G)
        # same seeds in the primary run, for the cap vs no-cap contrast
        p5, _ = load("vlsi_primary_10s", seeds=set(range(5)))
        S["fresh_vs_capped_gnn"] = paired(strat(fr, "gnn_global"), strat(p5, "gnn_global"))
        S["fresh_vs_capped_knn"] = paired(strat(fr, "knn_global"), strat(p5, "knn_global"))
        ops = {}
        for f in (KT / "vlsi_primary_fresh_10s").glob("*__s*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            for s, run in r["runs"].items():
                ops.setdefault(s, []).append((run["kicks"] / max(run["elapsed"], 1e-9),
                                              run["model_time"] / max(run["elapsed"], 1e-9),
                                              run["stale_cuts"] / max(run["guided_cuts"], 1),
                                              run.get("refreshes", 0)))
        S["fresh_ops"] = {s: dict(zip(("kicks_per_s", "model_share", "stale_share", "refreshes"),
                                      map(float, np.mean(v, axis=0)))) for s, v in ops.items()}
    # 60 s budget
    l6, l6_tr = load("vlsi_primary_60s")
    if l6:
        S["budget_60s"] = block(l6, l6_tr, G)
    # oracle
    orc, orc_tr = load("vlsi_primary_oracle_10s")
    if orc:
        S["oracle"] = block(orc, orc_tr, [("oracle_global", "knn_global")])
        p3, p3_tr = load("vlsi_primary_10s", seeds={0, 1, 2})
        S["oracle_vs_primary_seeds012"] = {
            "oracle-gnn_global": paired(strat(orc, "oracle_global"), strat(p3, "gnn_global")),
            "oracle-random_global": paired(strat(orc, "oracle_global"), strat(p3, "random_global")),
            "knn_rerun-knn_primary": paired(strat(orc, "knn_global"), strat(p3, "knn_global")),
            "primary_mean_gap": {s: float(np.mean([np.mean(v) for v in strat(p3, s).values()]))
                                 for s in ("random_global", "gnn_global", "knn_global")},
        }
        ops = []
        for f in (KT / "vlsi_primary_oracle_10s").glob("*__s*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))["runs"]["oracle_global"]
            ops.append((r["kicks"] / r["elapsed"], r["model_time"] / r["elapsed"], r["accepted"] / max(r["kicks"], 1)))
        S["oracle_ops"] = dict(zip(("kicks_per_s", "model_share", "accept_rate"), map(float, np.mean(ops, axis=0))))
    (KT / "exploratory_summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    print(json.dumps(S, indent=1)[:6000])


if __name__ == "__main__":
    main()
