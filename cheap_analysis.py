# -*- coding: utf-8 -*-
"""Analysis of the exploratory cheap-scorer and large-instance campaigns.

    py -3.13 cheap_analysis.py   -> results/kick_targeting/cheap_summary.json
"""
from __future__ import annotations

import json

import numpy as np

from kick_oracle_analysis import KT, block, load, paired, strat


def ops(tag):
    o = {}
    for f in (KT / tag).glob("*__s*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        for s, run in r["runs"].items():
            o.setdefault(s, []).append((run["kicks"] / max(run["elapsed"], 1e-9),
                                        run["model_time"] / max(run["elapsed"], 1e-9),
                                        run.get("refreshes", 0), run["accepted"],
                                        run["stale_cuts"] / max(run["guided_cuts"], 1)))
    return {s: dict(zip(("kicks_per_s", "model_share", "refreshes", "accepted", "stale_share"),
                        map(float, np.mean(v, axis=0)))) for s, v in o.items()}


def main():
    S = {}
    pairs_g = [("cheap_global", "knn_global"), ("cheap_global", "gnn_global"), ("cheap_global", "random_global"),
               ("gnn_global", "knn_global"), ("gnn_global", "random_global"), ("knn_global", "random_global")]
    res, tr = load("vlsi_primary_cheap_10s")
    if res:
        S["primary_cheap_10s"] = block(res, tr, pairs_g)
        S["primary_cheap_10s"]["ops"] = ops("vlsi_primary_cheap_10s")
    res, tr = load("vlsi_large_30s")
    if res:
        pairs = pairs_g + [("random_local", "random_global"), ("knn_local", "random_local"),
                           ("knn_local", "knn_global")]
        S["large_30s"] = block(res, tr, pairs)
        S["large_30s"]["ops"] = ops("vlsi_large_30s")
        n_of = {}
        for f in (KT / "vlsi_large_30s").glob("*__s*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            n_of[r["instance"]] = r["n"]
        S["large_30s"]["per_instance"] = {
            k: {"n": n_of[k], **{s: float(np.mean(v)) for s, v in res[k].items()}} for k in sorted(res, key=n_of.get)}
        for lo, hi, lab in ((0, 10000, "n<10000"), (10000, 10 ** 9, "n>=10000")):
            sub = {k: v for k, v in res.items() if lo <= n_of[k] < hi}
            S["large_30s"][lab] = {f"{a}-{b}": paired(strat(sub, a), strat(sub, b)) for a, b in pairs
                                   if all(a in v and b in v for v in sub.values())}
    (KT / "cheap_summary.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
    for key, b in S.items():
        print("==", key, {k: round(x, 3) for k, x in b["mean_gap"].items()})
        for k, v in b["pairs"].items():
            print(f"  {k:30s} d={v['mean_diff']:+.4f} A/T/B={v['a_better']}/{v['ties']}/{v['b_better']} "
                  f"p={v['wilcoxon_p']:.3g} CI=[{v['bootstrap95'][0]:+.3f},{v['bootstrap95'][1]:+.3f}]")
        for s, o in b["ops"].items():
            print(f"  ops {s:14s} " + " ".join(f"{k}={x:.3f}" for k, x in o.items()))
        for cp, a in b["anytime"].items():
            print(f"  t={cp:>5s} " + " ".join(f"{k}={x:.3f}" for k, x in a.items() if k != "pairs"))


if __name__ == "__main__":
    main()
