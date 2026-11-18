# -*- coding: utf-8 -*-
"""Statistics for the kick-targeting campaign (see results/kick_targeting/PROTOCOL.json).

    py -3.13 kick_targeting_analysis.py vlsi_primary_10s [more tags...]
Writes results/kick_targeting/<tag>_summary.json and prints a compact report.
Quality is the % gap to the optimum / BKS: LOWER IS BETTER.  A difference
"A vs B" is gap(A) - gap(B); negative means A is better.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE / "results" / "kick_targeting"

PRIMARY = {
    "H1": ("gnn_global", "random_global"),
    "H2": ("gnn_local", "random_local"),
    "H3": ("gnn_global", "knn_global"),
    "H4": ("gnn_local", "knn_local"),
}
SECONDARY = {
    "S1_knn_global_vs_random_global": ("knn_global", "random_global"),
    "S2_knn_local_vs_random_local": ("knn_local", "random_local"),
    "S3_random_local_vs_random_global": ("random_local", "random_global"),
}
MARGIN = 0.10


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        out[i] = running
    return out.tolist()


def load(tag):
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((ROOT / tag).glob("*__s*.json"))]
    if not recs:
        raise SystemExit(f"no records for {tag}")
    return recs


def per_instance(recs, key="gap_pct", checkpoint=None):
    """instance -> strategy -> mean over seeds."""
    acc: dict[str, dict[str, list[float]]] = {}
    meta = {}
    for r in recs:
        meta[r["instance"]] = r["n"]
        for s, run in r["runs"].items():
            v = run[key] if checkpoint is None else run["trace_gap_pct"][checkpoint]
            acc.setdefault(r["instance"], {}).setdefault(s, []).append(v)
    return {i: {s: float(np.mean(v)) for s, v in d.items()} for i, d in acc.items()}, meta, acc


def compare(pi, a, b, rng_seed=17):
    names = sorted(pi)
    d = np.array([pi[i][a] - pi[i][b] for i in names])
    nz = d[np.abs(d) > 1e-12]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 1 else 1.0
    # rank-biserial (positive = A better, i.e. lower gap)
    if len(nz):
        ranks = stats.rankdata(np.abs(nz))
        rb = float((ranks[nz < 0].sum() - ranks[nz > 0].sum()) / ranks.sum())
    else:
        rb = 0.0
    rng = np.random.default_rng(rng_seed)
    boots = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # TOST (paired t on instance means) with margin MARGIN
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.inf
    t1 = (d.mean() + MARGIN) / se
    t2 = (d.mean() - MARGIN) / se
    df = len(d) - 1
    p_tost = float(max(1 - stats.t.cdf(t1, df), stats.t.cdf(t2, df)))
    return {
        "a": a, "b": b, "n_instances": len(d),
        "mean_diff": float(d.mean()), "median_diff": float(np.median(d)),
        "a_better": int((d < -1e-12).sum()), "ties": int((np.abs(d) <= 1e-12).sum()),
        "b_better": int((d > 1e-12).sum()),
        "wilcoxon_p": p, "rank_biserial_a_better": rb,
        "bootstrap95": [float(lo), float(hi)],
        "tost_p_margin_0.10": p_tost,
    }


def summarize(tag):
    recs = load(tag)
    pi, meta, raw = per_instance(recs)
    strategies = sorted(next(iter(pi.values())).keys())
    out = {"tag": tag, "n_instances": len(pi), "n_records": len(recs),
           "seeds_per_instance": {i: len(raw[i][strategies[0]]) for i in raw}}
    out["mean_gap"] = {s: float(np.mean([pi[i][s] for i in pi])) for s in strategies}
    out["median_gap"] = {s: float(np.median([pi[i][s] for i in pi])) for s in strategies}
    # operational
    ops = {}
    for s in strategies:
        kicks = [r["runs"][s]["kicks"] / r["runs"][s]["elapsed"] for r in recs]
        acc = [r["runs"][s]["accepted"] for r in recs]
        mt = [r["runs"][s]["model_time"] / r["runs"][s]["elapsed"] for r in recs]
        gc = sum(r["runs"][s]["guided_cuts"] for r in recs)
        sc = sum(r["runs"][s]["stale_cuts"] for r in recs)
        ops[s] = {"kicks_per_s_mean": float(np.mean(kicks)), "accepted_mean": float(np.mean(acc)),
                  "model_time_frac_mean": float(np.mean(mt)),
                  "stale_cut_rate": (sc / gc) if gc else None,
                  "kicks_total_mean": float(np.mean([r["runs"][s]["kicks"] for r in recs]))}
    out["operational"] = ops
    out["start_gap_mean"] = float(np.mean([r["start_gap_pct"] for r in recs]))
    prim = {h: compare(pi, a, b) for h, (a, b) in PRIMARY.items() if a in strategies and b in strategies}
    hp = holm([prim[h]["wilcoxon_p"] for h in prim])
    for h, p in zip(prim, hp):
        prim[h]["holm_p"] = p
    out["primary"] = prim
    out["secondary"] = {h: compare(pi, a, b) for h, (a, b) in SECONDARY.items()
                        if a in strategies and b in strategies}
    # anytime
    cps = sorted(recs[0]["runs"][strategies[0]]["trace_gap_pct"], key=float)
    anytime = {}
    for cp in cps:
        pic, _, _ = per_instance(recs, checkpoint=cp)
        anytime[cp] = {"mean_gap": {s: float(np.mean([pic[i][s] for i in pic])) for s in strategies},
                       "H": {h: {k: compare(pic, a, b)[k] for k in ("mean_diff", "wilcoxon_p", "a_better", "b_better")}
                             for h, (a, b) in {**PRIMARY, **SECONDARY}.items()
                             if a in strategies and b in strategies}}
    out["anytime"] = anytime
    # size strata
    strata = {"n<2500": [i for i in pi if meta[i] < 2500], "n>=2500": [i for i in pi if meta[i] >= 2500]}
    out["strata"] = {}
    for k, ids in strata.items():
        if len(ids) >= 6:
            sub = {i: pi[i] for i in ids}
            out["strata"][k] = {h: compare(sub, a, b) for h, (a, b) in {**PRIMARY, **SECONDARY}.items()
                                if a in strategies and b in strategies}
    out["per_instance"] = {i: {"n": meta[i], **pi[i]} for i in sorted(pi, key=lambda x: meta[x])}
    (ROOT / f"{tag}_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def report(o):
    print(f"== {o['tag']}: {o['n_instances']} instances, {o['n_records']} records; start gap {o['start_gap_mean']:.3f}%")
    for s, g in sorted(o["mean_gap"].items(), key=lambda kv: kv[1]):
        op = o["operational"][s]
        print(f"  {s:14s} gap={g:.4f}% med={o['median_gap'][s]:.4f}  kicks/s={op['kicks_per_s_mean']:.0f} "
              f"model={100 * op['model_time_frac_mean']:.1f}% stale={op['stale_cut_rate']}")
    for grp in ("primary", "secondary"):
        for h, c in o[grp].items():
            extra = f" holm={c['holm_p']:.4f}" if "holm_p" in c else ""
            print(f"  {h:34s} {c['a']}-{c['b']}: d={c['mean_diff']:+.4f} CI[{c['bootstrap95'][0]:+.3f},{c['bootstrap95'][1]:+.3f}] "
                  f"A/T/B={c['a_better']}/{c['ties']}/{c['b_better']} p={c['wilcoxon_p']:.4f}{extra} tost={c['tost_p_margin_0.10']:.4f}")
    for cp, d in o["anytime"].items():
        print(f"  t={cp:>5s} " + " ".join(f"{s}={g:.3f}" for s, g in sorted(d["mean_gap"].items())))


if __name__ == "__main__":
    for tag in sys.argv[1:]:
        report(summarize(tag))
