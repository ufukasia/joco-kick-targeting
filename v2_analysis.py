# -*- coding: utf-8 -*-
"""Paired comparisons and power analysis for the PROTOCOL_v2 campaign.

    py -3.13 v2_analysis.py compare --tag dev_excess_vlsi_old --pairs excess_global:knn_global,gain_global:knn_global
    py -3.13 v2_analysis.py power   --tag vlsi_primary_10s --pair gnn_global:knn_global

Unit of inference: instance (seeds averaged within instance).  Difference = first - second in
gap percentage points at the final budget (negative: first is better).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

OUT_ROOT = Path(__file__).resolve().parent / "results" / "kick_targeting"


def per_instance(tag, checkpoint=None):
    """instance -> strategy -> mean gap over seeds (only seeds where every strategy finished)."""
    acc = defaultdict(lambda: defaultdict(list))
    ns = {}
    for f in sorted((OUT_ROOT / tag).glob("*__s*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        ns[d["instance"]] = d["n"]
        for s, r in d["runs"].items():
            g = r["gap_pct"] if checkpoint is None else r["trace_gap_pct"][str(checkpoint)]
            acc[d["instance"]][s].append(g)
    return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in acc.items()}, ns


def paired(tag, a, b, checkpoint=None, boot=10000, seed=0):
    data, ns = per_instance(tag, checkpoint)
    keys = [k for k in data if a in data[k] and b in data[k]]
    d = np.array([data[k][a] - data[k][b] for k in keys])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(len(d), size=(boot, len(d)))].mean(1)
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return {"pair": f"{a} - {b}", "n_inst": len(d), "mean": float(d.mean()), "median": float(np.median(d)),
            "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "wins_first": int((d < 0).sum()), "ties": int((d == 0).sum()), "wins_second": int((d > 0).sum()),
            "sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"), "p_wilcoxon": float(p)}


def tost(d, margin):
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    p_lo = 1 - stats.t.cdf((d.mean() + margin) / se, n - 1)
    p_hi = stats.t.cdf((d.mean() - margin) / se, n - 1)
    return float(max(p_lo, p_hi))


def n_required(sd, delta, alpha=0.05, power=0.8, are=0.955):
    """Instances needed for a two-sided Wilcoxon signed-rank test (normal approximation / ARE)."""
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return int(np.ceil((z * sd / delta) ** 2 / are))


def cmd_compare(a):
    rows = []
    for pr in a.pairs.split(","):
        x, y = pr.split(":")
        r = paired(a.tag, x, y, a.checkpoint)
        rows.append(r)
        print(f"{r['pair']:34s} n={r['n_inst']:3d} mean={r['mean']:+.4f} CI[{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] "
              f"W/T/L={r['wins_first']}/{r['ties']}/{r['wins_second']} p={r['p_wilcoxon']:.4g}")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")


def cmd_power(a):
    x, y = a.pair.split(":")
    r = paired(a.tag, x, y)
    sd = r["sd"]
    print(f"{r['pair']}: n={r['n_inst']} mean={r['mean']:+.4f} sd(per-instance diff)={sd:.4f}")
    for alpha in (0.05, 0.0125):
        for delta in (0.02, 0.03, 0.04, 0.05, 0.06):
            print(f"  alpha={alpha:<7} delta={delta:.2f}  instances for 80% power: {n_required(sd, delta, alpha)}")


def test_hypothesis(d, h, alpha):
    """d = per-instance differences a - b (negative: a better).
    superiority : two-sided Wilcoxon on d; success = p < alpha and mean(d) has the stated sign.
    noninferiority (a not worse than b by more than margin): one-sided Wilcoxon of d - margin < 0."""
    if h["test"] == "superiority":
        p = float(stats.wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
        sign_ok = d.mean() < 0 if h["direction"] == "a_better" else d.mean() > 0
        return p, bool(p < alpha and sign_ok)
    if h["test"] == "noninferiority":
        p = float(stats.wilcoxon(d - h["margin"], alternative="less").pvalue)
        return p, bool(p < alpha)
    raise ValueError(h["test"])


def cmd_confirm(a):
    """Fixed-sequence confirmatory analysis exactly as specified in the frozen PROTOCOL_v2.json."""
    proto = json.loads((OUT_ROOT / "PROTOCOL_v2.json").read_text(encoding="utf-8"))
    alpha = proto["alpha"]
    data, _ = per_instance(a.tag)
    open_gate, out = True, []
    for h in proto["confirmatory_hypotheses"]:
        keys = [k for k in data if h["a"] in data[k] and h["b"] in data[k]]
        d = np.array([data[k][h["a"]] - data[k][h["b"]] for k in keys])
        p, ok = test_hypothesis(d, h, alpha)
        r = {"id": h["id"], "a": h["a"], "b": h["b"], "test": h["test"], "n_inst": len(d),
             "mean": float(d.mean()), "p": p, "criterion_met": ok, "tested": open_gate,
             "rejected": bool(open_gate and ok)}
        out.append(r)
        print(f"{h['id']}: {h['a']} - {h['b']} [{h['test']}] n={len(d)} mean={d.mean():+.4f} p={p:.4g} "
              f"{'REJECTED H0' if r['rejected'] else ('not rejected' if open_gate else 'not tested (gate closed)')}")
        open_gate = open_gate and ok
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("compare")
    c.add_argument("--tag", required=True)
    c.add_argument("--pairs", required=True)
    c.add_argument("--checkpoint", type=float, default=None)
    c.add_argument("--out", default=None)
    p = sp.add_parser("power")
    p.add_argument("--tag", required=True)
    p.add_argument("--pair", required=True)
    f = sp.add_parser("confirm")
    f.add_argument("--tag", required=True)
    f.add_argument("--out", default=None)
    a = ap.parse_args()
    {"compare": cmd_compare, "power": cmd_power, "confirm": cmd_confirm}[a.cmd](a)
