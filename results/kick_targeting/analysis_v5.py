# -*- coding: utf-8 -*-
"""Analysis of the EXPLORATORY block registered in PROTOCOL_v4_exploratory.json (B1-B6).

Unit = instance (gap averaged over seeds); paired mean difference [percentage points], 10 000-sample
bootstrap 95 % CI, two-sided Wilcoxon signed-rank p, Holm correction within each family, Hodges-Lehmann
estimate, strata breakdown (statistics helpers imported unchanged from analysis_v4_robustness.py).
Non-inferiority statements, if any, are read from the CI upper bound against +0.05 pt.

    py -3.13 results/kick_targeting/analysis_v5.py   -> v5_analysis.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import v2_analysis as A  # noqa: E402
from analysis_v4_robustness import holm, pair  # noqa: E402

R, E, G, K = "random_global", "excess_global@0.06:0.5", "gnn_global@0.04:0.5", "knn_global@0.04:0.75"
SEL = json.loads((HERE / "v5_selected_rules.json").read_text(encoding="utf-8"))
GA_DEF = "gain_global"
GA = GA_DEF if SEL["gain"]["selected"] == "0.06:0.75" else f"gain_global@{SEL['gain']['selected']}"
NL = f"neurolkh_global@{SEL['neurolkh']['selected']}"
NLF = f"neurolkhfree_global@{SEL['neurolkh']['selected']}"
LP, OW, OP, OS = "linprior_global", "onlinewarm_global", "onlinepos_global", "onlinesucc_global"
LD, LR, LE = "lkh_default", "lkh_ils_random", "lkh_ils_excess"


def present(tag):
    f = sorted((A.OUT_ROOT / tag).glob("*__s*.json"))
    return set(json.loads(f[0].read_text(encoding="utf-8"))["runs"]) if f else set()


def family(tag, contrasts):
    have = present(tag)
    rows = [pair(tag, a, b) for a, b in contrasts if a in have and b in have and a != b]
    for r, ph in zip(rows, holm([r["p_wilcoxon"] for r in rows])):
        r["p_holm"] = ph
    return {"tag": tag, "contrasts": rows}


def counters(tag):
    acc = defaultdict(lambda: defaultdict(list))
    for f in sorted((A.OUT_ROOT / tag).glob("*__s*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s, r in d["runs"].items():
            for k in ("kicks", "accepted", "elapsed", "model_time", "stale_cuts", "guided_cuts", "trials"):
                if r.get(k) is not None:
                    acc[s][k].append(r[k])
            acc[s]["gap_pct"].append(r["gap_pct"])
    out = {}
    for s, c in acc.items():
        o = {"gap_mean": float(np.mean(c["gap_pct"]))}
        if c.get("kicks") and c.get("elapsed"):
            o["kicks_per_s"] = float(np.sum(c["kicks"]) / np.sum(c["elapsed"]))
            o["accept_rate"] = float(np.sum(c["accepted"]) / np.sum(c["kicks"]))
            o["elapsed_mean"] = float(np.mean(c["elapsed"]))
            o["model_share"] = float(np.sum(c["model_time"]) / np.sum(c["elapsed"]))
        if c.get("trials"):
            o["trials_mean"] = float(np.mean(c["trials"]))
        out[s] = o
    return out


def order_effect(tag):
    """Descriptive: kick rate of each arm relative to its own mean, by execution position."""
    rows = []
    for f in sorted((A.OUT_ROOT / tag).glob("*__s*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for i, s in enumerate(d.get("order", [])):
            r = d["runs"][s]
            if r.get("elapsed"):
                rows.append((s, i, r["kicks"] / r["elapsed"]))
    if not rows:
        return {}
    mean = defaultdict(list)
    for s, _, k in rows:
        mean[s].append(k)
    mean = {s: np.mean(v) for s, v in mean.items()}
    bypos = defaultdict(list)
    for s, i, k in rows:
        bypos[i].append(k / mean[s])
    from scipy import stats
    rho = stats.spearmanr([i for _, i, _ in rows], [k / mean[s] for s, _, k in rows])
    return {"relative_kick_rate_by_position": {str(i): float(np.mean(v)) for i, v in sorted(bypos.items())},
            "spearman_position_vs_relative_rate": {"rho": float(rho[0]), "p": float(rho[1])}}


def main():
    out = {"selected_rules": SEL}
    t = "holdout2_v5_equal_10s"   # B2 + B4 (equal time)
    out["S2_equal_time"] = family(t, [(E, NL), (E, NLF), (NL, R), (NLF, R), (E, GA), (GA, GA_DEF), (E, G), (E, R)])
    out["S2_counters"], out["S2_order"] = counters(t), order_effect(t)
    t = "holdout2_v5_kicks20k"    # B1 (equal kick count)
    out["S1_equal_kicks"] = family(t, [(E, G), (E, GA), (E, NL), (E, K), (G, R), (GA, R), (NL, R), (E, R)])
    out["S1_counters"] = counters(t)
    t = "holdout2_v5_credit_10s"  # B5 (online credit)
    out["S3_credit"] = family(t, [(OP, LP), (OS, LP), (OW, LP), (OP, OW), (OS, OW), (LP, R)])
    out["S3_counters"] = counters(t)
    t = "holdout2_v5_single_10s"  # B6 (one worker, random order)
    out["S4_single"] = family(t, [(E, R), (E, G)])
    out["S4_counters"] = counters(t)
    t = "holdout2_lkh_10s"        # B3 (LKH-3)
    out["S5_lkh"] = family(t, [(LE, LR), (LR, LD), (LE, LD)])
    out["S5_counters"] = counters(t)
    (HERE / "v5_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out.items():
        if isinstance(v, dict) and "contrasts" in v:
            print(k)
            for r in v["contrasts"]:
                print(f"  {r['pair']:58s} {r['mean']:+.4f} [{r['ci95'][0]:+.3f}; {r['ci95'][1]:+.3f}] "
                      f"{r['wins_first']}/{r['n_inst']} p_holm={r['p_holm']:.2g}")


if __name__ == "__main__":
    main()
