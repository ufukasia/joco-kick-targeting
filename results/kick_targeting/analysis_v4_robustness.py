# -*- coding: utf-8 -*-
"""Analysis of the EXPLORATORY robustness block registered in PROTOCOL_v3_exploratory.json.

Unit = instance (gap averaged over seeds); paired mean difference (percentage points), 10 000-sample
bootstrap 95% CI, Wilcoxon signed-rank p, Holm correction within each family (one family per item),
Hodges-Lehmann estimate.  None of this changes the confirmatory decisions (PROTOCOL_v2, V1-V4).

    py -3.13 results/kick_targeting/analysis_v4_robustness.py   -> v4_robustness_analysis.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import v2_analysis as A  # noqa: E402

R, E, G = "random_global", "excess_global@0.06:0.5", "gnn_global@0.04:0.5"
RL, EL, LL = "random_local", "excess_local@0.06:0.5", "length_local@0.06:0.5"
RULES = ["excess_global@0.03:0.25", "excess_global@0.03:1.0", "excess_global@0.12:0.25", "excess_global@0.12:1.0"]
ALL7 = ["excess_global@0.06:0.5", "knn_global@0.04:0.75", "gnn_global@0.04:0.5", "gain_global",
        "cheap_global", "linprior_global", "onlinewarm_global"]
CPS = ["0.1", "0.25", "0.5", "1.0", "2.0", "5.0", "10.0"]
CAT = json.loads((ROOT / "data_holdout2" / "catalog.json").read_text(encoding="utf-8"))["instances"]


def holm(ps):
    o = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(ps) - i) * ps[j]))
        adj[j] = run
    return adj.tolist()


def hl(d):
    w = (d[:, None] + d[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(d))]))


def pair(tag, a, b, cp=None):
    r = A.paired(tag, a, b, cp)
    data, _ = A.per_instance(tag, cp)
    keys = [k for k in data if a in data[k] and b in data[k]]
    d = np.array([data[k][a] - data[k][b] for k in keys])
    r["hl"] = hl(d)
    by = defaultdict(list)
    for k, x in zip(keys, d):
        by[CAT[k]["stratum"]].append(x)
    r["by_stratum"] = {s: {"n": len(v), "mean": float(np.mean(v)), "wins_first": int((np.array(v) < 0).sum())}
                       for s, v in sorted(by.items())}
    return r


def family(tag, contrasts, cp=None):
    if not (A.OUT_ROOT / tag).exists():
        return {"status": "not run"}
    rs = [pair(tag, a, b, cp) for a, b in contrasts]
    for r, p in zip(rs, holm([r["p_wilcoxon"] for r in rs])):
        r["p_holm"] = p
    data, _ = A.per_instance(tag, cp)
    means = {s: float(np.mean([data[k][s] for k in data if s in data[k]])) for s in next(iter(data.values()))}
    return {"tag": tag, "checkpoint": cp, "n_files": len(list((A.OUT_ROOT / tag).glob("*__s*.json"))),
            "mean_gap": means, "contrasts": rs}


def counters(tag):
    acc = defaultdict(lambda: defaultdict(list))
    for f in (A.OUT_ROOT / tag).glob("*__s*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        for s, r in d["runs"].items():
            acc[s]["kicks_per_s"].append(r["kicks"] / r["elapsed"])
            acc[s]["accept_rate"].append(r["accepted"] / max(r["kicks"], 1))
    return {s: {k: float(np.mean(v)) for k, v in m.items()} for s, m in acc.items()}


def structure():
    """Per-instance clustering index = coefficient of variation of the nearest-neighbour distance
    (uniform random ~0.52; clustered / VLSI larger), Spearman vs per-instance (excess - random)."""
    import fast_ils as F
    import v2_targeting_experiment as V
    data, _ = A.per_instance("holdout2_10s")
    rows = []
    for nm, p, _b in V.instances("holdout2"):
        xs, ys, kind = V.load_xy(Path(p))
        knn = F.build_knn(xs, ys, kind, 2)
        idx = np.arange(len(xs), dtype=np.int32)
        d1 = F._edge_lengths(idx, np.ascontiguousarray(knn[:, 0], dtype=np.int32), xs, ys, kind)
        cv = float(d1.std() / d1.mean())
        rows.append({"instance": nm, "stratum": CAT[nm]["stratum"], "n": len(xs), "cv_nn": cv,
                     "excess_minus_random": data[nm][E] - data[nm][R],
                     "excess_minus_gnn": data[nm][E] - data[nm][G]})
    cv = np.array([r["cv_nn"] for r in rows])
    out = {"rows": rows}
    for key in ("excess_minus_random", "excess_minus_gnn"):
        rho, p = stats.spearmanr(cv, [r[key] for r in rows])
        out[f"spearman_cv_vs_{key}"] = {"rho": float(rho), "p": float(p)}
    rho, p = stats.spearmanr([np.log(r["n"]) for r in rows], [r["excess_minus_random"] for r in rows])
    out["spearman_logn_vs_excess_minus_random"] = {"rho": float(rho), "p": float(p)}
    return out


def main():
    out = {"exploratory": True, "protocol": "PROTOCOL_v3_exploratory.json"}
    # item 1 (S1): classic local (segment) kick
    out["S1_local"] = family("holdout2_local_10s", [(E, RL), (EL, RL), (LL, RL), (EL, E), (RL, R)])
    if out["S1_local"].get("tag"):
        out["S1_local"]["counters"] = counters("holdout2_local_10s")
    # item 2: anytime curves from the stored confirmatory runs (no new runs)
    out["T_anytime"] = {cp: {s: A.paired("holdout2_10s", s, R, cp) for s in ALL7} for cp in CPS}
    # item 3 (S2): 60 s budget, 5 seeds, checkpoints inside the same runs
    out["S2_60s"] = {cp: family("holdout2_60s", [(E, R), (E, G)], cp) for cp in ["10.0", "20.0", "30.0", "60.0"]}
    if (A.OUT_ROOT / "holdout2_60s").exists():
        d = np.array([v - w for v, w in zip(*[[A.per_instance("holdout2_60s")[0][k][s] for k in
                      A.per_instance("holdout2_60s")[0]] for s in (E, G)])])
        out["S2_60s"]["noninferiority_excess_vs_gnn_margin0.05_tost_p"] = A.tost(d, 0.05)
        out["S2_60s"]["counters"] = counters("holdout2_60s")
    # item 4 (S3): pool-rule sensitivity
    out["S3_rule"] = family("holdout2_rule_10s", [(s, R) for s in RULES] + [(s, E) for s in RULES])
    # item 5: instance structure vs gain
    out["T_structure"] = structure()
    (A.OUT_ROOT / "v4_robustness_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k in ("S1_local", "S3_rule"):
        for r in out[k].get("contrasts", []):
            print(k, r["pair"], f"{r['mean']:+.4f}", [round(x, 4) for x in r["ci95"]], f"p_holm={r['p_holm']:.2g}",
                  f"{r['wins_first']}/{r['n_inst']}")
    for cp, f in out["S2_60s"].items():
        if isinstance(f, dict) and f.get("contrasts"):
            for r in f["contrasts"]:
                print("S2", cp, r["pair"], f"{r['mean']:+.4f}", f"p_holm={r['p_holm']:.2g}")
    for cp in CPS:
        print("T", cp, " ".join(f"{s.split('_')[0]}={out['T_anytime'][cp][s]['mean']:+.3f}" for s in ALL7))
    print({k: v for k, v in out["T_structure"].items() if k != "rows"})


if __name__ == "__main__":
    main()
