# -*- coding: utf-8 -*-
"""Exploratory length-arm analysis (review Ö4): holdout2_length_10s, paired by instance."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import v2_analysis as A
from scipy import stats

TAG = "holdout2_length_10s"
R, E, L = "random_global", "excess_global@0.06:0.5", "length_global@0.06:0.5"
cat = json.loads((A.OUT_ROOT.parents[1] / "data_holdout2" / "catalog.json").read_text(encoding="utf-8"))["instances"]
def hl(d):
    w = (d[:, None] + d[None, :]) / 2
    return float(np.median(w[np.triu_indices(len(d))]))
out = {"tag": TAG, "exploratory": True, "pairs": {}}
data, ns = A.per_instance(TAG)
for a, b in [(L, R), (L, E), (E, R)]:
    r = A.paired(TAG, a, b)
    keys = [k for k in data if a in data[k] and b in data[k]]
    d = np.array([data[k][a] - data[k][b] for k in keys])
    r["hl"] = hl(d)
    strat = defaultdict(list)
    for k, x in zip(keys, d):
        s = cat[k]["stratum"]; s = "synth" if s.startswith("synth_uniform") is False and s.startswith("synth") else s
        strat[cat[k]["stratum"]].append(x)
    r["by_stratum"] = {s: {"n": len(v), "mean": float(np.mean(v)), "wins_first": int((np.array(v) < 0).sum())} for s, v in strat.items()}
    out["pairs"][f"{a} - {b}"] = r
    print(json.dumps({k: v for k, v in r.items() if k != "by_stratum"}))
    for s, v in r["by_stratum"].items(): print("   ", s, v)
# kick counters
acc = defaultdict(lambda: defaultdict(list))
for f in (A.OUT_ROOT / TAG).glob("*__s*.json"):
    d = json.loads(f.read_text(encoding="utf-8"))
    for s, r in d["runs"].items():
        acc[s]["kicks_per_s"].append(r["kicks"] / r["elapsed"]); acc[s]["accept_rate"].append(r["accepted"] / max(r["kicks"], 1))
out["counters"] = {s: {k: float(np.mean(v)) for k, v in m.items()} for s, m in acc.items()}
print(out["counters"])
# consistency with confirmatory run: excess - random in holdout2_10s
c = A.paired("holdout2_10s", "excess_global@0.06:0.5", "random_global")
out["confirmatory_excess_random_reference"] = {"mean": c["mean"], "p": c["p_wilcoxon"]}
print("ref confirm E-R", c["mean"], c["p_wilcoxon"])
(A.OUT_ROOT / "holdout2_length_10s_analysis.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
