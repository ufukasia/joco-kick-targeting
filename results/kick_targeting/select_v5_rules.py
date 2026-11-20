# -*- coding: utf-8 -*-
"""Pool-rule selection for the gain-MLP and NeuroLKH (block B4, PROTOCOL_v4_exploratory).

Same grid and the same selection rule that PROTOCOL_v2 development used for kNN / excess / GNN:
grid f in {0.02, 0.04, 0.06, 0.10} x p in {0.5, 0.75, 0.9} on vlsi_primary + synth_1000 (3 seeds, 10 s);
minimise the equal-weight mean over both sets of the per-instance mean gap; keep the default 0.06/0.75
if it is within 0.01 pt of the best.  Writes v5_selected_rules.json (read by run_v5.ps1 and the analysis).

    py -3.13 results/kick_targeting/select_v5_rules.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import v2_analysis as A  # noqa: E402

TAGS = ["dev5_grid_vlsi_primary_10s", "dev5_grid_synth1000_10s"]
FRACS = ["0.02", "0.04", "0.06", "0.10"]
PROBS = ["0.5", "0.75", "0.9"]
SIGNALS = ["gain", "neurolkh"]
DEFAULT = "0.06:0.75"


def main():
    data = [A.per_instance(t)[0] for t in TAGS]
    out = {}
    for sig in SIGNALS:
        score = {}
        for f in FRACS:
            for p in PROBS:
                s = f"{sig}_global@{f}:{p}"
                score[f"{f}:{p}"] = float(np.mean([np.mean([d[k][s] for k in d]) for d in data]))
        best = min(score, key=score.get)
        chosen = DEFAULT if score[DEFAULT] - score[best] <= 0.01 else best
        out[sig] = {"selected": chosen, "best": best, "grid_mean_gap": score}
        print(sig, "best", best, f"{score[best]:.4f}", "default", f"{score[DEFAULT]:.4f}", "->", chosen)
    (HERE / "v5_selected_rules.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
