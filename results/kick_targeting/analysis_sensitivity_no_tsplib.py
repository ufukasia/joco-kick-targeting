# -*- coding: utf-8 -*-
"""Exploratory sensitivity check (post hoc, no new runs): the frozen PROTOCOL_v2 confirmatory
tests V1-V4, re-applied to holdout2 without the three TSPLIB instances (pcb3038, fl3795, fnl4461),
which were used in Phase 1 (slow CRINE-ILS skeleton) but never in Phase 2 or Phase 3.

Uses the same per-instance aggregation and test routine as v2_analysis.cmd_confirm.

    py -3.13 results/kick_targeting/analysis_sensitivity_no_tsplib.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import v2_analysis as A  # noqa: E402

EXCLUDE = {"pcb3038", "fl3795", "fnl4461"}


def main():
    proto = json.loads((HERE / "PROTOCOL_v2.json").read_text(encoding="utf-8"))
    data, _ = A.per_instance("holdout2_10s")
    keys = [k for k in data if k not in EXCLUDE]
    assert len(data) - len(keys) == len(EXCLUDE), sorted(data)
    out, gate = [], True
    for h in proto["confirmatory_hypotheses"]:
        d = np.array([data[k][h["a"]] - data[k][h["b"]] for k in keys])
        p, ok = A.test_hypothesis(d, h, proto["alpha"])
        r = {"id": h["id"], "a": h["a"], "b": h["b"], "test": h["test"], "n_inst": len(d),
             "mean": float(d.mean()), "p": p, "criterion_met": ok, "rejected": bool(gate and ok)}
        gate = gate and ok
        out.append(r)
        print(f"{r['id']}: n={len(d)} mean={r['mean']:+.4f} p={p:.3g} rejected={r['rejected']}")
    (HERE / "holdout2_10s_confirm_no_tsplib.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
