from __future__ import annotations

import argparse
import json
from pathlib import Path

import diversification_experiment as common

HERE = Path(__file__).resolve().parent


def regime_rows(per_instance: list[dict]) -> list[dict]:
    regimes = {
        "n_le_1000": [row for row in per_instance if row["n"] <= 1000],
        "n_1001_5999": [row for row in per_instance if 1000 < row["n"] < 6000],
        "n_ge_6000": [row for row in per_instance if row["n"] >= 6000],
        "kick_active_both": [
            row for row in per_instance
            if row["random"]["kicks"] > 0 and row["suspicious"]["kicks"] > 0
        ],
        "kick_inactive_both": [
            row for row in per_instance
            if row["random"]["kicks"] == 0 and row["suspicious"]["kicks"] == 0
        ],
    }
    output = []
    for name, rows in regimes.items():
        comparison = (common.paired_comparison("suspicious", "random", rows) if rows else {
            "comparison": "suspicious_minus_random", "method": "suspicious",
            "opponent": "random", "n_instances": 0, "n_nonzero": 0,
            "mean_quality_difference": 0.0, "median_quality_difference": 0.0,
            "wins": 0, "ties": 0, "losses": 0, "wilcoxon_statistic": 0.0,
            "p_raw": 1.0, "rank_biserial": 0.0, "paired_differences": [],
        })
        comparison["regime"] = name
        comparison["instances"] = [row["instance"] for row in rows]
        comparison["mean_kicks_random"] = (
            sum(row["random"]["kicks"] for row in rows) / len(rows) if rows else 0.0)
        comparison["mean_kicks_suspicious"] = (
            sum(row["suspicious"]["kicks"] for row in rows) / len(rows) if rows else 0.0)
        output.append(comparison)
    common.holm(output)
    common.add_bootstrap_intervals(output, 88421)
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = Path(args.summary)
    if not source.is_absolute():
        source = HERE / source
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = {
        "source": str(source.relative_to(HERE)),
        "suite": payload["suite"],
        "inference_unit": "instance; seeds averaged within instance",
        "multiplicity": "Holm across five descriptive size/activity regimes; no independent confirmatory claim",
        "regimes": regime_rows(payload["per_instance"]),
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = HERE / out
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
