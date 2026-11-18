from __future__ import annotations

import json
from pathlib import Path

import diversification_experiment as common

HERE = Path(__file__).resolve().parent
DRILLING = set("a280,d198,d493,d657,d1291,d1655,d2103,fl417,fl1400,fl1577,fl3795,p654,pcb442,pcb1173,pcb3038,u159,u574,u724,u1060,u1432,u1817,u2152,u2319".split(","))
RATTLED = set("rat99,rat195,rat575,rat783".split(","))


def load_rows() -> list[dict]:
    scale = json.loads((HERE / "results" / "tsplib_scale_72x5_summary.json").read_text(encoding="utf-8"))
    focused = json.loads((HERE / "results" / "focused_tsplib_9x10_summary.json").read_text(encoding="utf-8"))
    rows = []
    for source, payload in (("tsplib_scale_72x5", scale), ("focused_tsplib_9x10", focused)):
        for original in payload["per_instance"]:
            if original["instance"] not in DRILLING | RATTLED:
                continue
            if any(row["instance"] == original["instance"] for row in rows):
                continue
            row = dict(original)
            row["source"] = source
            row["family"] = "rattled_grid" if row["instance"] in RATTLED else "pcb_drilling"
            rows.append(row)
    return rows


def comparison_for(name: str, rows: list[dict]) -> dict:
    row = common.paired_comparison("suspicious", "random", rows)
    row["regime"] = name
    row["instances"] = [item["instance"] for item in rows]
    row["kick_active_both"] = sum(
        item["random"]["kicks"] > 0 and item["suspicious"]["kicks"] > 0 for item in rows)
    row["zero_kick_both"] = sum(
        item["random"]["kicks"] == 0 and item["suspicious"]["kicks"] == 0 for item in rows)
    row["mean_gap_random"] = -sum(item["random"]["quality"] for item in rows) / len(rows)
    row["mean_gap_suspicious"] = -sum(item["suspicious"]["quality"] for item in rows) / len(rows)
    row["mean_elapsed_random"] = sum(item["random"]["elapsed"] for item in rows) / len(rows)
    row["mean_elapsed_suspicious"] = sum(item["suspicious"]["elapsed"] for item in rows) / len(rows)
    return row


def main() -> None:
    rows = load_rows()
    drilling = [row for row in rows if row["family"] == "pcb_drilling"]
    rattled = [row for row in rows if row["family"] == "rattled_grid"]
    comparisons = [
        comparison_for("pcb_drilling_all", drilling),
        comparison_for("pcb_drilling_n_le_999", [row for row in drilling if row["n"] <= 999]),
        comparison_for("pcb_drilling_n_ge_1000", [row for row in drilling if row["n"] >= 1000]),
        comparison_for("rattled_grid", rattled),
    ]
    common.holm(comparisons)
    common.add_bootstrap_intervals(comparisons, 93017)
    output = {
        "status": "post-hoc exploratory analysis; not independent confirmatory evidence",
        "sources": ["results/tsplib_scale_72x5_summary.json", "results/focused_tsplib_9x10_summary.json"],
        "classification_basis": "TSPLIB file COMMENT fields: drilling problem or rattled grid",
        "multiplicity": "Holm across four exploratory regime comparisons",
        "inference_unit": "instance; seeds averaged within instance",
        "n_unique_instances": len(rows),
        "comparisons": comparisons,
        "per_instance": rows,
    }
    out = HERE / "results" / "tsplib_circuit_regimes.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"n_unique_instances": len(rows), "comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    main()
