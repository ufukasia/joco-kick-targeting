# -*- coding: utf-8 -*-
"""Repeated wall-clock budget comparison of suspicious CRINE-ILS and LKH-3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from adaptive_ils import run_adaptive_ils_search
import diversification_experiment as common
import entropy_experiment as base
import external_solvers as ES
import focused_experiment as focused
import gnn_trained as gt

HERE = Path(__file__).resolve().parent
METHODS = ("suspicious", "lkh3")


def summarize_lkh(payload: dict) -> dict:
    seeds = payload["config"]["seeds"]
    budgets = payload["config"]["budgets"]
    records = payload["records"]
    suites = list(dict.fromkeys(record["suite"] for record in records))
    analyses = []
    for suite in suites:
        suite_records = [record for record in records if record["suite"] == suite]
        names = list(dict.fromkeys(record["instance"] for record in suite_records))
        for budget in budgets:
            per_instance = []
            incomplete = []
            for name in names:
                selected = [record for record in suite_records
                            if record["instance"] == name and record["budget"] == budget]
                by_method = {
                    method: [record for record in selected if record["method"] == method]
                    for method in METHODS
                }
                if any(len(runs) != seeds or any(run["cost"] is None for run in runs)
                       for runs in by_method.values()):
                    incomplete.append(name)
                    continue
                row = {"instance": name, "n": selected[0]["n"]}
                for method, runs in by_method.items():
                    row[method] = {
                        "quality": float(np.mean([run["quality"] for run in runs])),
                        "cost": float(np.mean([run["cost"] for run in runs])),
                        "elapsed": float(np.mean([run["elapsed"] for run in runs])),
                        "target_gap_success": {
                            str(target): float(np.mean([
                                -run["quality"] <= target for run in runs
                            ])) for target in (0.1, 1.0, 5.0)
                        },
                    }
                row["suspicious"]["kicks"] = float(np.mean([
                    run["kicks"] for run in by_method["suspicious"]
                ]))
                per_instance.append(row)
            comparison = None
            aggregate = {}
            if per_instance:
                comparison = common.paired_comparison("lkh3", "suspicious", per_instance)
                common.add_bootstrap_intervals([comparison], payload["config"]["seed"] + 211)
                aggregate = {
                    method: {
                        "mean_quality": float(np.mean([
                            row[method]["quality"] for row in per_instance
                        ])),
                        "mean_gap_pct": float(-np.mean([
                            row[method]["quality"] for row in per_instance
                        ])),
                        "mean_elapsed_s": float(np.mean([
                            row[method]["elapsed"] for row in per_instance
                        ])),
                        "target_gap_success": {
                            target: float(np.mean([
                                row[method]["target_gap_success"][target]
                                for row in per_instance
                            ])) for target in ("0.1", "1.0", "5.0")
                        },
                    } for method in METHODS
                }
            analyses.append({
                "suite": suite, "budget": budget,
                "n_complete_instances": len(per_instance),
                "incomplete_instances": incomplete,
                "aggregate": aggregate, "lkh3_minus_suspicious": comparison,
                "per_instance": per_instance,
            })
    comparisons = [analysis["lkh3_minus_suspicious"] for analysis in analyses
                   if analysis["lkh3_minus_suspicious"] is not None]
    common.holm(comparisons)
    for comparison in comparisons:
        comparison["classification"] = (
            "significant_benefit" if comparison["significant_0_05"]
            and comparison["mean_quality_difference"] > 0
            else "significant_harm" if comparison["significant_0_05"]
            and comparison["mean_quality_difference"] < 0
            else "inconclusive"
        )
    return {
        "experiment": payload["experiment"],
        "protocol": "independent repeated budgets; not a single-run internal anytime trace",
        "inference_unit": "instance; seeds are averaged within each instance",
        "multiplicity": "Holm across the suite-by-budget LKH-3 versus suspicious comparisons",
        "quality_direction": "negative BKS gap; higher is better",
        "analyses": analyses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("tsplib", "world", "both"), default="both")
    ap.add_argument("--datasets")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--budgets", default="1,5")
    ap.add_argument("--seed", type=int, default=42577)
    ap.add_argument("--max-kicks", type=int, default=100000)
    ap.add_argument("--outer-timeout", type=float)
    ap.add_argument("--output", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    args.budgets = [float(value) for value in args.budgets.split(",")]
    if args.seeds < 1 or not args.budgets or any(value <= 0 for value in args.budgets):
        raise ValueError("seeds and budgets must be positive")
    out = Path(args.output)
    if not out.is_absolute():
        out = HERE / out
    if args.summarize:
        payload = json.loads(out.read_text(encoding="utf-8"))
    else:
        started = time.perf_counter()
        payload = {
            "experiment": "suspicious_crine_ils_vs_lkh3_repeated_budgets",
            "config": vars(args), "records": [],
        }
        previous_wall = 0.0
        if args.resume and out.exists():
            payload = json.loads(out.read_text(encoding="utf-8"))
            ignored = {"resume", "summarize", "outer_timeout"}
            previous_config = {key: value for key, value in payload["config"].items()
                               if key not in ignored}
            current_config = {key: value for key, value in vars(args).items()
                              if key not in ignored}
            if previous_config != current_config:
                raise ValueError("resume configuration differs")
            previous_wall = payload.get("wall_seconds", 0.0)
        if args.datasets:
            names = [(args.suite, name) for name in args.datasets.split(",")]
            if args.suite == "both":
                raise ValueError("--datasets requires an explicit suite")
        else:
            names = ([("tsplib", name) for name in common.TSPLIB_NAMES.split(",")]
                     if args.suite in ("tsplib", "both") else [])
            names += ([("world", name) for name in focused.FOCUSED_WORLD_NAMES.split(",")]
                      if args.suite in ("world", "both") else [])
        crine_weights = gt.get_weights(HERE / "models" / "crine_trained.npz")[0]
        kick_weights = gt.get_weights(HERE / "models" / "gnn_trained.npz")[0]
        completed = {(record["suite"], record["instance"], record["budget"],
                      record["seed"], record["method"]): record
                     for record in payload["records"]}
        for suite, name in names:
            inst, bks, initial_source = common.load_real_instance(name, suite, None)
            tsp_path = HERE / ("data" if suite == "tsplib" else "data_world") / f"{name}.tsp"
            for budget in args.budgets:
                for seed in range(args.seeds):
                    run_seed = args.seed + 101 * seed
                    key = (suite, name, budget, seed, "suspicious")
                    if key not in completed:
                        initial = (base.initial_for_seed(inst, seed) if initial_source is None
                                   else initial_source[:])
                        result = run_adaptive_ils_search(
                            inst, initial, crine_weights, kick_weights,
                            strategy="suspicious", seed=run_seed,
                            max_kicks=args.max_kicks, max_time_seconds=budget,
                        )
                        record = {
                            "suite": suite, "instance": name, "n": inst.size, "bks": bks,
                            "budget": budget, "seed": seed, "method": "suspicious",
                            "cost": result.cost,
                            "quality": -100.0 * (result.cost - bks) / bks,
                            "elapsed": result.elapsed, "kicks": result.rounds,
                            "accepted_kicks": result.accepted_kicks,
                            "history": [{"kind": row["kind"], "cost": row["cost"],
                                         "elapsed": row["elapsed"]} for row in result.history],
                            "status": "ok",
                        }
                        payload["records"].append(record)
                        completed[key] = record
                        out.parent.mkdir(parents=True, exist_ok=True)
                        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    key = (suite, name, budget, seed, "lkh3")
                    if key not in completed:
                        tour, status = ES.run_lkh(
                            tsp_path, inst.size, time_limit=budget, seed=run_seed,
                            outer_timeout=args.outer_timeout,
                        )
                        cost = inst.tour_cost(tour) if tour is not None else None
                        record = {
                            "suite": suite, "instance": name, "n": inst.size, "bks": bks,
                            "budget": budget, "seed": seed, "method": "lkh3",
                            "cost": cost,
                            "quality": -100.0 * (cost - bks) / bks if cost is not None else None,
                            "elapsed": None, "status": status,
                        }
                        try:
                            record["elapsed"] = float(status.split(" ok in ", 1)[1].split("s ", 1)[0])
                        except (IndexError, ValueError):
                            pass
                        payload["records"].append(record)
                        completed[key] = record
                        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"{suite} {name} budget={budget:g}s records="
                      f"{sum(1 for record in payload['records'] if record['instance'] == name and record['budget'] == budget)}",
                      flush=True)
        payload["wall_seconds"] = previous_wall + time.perf_counter() - started
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize_lkh(payload)
    summary_out = out.with_name(out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
