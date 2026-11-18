# -*- coding: utf-8 -*-
"""Validation and external tests for suspicious-centered low-overhead variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from adaptive_ils import FOCUSED_STRATEGIES, TRANSFER_STRATEGIES, run_adaptive_ils_search
import diversification_experiment as common
import entropy_experiment as base
import gnn_trained as gt

HERE = Path(__file__).resolve().parent
DEFAULT_METHODS = ("random", "suspicious") + FOCUSED_STRATEGIES
ALLOWED_METHODS = DEFAULT_METHODS + TRANSFER_STRATEGIES
FOCUSED_WORLD_NAMES = "wi29,tz6117,ym7663,pm8079,ei8246,ja9847,kz9976,fi10639,mo14185,ho14473"


def summarize_focused(payload: dict) -> dict:
    summary = common.summarize(payload)
    candidates = [method for method in payload["config"]["methods"]
                  if method not in ("random", "suspicious")]
    rows = [common.paired_comparison(method, "suspicious", summary["per_instance"])
            for method in candidates]
    common.holm(rows)
    for row in rows:
        row["classification"] = (
            "significant_benefit" if row["significant_0_05"] and row["mean_quality_difference"] > 0
            else "significant_harm" if row["significant_0_05"] and row["mean_quality_difference"] < 0
            else "inconclusive"
        )
    common.add_bootstrap_intervals(rows, payload["config"]["seed"] + 179)
    summary["comparisons_vs_suspicious"] = rows
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=("synthetic", "tsplib", "world", "vlsi", "art"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    ap.add_argument("--instances", type=int, default=40)
    ap.add_argument("--datasets")
    ap.add_argument("--data-dir")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=57721)
    ap.add_argument("--nmin", type=int, default=80)
    ap.add_argument("--nmax", type=int, default=160)
    ap.add_argument("--method-time", type=float, default=0.5)
    ap.add_argument("--max-kicks", type=int, default=100000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    methods = tuple(args.methods.split(","))
    if "random" not in methods or "suspicious" not in methods:
        raise ValueError("random and suspicious baselines are required")
    if any(method not in ALLOWED_METHODS for method in methods):
        raise ValueError("unknown focused method")
    args.methods = list(methods)
    out = Path(args.output)
    if not out.is_absolute():
        out = HERE / out
    if args.summarize:
        payload = json.loads(out.read_text(encoding="utf-8"))
    else:
        started = time.perf_counter()
        crine_weights = gt.get_weights(HERE / "models" / "crine_trained.npz")[0]
        kick_weights = gt.get_weights(HERE / "models" / "gnn_trained.npz")[0]
        payload = {
            "experiment": "suspicious_centered_focused_ils", "config": vars(args),
            "metric": "improvement_pct" if args.suite == "synthetic" else "negative_gap_pct",
            "records": [],
        }
        completed = set()
        previous_wall = 0.0
        if args.resume and out.exists():
            payload = json.loads(out.read_text(encoding="utf-8"))
            ignored = {"resume", "summarize"}
            previous_config = {k: v for k, v in payload["config"].items() if k not in ignored}
            current_config = {k: v for k, v in vars(args).items() if k not in ignored}
            if previous_config != current_config:
                raise ValueError("resume configuration differs")
            previous_wall = payload.get("wall_seconds", 0.0)
            completed = {record["instance"] for record in payload["records"]}
        names = (args.datasets.split(",") if args.datasets
                 else list(range(args.instances)) if args.suite == "synthetic"
                 else FOCUSED_WORLD_NAMES.split(",") if args.suite == "world"
                 else common.TSPLIB_NAMES.split(","))
        for idx, name in enumerate(names):
            if name in completed:
                continue
            if args.suite == "synthetic":
                inst, kind = base.synthetic_instance(idx, args)
                bks = initial_source = None
            else:
                inst, bks, initial_source = common.load_real_instance(name, args.suite, args.data_dir)
                kind = None
            runs = []
            for seed in range(args.seeds):
                initial = (base.initial_for_seed(inst, seed) if initial_source is None
                           else initial_source[:])
                initial_cost = inst.tour_cost(initial)
                for method in methods:
                    result = run_adaptive_ils_search(
                        inst, initial, crine_weights, kick_weights, strategy=method,
                        seed=args.seed + 101 * seed, max_kicks=args.max_kicks,
                        max_time_seconds=args.method_time,
                    )
                    improvement = 100.0 * (initial_cost - result.cost) / max(initial_cost, 1e-9)
                    kicks = [row for row in result.history if row["kind"] == "kick"]
                    entropies = [row["model_entropy"] for row in kicks
                                 if row["model_entropy"] is not None]
                    runs.append({
                        "method": method, "seed": seed, "cost": result.cost,
                        "initial_cost": initial_cost, "improvement_pct": improvement,
                        "quality": improvement if bks is None else -100.0 * (result.cost - bks) / bks,
                        "elapsed": result.elapsed, "kicks": result.rounds,
                        "accepted_kicks": result.accepted_kicks,
                        "expansions": result.expansions,
                        "operator_counts": result.operator_counts,
                        "mean_model_entropy": (sum(entropies) / len(entropies)
                                               if entropies else None),
                    })
            payload["records"].append({
                "instance": name if args.suite != "synthetic" else idx,
                "kind": kind, "n": inst.size, "bks": bks, "runs": runs,
            })
            payload["records"].sort(key=lambda record: names.index(record["instance"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            summary = summarize_focused(payload)
            leader = summary["quality_ranking"][0]
            print(f"{args.suite} [{idx + 1}/{len(names)}] {name} "
                  f"leader={leader} quality={summary['aggregate'][leader]['mean_quality']:.6f} "
                  f"suspicious={summary['aggregate']['suspicious']['mean_quality']:.6f}", flush=True)
        payload["wall_seconds"] = previous_wall + time.perf_counter() - started
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize_focused(payload)
    summary_out = out.with_name(out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": summary["quality_ranking"],
                      "friedman": summary["friedman"],
                      "versus_random": summary["comparisons"],
                      "versus_suspicious": summary["comparisons_vs_suspicious"]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
