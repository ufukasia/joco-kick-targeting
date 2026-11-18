# -*- coding: utf-8 -*-
"""Validation and untouched-holdout tests for entropy/suspicious rank fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import wilcoxon

import entropy_experiment as base
import gnn_trained as gt
import tsplib_engine as E

HERE = Path(__file__).resolve().parent


def parse_variants(text: str) -> list[dict]:
    variants = []
    for name in text.split(","):
        name = name.strip()
        if name in ("random", "suspicious", "entropy", "elite_entropy"):
            variants.append({"name": name, "strategy": name, "weight": 0.0})
        elif name.startswith("hybrid"):
            weight = int(name.removeprefix("hybrid")) / 100.0
            variants.append({"name": name, "strategy": "hybrid", "weight": weight})
        else:
            raise ValueError(f"unknown variant: {name}")
    return variants


def holm(rows: list[dict]) -> None:
    order = np.argsort([r["p_raw"] for r in rows])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(rows) - rank) * rows[idx]["p_raw"])
        rows[idx]["p_holm"] = min(1.0, running)
        rows[idx]["significant_0_05"] = bool(rows[idx]["p_holm"] < 0.05)


def summarize(payload: dict) -> dict:
    records = payload["records"]
    variants = payload["config"]["variants"]
    seeds = payload["config"]["seeds"]
    per_instance = []
    for record in records:
        row = {"instance": record["instance"], "kind": record.get("kind")}
        for variant in variants:
            runs = [r for r in record["runs"] if r["variant"] == variant]
            if len(runs) != seeds:
                raise ValueError(f"incomplete instance: {record['instance']}")
            row[variant] = {
                "quality": float(np.mean([r["quality"] for r in runs])),
                "cost": float(np.mean([r["cost"] for r in runs])),
                "elapsed": float(np.mean([r["elapsed"] for r in runs])),
            }
        per_instance.append(row)
    focus = payload["config"].get("focus")
    comparisons = []
    opponents = ([v for v in variants if v != focus] if focus
                 else [v for v in variants if v.startswith("hybrid")])
    for variant in opponents:
        other = "suspicious" if not focus else focus
        a, b = (variant, other) if not focus else (focus, variant)
        diffs = np.asarray([r[a]["quality"] - r[b]["quality"] for r in per_instance])
        nonzero = diffs[np.abs(diffs) > 1e-12]
        test = wilcoxon(diffs, alternative="two-sided", method="auto") if len(nonzero) else None
        comparisons.append({
            "comparison": f"{a}_minus_{b}", "n_instances": len(diffs),
            "n_nonzero": len(nonzero),
            "mean_quality_difference": float(np.mean(diffs)),
            "median_quality_difference": float(np.median(diffs)),
            "wins": int(np.sum(diffs > 1e-12)),
            "ties": int(np.sum(np.abs(diffs) <= 1e-12)),
            "losses": int(np.sum(diffs < -1e-12)),
            "wilcoxon_statistic": float(test.statistic) if test else 0.0,
            "p_raw": float(test.pvalue) if test else 1.0,
        })
    holm(comparisons)
    return {
        "metric": payload["metric"], "n_complete_instances": len(records),
        "quality_direction": "higher is better; TSPLIB quality is negative gap",
        "aggregate": {
            variant: {
                "mean_quality": float(np.mean([r[variant]["quality"] for r in per_instance])),
                "median_quality": float(np.median([r[variant]["quality"] for r in per_instance])),
                "mean_cost": float(np.mean([r[variant]["cost"] for r in per_instance])),
                "mean_elapsed_s": float(np.mean([r[variant]["elapsed"] for r in per_instance])),
            } for variant in variants
        },
        "comparisons": comparisons, "per_instance": per_instance,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=("synthetic", "tsplib"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--variants", default="random,suspicious,entropy,hybrid25,hybrid50,hybrid75")
    ap.add_argument("--focus")
    ap.add_argument("--instances", type=int, default=40)
    ap.add_argument("--datasets", default="kroA100,rat783,ali535,gr666,pcb3038,fl3795,fnl4461,rl5915,usa13509")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=8675309)
    ap.add_argument("--nmin", type=int, default=80)
    ap.add_argument("--nmax", type=int, default=160)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--no-improve", type=int, default=25)
    ap.add_argument("--method-time", type=float, default=0.5)
    ap.add_argument("--weights", default="models/gnn_trained.npz")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    variants = parse_variants(args.variants)
    by_name = {v["name"]: v for v in variants}
    args.variants = [v["name"] for v in variants]
    out = Path(args.output)
    if not out.is_absolute():
        out = HERE / out
    if args.summarize:
        payload = json.loads(out.read_text(encoding="utf-8"))
    else:
        started = time.perf_counter()
        weights = gt.get_weights(HERE / args.weights)[0]
        payload = {
            "experiment": "entropy_rank_fusion", "config": vars(args),
            "metric": "improvement_pct" if args.suite == "synthetic" else "negative_gap_pct",
            "records": [],
        }
        completed = set()
        if args.resume and out.exists():
            previous_payload = json.loads(out.read_text(encoding="utf-8"))
            if previous_payload["config"]["variants"] != args.variants:
                raise ValueError("resume variant configuration differs")
            payload = previous_payload
            completed = {r["instance"] for r in payload["records"]}
        names = (args.datasets.split(",") if args.suite == "tsplib"
                 else list(range(args.instances)))
        for idx, name in enumerate(names):
            if name in completed:
                continue
            if args.suite == "synthetic":
                inst, kind = base.synthetic_instance(idx, args)
                bks = initial_source = None
            else:
                _, coords, ewt = E.parse_tsp(HERE / "data" / f"{name}.tsp")
                inst = E.make_instance(coords, ewt)[0]
                kind = None
                bks = E.BKS[name]
                previous = json.loads((HERE / "results" / "crine_runner" / f"{name}.json").read_text(encoding="utf-8"))
                constructors = [m for m in previous["methods"]
                                if m["key"] in ("snake_greedy_band", "snake_farthest_band")]
                initial_source = min(constructors, key=lambda m: m["cost"])["tour"]
            runs = []
            for seed in range(args.seeds):
                initial = (base.initial_for_seed(inst, seed) if initial_source is None
                           else initial_source[:])
                for variant in variants:
                    result = gt.run_trained_gnn_local_search(
                        inst, initial, weights,
                        max_iterations=args.iterations,
                        max_no_improve=args.no_improve,
                        max_time_seconds=args.method_time,
                        seed=args.seed + 101 * seed,
                        kick_strategy=variant["strategy"],
                        entropy_weight=variant["weight"],
                    )
                    initial_cost = inst.tour_cost(initial)
                    improvement = 100.0 * (initial_cost - result.cost) / max(initial_cost, 1e-9)
                    runs.append({
                        "variant": variant["name"], "seed": seed,
                        "cost": result.cost, "initial_cost": initial_cost,
                        "improvement_pct": improvement,
                        "quality": improvement if bks is None else -100.0 * (result.cost - bks) / bks,
                        "elapsed": result.frames[-1].elapsed_time,
                        "iterations": result.iterations_completed,
                    })
            payload["records"].append({
                "instance": name if args.suite == "tsplib" else idx,
                "kind": kind, "n": inst.size, "bks": bks, "runs": runs,
            })
            payload["records"].sort(key=lambda r: names.index(r["instance"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            summary = summarize(payload)
            print(f"{args.suite} [{idx + 1}/{len(names)}] {name} " + " ".join(
                f"{v}={summary['aggregate'][v]['mean_quality']:.5f}" for v in args.variants
            ), flush=True)
        payload["wall_seconds"] = time.perf_counter() - started
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize(payload)
    out.with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
