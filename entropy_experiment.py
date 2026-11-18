# -*- coding: utf-8 -*-
"""Paired holdout evaluation of entropy, suspicious-edge, and random ILS kicks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy.stats import wilcoxon

import gnn_generate_data as gd
import gnn_trained as gt
import line_reassign_optimizer as lro
import tsplib_engine as E
from crine_ls import deterministic_start

HERE = Path(__file__).resolve().parent
STRATEGIES = ("random", "suspicious", "entropy")
lro.VERBOSE = False
lro.RECORD_FRAME_TOURS = False


def initial_for_seed(inst, seed: int) -> list[int]:
    start = deterministic_start(inst)
    if seed == 0:
        return start
    return lro._double_bridge_perturbation(start, random.Random(91_733 + seed))


def synthetic_instance(idx: int, args):
    kind, generator = gd.KINDS[idx % len(gd.KINDS)]
    rng = random.Random(args.seed * 1_000_003 + idx)
    coords = generator(rng, rng.randint(args.nmin, args.nmax))
    inst = E.make_instance(coords, "EUC_2D", force_dense=True)[0]
    return inst, kind


def run_one(inst, initial, weights, strategy: str, args) -> dict:
    result = gt.run_trained_gnn_local_search(
        inst, initial, weights,
        max_iterations=args.iterations,
        max_no_improve=args.no_improve,
        max_time_seconds=args.method_time,
        seed=args.seed + 101 * args.current_seed,
        kick_strategy=strategy,
    )
    initial_cost = inst.tour_cost(initial)
    improvement = 100.0 * (initial_cost - result.cost) / max(initial_cost, 1e-9)
    row = {
        "strategy": strategy, "seed": args.current_seed,
        "cost": result.cost, "initial_cost": initial_cost,
        "improvement_pct": improvement, "elapsed": result.frames[-1].elapsed_time,
        "iterations": result.iterations_completed,
    }
    return row


def summarize(payload: dict) -> dict:
    records = payload["records"]
    if not records:
        raise ValueError("no complete instances to analyze")
    per_instance = []
    for record in records:
        row = {"instance": record["instance"], "kind": record.get("kind")}
        for strategy in STRATEGIES:
            runs = [r for r in record["runs"] if r["strategy"] == strategy]
            if len(runs) != payload["config"]["seeds"]:
                raise ValueError(f"incomplete instance: {record['instance']}")
            row[strategy] = {
                "quality": float(np.mean([r["quality"] for r in runs])),
                "cost": float(np.mean([r["cost"] for r in runs])),
                "elapsed": float(np.mean([r["elapsed"] for r in runs])),
            }
        per_instance.append(row)
    raw = []
    for other in ("random", "suspicious"):
        diffs = np.asarray([
            row["entropy"]["quality"] - row[other]["quality"]
            for row in per_instance
        ])
        nonzero = diffs[np.abs(diffs) > 1e-12]
        test = wilcoxon(diffs, alternative="two-sided", method="auto") if len(nonzero) else None
        raw.append({
            "comparison": f"entropy_minus_{other}",
            "n_instances": len(diffs), "n_nonzero": len(nonzero),
            "mean_quality_difference": float(np.mean(diffs)),
            "median_quality_difference": float(np.median(diffs)),
            "wins": int(np.sum(diffs > 1e-12)),
            "ties": int(np.sum(np.abs(diffs) <= 1e-12)),
            "losses": int(np.sum(diffs < -1e-12)),
            "wilcoxon_statistic": float(test.statistic) if test else 0.0,
            "p_raw": float(test.pvalue) if test else 1.0,
        })
    order = np.argsort([r["p_raw"] for r in raw])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(raw) - rank) * raw[idx]["p_raw"])
        raw[idx]["p_holm"] = min(1.0, running)
        raw[idx]["significant_0_05"] = bool(raw[idx]["p_holm"] < 0.05)
    return {
        "metric": payload["metric"],
        "quality_direction": "higher is better; TSPLIB quality is negative gap",
        "n_complete_instances": len(records),
        "aggregate": {
            strategy: {
                "mean_quality": float(np.mean([r[strategy]["quality"] for r in per_instance])),
                "median_quality": float(np.median([r[strategy]["quality"] for r in per_instance])),
                "mean_cost": float(np.mean([r[strategy]["cost"] for r in per_instance])),
                "mean_elapsed_s": float(np.mean([r[strategy]["elapsed"] for r in per_instance])),
            } for strategy in STRATEGIES
        },
        "comparisons": raw,
        "per_instance": per_instance,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=("synthetic", "tsplib"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--datasets", default="kroA100,rat783,ali535,gr666,pcb3038,fl3795,fnl4461,rl5915,usa13509")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=481516)
    ap.add_argument("--nmin", type=int, default=80)
    ap.add_argument("--nmax", type=int, default=160)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--no-improve", type=int, default=25)
    ap.add_argument("--method-time", type=float, default=0.0)
    ap.add_argument("--weights", default="models/gnn_trained.npz")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    out = Path(args.output)
    if not out.is_absolute():
        out = HERE / out
    if args.summarize:
        payload = json.loads(out.read_text(encoding="utf-8"))
    else:
        started = time.perf_counter()
        weights = gt.get_weights(HERE / args.weights)[0]
        payload = {
            "experiment": "entropy_kick_ablation", "config": vars(args),
            "metric": "improvement_pct" if args.suite == "synthetic" else "negative_gap_pct",
            "records": [],
        }
        names = (args.datasets.split(",") if args.suite == "tsplib"
                 else list(range(args.instances)))
        for idx, name in enumerate(names):
            if args.suite == "synthetic":
                inst, kind = synthetic_instance(idx, args)
                bks = None
                initial_source = None
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
                args.current_seed = seed
                initial = (initial_for_seed(inst, seed) if initial_source is None
                           else initial_source[:])
                for strategy in STRATEGIES:
                    row = run_one(inst, initial, weights, strategy, args)
                    row["quality"] = (row["improvement_pct"] if bks is None
                                      else -100.0 * (row["cost"] - bks) / bks)
                    runs.append(row)
            payload["records"].append({
                "instance": name if args.suite == "tsplib" else idx,
                "kind": kind, "n": inst.size, "bks": bks, "runs": runs,
            })
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            summary = summarize(payload)
            print(f"{args.suite} [{idx + 1}/{len(names)}] {name} " + " ".join(
                f"{s}={summary['aggregate'][s]['mean_quality']:.5f}" for s in STRATEGIES
            ), flush=True)
        payload["wall_seconds"] = time.perf_counter() - started
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize(payload)
    summary_out = out.with_name(out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["comparisons"], indent=2), flush=True)


if __name__ == "__main__":
    main()
