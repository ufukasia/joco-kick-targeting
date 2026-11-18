# -*- coding: utf-8 -*-
"""Paired comparison of ten diversification strategies on the CRINE-ILS skeleton."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from adaptive_ils import STRATEGIES, run_adaptive_ils_search
from crine_ls import run_crine_search
import entropy_experiment as base
import gnn_trained as gt
import tsplib_engine as E
import art_datasets
import vlsi_datasets
import world_datasets

HERE = Path(__file__).resolve().parent
METHODS = STRATEGIES + ("crine",)
ALTERNATIVES = tuple(method for method in STRATEGIES if method != "random")
TSPLIB_NAMES = "kroA100,rat783,ali535,gr666,pcb3038,fl3795,fnl4461,rl5915,usa13509"
WORLD_NAMES = "dj38,qa194,uy734,zi929,lu980,rw1621,mu1979,nu3496,ca4663,gr9882"


def holm(rows: list[dict]) -> None:
    order = np.argsort([row["p_raw"] for row in rows])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(rows) - rank) * rows[idx]["p_raw"])
        rows[idx]["p_holm"] = min(1.0, running)
        rows[idx]["significant_0_05"] = bool(rows[idx]["p_holm"] < 0.05)


def paired_comparison(method: str, opponent: str, per_instance: list[dict]) -> dict:
    diffs = np.asarray([
        row[method]["quality"] - row[opponent]["quality"] for row in per_instance
    ], dtype=np.float64)
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero):
        test = wilcoxon(diffs, alternative="two-sided", method="auto")
        ranks = rankdata(np.abs(nonzero))
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        effect = (positive - negative) / (positive + negative)
        p_raw = float(test.pvalue)
        statistic = float(test.statistic)
    else:
        effect, p_raw, statistic = 0.0, 1.0, 0.0
    return {
        "comparison": f"{method}_minus_{opponent}", "method": method,
        "opponent": opponent, "n_instances": len(diffs), "n_nonzero": len(nonzero),
        "mean_quality_difference": float(np.mean(diffs)),
        "median_quality_difference": float(np.median(diffs)),
        "wins": int(np.sum(diffs > 1e-12)),
        "ties": int(np.sum(np.abs(diffs) <= 1e-12)),
        "losses": int(np.sum(diffs < -1e-12)),
        "wilcoxon_statistic": statistic, "p_raw": p_raw,
        "rank_biserial": float(effect), "paired_differences": diffs.tolist(),
    }


def add_bootstrap_intervals(rows: list[dict], seed: int, resamples: int = 10000) -> None:
    rng = np.random.default_rng(seed)
    for row in rows:
        diffs = np.asarray(row.pop("paired_differences"), dtype=np.float64)
        if not len(diffs):
            row["bootstrap_95_ci"] = [None, None]
            continue
        indices = rng.integers(0, len(diffs), size=(resamples, len(diffs)))
        means = diffs[indices].mean(axis=1)
        row["bootstrap_95_ci"] = [float(x) for x in np.percentile(means, [2.5, 97.5])]


def summarize(payload: dict) -> dict:
    records = payload["records"]
    if not records:
        raise ValueError("no complete instances to analyze")
    seeds = payload["config"]["seeds"]
    methods = tuple(payload["config"].get("methods", METHODS))
    strategies = tuple(method for method in methods if method != "crine")
    alternatives = tuple(method for method in strategies if method != "random")
    per_instance = []
    for record in records:
        row = {"instance": record["instance"], "kind": record.get("kind"), "n": record["n"]}
        for method in methods:
            runs = [run for run in record["runs"] if run["method"] == method]
            if len(runs) != seeds:
                raise ValueError(f"incomplete instance: {record['instance']} / {method}")
            row[method] = {
                "quality": float(np.mean([run["quality"] for run in runs])),
                "cost": float(np.mean([run["cost"] for run in runs])),
                "elapsed": float(np.mean([run["elapsed"] for run in runs])),
                "kicks": float(np.mean([run["kicks"] for run in runs])),
                "accepted_kicks": float(np.mean([run["accepted_kicks"] for run in runs])),
            }
        per_instance.append(row)
    comparisons = [paired_comparison(method, "random", per_instance)
                   for method in alternatives]
    holm(comparisons)
    for row in comparisons:
        row["classification"] = (
            "significant_benefit" if row["significant_0_05"] and row["mean_quality_difference"] > 0
            else "significant_harm" if row["significant_0_05"] and row["mean_quality_difference"] < 0
            else "inconclusive"
        )
    add_bootstrap_intervals(comparisons, payload["config"]["seed"] + 77)
    secondary = None
    if "crine" in methods:
        secondary = paired_comparison("crine", "random", per_instance)
        add_bootstrap_intervals([secondary], payload["config"]["seed"] + 78)
    matrix = [[row[method]["quality"] for method in strategies] for row in per_instance]
    try:
        friedman = friedmanchisquare(*np.asarray(matrix).T)
        if not np.isfinite(friedman.statistic) or not np.isfinite(friedman.pvalue):
            raise ValueError("undefined Friedman statistic")
        omnibus = {
            "statistic": float(friedman.statistic), "p_value": float(friedman.pvalue),
            "kendalls_w": float(friedman.statistic / (len(records) * (len(strategies) - 1))),
            "n_instances": len(records), "n_methods": len(strategies),
        }
    except ValueError:
        omnibus = {"statistic": None, "p_value": 1.0, "kendalls_w": 0.0,
                   "n_instances": len(records), "n_methods": len(strategies)}
    aggregate = {
        method: {
            "mean_quality": float(np.mean([row[method]["quality"] for row in per_instance])),
            "median_quality": float(np.median([row[method]["quality"] for row in per_instance])),
            "mean_cost": float(np.mean([row[method]["cost"] for row in per_instance])),
            "mean_elapsed_s": float(np.mean([row[method]["elapsed"] for row in per_instance])),
            "mean_kicks": float(np.mean([row[method]["kicks"] for row in per_instance])),
            "mean_accepted_kicks": float(np.mean([row[method]["accepted_kicks"] for row in per_instance])),
        } for method in methods
    }
    ranking = sorted(strategies, key=lambda method: aggregate[method]["mean_quality"], reverse=True)
    return {
        "experiment": payload["experiment"], "suite": payload["config"]["suite"],
        "metric": payload["metric"], "n_complete_instances": len(records),
        "quality_direction": "higher is better; real-world quality is negative BKS gap",
        "inference_unit": "instance; seeds are averaged within each instance",
        "multiplicity": f"Holm across {len(alternatives)} alternative-versus-random comparisons",
        "aggregate": aggregate, "quality_ranking": ranking,
        "friedman": omnibus, "comparisons": comparisons,
        "secondary_crine_comparison": secondary, "per_instance": per_instance,
    }


def load_real_instance(name: str, suite: str, data_dir: str | None):
    directories = {"tsplib": "data", "world": "data_world", "vlsi": "data_vlsi", "art": "data_art"}
    catalogs = {"world": world_datasets, "vlsi": vlsi_datasets, "art": art_datasets}
    if suite not in directories:
        raise ValueError(f"unknown real-instance suite: {suite}")
    directory = data_dir or directories[suite]
    _, coords, edge_weight_type = E.parse_tsp(HERE / directory / f"{name}.tsp")
    inst = E.make_instance(coords, edge_weight_type)[0]
    bks = E.BKS.get(name) if suite == "tsplib" else catalogs[suite].BKS[name]
    if bks is None and suite == "tsplib":
        catalog_path = HERE / directory / "catalog.json"
        if catalog_path.exists():
            catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
            bks = catalog["instances"].get(name, {}).get("bks")
    if bks is None:
        raise KeyError(f"missing BKS: {name}")
    initial_source = None
    previous_path = HERE / "results" / "crine_runner" / f"{name}.json"
    if suite == "tsplib" and previous_path.exists():
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        constructors = [m for m in previous["methods"]
                        if m["key"] in ("snake_greedy_band", "snake_farthest_band")]
        initial_source = min(constructors, key=lambda m: m["cost"])["tour"]
    return inst, bks, initial_source


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=("synthetic", "tsplib", "world"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--datasets")
    ap.add_argument("--data-dir")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--nmin", type=int, default=80)
    ap.add_argument("--nmax", type=int, default=160)
    ap.add_argument("--method-time", type=float, default=1.0)
    ap.add_argument("--max-kicks", type=int, default=100000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
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
            "experiment": "ten_strategy_crine_ils", "config": vars(args),
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
        default_names = WORLD_NAMES if args.suite == "world" else TSPLIB_NAMES
        names = (args.datasets.split(",") if args.datasets
                 else list(range(args.instances)) if args.suite == "synthetic"
                 else default_names.split(","))
        for idx, name in enumerate(names):
            if name in completed:
                continue
            if args.suite == "synthetic":
                inst, kind = base.synthetic_instance(idx, args)
                bks = initial_source = None
            else:
                inst, bks, initial_source = load_real_instance(name, args.suite, args.data_dir)
                kind = None
            runs = []
            for seed in range(args.seeds):
                initial = (base.initial_for_seed(inst, seed) if initial_source is None
                           else initial_source[:])
                initial_cost = inst.tour_cost(initial)
                for method in METHODS:
                    if method == "crine":
                        result = run_crine_search(
                            inst, initial, mode="crine", weights=crine_weights,
                            seed=args.seed + 101 * seed, max_rounds=6,
                            max_time_seconds=args.method_time,
                        )
                        accepted = 0
                        operator_counts = {}
                    else:
                        result = run_adaptive_ils_search(
                            inst, initial, crine_weights, kick_weights, strategy=method,
                            seed=args.seed + 101 * seed, max_kicks=args.max_kicks,
                            max_time_seconds=args.method_time,
                        )
                        accepted = result.accepted_kicks
                        operator_counts = result.operator_counts
                    improvement = 100.0 * (initial_cost - result.cost) / max(initial_cost, 1e-9)
                    runs.append({
                        "method": method, "seed": seed, "cost": result.cost,
                        "initial_cost": initial_cost, "improvement_pct": improvement,
                        "quality": improvement if bks is None else -100.0 * (result.cost - bks) / bks,
                        "elapsed": result.elapsed,
                        "kicks": 0 if method == "crine" else result.rounds,
                        "accepted_kicks": accepted, "expansions": result.expansions,
                        "operator_counts": operator_counts,
                    })
            payload["records"].append({
                "instance": name if args.suite != "synthetic" else idx,
                "kind": kind, "n": inst.size, "bks": bks, "runs": runs,
            })
            payload["records"].sort(key=lambda record: names.index(record["instance"]))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            summary = summarize(payload)
            leader = summary["quality_ranking"][0]
            print(f"{args.suite} [{idx + 1}/{len(names)}] {name} "
                  f"leader={leader} quality={summary['aggregate'][leader]['mean_quality']:.6f} "
                  f"random={summary['aggregate']['random']['mean_quality']:.6f}", flush=True)
        payload["wall_seconds"] = previous_wall + time.perf_counter() - started
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary = summarize(payload)
    summary_out = out.with_name(out.stem + "_summary.json")
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": summary["quality_ranking"],
                      "friedman": summary["friedman"],
                      "comparisons": summary["comparisons"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
