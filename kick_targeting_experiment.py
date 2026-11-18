# -*- coding: utf-8 -*-
"""Equal-wall-clock kick-targeting experiment on the fast ILS testbed.

One task = (instance, seed).  Every strategy in a task starts from the SAME
locally optimal start tour (nearest neighbour from a seed-chosen city + 2-opt/
Or-opt) and uses the SAME kick RNG seed (common random numbers).  Strategy
order is rotated by seed.  Results are written one JSON file per task so the
campaign is resumable; `--summarize` builds the statistics.

    py -3.13 kick_targeting_experiment.py run --suite vlsi_primary --seeds 10 --budget 10 --workers 8
    py -3.13 kick_targeting_experiment.py summarize --suite vlsi_primary
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "results" / "kick_targeting"

OLD_VLSI = ["xqf131", "xqg237", "pma343", "pka379", "pbl395", "pbk411", "pbn423",
            "pbm436", "xql662", "rbx711", "rbu737", "dkg813", "lim963", "pbd984",
            "xit1083", "dca1389", "bcl380", "bck2217", "bch2762", "beg3293"]


def suite_instances(suite: str) -> list[tuple[str, Path, float]]:
    if suite.startswith("vlsi"):
        import vlsi_datasets as v
        items = sorted(v.CATALOG.items(), key=lambda kv: kv[1][0])
        rows = [(k, HERE / "data_vlsi" / f"{k}.tsp", float(b)) for k, (n, b, _) in items
                if n <= 4000 and (HERE / "data_vlsi" / f"{k}.tsp").exists()]
        if suite == "vlsi_primary":
            return [r for r in rows if r[0] not in OLD_VLSI]
        if suite == "vlsi_old":
            return [r for r in rows if r[0] in OLD_VLSI]
        return rows
    if suite == "tsplib72":
        cat = json.loads((HERE / "data_tsplib_scale" / "catalog.json").read_text(encoding="utf-8"))
        rows = [(k, HERE / "data_tsplib_scale" / f"{k}.tsp", float(d["bks"]))
                for k, d in cat["instances"].items()]
        return sorted(rows, key=lambda r: cat["instances"][r[0]]["n"])
    if suite in ("synth_holdout", "synth_1000"):
        cat = json.loads((HERE / "data_synth_fast" / "catalog.json").read_text(encoding="utf-8"))
        return [(k, HERE / "data_synth_fast" / f"{k}.tsp", float(d["ref"]))
                for k, d in cat["instances"].items() if d["suite"] == suite]
    raise ValueError(suite)


def load_xy(path: Path):
    import tsplib_engine as E
    import fast_ils as F
    _, coords, ewt = E.parse_tsp(path)
    xs = np.array([c[0] for c in coords], dtype=np.float64)
    ys = np.array([c[1] for c in coords], dtype=np.float64)
    if "GEO" in ewt:
        kind = F.GEO
    elif ewt.startswith("EUC_2D"):
        kind = F.EUC
    else:
        raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {ewt}")
    return xs, ys, kind


_W = None


def _worker_init():
    global _W
    import fast_ils as F
    import gnn_trained as gt
    F.warmup()
    _W, _ = gt.get_weights(HERE / "models" / "gnn_trained.npz")


def run_task(args) -> str:
    suite, name, path, bks, seed, budget, strategies, out_dir, extra = args
    import fast_ils as F
    out = Path(out_dir) / f"{name}__s{seed}.json"
    if out.exists():
        return f"skip {out.name}"
    xs, ys, kind = load_xy(Path(path))
    n = len(xs)
    nbr = F.build_knn(xs, ys, kind, 10)
    knn24 = F.build_knn(xs, ys, kind, min(24, n - 1))
    gnn = F.GNNScorer(xs, ys, kind, _W)
    start_seed = 7919 * (seed + 1) + n
    init, c0, t_start = F.start_tour(xs, ys, kind, nbr, start_seed)
    kick_seed = 104729 + 1009 * seed
    order = list(strategies)
    r = seed % len(order)
    order = order[r:] + order[:r]
    runs = {}
    for s in order:
        res = F.run_strategy(xs, ys, kind, nbr, init, c0, s, kick_seed, budget,
                             gnn=gnn, knn24=knn24, **extra)
        res["gap_pct"] = 100.0 * (res["cost"] - bks) / bks
        res["trace_gap_pct"] = {t: 100.0 * (c - bks) / bks for t, c in res["trace"].items()}
        runs[s] = res
    payload = {"suite": suite, "instance": name, "n": n, "bks": bks, "seed": seed,
               "budget": budget, "start_cost": c0, "start_gap_pct": 100.0 * (c0 - bks) / bks,
               "start_time": t_start, "extra": extra, "runs": runs,
               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp, out)
    best = min(runs, key=lambda k: runs[k]["cost"])
    return f"{name} s{seed} n={n} best={best} " + " ".join(
        f"{k}={runs[k]['gap_pct']:.3f}" for k in strategies)


def cmd_run(a):
    import fast_ils as F
    strategies = a.strategies.split(",") if a.strategies else list(F.STRATEGIES)
    out_dir = OUT_ROOT / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    insts = suite_instances(a.suite)
    if a.only:
        keep = set(a.only.split(","))
        insts = [r for r in insts if r[0] in keep]
    extra = json.loads(a.extra) if a.extra else {}
    cfg = {"suite": a.suite, "seeds": a.seeds, "budget": a.budget, "strategies": strategies,
           "instances": [r[0] for r in insts], "extra": extra, "workers": a.workers}
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    # big instances first for load balance
    tasks = [(a.suite, nm, str(p), b, s, a.budget, strategies, str(out_dir), extra)
             for nm, p, b in reversed(insts) for s in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers, initializer=_worker_init) as pool:
        for i, msg in enumerate(pool.imap_unordered(run_task, tasks)):
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:7.0f}s] {msg}", flush=True)


def load_records(tag: str):
    rows = []
    for f in sorted((OUT_ROOT / tag).glob("*__s*.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--seeds", type=int, default=10)
    r.add_argument("--budget", type=float, default=10.0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--strategies")
    r.add_argument("--only")
    r.add_argument("--extra", help="JSON dict of run_strategy kwargs")
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a)


if __name__ == "__main__":
    main()
