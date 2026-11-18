# -*- coding: utf-8 -*-
"""LKH-3 runs with explicit parameter sets and measured wall-clock time.

    py -3.13 lkh_runner.py ref      --workers 8            # near-optimal reference tours (mechanism analysis)
    py -3.13 lkh_runner.py baseline --workers 4            # time-to-quality baseline on the primary VLSI set
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "kick_targeting"

CONFIGS = {
    # LKH-3 defaults (alpha-nearness candidates after subgradient ascent)
    "default": "",
    # cheap preprocessing: POPMUSIC candidate sets, no subgradient ascent
    "fast": "CANDIDATE_SET_TYPE = POPMUSIC\nSUBGRADIENT = NO\n",
}


def run_lkh(tsp: Path, n: int, time_limit: float, seed: int, extra: str = "",
            hard_cap: float = 900.0) -> dict:
    exe = str(HERE / "bin" / "LKH.exe")
    with tempfile.TemporaryDirectory(prefix="lkh_") as td:
        prob = Path(td) / "p.tsp"
        prob.write_bytes(tsp.read_bytes())
        tour = Path(td) / "o.tour"
        par = Path(td) / "r.par"
        par.write_text(f"PROBLEM_FILE = {prob}\nOUTPUT_TOUR_FILE = {tour}\nRUNS = 1\n"
                       f"SEED = {seed}\nTIME_LIMIT = {time_limit}\nTRACE_LEVEL = 0\n{extra}",
                       encoding="ascii")
        t0 = time.perf_counter()
        p = subprocess.run([exe, str(par)], capture_output=True, text=True, timeout=hard_cap)
        wall = time.perf_counter() - t0
        if not tour.exists():
            return {"ok": False, "wall": wall, "stderr": p.stderr[-300:], "stdout": p.stdout[-300:]}
        lines = tour.read_text().splitlines()
        i = next(k for k, l in enumerate(lines) if l.strip().startswith("TOUR_SECTION"))
        t = []
        for l in lines[i + 1:]:
            v = int(l.split()[0])
            if v == -1:
                break
            t.append(v - 1)
        assert len(t) == n
        return {"ok": True, "wall": wall, "tour": t}


def _task(args):
    mode, name, path, bks, cfg, tl, seed = args
    import fast_ils as F
    from kick_targeting_experiment import load_xy
    xs, ys, kind = load_xy(Path(path))
    n = len(xs)
    sub = OUT / ("lkh_ref" if mode == "ref" else "lkh_baseline")
    sub.mkdir(parents=True, exist_ok=True)
    f = sub / (f"{name}.json" if mode == "ref" else f"{name}__{cfg}__t{tl:g}__s{seed}.json")
    if f.exists():
        return f"skip {f.name}"
    r = run_lkh(Path(path), n, tl, seed + 1, CONFIGS[cfg])
    if r["ok"]:
        c = float(F.tour_cost(np.array(r["tour"], dtype=np.int32), xs, ys, kind))
        r["cost"] = c
        r["gap_pct"] = 100.0 * (c - bks) / bks
        if mode != "ref":
            r.pop("tour")
    r.update({"instance": name, "n": n, "bks": bks, "config": cfg, "time_limit": tl, "seed": seed})
    f.write_text(json.dumps(r), encoding="utf-8")
    return f"{name} {cfg} tl={tl} s{seed} ok={r['ok']} wall={r['wall']:.1f} gap={r.get('gap_pct', float('nan')):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("ref", "baseline"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--suite", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    from kick_targeting_experiment import suite_instances
    if a.mode == "ref":
        insts = suite_instances(a.suite or "vlsi")
        tasks = [("ref", nm, str(p), b, "default", 60.0, 0) for nm, p, b in reversed(insts)]
    else:
        insts = suite_instances(a.suite or "vlsi_primary")
        tasks = [("baseline", nm, str(p), b, cfg, tl, s) for nm, p, b in reversed(insts)
                 for cfg in CONFIGS for tl in (1.0, 10.0) for s in range(a.seeds)]
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        for i, m in enumerate(pool.imap_unordered(_task, tasks)):
            print(f"[{i + 1}/{len(tasks)} {time.time() - t0:6.0f}s] {m}", flush=True)


if __name__ == "__main__":
    main()
