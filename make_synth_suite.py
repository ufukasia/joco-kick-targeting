# -*- coding: utf-8 -*-
"""Writes the synthetic suites of the kick-targeting study as TSPLIB files.

synth_holdout : the 100-instance focused holdout of the original study
                (gnn_generate_data.KINDS, seed 7447, n in [80,160])
synth_1000    : 40 instances, n = 1000, same five generators (8 each),
                seed 99173 -- out-of-size for the GNN trained on n in [101, 400]

Reference values: LKH-3 (RUNS=1, defaults, TIME_LIMIT 30 s) cost, stored in
data_synth_fast/catalog.json; gaps on these suites are relative to LKH.

    py -3.13 make_synth_suite.py write
    py -3.13 make_synth_suite.py ref --workers 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIR = HERE / "data_synth_fast"


def write_tsp(path: Path, name: str, coords):
    lines = [f"NAME : {name}", "TYPE : TSP", f"DIMENSION : {len(coords)}",
             "EDGE_WEIGHT_TYPE : EUC_2D", "NODE_COORD_SECTION"]
    lines += [f"{i + 1} {x:.6f} {y:.6f}" for i, (x, y) in enumerate(coords)]
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def cmd_write():
    import gnn_generate_data as gd
    DIR.mkdir(exist_ok=True)
    cat = {"instances": {}}
    for idx in range(100):
        kind, gen = gd.KINDS[idx % len(gd.KINDS)]
        rng = random.Random(7447 * 1_000_003 + idx)
        coords = gen(rng, rng.randint(80, 160))
        nm = f"h{idx:03d}_{kind}"
        write_tsp(DIR / f"{nm}.tsp", nm, coords)
        cat["instances"][nm] = {"suite": "synth_holdout", "n": len(coords), "kind": kind}
    for idx in range(40):
        kind, gen = gd.KINDS[idx % len(gd.KINDS)]
        rng = random.Random(99173 * 1_000_003 + idx)
        coords = gen(rng, 1000)
        nm = f"k{idx:03d}_{kind}"
        write_tsp(DIR / f"{nm}.tsp", nm, coords)
        cat["instances"][nm] = {"suite": "synth_1000", "n": len(coords), "kind": kind}
    (DIR / "catalog.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")
    print(len(cat["instances"]), "instances written")


def _ref(nm):
    import fast_ils as F
    from kick_targeting_experiment import load_xy
    from lkh_runner import run_lkh
    p = DIR / f"{nm}.tsp"
    xs, ys, kind = load_xy(p)
    r = run_lkh(p, len(xs), 30.0, 1)
    c = float(F.tour_cost(np.array(r["tour"], dtype=np.int32), xs, ys, kind))
    return nm, c, r["wall"]


def cmd_ref(workers):
    cat = json.loads((DIR / "catalog.json").read_text(encoding="utf-8"))
    todo = [k for k, v in cat["instances"].items() if "ref" not in v]
    with mp.Pool(workers) as pool:
        for nm, c, w in pool.imap_unordered(_ref, todo):
            cat["instances"][nm]["ref"] = c
            cat["instances"][nm]["ref_wall"] = w
            print(nm, c, round(w, 1), flush=True)
            (DIR / "catalog.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("write", "ref"))
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    cmd_write() if a.cmd == "write" else cmd_ref(a.workers)
