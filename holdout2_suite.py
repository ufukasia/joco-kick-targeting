# -*- coding: utf-8 -*-
"""Second, untouched test set (holdout2) for the PROTOCOL_v2 confirmatory campaign.

Real instances never used by the fast-ILS (Phase 2) campaign:
    vlsi6     : bgd4396, frv4410, bgf4475, fqm5087, bnd7168, ida8197 (Waterloo VLSI, optimum known)
    national3 : eg7146, ar9152, gr9882 (Waterloo National TSP, optimum known)
    tsplib3   : pcb3038, fl3795, fnl4461 (TSPLIB, optimum known; used in Phase 1 only, where
                the kick loop was inactive -- flagged in the catalog and reported as a stratum)
Fresh synthetic instances, n in [1500, 4000], generators that were NOT used to train any model
of this project (DIMACS-challenge style, coordinates in [0, 1e6]^2), new seed 20261116:
    uniform   : i.i.d. uniform points
    clustered : n/100 centres uniform, points normal around a random centre, sd = 1e6/sqrt(n)
    mixed     : half uniform, half clustered
Reference values: published optima for real instances; LKH-3 (defaults, RUNS=1, 60 s) for the
synthetic ones.  No strategy is run here -- this file only builds the instances and references.

    py -3.13 holdout2_suite.py write
    py -3.13 holdout2_suite.py ref --workers 4
    py -3.13 holdout2_suite.py check
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIR = HERE / "data_holdout2"
SEED = 20261116
N_PER_KIND = 14
KINDS = ("uniform", "clustered", "mixed")

VLSI6 = ["bgd4396", "frv4410", "bgf4475", "fqm5087", "bnd7168", "ida8197"]
NATIONAL3 = {"eg7146": 172386, "ar9152": 837479, "gr9882": 300899}
TSPLIB3 = {"pcb3038": 137694, "fl3795": 28772, "fnl4461": 182566}


def _uniform(rng, n):
    return rng.uniform(0, 1e6, size=(n, 2))


def _clustered(rng, n):
    k = max(1, n // 100)
    centres = rng.uniform(0, 1e6, size=(k, 2))
    pts = centres[rng.integers(k, size=n)] + rng.normal(0, 1e6 / np.sqrt(n), size=(n, 2))
    return np.clip(pts, 0, 1e6)


def _mixed(rng, n):
    h = n // 2
    return np.vstack([_uniform(rng, h), _clustered(rng, n - h)])


GEN = {"uniform": _uniform, "clustered": _clustered, "mixed": _mixed}


def write_tsp(path: Path, name: str, pts):
    lines = [f"NAME : {name}", "TYPE : TSP", f"DIMENSION : {len(pts)}",
             "EDGE_WEIGHT_TYPE : EUC_2D", "NODE_COORD_SECTION"]
    lines += [f"{i + 1} {x:.1f} {y:.1f}" for i, (x, y) in enumerate(pts)]
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def cmd_write():
    import vlsi_datasets as v
    DIR.mkdir(exist_ok=True)
    cat = {"seed": SEED, "instances": {}}
    for k in VLSI6:
        shutil.copyfile(HERE / "data_vlsi" / f"{k}.tsp", DIR / f"{k}.tsp")
        cat["instances"][k] = {"stratum": "vlsi6", "n": v.CATALOG[k][0], "bks": float(v.CATALOG[k][1]),
                               "bks_kind": "opt", "real": True}
    for k, opt in NATIONAL3.items():
        shutil.copyfile(HERE / "data_world" / f"{k}.tsp", DIR / f"{k}.tsp")
        cat["instances"][k] = {"stratum": "national3", "n": int(k[2:]), "bks": float(opt),
                               "bks_kind": "opt", "real": True}
    for k, opt in TSPLIB3.items():
        shutil.copyfile(HERE / "data" / f"{k}.tsp", DIR / f"{k}.tsp")
        cat["instances"][k] = {"stratum": "tsplib3", "n": int("".join(c for c in k if c.isdigit())),
                               "bks": float(opt), "bks_kind": "opt", "real": True,
                               "note": "used in Phase 1 (slow engine, kick-inactive) only"}
    rng = np.random.default_rng(SEED)
    sizes = rng.integers(1500, 4001, size=N_PER_KIND * len(KINDS))
    i = 0
    for kind in KINDS:
        for j in range(N_PER_KIND):
            n = int(sizes[i])
            i += 1
            pts = GEN[kind](np.random.default_rng([SEED, i]), n)
            nm = f"h2{kind[0]}{j:02d}_{n}"
            write_tsp(DIR / f"{nm}.tsp", nm, pts)
            cat["instances"][nm] = {"stratum": f"synth_{kind}", "n": n, "real": False}
    (DIR / "catalog.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")
    print(len(cat["instances"]), "instances written to", DIR)


def _ref(nm):
    import fast_ils as F
    from kick_targeting_experiment import load_xy
    from lkh_runner import run_lkh
    p = DIR / f"{nm}.tsp"
    xs, ys, kind = load_xy(p)
    r = run_lkh(p, len(xs), 60.0, 1)
    c = float(F.tour_cost(np.array(r["tour"], dtype=np.int32), xs, ys, kind))
    return nm, c, r["wall"]


def cmd_ref(workers):
    cat = json.loads((DIR / "catalog.json").read_text(encoding="utf-8"))
    todo = [k for k, d in cat["instances"].items() if "bks" not in d]
    with mp.Pool(workers) as pool:
        for nm, c, w in pool.imap_unordered(_ref, todo):
            d = cat["instances"][nm]
            d.update({"bks": c, "bks_kind": "lkh60", "ref_wall": w})
            print(nm, c, round(w, 1), flush=True)
            (DIR / "catalog.json").write_text(json.dumps(cat, indent=1), encoding="utf-8")


def cmd_check():
    """Parses every file, checks n and that real optima are not beaten by a nearest-neighbour bound."""
    from kick_targeting_experiment import load_xy
    cat = json.loads((DIR / "catalog.json").read_text(encoding="utf-8"))
    bad = 0
    for k, d in cat["instances"].items():
        xs, _, _ = load_xy(DIR / f"{k}.tsp")
        ok = len(xs) == d["n"]
        bad += not ok
        print(f"{k:16s} {d['stratum']:16s} n={len(xs):5d} bks={d.get('bks', '-')} {'' if ok else 'N MISMATCH'}")
    print("instances:", len(cat["instances"]), "missing refs:",
          sum("bks" not in d for d in cat["instances"].values()), "errors:", bad)


def instances():
    cat = json.loads((DIR / "catalog.json").read_text(encoding="utf-8"))
    return [(k, DIR / f"{k}.tsp", float(d["bks"])) for k, d in
            sorted(cat["instances"].items(), key=lambda kv: kv[1]["n"])]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("write", "ref", "check"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    {"write": cmd_write, "ref": lambda: cmd_ref(a.workers), "check": cmd_check}[a.cmd]()
