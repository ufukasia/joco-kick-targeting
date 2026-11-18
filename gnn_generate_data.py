# -*- coding: utf-8 -*-
"""Training-data generator for the Trained GNN-Guided Local Search.

For each synthetic instance (uniform / clustered / mixed — TSPLIB is NEVER
used, so the benchmark evaluation stays zero-shot):

  1. Build the deterministic base tour with the user's own pipeline
     (snake-grid construction -> neighborhood VND), the same distribution the
     GNN methods are seeded from in the benchmark.
  2. Label edges with EXACT ground truth (default `--labels lkh`): the
     instance is written to a temporary TSPLIB file and solved with the
     bundled LKH-3 (see external_solvers.py) — effectively optimal at these
     sizes — and an edge's label is 1.0 iff it belongs to that optimal tour.
     The legacy `--labels ils` mode instead uses per-edge CONSENSUS
     frequencies over the best half of S independent ILS local optima (the
     original, weaker "backbone" signal).
  3. Store training snapshots drawn from the states the search actually
     visits at inference time: the VND tour, perturbed+polished tours
     (double-bridge / segment-reverse followed by a short local search), a
     couple of the raw ILS optima themselves, and (in lkh mode) the optimal
     tour itself so the model also sees what a perfect state looks like.

Output: one .npz per instance in --out (coords, snapshots, edges, freqs, meta).

Usage:
  python gnn_generate_data.py --out gnn_train_data_lkh --count 120 \
      --nmin 100 --nmax 400 --ils-seeds 4 --ils-time 4 --snapshots 8
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import tsplib_engine as E  # inserts the sneak-yazilim dir into sys.path
import external_solvers as ES
import line_reassign_optimizer as lro
from app import snake_path_initialization_result

lro.VERBOSE = False
lro.RECORD_FRAME_TOURS = False

HERE = Path(__file__).resolve().parent


@contextlib.contextmanager
def _quiet():
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()
    try:
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def _log(msg: str) -> None:
    print(msg, file=sys.__stderr__, flush=True)


# ---------------------------------------------------------------------------
#  Synthetic instance generators (coordinates in [0, 1000]^2, EUC_2D)
# ---------------------------------------------------------------------------
def gen_uniform(rng: random.Random, n: int) -> list[tuple[float, float]]:
    return [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]


def gen_clustered(rng: random.Random, n: int) -> list[tuple[float, float]]:
    nc = rng.randint(3, 9)
    centers = [(rng.uniform(120, 880), rng.uniform(120, 880)) for _ in range(nc)]
    stds = [rng.uniform(20, 90) for _ in range(nc)]
    pts = []
    for _ in range(n):
        c = rng.randrange(nc)
        pts.append((
            min(1000.0, max(0.0, rng.gauss(centers[c][0], stds[c]))),
            min(1000.0, max(0.0, rng.gauss(centers[c][1], stds[c]))),
        ))
    return pts


def gen_mixed(rng: random.Random, n: int) -> list[tuple[float, float]]:
    n_cl = int(n * 0.6)
    return gen_clustered(rng, n_cl) + gen_uniform(rng, n - n_cl)


def gen_grid(rng: random.Random, n: int) -> list[tuple[float, float]]:
    """PCB-drilling style (pcb3038-like): points along horizontal grid lines
    with small jitter, plus vertical connector columns."""
    pts: list[tuple[float, float]] = []
    n_lines = rng.randint(6, 16)
    line_ys = sorted(rng.uniform(0, 1000) for _ in range(n_lines))
    n_h = int(n * rng.uniform(0.65, 0.9))
    for _ in range(n_h):
        y = line_ys[rng.randrange(n_lines)]
        pts.append((rng.uniform(0, 1000),
                    min(1000.0, max(0.0, rng.gauss(y, 4.0)))))
    col_xs = sorted(rng.uniform(0, 1000) for _ in range(max(3, n_lines // 2)))
    for _ in range(n - n_h):
        x = col_xs[rng.randrange(len(col_xs))]
        pts.append((min(1000.0, max(0.0, rng.gauss(x, 4.0))),
                    rng.uniform(0, 1000)))
    return pts


def gen_flcluster(rng: random.Random, n: int) -> list[tuple[float, float]]:
    """fl3795-style pathology: a few VERY dense blobs separated by large empty
    regions, plus a thin sparse background scatter."""
    nc = rng.randint(2, 5)
    centers = [(rng.uniform(100, 900), rng.uniform(100, 900)) for _ in range(nc)]
    stds = [rng.uniform(8, 35) for _ in range(nc)]
    n_bg = max(5, int(n * rng.uniform(0.05, 0.15)))
    pts = []
    for _ in range(n - n_bg):
        c = rng.randrange(nc)
        pts.append((
            min(1000.0, max(0.0, rng.gauss(centers[c][0], stds[c]))),
            min(1000.0, max(0.0, rng.gauss(centers[c][1], stds[c]))),
        ))
    for _ in range(n_bg):
        pts.append((rng.uniform(0, 1000), rng.uniform(0, 1000)))
    return pts


KINDS = [("uniform", gen_uniform), ("clustered", gen_clustered), ("mixed", gen_mixed),
         ("grid", gen_grid), ("flcluster", gen_flcluster)]
KIND_BY_NAME = dict(KINDS)


# ---------------------------------------------------------------------------
#  Exact ground truth via LKH-3
# ---------------------------------------------------------------------------
def _write_tsp(path: Path, coords: list[tuple[float, float]]) -> None:
    """Writes a minimal TSPLIB EUC_2D file. LKH rounds EUC_2D distances with
    nint exactly like tsplib_engine/make_instance, so its optimal tour is
    optimal under the same cost function the labels are defined on."""
    lines = [
        "NAME : gnn_train_instance",
        "TYPE : TSP",
        f"DIMENSION : {len(coords)}",
        "EDGE_WEIGHT_TYPE : EUC_2D",
        "NODE_COORD_SECTION",
    ]
    lines += [f"{i + 1} {x:.6f} {y:.6f}" for i, (x, y) in enumerate(coords)]
    lines.append("EOF")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def lkh_optimal_tour(coords: list[tuple[float, float]], time_limit: float,
                     runs: int, seed: int) -> list[int]:
    """Solves the instance with the bundled LKH-3 and returns its tour
    (effectively the optimum at training sizes). Raises on failure so the
    caller skips the instance rather than training on a silent fallback."""
    with tempfile.TemporaryDirectory(prefix="gnn_lkh_") as td:
        tsp_path = Path(td) / "inst.tsp"
        _write_tsp(tsp_path, coords)
        tour, status = ES.run_lkh(tsp_path, len(coords),
                                  time_limit=time_limit, seed=seed, runs=runs)
    if tour is None:
        raise RuntimeError(f"LKH-3 labeling failed: {status}")
    return tour


# ---------------------------------------------------------------------------
#  One instance -> one training file
# ---------------------------------------------------------------------------
def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def tour_edges(tour: list[int]):
    n = len(tour)
    for i in range(n):
        yield edge_key(tour[i], tour[(i + 1) % n])


def build_one(idx: int, args, rng: random.Random,
              kinds: list[tuple] | None = None) -> dict | None:
    kinds = kinds or KINDS
    kind_name, gen = kinds[idx % len(kinds)]
    n = rng.randint(args.nmin, args.nmax)
    coords = gen(rng, n)
    inst, _ = E.make_instance(coords, "EUC_2D")

    t_start = time.perf_counter()
    with _quiet():
        snake = snake_path_initialization_result(inst)
        vnd = lro.run_neighborhood_vnd(
            inst, snake.tour,
            neighbor_limit_2opt=20, neighbor_limit_oropt=20,
            max_oropt_segment=3, k_relocate=30,
            relocate_neighbor_limit=120, max_rounds=40)
    t0_tour = vnd.tour
    t0_cost = vnd.cost

    # --- S independent strong local optima (the label source) ---
    optima: list[tuple[float, list[int]]] = []
    for s in range(1, args.ils_seeds + 1):
        with _quiet():
            _, diag = lro.build_ils_result(
                inst, t0_tour,
                max_iterations=args.ils_iters,
                max_no_improve=max(25, args.ils_iters // 3),
                max_time_seconds=args.ils_time,
                seed=s, max_ls_rounds=8)
        optima.append((diag.cost, diag.tour))
    optima.sort(key=lambda cw: cw[0])

    lkh_tour: list[int] | None = None
    if args.labels == "lkh":
        # exact ground truth: an edge is a "backbone" edge iff it is in the
        # LKH-3 (near-)optimal tour — binary labels, no consensus uncertainty
        lkh_tour = lkh_optimal_tour(coords, args.lkh_time, args.lkh_runs,
                                    seed=1_000 + idx)
        edges = np.array(sorted(tour_edges(lkh_tour)), dtype=np.int32)
        freqs = np.ones(len(edges), dtype=np.float32)
    else:
        # legacy: consensus frequencies over the BEST half of the ILS optima
        top = optima[: max(2, math.ceil(len(optima) / 2))]
        counts: dict[tuple[int, int], int] = {}
        for _, tr in top:
            for e in tour_edges(tr):
                counts[e] = counts.get(e, 0) + 1
        m_top = len(top)
        edges = np.array(sorted(counts.keys()), dtype=np.int32)
        freqs = np.array([counts[tuple(e)] / m_top for e in edges.tolist()],
                         dtype=np.float32)

    # --- snapshots: the state distribution the model sees at inference ---
    projected = lro.compute_projected_coords(inst)
    dist_matrix = inst.distance_matrix
    cnn = lro._precompute_city_nearest_neighbors(inst, min(n - 1, 120))
    best_tour = optima[0][1]

    def polish(tour: list[int], rounds: int) -> list[int]:
        cost = lro.tour_cost_from_matrix(tour, dist_matrix)
        out, _, _, _ = lro._combined_local_search(
            tour=tour, current_cost=cost, dist_matrix=dist_matrix,
            projected=projected, city_nearest_neighbors=cnn,
            neighbor_limit_2opt=20, neighbor_limit_oropt=20,
            k_nearest_edges_relocate=30, city_neighbor_limit_relocate=120,
            max_ls_rounds=rounds)
        return out

    snapshots: list[list[int]] = [list(t0_tour)]
    prng = random.Random(10_000 + idx)
    for j in range(1, max(1, args.snapshots - 2)):
        base = best_tour if j % 2 == 0 else t0_tour
        if j % 3 == 0:
            pert = lro._double_bridge_perturbation(list(base), prng)
        else:
            pert = lro._segment_reverse_perturbation(list(base), prng)
        snapshots.append(polish(pert, 1 + (j % 2)))
    # two raw ILS optima: the consensus-best and the WORST descent — the
    # latter is a genuine local minimum whose divergence from the backbone
    # is exactly what the misplacement head must learn to flag.
    snapshots.append(list(best_tour))
    snapshots.append(list(optima[-1][1]))
    if lkh_tour is not None:
        # the optimum itself: every edge label 1 / every node label 0 — shows
        # the model what a state with nothing left to fix looks like
        snapshots.append(list(lkh_tour))

    dt = time.perf_counter() - t_start
    meta = {
        "n": n, "kind": kind_name,
        "t0_cost": round(t0_cost, 2),
        "ils_costs": [round(c, 2) for c, _ in optima],
        "labels": args.labels,
        "gen_time_s": round(dt, 1),
    }
    if lkh_tour is not None:
        lkh_cost = inst.tour_cost(lkh_tour)
        meta["lkh_cost"] = round(lkh_cost, 2)
        meta["ils_best_gap_pct"] = round(
            100.0 * (optima[0][0] - lkh_cost) / max(lkh_cost, 1e-9), 3)
    _log(f"[{idx:04d}] {kind_name:9s} n={n:4d} t0={t0_cost:9.0f} "
         f"best={optima[0][0]:9.0f} worst={optima[-1][0]:9.0f} "
         + (f"lkh={meta['lkh_cost']:9.0f} gap={meta['ils_best_gap_pct']:5.2f}% "
            if lkh_tour is not None else "")
         + f"edges={len(edges)} snaps={len(snapshots)} t={dt:.0f}s")
    return {
        "coords": np.asarray(coords, dtype=np.float64),
        "snapshots": np.asarray(snapshots, dtype=np.int32),
        "edges": edges, "freqs": freqs,
        "meta": meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="gnn_train_data_lkh")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--nmin", type=int, default=100)
    ap.add_argument("--nmax", type=int, default=400)
    ap.add_argument("--ils-seeds", type=int, default=4,
                    help="ILS descents per instance (snapshot diversity; also "
                         "the label source in --labels ils mode)")
    ap.add_argument("--ils-iters", type=int, default=150)
    ap.add_argument("--ils-time", type=float, default=4.0,
                    help="per-seed ILS time budget (s)")
    ap.add_argument("--labels", choices=("lkh", "ils"), default="lkh",
                    help="edge-label source: 'lkh' = exact LKH-3 optimum "
                         "(default), 'ils' = legacy ILS consensus frequencies")
    ap.add_argument("--lkh-time", type=float, default=10.0,
                    help="LKH-3 TIME_LIMIT per instance (s)")
    ap.add_argument("--lkh-runs", type=int, default=2,
                    help="LKH-3 RUNS per instance")
    ap.add_argument("--snapshots", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", type=int, default=0,
                    help="first instance index (resume support)")
    ap.add_argument("--kinds", type=str, default=None,
                    help="comma list of generator kinds to cycle (default: all)")
    args = ap.parse_args()

    if args.kinds:
        try:
            kinds = [(k, KIND_BY_NAME[k]) for k in args.kinds.split(",")]
        except KeyError as exc:
            raise SystemExit(f"unknown kind {exc}; choices: "
                             f"{', '.join(KIND_BY_NAME)}")
    else:
        kinds = KINDS

    out_dir = (HERE / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    for idx in range(args.start, args.count):
        path = out_dir / f"inst_{idx:04d}.npz"
        if path.exists():
            _log(f"[{idx:04d}] exists, skipping")
            continue
        rng = random.Random(args.seed * 1_000_003 + idx)
        try:
            rec = build_one(idx, args, rng, kinds=kinds)
        except Exception as exc:
            import traceback
            _log(f"[{idx:04d}] FAILED: {exc}")
            traceback.print_exc(file=sys.__stderr__)
            continue
        np.savez_compressed(
            path,
            coords=rec["coords"], snapshots=rec["snapshots"],
            edges=rec["edges"], freqs=rec["freqs"],
            meta=np.frombuffer(json.dumps(rec["meta"]).encode("utf-8"),
                               dtype=np.uint8))
    _log(f"done: {args.count - args.start} instances in "
         f"{(time.perf_counter() - t_all) / 60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
