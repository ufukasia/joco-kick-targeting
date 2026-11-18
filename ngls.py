# -*- coding: utf-8 -*-
"""NGLS — Neural-Prior Guided Local Search for TSP tours.

Fast Guided Local Search (GLS) whose edge penalties are BIASED by the trained
GNN edge head from `gnn_trained.py` and updated ONLINE from the search
trajectory:

  * The model's edge head gives a prior  p_e = P(edge is NOT a backbone edge)
    for every edge of the incumbent tour (snapshot recomputed only after a
    meaningful incumbent improvement, like the v2 trained-GNN cache).
  * Each GLS cycle penalises the max-utility edges of the current tour, where
    the classic GLS utility is weighted by the learned prior:

        util(e) = c(e) * (1 + BETA * p_e) / (1 + pen(e))

    so edges the model flags as non-backbone are penalised EARLIER, while the
    online counter pen(e) corrects the model when it is wrong (the prior is a
    starting bias, not a decision maker).
  * Local search runs on the AUGMENTED cost c'(e) = c(e) + lambda*pen(e), but
    in the FAST-GLS style (Voudouris & Tsang): after each penalty step only
    the cities incident to the penalised edges are activated and a RESTRICTED
    2-opt / Or-opt repair propagates from them, so one cycle costs O(k), not
    O(n*k), and thousands of penalty cycles fit into the benchmark budgets.
    The true cost is tracked incrementally alongside the augmented deltas.
  * A new incumbent is occasionally polished with the full TRUE-cost local
    search (2-opt + Or-opt + relocate), and on prolonged true-cost stagnation
    a LEARNED KICK (the guided double-bridge of `gnn_trained.py`, cuts
    sampled from the model's most suspicious edges) plus a deep TRUE-cost
    descent restarts the region while the penalty table decays.

Motivation (paper): a purely static learned score does not transfer reliably
from synthetic training instances to TSPLIB (shown experimentally by the
failed retraining sweep); NGLS therefore demotes the learned model to a PRIOR
over edge badness and lets the online GLS penalties act as the posterior.

The module reuses the model and kick code of `gnn_trained.py` and the
local-search primitives of `line_reassign_optimizer.py`, so ILS / trained-GNN
/ NGLS share one move set and differ only in guidance (fair comparison)."""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

import line_reassign_optimizer as lro
import gnn_trained as gt
from app import (
    HistoryPoint,
    RunMetrics,
    TSPInstance,
    TSPRunResult,
    TourFrame,
    tour_cost_from_matrix,
)
from line_reassign_optimizer import LineReassignResult

# ---------------------------------------------------------------------------
#  GLS constants
# ---------------------------------------------------------------------------
_GLS_ALPHA = 0.3        # lambda = ALPHA * best_cost / n   (Voudouris & Tsang)
_PRIOR_BETA = 1.0       # weight of the learned prior inside the utility
_PRIOR_DEFAULT = 0.5    # prior for edges the model has not scored yet
_TOPK_DIVISOR = 1500    # edges penalised per cycle = max(1, n // _TOPK_DIVISOR)
_KICK_PATIENCE = 300    # cycles without true improvement before a kick
_KICK_LS_ROUNDS = 6     # deep TRUE-cost descent after a kick (ILS cadence)
_POLISH_LS_ROUNDS = 2   # full TRUE-cost polish of a new incumbent
_POLISH_MIN_GAP = 25    # min cycles between two incumbent polishes
_PEN_DECAY = 0.5        # penalty decay applied at each kick
_PEN_FLOOR = 0.05       # decayed penalties below this are dropped
_SNAP_REL_GAIN = 5e-4   # refresh the model snapshot after this relative gain
_MAX_STALE_KICKS = 8    # untimed runs stop after this many fruitless kicks
_OROPT_MAX_SEG = 3      # restricted Or-opt segment length


# ---------------------------------------------------------------------------
#  Augmented distances: dist[a][b] + lambda * pen[a][b]
# ---------------------------------------------------------------------------
class _AugRow:
    """Row proxy adding the penalty term only for the few penalised partners."""
    __slots__ = ("_base", "_pen", "_lam")

    def __init__(self, base_row, pen_row, lam):
        self._base = base_row
        self._pen = pen_row
        self._lam = lam

    def __getitem__(self, j):
        p = self._pen.get(j)
        if p:
            return self._base[j] + self._lam * p
        return self._base[j]


class AugmentedDistance:
    """Behaves like `dist_matrix[a][b]` but returns the GLS-augmented cost.

    Rows with no penalised edge return the RAW base row (zero overhead), so
    hot loops only pay for the wrapper on the small set of cities incident to
    penalised edges. Works with both the dense matrix and the sparse
    LazyDistance (anything supporting `base[a][b]`)."""
    __slots__ = ("base", "pen", "lam")

    def __init__(self, base):
        self.base = base
        self.pen: dict[int, dict[int, float]] = {}
        self.lam = 0.0

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        prow = self.pen.get(i)
        if prow is None or self.lam <= 0.0:
            return self.base[i]
        return _AugRow(self.base[i], prow, self.lam)

    def penalty(self, a: int, b: int) -> float:
        row = self.pen.get(a)
        return row.get(b, 0.0) if row else 0.0

    def add_penalty(self, a: int, b: int, amount: float = 1.0) -> None:
        for i, j in ((a, b), (b, a)):
            row = self.pen.get(i)
            if row is None:
                row = self.pen[i] = {}
            row[j] = row.get(j, 0.0) + amount

    def decay(self, factor: float) -> None:
        if factor <= 0.0:
            self.pen.clear()
            return
        for i in list(self.pen):
            row = self.pen[i]
            for j in list(row):
                v = row[j] * factor
                if v < _PEN_FLOOR:
                    del row[j]
                else:
                    row[j] = v
            if not row:
                del self.pen[i]


def _penalize_max_utility(
    aug: AugmentedDistance,
    tour: list[int],
    true_dm,
    prior_map: dict[tuple[int, int], float],
    top_k: int,
) -> list[int]:
    """Classic GLS penalty step with the learned-prior-weighted utility.
    Returns the cities incident to the penalised edges (activation seeds)."""
    n = len(tour)
    utils = []
    for i in range(n):
        a = tour[i]
        b = tour[(i + 1) % n]
        key = (a, b) if a < b else (b, a)
        prior = prior_map.get(key, _PRIOR_DEFAULT)
        u = (true_dm[a][b] * (1.0 + _PRIOR_BETA * prior)
             / (1.0 + aug.penalty(a, b)))
        utils.append((u, a, b))
    utils.sort(reverse=True)
    seeds: list[int] = []
    for _, a, b in utils[:top_k]:
        aug.add_penalty(a, b)
        seeds.append(a)
        seeds.append(b)
    return seeds


# ---------------------------------------------------------------------------
#  Fast-GLS restricted repair: 2-opt + Or-opt propagating from activated
#  cities only (augmented acceptance, incremental true-cost tracking)
# ---------------------------------------------------------------------------
def _restricted_repair(
    tour: list[int],
    aug: AugmentedDistance,
    true_dm,
    city_nearest_neighbors: list[list[int]],
    neighbor_limit_2opt: int,
    neighbor_limit_oropt: int,
    seed_cities: list[int],
    deadline: Optional[float] = None,
) -> tuple[list[int], float, int]:
    """Restricted local search seeded at `seed_cities` on the AUGMENTED cost.

    Maintains an activation queue (fast-GLS "activation bits"): each accepted
    move re-activates the cities whose tour edges changed, so the repair
    propagates exactly as far as the penalty perturbation reaches and no
    further. Returns (tour, true_cost_delta, accepted_moves)."""
    n = len(tour)
    max_city_id = max(tour)
    city_to_pos = [0] * (max_city_id + 1)
    for pos, city in enumerate(tour):
        city_to_pos[city] = pos

    queue: deque[int] = deque()
    queued = bytearray(max_city_id + 1)

    def _activate(city: int) -> None:
        if not queued[city]:
            queued[city] = 1
            queue.append(city)

    for c in seed_cities:
        _activate(c)

    true_delta = 0.0
    accepted = 0
    guard = 0
    max_steps = 40 * max(4, len(seed_cities))  # runaway-propagation guard

    while queue:
        guard += 1
        if guard > max_steps or lro._past_deadline(deadline):
            break
        city = queue.popleft()
        queued[city] = 0
        i = city_to_pos[city]

        # ---- restricted 2-opt around `city` ----
        moved = True
        while moved:
            moved = False
            i = city_to_pos[city]
            for nb in city_nearest_neighbors[city][:neighbor_limit_2opt]:
                j = city_to_pos[nb]
                if j > i + 1 and not (i == 0 and j == n - 1):
                    a, b = i, j
                elif i > j + 1 and not (j == 0 and i == n - 1):
                    a, b = j, i
                else:
                    continue
                d_aug = lro._two_opt_delta(tour, aug, a, b)
                if d_aug < -1e-9:
                    true_delta += lro._two_opt_delta(tour, true_dm, a, b)
                    tour = lro._apply_two_opt(tour, a, b)
                    for pos in range(a + 1, b + 1):
                        city_to_pos[tour[pos]] = pos
                    for c in (tour[a], tour[a + 1], tour[b], tour[(b + 1) % n]):
                        _activate(c)
                    accepted += 1
                    moved = True
                    break

        # ---- restricted Or-opt: move the short segment starting at `city`
        #      next to one of its geometric neighbours ----
        i = city_to_pos[city]
        best = None
        for seg_len in range(1, min(_OROPT_MAX_SEG, n - 2) + 1):
            for nb in city_nearest_neighbors[city][:neighbor_limit_oropt]:
                insert_after = city_to_pos[nb]
                offset = (insert_after - i) % n
                if offset < seg_len or insert_after == (i - 1) % n:
                    continue  # inside the segment or a no-op reinsertion
                d_aug = lro._or_opt_delta(tour, aug, i, seg_len, insert_after)
                if d_aug < -1e-9 and (best is None or d_aug < best[0]):
                    best = (d_aug, seg_len, insert_after)
        if best is not None:
            _, seg_len, insert_after = best
            true_delta += lro._or_opt_delta(tour, true_dm, i, seg_len, insert_after)
            neighbors_of_move = (
                tour[(i - 1) % n], tour[i],
                tour[(i + seg_len - 1) % n], tour[(i + seg_len) % n],
                tour[insert_after], tour[(insert_after + 1) % n],
            )
            tour = lro._apply_or_opt(tour, i, seg_len, insert_after)
            for pos, c in enumerate(tour):   # segment shifts: full rebuild
                city_to_pos[c] = pos
            for c in neighbors_of_move:
                _activate(c)
            accepted += 1

    return tour, true_delta, accepted


# ---------------------------------------------------------------------------
#  Search
# ---------------------------------------------------------------------------
def run_ngls(
    instance: TSPInstance,
    initial_tour: list[int],
    weights: dict[str, np.ndarray],
    max_iterations: int = 6000,
    max_no_improve: int = 200,
    max_time_seconds: float = 0.0,
    seed: int = 42,
) -> LineReassignResult:
    """Neural-Prior Guided Local Search (fast-GLS implementation).

    One iteration = one GLS penalty cycle: penalise the max-utility edges of
    the current tour (learned-prior-weighted utility), then run the RESTRICTED
    augmented-cost repair seeded at the penalised edges. True cost is tracked
    incrementally; the incumbent updates only on a true improvement and is
    occasionally polished by the full true-cost local search. On
    `_KICK_PATIENCE` cycles without true improvement a learned kick + deep
    true-cost descent fires and the penalties decay."""
    import random as random_module

    rng = random_module.Random(seed)
    instance.validate_tour(initial_tour)
    projected = lro.compute_projected_coords(instance)
    dist_matrix = instance.distance_matrix
    n_cities = len(initial_tour)

    # identical neighbourhood sizing to the v2 trained-GNN skeleton
    k_geo = gt.K_GEO
    neighbor_limit_2opt = min(n_cities - 1, max(40, k_geo * 2))
    neighbor_limit_oropt = min(n_cities - 1, max(24, k_geo))
    relocate_k = min(n_cities - 1, max(30, gt.DEFAULT_K_INSERT_CANDIDATES))
    relocate_neighbor_limit = min(
        n_cities - 1, max(160, gt.DEFAULT_K_INSERT_CANDIDATES * 4))
    max_nn = min(
        n_cities - 1,
        max(k_geo, gt.K_DENSITY, neighbor_limit_2opt, neighbor_limit_oropt,
            relocate_neighbor_limit),
    )

    if lro.VERBOSE:
        print(f"\n{'=' * 65}")
        print(f"[{time.strftime('%H:%M:%S')}] NEURAL-PRIOR GLS (NGLS) STARTED")
        print(f"{'=' * 65}")

    t0 = time.perf_counter()
    city_nearest_neighbors = lro._precompute_city_nearest_neighbors(instance, max_nn)
    knn_arr = gt.knn_table_to_array(city_nearest_neighbors, min(max_nn, n_cities - 1))
    xs, ys, is_geo = gt.extract_xy(instance)
    proj_x = np.asarray([p[0] for p in projected])
    proj_y = np.asarray([p[1] for p in projected])
    if lro.VERBOSE:
        sys.stdout.write(
            f"  [Info] Cached nearest-neighbor lists up to {max_nn} cities "
            f"in {time.perf_counter() - t0:.2f}s.\n")

    start_time = time.perf_counter()
    ls_deadline = (start_time + max_time_seconds * lro._LS_DEADLINE_SAFETY
                   if max_time_seconds > 0 else None)
    initial_cost = tour_cost_from_matrix(initial_tour, dist_matrix)

    def _true_ls(tour, cost, rounds):
        return lro._combined_local_search(
            tour=tour, current_cost=cost,
            dist_matrix=dist_matrix, projected=projected,
            city_nearest_neighbors=city_nearest_neighbors,
            neighbor_limit_2opt=neighbor_limit_2opt,
            neighbor_limit_oropt=neighbor_limit_oropt,
            k_nearest_edges_relocate=relocate_k,
            city_neighbor_limit_relocate=relocate_neighbor_limit,
            max_ls_rounds=rounds, deadline=ls_deadline,
        )

    # Phase 0: deep true-cost descent (identical to the ILS / v2 phase 0)
    working_tour, working_cost, init_accepted, _ = _true_ls(
        list(initial_tour), initial_cost, lro.DEFAULT_GNN_POLISH_LS_ROUNDS)
    elapsed_init = time.perf_counter() - start_time
    if lro.VERBOSE:
        sys.stdout.write(
            f"  [NGLS] Initial LS: {initial_cost:.2f} -> {working_cost:.2f} "
            f"({init_accepted} moves, {elapsed_init:.1f}s)\n")

    best_tour = list(working_tour)
    best_cost = working_cost

    aug = AugmentedDistance(dist_matrix)
    aug.lam = _GLS_ALPHA * best_cost / n_cities
    top_k = max(1, n_cities // _TOPK_DIVISOR)
    prior_map: dict[tuple[int, int], float] = {}

    frames: list[TourFrame] = [
        TourFrame(step=0, phase="initial", stage_index=0, elapsed_time=0.0,
                  cost=initial_cost, best_cost=best_cost, constructed_cities=0,
                  tour=lro._frame_tour(list(initial_tour))),
        TourFrame(step=1, phase="initial_ls", stage_index=0,
                  elapsed_time=elapsed_init, cost=working_cost,
                  best_cost=best_cost, constructed_cities=init_accepted,
                  tour=lro._frame_tour(working_tour)),
    ]
    history: list[HistoryPoint] = [
        HistoryPoint(iteration=0, elapsed_time=0.0, cost=initial_cost),
        HistoryPoint(iteration=1, elapsed_time=elapsed_init, cost=best_cost),
    ]

    total_improvements = 0
    no_improve_cycles = 0     # penalty cycles since the last TRUE improvement
    stale_kicks = 0           # consecutive kicks that failed to improve best
    restart_count = 0
    cycles_since_polish = _POLISH_MIN_GAP
    iteration = 0

    snap_edge_quality = None  # [(P(backbone), pos)] on best_tour, for kicks
    snap_cost = None

    def _record_best(elapsed, moves):
        frames.append(TourFrame(
            step=len(frames), phase="repair", stage_index=total_improvements,
            elapsed_time=elapsed, cost=best_cost, best_cost=best_cost,
            constructed_cities=moves, tour=lro._frame_tour(best_tour),
        ))
        history.append(HistoryPoint(
            iteration=total_improvements + 1, elapsed_time=elapsed,
            cost=best_cost,
        ))

    for iteration in range(1, max_iterations + 1):
        elapsed = time.perf_counter() - start_time
        if max_time_seconds > 0 and elapsed >= max_time_seconds:
            if lro.VERBOSE:
                sys.stdout.write("\n  [NGLS] Time limit reached.\n")
            break
        if max_time_seconds <= 0 and stale_kicks >= _MAX_STALE_KICKS:
            if lro.VERBOSE:
                sys.stdout.write("\n  [NGLS] Stagnation limit reached.\n")
            break

        # ---- model snapshot on the incumbent; refreshed only after a
        #      meaningful gain so the forward pass stays off the hot path ----
        if snap_cost is None or best_cost < snap_cost * (1.0 - _SNAP_REL_GAIN):
            tour_arr = np.asarray(best_tour, dtype=np.int64)
            g = gt.compute_graph_tensors(
                tour_arr, xs, ys, is_geo, proj_x, proj_y, knn_arr)
            _, _, edge_logit = gt.model_forward(weights, g)
            edge_good = gt._sigmoid(edge_logit)      # P(backbone edge)
            nxt = np.roll(tour_arr, -1)
            for i in range(n_cities):
                a = int(tour_arr[i]); b = int(nxt[i])
                key = (a, b) if a < b else (b, a)
                prior_map[key] = 1.0 - float(edge_good[i])   # P(non-backbone)
            snap_edge_quality = sorted(
                (float(edge_good[i]), i) for i in range(n_cities))
            snap_cost = best_cost

        # ---- GLS cycle: penalise max-utility edges, restricted repair ----
        seeds = _penalize_max_utility(
            aug, working_tour, dist_matrix, prior_map, top_k)
        working_tour, true_delta, cyc_moves = _restricted_repair(
            working_tour, aug, dist_matrix, city_nearest_neighbors,
            neighbor_limit_2opt, neighbor_limit_oropt, seeds,
            deadline=ls_deadline,
        )
        working_cost += true_delta
        cycles_since_polish += 1

        if working_cost < best_cost - 1e-9:
            # exact resync (increments are float-safe, but stay exact)
            working_cost = tour_cost_from_matrix(working_tour, dist_matrix)
        improved = working_cost < best_cost - 1e-9

        if improved and cycles_since_polish >= _POLISH_MIN_GAP:
            # full true-cost polish of the new incumbent (adds the relocate
            # neighbourhood the restricted repair does not search)
            working_tour, working_cost, p_moves, _ = _true_ls(
                working_tour, working_cost, _POLISH_LS_ROUNDS)
            cyc_moves += p_moves
            cycles_since_polish = 0

        if improved:
            best_tour = list(working_tour)
            best_cost = working_cost
            total_improvements += 1
            no_improve_cycles = 0
            stale_kicks = 0
            aug.lam = _GLS_ALPHA * best_cost / n_cities
            _record_best(time.perf_counter() - start_time, cyc_moves)
        else:
            no_improve_cycles += 1

        # ---- learned kick on prolonged true-cost stagnation ----
        if no_improve_cycles >= _KICK_PATIENCE:
            kicked = gt._guided_double_bridge(best_tour, snap_edge_quality, rng)
            kicked_cost = tour_cost_from_matrix(kicked, dist_matrix)
            working_tour, working_cost, _k_moves, _ = _true_ls(
                kicked, kicked_cost, _KICK_LS_ROUNDS)
            aug.decay(_PEN_DECAY)
            restart_count += 1
            no_improve_cycles = 0
            cycles_since_polish = _POLISH_MIN_GAP
            if working_cost < best_cost - 1e-9:
                best_tour = list(working_tour)
                best_cost = working_cost
                total_improvements += 1
                stale_kicks = 0
                aug.lam = _GLS_ALPHA * best_cost / n_cities
                _record_best(time.perf_counter() - start_time, _k_moves)
            else:
                stale_kicks += 1

        if lro.VERBOSE and (iteration & 127) == 0:
            sys.stdout.write(
                "\r"
                f"[NGLS] Cycle: {iteration}/{max_iterations} | "
                f"Cost: {working_cost:.2f} | Best: {best_cost:.2f} | "
                f"Improved: {total_improvements} | "
                f"Stall: {no_improve_cycles}/{_KICK_PATIENCE} | "
                f"Kicks: {restart_count} | "
                f"Elapsed: {time.perf_counter() - start_time:.1f}s   ")
            sys.stdout.flush()

    if lro.VERBOSE:
        sys.stdout.write("\n")
    elapsed = time.perf_counter() - start_time
    frames.append(TourFrame(
        step=len(frames), phase="final", stage_index=total_improvements,
        elapsed_time=elapsed, cost=best_cost, best_cost=best_cost,
        constructed_cities=0, tour=best_tour[:],
    ))
    history.append(HistoryPoint(
        iteration=total_improvements + 1, elapsed_time=elapsed, cost=best_cost,
    ))

    if lro.VERBOSE:
        print(f"\n{'=' * 65}")
        print("[NGLS] COMPLETED")
        print(f"  Initial cost  : {initial_cost:.2f}")
        print(f"  Best cost     : {best_cost:.2f}")
        print(f"  Total saving  : {initial_cost - best_cost:.2f}")
        print(f"  Improvements  : {total_improvements}")
        print(f"  Cycles        : {iteration}")
        print(f"  Total time    : {elapsed:.1f}s")
        print(f"{'=' * 65}\n")

    return LineReassignResult(
        tour=best_tour,
        cost=best_cost,
        iterations_completed=min(iteration, max_iterations),
        improvements_accepted=total_improvements,
        total_candidates_evaluated=0,
        total_saving=initial_cost - best_cost,
        frames=frames,
        history=history,
    )


def build_ngls_result(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int = 6000,
    max_no_improve: int = 200,
    max_time_seconds: float = 0.0,
    seed: int = 42,
    weights_path: Optional[str | Path] = None,
) -> tuple[TSPRunResult, LineReassignResult]:
    """Dashboard-compatible result for NGLS. `weights_path` defaults to
    $GNN_TRAINED_WEIGHTS or gnn_trained_weights.npz next to this module
    (the SAME weights file as the trained GNN — NGLS adds no new training)."""
    if weights_path is None:
        weights_path = os.environ.get(
            "GNN_TRAINED_WEIGHTS",
            str(Path(__file__).resolve().parent / "gnn_trained_weights.npz"))
    weights, _meta = gt.get_weights(weights_path)

    result = run_ngls(
        instance=instance,
        initial_tour=initial_tour,
        weights=weights,
        max_iterations=max_iterations,
        max_no_improve=max_no_improve,
        max_time_seconds=max_time_seconds,
        seed=seed,
    )
    metrics = RunMetrics(
        initialization_name="Neural-Prior Guided Local Search",
        initial_cost=instance.tour_cost(initial_tour),
        final_cost=result.cost,
        cpu_time_seconds=result.frames[-1].elapsed_time,
        frame_count=len(result.frames),
        history=result.history,
    )
    return (
        TSPRunResult(
            initial_tour=list(initial_tour),
            final_tour=result.tour,
            metrics=metrics,
            animation_frames=result.frames,
        ),
        result,
    )
