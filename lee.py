# -*- coding: utf-8 -*-
"""LEE -- Longest-Edge Elimination, a deterministic GLOBAL finishing pass.

Motivation: every local search in this codebase (2-opt / Or-opt / relocate,
whether inside ILS, GNN-LS or NGLS) only considers a fixed, small candidate
list of geometric neighbours per city. That is what makes them fast, but it
also means the small number of very long "outlier" edges left in a tour
(visible as long diagonal jumps, e.g. on gr666) are sometimes NOT a genuine
local optimum -- their true improving partner simply falls outside the
truncated candidate list of either endpoint, so ordinary 2-opt/Or-opt never
even considers the move that would fix them.

LEE runs LAST, over the run's own best tour so far (any stage/method), and
repeatedly:
  1. Ranks all tour edges by length and takes the outliers (mean + Z*std,
     with a floor so it always has SOMETHING to target even on an already
     very uniform tour).
  2. For just the endpoints of those few outlier edges, computes an
     EXPANDED (much larger-k) nearest-neighbour list on demand -- cheap,
     since it is only ever done for a handful of cities, not all of them.
  3. Runs the SAME restricted 2-opt/Or-opt repair machinery as NGLS
     (`ngls._restricted_repair`, reused verbatim with a zero-penalty
     augmented distance so it performs a plain TRUE-cost repair), seeded at
     those endpoints, so the widened candidate list is exactly what breaks
     the truncation that stranded the long edge in the first place.
  4. Runs the GLOBAL move local search cannot express: DETOUR RELOCATION
     (`_long_segment_pass`). When TWO outlier edges bracket a whole side
     path (the tour jumps out over one long edge, walks a line of cities,
     and jumps back over another), the entire segment is cut out (closing
     the gap directly) and re-inserted -- forward or reversed -- into the
     edge where it spatially belongs. This is an unbounded-length Or-opt
     restricted to outlier-edge pairs, so it stays O(targets^2 * k).
  5. Repeats until a round finds nothing left to fix, a round cap is hit, or
     the deadline passes.

No new training, no new move types: this is a targeting/candidate-list
strategy layered on the exact same primitives already used everywhere else,
so it stays directly comparable in the record()-based tour+cost contract."""
from __future__ import annotations

import heapq
import statistics
import time

import line_reassign_optimizer as lro
import ngls
from app import HistoryPoint, RunMetrics, TSPInstance, TSPRunResult, TourFrame, tour_cost_from_matrix
from line_reassign_optimizer import LineReassignResult

# ---------------------------------------------------------------------------
#  LEE constants
# ---------------------------------------------------------------------------
_Z_THRESH = 2.5        # an edge longer than mean + Z*std of tour-edge lengths is an "outlier"
_MIN_TARGETS = 20      # floor: always target at least this many of the longest edges
_MAX_ROUNDS = 8        # stop once a round fixes nothing (this is just a safety cap)
_BASE_K = 24           # candidate-list size for cities NOT an outlier-edge endpoint
_EXPAND_FLOOR = 80     # expanded candidate-list size floor for outlier-edge endpoints
_EXPAND_FRACTION = 15  # ... and n // this, whichever is larger
_SEG_MAX_DIV = 3       # detour relocation: max segment length = n // _SEG_MAX_DIV


def _target_edge_positions(
    tour: list[int], dist_matrix, z_thresh: float, min_targets: int,
) -> list[int]:
    """Positions i such that edge (tour[i], tour[i+1]) is a length outlier
    (or, on a very uniform tour with no outliers, the `min_targets` longest
    edges) -- worst edge first."""
    n = len(tour)
    lens = [dist_matrix[tour[i]][tour[(i + 1) % n]] for i in range(n)]
    mean = statistics.fmean(lens)
    std = statistics.pstdev(lens) if n > 1 else 0.0
    thresh = mean + z_thresh * std
    order = sorted(range(n), key=lambda i: lens[i], reverse=True)
    picked = [i for i in order if lens[i] > thresh]
    if len(picked) < min_targets:
        picked = order[: min(min_targets, n)]
    return picked


def _expanded_neighbors(city: int, dist_matrix, k: int, n: int) -> list[int]:
    """On-demand, larger-k nearest-neighbour list for ONE city (only ever
    called for outlier-edge endpoints, so the O(n log k) cost is paid a
    handful of times per round, not once per city in the instance)."""
    row = dist_matrix[city]
    take = min(k + 1, n)
    nearest = heapq.nsmallest(take, range(n), key=lambda other: row[other])
    return [c for c in nearest if c != city][:k]


def _long_segment_pass(
    tour: list[int],
    dm,
    z_thresh: float,
    min_targets: int,
    max_seg_len: int,
    neighbors_of,
    deadline,
) -> tuple[list[int], float, int, list[int]]:
    """Global DETOUR RELOCATION over pairs of outlier-length edges.

    The move a human sees on gr666-style tours but 2-opt / short Or-opt cannot
    express: when the tour leaves its "own line" over a long edge, traverses a
    whole side path (dozens of cities) and comes back over another long edge,
    the entire bracketed segment can be cut out (closing x->y directly) and
    re-inserted -- forward or REVERSED -- into the edge (u, v) where it
    geographically belongs. This is an Or-opt with UNBOUNDED segment length,
    restricted to segments delimited by two outlier edges, so the pair count
    stays tiny (targets^2) and each candidate insertion is O(1).

    Best-improvement per iteration, repeated until no move helps. Returns
    (tour, true_delta, moves, touched_cities) -- touched cities should be fed
    back into the restricted repair so the six new junctions get polished."""
    n = len(tour)
    total_delta = 0.0
    moves = 0
    touched: list[int] = []
    while True:
        if lro._past_deadline(deadline):
            break
        targets = _target_edge_positions(tour, dm, z_thresh, min_targets)
        if len(targets) < 2:
            break
        pos = [0] * n
        for p, c in enumerate(tour):
            pos[c] = p
        best = None  # (delta, i, j, k, reversed)
        for i in targets:
            x = tour[i]
            a = tour[(i + 1) % n]
            d_xa = dm[x][a]
            for j in targets:
                if j == i:
                    continue
                seg_len = (j - i) % n   # cities tour[i+1 .. j], cyclic
                if seg_len < 1 or seg_len > max_seg_len:
                    continue
                b = tour[j]
                y = tour[(j + 1) % n]
                removal_gain = d_xa + dm[b][y] - dm[x][y]
                if removal_gain <= 1e-9:
                    continue  # cutting the detour out cannot pay by itself
                # candidate insertion edges come from the EXPANDED neighbour
                # lists of the segment's endpoints (where it belongs, spatially)
                for u in set(neighbors_of(a)) | set(neighbors_of(b)):
                    # rest[k] = tour[(j+1+k) % n]; k >= n-seg_len-1 means u is
                    # inside the segment (or u == x, whose successor edge is the
                    # freshly closed x->y -- reinserting there is the identity)
                    k = (pos[u] - (j + 1)) % n
                    if k >= n - seg_len - 1:
                        continue
                    v = tour[(j + 2 + k) % n]   # rest[k+1]
                    base_ins = -removal_gain - dm[u][v]
                    delta_f = dm[u][a] + dm[b][v] + base_ins
                    if delta_f < -1e-9 and (best is None or delta_f < best[0]):
                        best = (delta_f, i, j, k, False)
                    delta_r = dm[u][b] + dm[a][v] + base_ins
                    if delta_r < -1e-9 and (best is None or delta_r < best[0]):
                        best = (delta_r, i, j, k, True)
        if best is None:
            break
        delta, i, j, k, rev = best
        seg_len = (j - i) % n
        x, a = tour[i], tour[(i + 1) % n]
        b, y = tour[j], tour[(j + 1) % n]
        rot = tour[(i + 1) % n:] + tour[:(i + 1) % n]  # segment now at rot[0:seg_len]
        seg = rot[:seg_len]
        rest = rot[seg_len:]                            # starts at y, ends at x
        if rev:
            seg = seg[::-1]
        u, v = rest[k], rest[k + 1]
        tour = rest[:k + 1] + seg + rest[k + 1:]
        total_delta += delta
        moves += 1
        touched.extend((x, y, u, v, a, b))
    return tour, total_delta, moves, touched


def run_lee(
    instance: TSPInstance,
    initial_tour: list[int],
    max_time_seconds: float = 0.0,
    z_thresh: float = _Z_THRESH,
    min_targets: int = _MIN_TARGETS,
    max_rounds: int = _MAX_ROUNDS,
    base_k: int = _BASE_K,
) -> LineReassignResult:
    """Deterministic longest-edge finishing pass (see module docstring)."""
    n = len(initial_tour)
    instance.validate_tour(initial_tour)
    dist_matrix = instance.distance_matrix
    initial_cost = tour_cost_from_matrix(initial_tour, dist_matrix)

    expand_k = min(n - 1, max(_EXPAND_FLOOR, n // _EXPAND_FRACTION))
    k_base = min(n - 1, base_k)
    base_nn = lro._precompute_city_nearest_neighbors(instance, k_base)
    # LEE hep lam=0 (ceza yok) kosar ve _restricted_repair `aug`'u yalnizca
    # dm[a][b] olarak okur (hicbir penalty metodu cagirmaz), dolayisiyla ham
    # dist_matrix'i dogrudan gecmek AugmentedDistance sarmalayici katmanini
    # (buyuk n'de LazyDistance uzerinde sicak yol) tamamen atlar -- BIT-AYNI.

    start_time = time.perf_counter()
    deadline = (start_time + max_time_seconds * lro._LS_DEADLINE_SAFETY
                if max_time_seconds > 0 else None)

    best_tour = list(initial_tour)
    best_cost = initial_cost
    total_moves = 0
    rounds_run = 0
    expanded_cache: dict[int, list[int]] = {}

    frames: list[TourFrame] = [TourFrame(
        step=0, phase="initial", stage_index=0, elapsed_time=0.0,
        cost=initial_cost, best_cost=initial_cost, constructed_cities=0,
        tour=lro._frame_tour(list(best_tour)))]
    history: list[HistoryPoint] = [
        HistoryPoint(iteration=0, elapsed_time=0.0, cost=initial_cost)]

    def _nb(c: int) -> list[int]:
        if c not in expanded_cache:
            expanded_cache[c] = _expanded_neighbors(c, dist_matrix, expand_k, n)
        return expanded_cache[c]

    max_seg_len = max(2, n // _SEG_MAX_DIV)

    for rnd in range(1, max_rounds + 1):
        if lro._past_deadline(deadline):
            break
        targets = _target_edge_positions(best_tour, dist_matrix, z_thresh, min_targets)
        if not targets:
            break

        seed_cities: list[int] = []
        seen: set[int] = set()
        city_nn = list(base_nn)
        for i in targets:
            for c in (best_tour[i], best_tour[(i + 1) % n]):
                if c not in seen:
                    seen.add(c)
                    seed_cities.append(c)
                city_nn[c] = _nb(c)

        # phase 1: expanded-candidate restricted 2-opt/Or-opt at the outliers
        new_tour, d1, m1 = ngls._restricted_repair(
            best_tour, dist_matrix, dist_matrix, city_nn,
            neighbor_limit_2opt=expand_k, neighbor_limit_oropt=expand_k,
            seed_cities=seed_cities, deadline=deadline,
        )
        # phase 2: global detour relocation between pairs of long edges (the
        # move the restricted repair fundamentally cannot express)
        new_tour, d2, m2, touched = _long_segment_pass(
            new_tour, dist_matrix, z_thresh, min_targets, max_seg_len,
            _nb, deadline)
        # phase 3: polish the fresh junctions the relocation created
        d3 = 0.0
        m3 = 0
        if m2:
            for c in touched:
                city_nn[c] = _nb(c)
            new_tour, d3, m3 = ngls._restricted_repair(
                new_tour, dist_matrix, dist_matrix, city_nn,
                neighbor_limit_2opt=expand_k, neighbor_limit_oropt=expand_k,
                seed_cities=list(dict.fromkeys(touched)), deadline=deadline,
            )
        rounds_run = rnd
        round_delta = d1 + d2 + d3
        round_moves = m1 + m2 + m3
        if round_moves == 0 or round_delta >= -1e-9:
            break
        best_tour = new_tour
        best_cost += round_delta
        total_moves += round_moves
        elapsed = time.perf_counter() - start_time
        frames.append(TourFrame(
            step=len(frames), phase="lee_round", stage_index=rnd,
            elapsed_time=elapsed, cost=best_cost, best_cost=best_cost,
            constructed_cities=round_moves, tour=lro._frame_tour(best_tour)))
        history.append(HistoryPoint(iteration=rnd, elapsed_time=elapsed, cost=best_cost))

    # exact resync (increments are float-safe, but stay exact -- same
    # convention as ngls.run_ngls)
    best_cost = tour_cost_from_matrix(best_tour, dist_matrix)

    elapsed = time.perf_counter() - start_time
    frames.append(TourFrame(
        step=len(frames), phase="final", stage_index=rounds_run,
        elapsed_time=elapsed, cost=best_cost, best_cost=best_cost,
        constructed_cities=0, tour=best_tour[:]))
    history.append(HistoryPoint(iteration=rounds_run + 1, elapsed_time=elapsed, cost=best_cost))

    return LineReassignResult(
        tour=best_tour,
        cost=best_cost,
        iterations_completed=rounds_run,
        improvements_accepted=total_moves,
        total_candidates_evaluated=0,
        total_saving=initial_cost - best_cost,
        frames=frames,
        history=history,
    )


def build_lee_result(
    instance: TSPInstance,
    initial_tour: list[int],
    max_time_seconds: float = 0.0,
    **kwargs,
) -> tuple[TSPRunResult, LineReassignResult]:
    """Dashboard-compatible result for LEE -- same tour/cost contract (via
    record()) as every other benchmarked method."""
    result = run_lee(instance, initial_tour, max_time_seconds=max_time_seconds, **kwargs)
    metrics = RunMetrics(
        initialization_name="Longest-Edge Elimination (LEE)",
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
