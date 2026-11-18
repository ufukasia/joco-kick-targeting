# -*- coding: utf-8 -*-
"""Counterfactual Residual Improvement through Neighborhood Expansion.

The learned model only ranks expansion anchors. Every accepted move is still
validated by the exact TSPLIB tour cost through the existing local-search
operators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
import time

import numpy as np

import lee
import line_reassign_optimizer as lro
from line_reassign_optimizer import LineReassignResult
import ngls
from app import (
    HistoryPoint, RunMetrics, TSPInstance, TSPRunResult, TourFrame,
    snake_path_initialization_result, tour_cost_from_matrix,
)


@dataclass
class BaseState:
    tour: list[int]
    cost: float
    neighbors: list[list[int]]
    projected: list[tuple[float, float]]


@dataclass
class ExpansionResult:
    tour: list[int]
    cost: float
    gain: float
    moves: int
    anchors: list[int]
    expanded_cities: list[int]
    elapsed: float


@dataclass
class CrineResult:
    tour: list[int]
    cost: float
    initial_cost: float
    rounds: int
    expansions: int
    moves: int
    elapsed: float
    history: list[dict] = field(default_factory=list)


def _polish(
    instance: TSPInstance,
    tour: list[int],
    neighbors: list[list[int]],
    projected: list[tuple[float, float]],
    k0: int,
    max_rounds: int,
    deadline: float | None = None,
) -> tuple[list[int], float, int]:
    dm = instance.distance_matrix
    cost = tour_cost_from_matrix(tour, dm)
    out, _, accepted, _ = lro._combined_local_search(
        tour=list(tour), current_cost=cost, dist_matrix=dm,
        projected=projected, city_nearest_neighbors=neighbors,
        neighbor_limit_2opt=k0, neighbor_limit_oropt=k0,
        k_nearest_edges_relocate=k0, city_neighbor_limit_relocate=k0,
        max_ls_rounds=max_rounds, deadline=deadline,
    )
    exact = tour_cost_from_matrix(out, dm)
    if exact > cost + 1e-9:
        raise RuntimeError("base polish worsened the true tour cost")
    return out, exact, accepted


def prepare_base_state(
    instance: TSPInstance,
    initial_tour: list[int],
    k0: int = 20,
    max_rounds: int = 40,
    deadline: float | None = None,
) -> BaseState:
    instance.validate_tour(initial_tour)
    n = len(initial_tour)
    k0 = min(k0, n - 1)
    neighbors = lro._precompute_city_nearest_neighbors(instance, k0)
    projected = lro.compute_projected_coords(instance)
    tour, cost, _ = _polish(
        instance, initial_tour, neighbors, projected, k0, max_rounds, deadline,
    )
    return BaseState(tour=tour, cost=cost, neighbors=neighbors, projected=projected)


def deterministic_start(instance: TSPInstance) -> list[int]:
    return snake_path_initialization_result(instance).tour


def region_cities(tour: list[int], anchor: int) -> list[int]:
    pos = tour.index(anchor)
    return list(dict.fromkeys((tour[pos - 1], anchor, tour[(pos + 1) % len(tour)])))


def expand_and_repair(
    instance: TSPInstance,
    base: BaseState,
    anchors: list[int],
    k1: int = 80,
    polish_rounds: int = 4,
    deadline: float | None = None,
) -> ExpansionResult:
    start = time.perf_counter()
    anchors = list(dict.fromkeys(int(c) for c in anchors))
    if not anchors:
        return ExpansionResult(base.tour[:], base.cost, 0.0, 0, [], [], 0.0)
    n = len(base.tour)
    k1 = min(max(k1, 1), n - 1)
    expanded = list(dict.fromkeys(
        c for anchor in anchors for c in region_cities(base.tour, anchor)
    ))
    neighbors = list(base.neighbors)
    for city in expanded:
        neighbors[city] = lee._expanded_neighbors(
            city, instance.distance_matrix, k1, n,
        )
    tour, _, moves = ngls._restricted_repair(
        base.tour[:], instance.distance_matrix, instance.distance_matrix,
        neighbors, neighbor_limit_2opt=k1, neighbor_limit_oropt=k1,
        seed_cities=expanded, deadline=deadline,
    )
    cost = tour_cost_from_matrix(tour, instance.distance_matrix)
    total_moves = moves
    if cost < base.cost - 1e-9 and polish_rounds:
        tour, cost, accepted = _polish(
            instance, tour, base.neighbors, base.projected,
            len(base.neighbors[0]), polish_rounds, deadline,
        )
        total_moves += accepted
    if cost > base.cost + 1e-9:
        raise RuntimeError("expanded repair worsened the true tour cost")
    if cost >= base.cost - 1e-9:
        tour, cost, total_moves = base.tour[:], base.cost, 0
    instance.validate_tour(tour)
    return ExpansionResult(
        tour=tour, cost=cost, gain=base.cost - cost, moves=total_moves,
        anchors=anchors, expanded_cities=expanded,
        elapsed=time.perf_counter() - start,
    )


def probe_anchor(
    instance: TSPInstance,
    base: BaseState,
    anchor: int,
    k1: int = 80,
) -> ExpansionResult:
    return expand_and_repair(instance, base, [anchor], k1=k1)


def _trained_scores(
    instance: TSPInstance,
    base: BaseState,
    weights: dict[str, np.ndarray],
) -> list[tuple[float, int, int]]:
    import gnn_trained as gt

    xs, ys, is_geo = gt.extract_xy(instance)
    px = np.asarray([p[0] for p in base.projected])
    py = np.asarray([p[1] for p in base.projected])
    knn = gt.knn_table_to_array(base.neighbors, len(base.neighbors[0]))
    snapshot = gt._compute_trained_snapshot(
        weights, base.tour, xs, ys, is_geo, px, py, knn,
    )
    return snapshot[3]


def rank_anchors(
    mode: str,
    instance: TSPInstance,
    base: BaseState,
    rng: random.Random,
    weights: dict[str, np.ndarray] | None = None,
) -> list[int]:
    tour = base.tour
    if mode == "random":
        ranked = tour[:]
        rng.shuffle(ranked)
        return ranked
    if mode == "lee":
        targets = lee._target_edge_positions(
            tour, instance.distance_matrix, z_thresh=2.5,
            min_targets=min(20, len(tour)),
        )
        ranked: list[int] = []
        for pos in targets:
            for city in (tour[pos], tour[(pos + 1) % len(tour)]):
                if city not in ranked:
                    ranked.append(city)
        return ranked
    if mode == "free_gnn":
        snapshot = lro._compute_gnn_snapshot(
            tour, instance.distance_matrix, base.projected, base.neighbors,
            rounds=4, k_geo=min(24, len(base.neighbors[0])),
            k_density=min(16, len(base.neighbors[0])),
        )
        return [city for _, _, city in snapshot[3]]
    if mode in ("lkh_gnn", "crine"):
        if weights is None:
            raise ValueError(f"{mode} requires an explicit weights file")
        return [city for _, _, city in _trained_scores(instance, base, weights)]
    raise ValueError(f"unknown selector: {mode}")


def run_crine_search(
    instance: TSPInstance,
    initial_tour: list[int],
    mode: str = "crine",
    weights: dict[str, np.ndarray] | None = None,
    seed: int = 42,
    k0: int = 20,
    k1: int = 80,
    anchors_per_round: int = 8,
    max_rounds: int = 3,
    oracle_gains: list[dict[int, float]] | None = None,
    max_time_seconds: float = 0.0,
) -> CrineResult:
    start = time.perf_counter()
    deadline = (start + max_time_seconds * lro._LS_DEADLINE_SAFETY
                if max_time_seconds > 0 else None)
    base = prepare_base_state(instance, initial_tour, k0=k0, deadline=deadline)
    initial_cost = base.cost
    rng = random.Random(seed)
    history = [{"round": 0, "cost": base.cost, "elapsed": time.perf_counter() - start}]
    rounds_run = expansions = total_moves = 0

    if mode == "base":
        return CrineResult(base.tour, base.cost, initial_cost, 0, 0, 0,
                           time.perf_counter() - start, history)
    if mode not in ("full", "oracle", "random", "lee", "free_gnn", "lkh_gnn", "crine"):
        raise ValueError(f"unknown search mode: {mode}")

    for rnd in range(max_rounds):
        if max_time_seconds > 0 and time.perf_counter() - start >= max_time_seconds:
            break
        if mode == "full":
            anchors = base.tour[:]
        elif mode == "oracle":
            if oracle_gains is None or rnd >= len(oracle_gains):
                raise ValueError("oracle mode requires measured gains for every round")
            gains = oracle_gains[rnd]
            anchors = sorted(base.tour, key=lambda c: gains.get(c, 0.0), reverse=True)
            anchors = anchors[:anchors_per_round]
        else:
            ranked = rank_anchors(mode, instance, base, rng, weights)
            anchors = ranked[:anchors_per_round]
        outcome = expand_and_repair(instance, base, anchors, k1=k1, deadline=deadline)
        expansions += len(anchors)
        rounds_run = rnd + 1
        total_moves += outcome.moves
        history.append({
            "round": rounds_run, "cost": outcome.cost,
            "gain": outcome.gain, "anchors": outcome.anchors,
            "elapsed": time.perf_counter() - start,
        })
        if outcome.gain <= 1e-9:
            break
        base = BaseState(outcome.tour, outcome.cost, base.neighbors, base.projected)
    return CrineResult(
        tour=base.tour, cost=base.cost, initial_cost=initial_cost,
        rounds=rounds_run, expansions=expansions, moves=total_moves,
        elapsed=time.perf_counter() - start, history=history,
    )


def build_crine_result(
    instance: TSPInstance,
    initial_tour: list[int],
    weights_path: str | Path,
    max_time_seconds: float = 0.0,
    seed: int = 42,
    k0: int = 20,
    k1: int = 80,
    anchors_per_round: int = 8,
    max_rounds: int = 6,
) -> tuple[TSPRunResult, LineReassignResult]:
    import gnn_trained as gt

    weights, _ = gt.get_weights(weights_path)
    result = run_crine_search(
        instance, initial_tour, mode="crine", weights=weights, seed=seed,
        k0=k0, k1=k1, anchors_per_round=anchors_per_round,
        max_rounds=max_rounds, max_time_seconds=max_time_seconds,
    )
    raw_initial = instance.tour_cost(initial_tour)
    history = [HistoryPoint(iteration=0, elapsed_time=0.0, cost=raw_initial)]
    history.extend(HistoryPoint(
        iteration=int(row["round"]) + 1, elapsed_time=float(row["elapsed"]),
        cost=float(row["cost"]),
    ) for row in result.history)
    frames = [
        TourFrame(step=0, phase="initial", stage_index=0, elapsed_time=0.0,
                  cost=raw_initial, best_cost=raw_initial, constructed_cities=0,
                  tour=lro._frame_tour(list(initial_tour))),
        TourFrame(step=1, phase="final", stage_index=result.rounds,
                  elapsed_time=result.elapsed, cost=result.cost,
                  best_cost=result.cost, constructed_cities=result.moves,
                  tour=lro._frame_tour(result.tour)),
    ]
    stop_reason = ("time" if max_time_seconds > 0 and result.elapsed >= max_time_seconds
                   else "converged" if result.rounds < max_rounds else "max_iter")
    diag = LineReassignResult(
        tour=result.tour, cost=result.cost,
        iterations_completed=result.rounds,
        improvements_accepted=result.moves,
        total_candidates_evaluated=result.expansions,
        total_saving=raw_initial - result.cost,
        frames=frames, history=history, stop_reason=stop_reason,
        phase0_converged=True, perturbation_iterations=result.rounds,
    )
    metrics = RunMetrics(
        initialization_name="CRINE-LS (Counterfactual Residual Improvement)",
        initial_cost=raw_initial, final_cost=result.cost,
        cpu_time_seconds=result.elapsed, frame_count=len(frames), history=history,
    )
    return TSPRunResult(
        initial_tour=list(initial_tour), final_tour=result.tour,
        metrics=metrics, animation_frames=frames,
    ), diag
