# -*- coding: utf-8 -*-
"""Controlled diversification strategies on the shared CRINE-ILS skeleton."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import time

import numpy as np

import gnn_trained as gt
import line_reassign_optimizer as lro
from app import HistoryPoint, RunMetrics, TSPInstance, TSPRunResult, TourFrame
from crine_ls import BaseState, expand_and_repair, prepare_base_state, rank_anchors, _polish
from line_reassign_optimizer import LineReassignResult


STRATEGIES = (
    "random", "suspicious", "entropy", "hybrid", "elite_entropy",
    "long_edge", "backbone", "reactive", "bandit", "adaptive_portfolio",
)
FOCUSED_STRATEGIES = (
    "suspicious_reactive", "confidence_gate", "confidence_reactive",
    "geo_fusion", "focused_bandit", "focused_adaptive",
)
TRANSFER_STRATEGIES = (
    "gls_geometry", "gls_gnn", "gls_fusion", "gls_gnn_reactive",
    "suspicious_sideways", "gls_gnn_sideways", "suspicious_rtr", "gls_gnn_rtr",
)
PORTFOLIO_OPERATORS = ("random", "suspicious", "entropy", "long_edge", "backbone")
FOCUSED_PORTFOLIO = ("random", "suspicious", "long_edge")
GLS_STRATEGIES = frozenset(name for name in TRANSFER_STRATEGIES if name.startswith("gls_"))
SIDEWAYS_STRATEGIES = frozenset(("suspicious_sideways", "gls_gnn_sideways"))
RTR_STRATEGIES = frozenset(("suspicious_rtr", "gls_gnn_rtr"))


@dataclass
class AdaptiveILSResult:
    tour: list[int]
    cost: float
    initial_cost: float
    rounds: int
    expansions: int
    moves: int
    elapsed: float
    accepted_kicks: int
    archive_size: int
    operator_counts: dict[str, int]
    history: list[dict] = field(default_factory=list)


@dataclass
class UCB1:
    operators: tuple[str, ...] = PORTFOLIO_OPERATORS
    exploration: float = math.sqrt(2.0)
    counts: np.ndarray = field(init=False)
    rewards: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.counts = np.zeros(len(self.operators), dtype=np.int64)
        self.rewards = np.zeros(len(self.operators), dtype=np.float64)

    def select(self, rng: random.Random) -> str:
        untried = np.flatnonzero(self.counts == 0)
        if len(untried):
            index = int(untried[rng.randrange(len(untried))])
        else:
            means = self.rewards / self.counts
            bonuses = self.exploration * np.sqrt(
                math.log(float(self.counts.sum())) / self.counts)
            scores = means + bonuses
            best = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0.0, atol=1e-15))
            index = int(best[rng.randrange(len(best))])
        return self.operators[index]

    def update(self, operator: str, reward: float) -> None:
        index = self.operators.index(operator)
        self.counts[index] += 1
        self.rewards[index] += min(1.0, max(0.0, float(reward)))


def elite_frequency_edge_quality(
    tour: list[int], archive: list[tuple[float, list[int]]],
) -> list[tuple[float, int]]:
    if len(archive) < 2:
        return []
    edge_sets = [gt._tour_edge_set(row[1]) for row in archive]
    ranked = []
    for pos, (a, b) in enumerate(zip(tour, tour[1:] + tour[:1])):
        edge = (min(a, b), max(a, b))
        frequency = sum(edge in edges for edges in edge_sets) / len(edge_sets)
        ranked.append((frequency, pos))
    return sorted(ranked)


def long_edge_quality(
    tour: list[int], xs: np.ndarray, ys: np.ndarray, is_geo: bool,
) -> list[tuple[float, int]]:
    cities = np.asarray(tour, dtype=np.int64)
    lengths = gt.pair_dist(xs, ys, is_geo, cities, np.roll(cities, -1))
    return sorted((-float(length), pos) for pos, length in enumerate(lengths))


def gls_edge_quality(
    tour: list[int], xs: np.ndarray, ys: np.ndarray, is_geo: bool,
    model_edge_quality: list[tuple[float, int]] | None,
    penalties: dict[tuple[int, int], int], mode: str,
) -> list[tuple[float, int]]:
    cities = np.asarray(tour, dtype=np.int64)
    lengths = gt.pair_dist(xs, ys, is_geo, cities, np.roll(cities, -1))
    normalized_lengths = lengths / max(float(np.mean(lengths)), 1e-12)
    suspicion = np.ones(len(tour), dtype=np.float64)
    if model_edge_quality is not None:
        for probability, pos in model_edge_quality:
            suspicion[pos] = max(1e-9, 1.0 - probability)
    if mode == "geometry":
        base_score = normalized_lengths
    elif mode == "gnn":
        base_score = suspicion
    elif mode == "fusion":
        base_score = np.sqrt(normalized_lengths * suspicion)
    else:
        raise ValueError(f"unknown GLS score: {mode}")
    edges = [(min(a, b), max(a, b))
             for a, b in zip(tour, tour[1:] + tour[:1])]
    counts = np.asarray([penalties.get(edge, 0) for edge in edges], dtype=np.float64)
    utilities = base_score / (1.0 + counts)
    maximum = float(np.max(utilities))
    for edge, utility in zip(edges, utilities):
        if math.isclose(float(utility), maximum, rel_tol=0.0, abs_tol=1e-15):
            penalties[edge] = penalties.get(edge, 0) + 1
    updated = np.asarray([penalties.get(edge, 0) for edge in edges], dtype=np.float64)
    return sorted((-float(score), pos)
                  for pos, score in enumerate(base_score / (1.0 + updated)))


def stagnation_strength(stagnation: int, interval: int = 8, maximum: int = 4) -> int:
    return min(maximum, 1 + max(0, stagnation) // interval)


def mean_model_entropy(edge_quality: list[tuple[float, int]]) -> float:
    if not edge_quality:
        return 0.0
    return float(np.mean([-score for score, _ in gt.entropy_edge_quality(edge_quality)]))


def fuse_edge_ranks(
    primary: list[tuple[float, int]], secondary: list[tuple[float, int]],
    secondary_weight: float = 0.25,
) -> list[tuple[float, int]]:
    first = sorted(primary)
    second = sorted(secondary)
    first_rank = {pos: rank for rank, (_, pos) in enumerate(first)}
    second_rank = {pos: rank for rank, (_, pos) in enumerate(second)}
    return sorted(
        ((1.0 - secondary_weight) * first_rank[pos]
         + secondary_weight * second_rank[pos], pos)
        for _, pos in first
    )


def _edge_quality(
    operator: str,
    base: BaseState,
    kick_weights: dict[str, np.ndarray],
    archive: list[tuple[float, list[int]]],
    xs: np.ndarray,
    ys: np.ndarray,
    is_geo: bool,
    px: np.ndarray,
    py: np.ndarray,
    knn: np.ndarray,
    snapshot=None,
) -> list[tuple[float, int]]:
    if operator == "random":
        return []
    if operator == "long_edge":
        return long_edge_quality(base.tour, xs, ys, is_geo)
    if operator == "elite_entropy":
        return gt.elite_entropy_edge_quality(base.tour, archive)
    if operator == "backbone":
        return elite_frequency_edge_quality(base.tour, archive)
    if snapshot is None:
        snapshot = gt._compute_trained_snapshot(
            kick_weights, base.tour, xs, ys, is_geo, px, py, knn)
    if operator in ("suspicious", "suspicious_sideways", "suspicious_rtr"):
        return snapshot[4]
    if operator == "entropy":
        return gt.entropy_edge_quality(snapshot[4])
    if operator == "hybrid":
        return gt.fused_entropy_edge_quality(snapshot[4], entropy_weight=0.5)
    if operator == "geo_fusion":
        return fuse_edge_ranks(
            snapshot[4], long_edge_quality(base.tour, xs, ys, is_geo))
    raise ValueError(f"unknown kick operator: {operator}")


def run_adaptive_ils_search(
    instance: TSPInstance,
    initial_tour: list[int],
    crine_weights: dict[str, np.ndarray],
    kick_weights: dict[str, np.ndarray] | None = None,
    strategy: str = "random",
    seed: int = 42,
    k0: int = 20,
    k1: int = 80,
    anchors_per_round: int = 8,
    max_kicks: int = 100000,
    kick_polish_rounds: int = 6,
    max_time_seconds: float = 0.0,
    elite_limit: int = 8,
) -> AdaptiveILSResult:
    if strategy not in STRATEGIES + FOCUSED_STRATEGIES + TRANSFER_STRATEGIES:
        raise ValueError(f"unknown adaptive ILS strategy: {strategy}")
    if strategy in ("suspicious", "entropy", "hybrid", "bandit", "adaptive_portfolio") + FOCUSED_STRATEGIES + TRANSFER_STRATEGIES and kick_weights is None:
        raise ValueError(f"{strategy} requires LKH-GNN weights")
    start = time.perf_counter()
    deadline = (start + max_time_seconds * lro._LS_DEADLINE_SAFETY
                if max_time_seconds > 0 else None)
    base = prepare_base_state(instance, initial_tour, k0=k0, deadline=deadline)
    initial_cost = base.cost
    best_tour = base.tour[:]
    best_cost = base.cost
    rng = random.Random(seed)
    xs, ys, is_geo = gt.extract_xy(instance)
    px = np.asarray([p[0] for p in base.projected])
    py = np.asarray([p[1] for p in base.projected])
    knn = gt.knn_table_to_array(base.neighbors, len(base.neighbors[0]))
    archive = gt.update_elite_archive([], base.tour, base.cost, limit=elite_limit)
    if strategy in ("bandit", "adaptive_portfolio"):
        bandit = UCB1()
    elif strategy in ("focused_bandit", "focused_adaptive"):
        bandit = UCB1(operators=FOCUSED_PORTFOLIO)
    else:
        bandit = None
    cached_tour = None
    cached_snapshot = None
    penalties: dict[tuple[int, int], int] = {}
    expansion_pending = True
    stagnation = accepted_kicks = expansions = total_moves = 0
    operator_counts = {name: 0 for name in STRATEGIES + FOCUSED_STRATEGIES + TRANSFER_STRATEGIES}
    history = [{"round": 0, "kind": "base", "cost": base.cost,
                "elapsed": time.perf_counter() - start}]

    while history[-1]["round"] < max_kicks:
        if max_time_seconds > 0 and time.perf_counter() - start >= max_time_seconds:
            break
        if expansion_pending:
            ranked = rank_anchors("crine", instance, base, rng, crine_weights)
            anchors = ranked[:anchors_per_round]
            outcome = expand_and_repair(
                instance, base, anchors, k1=k1, deadline=deadline)
            expansions += len(anchors)
            total_moves += outcome.moves
            if outcome.gain > 1e-9:
                base = BaseState(outcome.tour, outcome.cost, base.neighbors, base.projected)
                if base.cost < best_cost - 1e-9:
                    best_tour = base.tour[:]
                    best_cost = base.cost
                archive = gt.update_elite_archive(
                    archive, base.tour, base.cost, limit=elite_limit)
            else:
                expansion_pending = False
            history.append({
                "round": history[-1]["round"], "kind": "expansion",
                "cost": base.cost, "candidate_cost": outcome.cost,
                "gain": outcome.gain, "elapsed": time.perf_counter() - start,
            })
            continue

        round_index = max(1, history[-1]["round"] + 1)
        requested_operator = (bandit.select(rng) if bandit is not None
                              else "random" if strategy == "reactive"
                              else "suspicious" if strategy == "suspicious_reactive"
                              else strategy)
        strength = (stagnation_strength(stagnation)
                    if strategy in ("reactive", "adaptive_portfolio", "suspicious_reactive",
                                    "confidence_reactive", "focused_adaptive",
                                    "gls_gnn_reactive") else 1)
        if requested_operator in ("suspicious", "entropy", "hybrid", "confidence_gate",
                                  "confidence_reactive", "geo_fusion", "gls_gnn",
                                  "gls_fusion", "gls_gnn_reactive", "suspicious_sideways",
                                  "gls_gnn_sideways", "suspicious_rtr", "gls_gnn_rtr"):
            if cached_tour != base.tour:
                cached_snapshot = gt._compute_trained_snapshot(
                    kick_weights, base.tour, xs, ys, is_geo, px, py, knn)
                cached_tour = base.tour[:]
        model_entropy = None
        operator = requested_operator
        if requested_operator in ("confidence_gate", "confidence_reactive"):
            model_entropy = mean_model_entropy(cached_snapshot[4])
            operator = "random" if model_entropy > 0.5 else "suspicious"
        if operator in GLS_STRATEGIES:
            mode = ("geometry" if operator == "gls_geometry"
                    else "gnn" if operator in ("gls_gnn", "gls_gnn_reactive",
                                               "gls_gnn_sideways", "gls_gnn_rtr")
                    else "fusion")
            quality = gls_edge_quality(
                base.tour, xs, ys, is_geo,
                None if mode == "geometry" else cached_snapshot[4], penalties, mode)
        else:
            quality = _edge_quality(
                operator, base, kick_weights, archive,
                xs, ys, is_geo, px, py, knn, snapshot=cached_snapshot)
        kicked = base.tour[:]
        for _ in range(strength):
            kicked = (lro._double_bridge_perturbation(kicked, rng)
                      if not quality else gt._guided_double_bridge(kicked, quality, rng))
        tour, cost, accepted = _polish(
            instance, kicked, base.neighbors, base.projected, k0,
            kick_polish_rounds, deadline)
        total_moves += accepted
        gain = base.cost - cost
        reward = min(1.0, max(0.0, 100.0 * gain / max(abs(base.cost), 1e-9)))
        if bandit is not None:
            bandit.update(operator, reward)
        operator_counts[operator] += 1
        archive = gt.update_elite_archive(archive, tour, cost, limit=elite_limit)
        if cost < best_cost - 1e-9:
            best_tour = tour[:]
            best_cost = cost
        acceptance_threshold = (
            min(0.02, 0.002 * (1 + stagnation // 8)) if strategy in RTR_STRATEGIES else 0.0)
        accept_candidate = (
            gain > 1e-9
            or (strategy in SIDEWAYS_STRATEGIES and abs(gain) <= 1e-9 and tour != base.tour)
            or (strategy in RTR_STRATEGIES
                and cost <= best_cost + acceptance_threshold * max(abs(best_cost), 1e-9)))
        if accept_candidate:
            base = BaseState(tour, cost, base.neighbors, base.projected)
            expansion_pending = True
            accepted_kicks += 1
            stagnation = 0
        else:
            stagnation += 1
        history.append({
            "round": round_index, "kind": "kick", "operator": operator,
            "requested_operator": requested_operator, "model_entropy": model_entropy,
            "strength": strength, "cost": base.cost, "candidate_cost": cost,
            "best_cost": best_cost, "accepted": accept_candidate,
            "acceptance_threshold": acceptance_threshold,
            "gain": gain, "reward": reward, "stagnation": stagnation,
            "elapsed": time.perf_counter() - start,
        })

    kicks = sum(operator_counts.values())
    return AdaptiveILSResult(
        tour=best_tour, cost=best_cost, initial_cost=initial_cost,
        rounds=kicks, expansions=expansions, moves=total_moves,
        elapsed=time.perf_counter() - start, accepted_kicks=accepted_kicks,
        archive_size=len(archive), operator_counts=operator_counts,
        history=history,
    )


def build_adaptive_ils_result(
    instance: TSPInstance,
    initial_tour: list[int],
    crine_weights_path: str,
    kick_weights_path: str | None,
    strategy: str = "suspicious",
    max_time_seconds: float = 0.0,
    seed: int = 42,
) -> tuple[TSPRunResult, LineReassignResult]:
    crine_weights = gt.get_weights(crine_weights_path)[0]
    kick_weights = (gt.get_weights(kick_weights_path)[0]
                    if kick_weights_path is not None else None)
    result = run_adaptive_ils_search(
        instance, initial_tour, crine_weights, kick_weights,
        strategy=strategy, seed=seed, max_time_seconds=max_time_seconds,
    )
    raw_initial = instance.tour_cost(initial_tour)
    history = [HistoryPoint(iteration=0, elapsed_time=0.0, cost=raw_initial)]
    history.extend(HistoryPoint(
        iteration=index + 1,
        elapsed_time=float(row["elapsed"]),
        cost=float(row["cost"]),
    ) for index, row in enumerate(result.history))
    if abs(history[-1].cost - result.cost) > 1e-9:
        history.append(HistoryPoint(
            iteration=len(history), elapsed_time=result.elapsed, cost=result.cost))
    frames = [
        TourFrame(
            step=0, phase="initial", stage_index=0, elapsed_time=0.0,
            cost=raw_initial, best_cost=raw_initial, constructed_cities=0,
            tour=lro._frame_tour(list(initial_tour)),
        ),
        TourFrame(
            step=1, phase="final", stage_index=result.rounds,
            elapsed_time=result.elapsed, cost=result.cost, best_cost=result.cost,
            constructed_cities=result.moves, tour=lro._frame_tour(result.tour),
        ),
    ]
    diag = LineReassignResult(
        tour=result.tour, cost=result.cost,
        iterations_completed=result.rounds,
        improvements_accepted=result.moves,
        total_candidates_evaluated=result.expansions,
        total_saving=raw_initial - result.cost,
        frames=frames, history=history,
        stop_reason=("time" if max_time_seconds > 0 and result.elapsed >= max_time_seconds
                     else "max_iter"),
        phase0_converged=True, perturbation_iterations=result.rounds,
    )
    metrics = RunMetrics(
        initialization_name=f"Adaptive CRINE-ILS ({strategy})",
        initial_cost=raw_initial, final_cost=result.cost,
        cpu_time_seconds=result.elapsed, frame_count=len(frames), history=history,
    )
    return TSPRunResult(
        initial_tour=list(initial_tour), final_tour=result.tour,
        metrics=metrics, animation_frames=frames,
    ), diag
