# -*- coding: utf-8 -*-
"""Trained GNN-Guided Local Search for TSP tours.

The learned counterpart of the training-free GNN-LS in
`line_reassign_optimizer.py`. v2 skeleton ("learned-kick ILS"): each iteration
first attempts guided intensification (one trained relocate + one trained
2-opt, both gated by the TRUE cost delta, plus a short classical polish);
when intensification is exhausted the iteration performs a double-bridge /
segment-reverse kick followed by a deep local-search descent. Cut points can
target predicted non-backbone edges (default), high-predictive-entropy edges,
or uniform random positions. Differences vs the training-free `gnn`: (1)
learned move scores, (2) the learned kick schedule.

Model: a small GraphSAGE-style message-passing network implemented in pure
numpy (no deep-learning framework needed at inference OR training time):

    h0   = relu(X @ W_in + b_in)                             X: 10 features/node
    h_l  = relu(h W_self + tour_mean(h) W_tour + geo_mean(h) W_geo + b)   x3
    node head: sigmoid(h w_node + b)   -> P(city is misplaced)
    edge head: sigmoid(MLP([h_i*h_j, |h_i-h_j|, edge_feats])) -> P(edge is
               a "backbone" edge, i.e. present in robust local minima)

Training (see tsplib/gnn_train.py) fits the two heads to consensus ("backbone")
edge frequencies computed from many independent strong ILS local optima on
synthetic instances; TSPLIB is never used for training (zero-shot evaluation).

The graph tensors are indexed by TOUR POSITION (like the training-free GNN):
tour neighbours are positions i-1 / i+1, geometric neighbours are the kNN
cities mapped to their current positions.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

import line_reassign_optimizer as lro
from app import (
    HistoryPoint,
    RunMetrics,
    TSPInstance,
    TSPRunResult,
    TourFrame,
    tour_cost_from_matrix,
)
from line_reassign_optimizer import LineReassignResult, PointSensitivity

# ---------------------------------------------------------------------------
#  Model configuration (must match between training and inference; stored in
#  the weights file and validated at load time)
# ---------------------------------------------------------------------------
FEATURE_DIM = 10          # per-node input features
EDGE_FEATURE_DIM = 2      # per-tour-edge input features
HIDDEN_DIM = 16
MP_LAYERS = 3
K_GEO = 24                # geometric neighbours in message passing
K_DENSITY = 16            # neighbours for density features

MODEL_CONFIG = {
    "feature_dim": FEATURE_DIM,
    "edge_feature_dim": EDGE_FEATURE_DIM,
    "hidden_dim": HIDDEN_DIM,
    "mp_layers": MP_LAYERS,
    "k_geo": K_GEO,
    "k_density": K_DENSITY,
}

# Search-skeleton defaults mirrored from the training-free GNN-LS.
DEFAULT_TOP_MISPLACED = 400
DEFAULT_K_INSERT_CANDIDATES = 48
DEFAULT_TOP_EDGES_2OPT = 320


# ---------------------------------------------------------------------------
#  Vectorised TSPLIB-exact distances
# ---------------------------------------------------------------------------
_RRR = 6378.388  # TSPLIB earth radius (km)


def pair_dist(xs: np.ndarray, ys: np.ndarray, is_geo: bool,
              i_idx: np.ndarray, j_idx: np.ndarray) -> np.ndarray:
    """TSPLIB-exact distances for index arrays (broadcasting allowed).
    EUC_2D: nint rounding. GEO: the TSPLIB great-circle formula."""
    if not is_geo:
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        return np.floor(np.hypot(dx, dy) + 0.5)
    la1 = np.radians(xs[i_idx]); lo1 = np.radians(ys[i_idx])
    la2 = np.radians(xs[j_idx]); lo2 = np.radians(ys[j_idx])
    q1 = np.cos(lo1 - lo2)
    q2 = np.cos(la1 - la2)
    q3 = np.cos(la1 + la2)
    val = np.clip(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3), -1.0, 1.0)
    d = np.floor(_RRR * np.arccos(val) + 1.0)
    # i == j must give 0 (arccos rounding gives 1.0 otherwise)
    same = (i_idx == j_idx)
    if np.any(same):
        d = np.where(same, 0.0, d)
    return d


def extract_xy(instance: TSPInstance) -> tuple[np.ndarray, np.ndarray, bool]:
    """Pulls the TRUE-distance coordinates out of any of the benchmark's
    instance flavours (dense `_RoundedInstance`, sparse `_SparseInstance`,
    plain `app.TSPInstance`)."""
    dm = instance.distance_matrix
    if hasattr(dm, "xs"):  # tsplib_engine.LazyDistance (sparse instance)
        return (np.asarray(dm.xs, dtype=np.float64),
                np.asarray(dm.ys, dtype=np.float64), bool(dm.is_geo))
    if hasattr(instance, "_dx"):  # tsplib_engine._RoundedInstance
        return (np.asarray(instance._dx, dtype=np.float64),
                np.asarray(instance._dy, dtype=np.float64),
                bool(getattr(instance, "_is_geo", False)))
    xs = np.asarray([c.x for c in instance.cities], dtype=np.float64)
    ys = np.asarray([c.y for c in instance.cities], dtype=np.float64)
    return xs, ys, False


def knn_table_to_array(knn: list[list[int]], k: int) -> np.ndarray:
    """list-of-lists kNN table -> (n, k) int array; short rows are padded by
    repeating their last entry (their weight is tiny after normalisation)."""
    n = len(knn)
    out = np.empty((n, k), dtype=np.int64)
    for i, row in enumerate(knn):
        m = min(len(row), k)
        if m == 0:  # degenerate; point at itself
            out[i, :] = i
            continue
        out[i, :m] = row[:m]
        if m < k:
            out[i, m:] = row[m - 1]
    return out


# ---------------------------------------------------------------------------
#  Graph tensors: node/edge features + message-passing structure
# ---------------------------------------------------------------------------
def _zscore_cols(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    std = X.std(axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1])
    std = np.where(std > 1e-9, std, 1.0)
    return (X - mu) / std


def compute_graph_tensors(
    tour: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    is_geo: bool,
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    knn: np.ndarray,
    k_geo: int = K_GEO,
    k_density: int = K_DENSITY,
) -> dict:
    """Builds all model inputs for one tour state (indexed by tour position).

    Returns dict with:
      X         (n, FEATURE_DIM)      z-scored node features
      edge_feats(n, EDGE_FEATURE_DIM) features of tour edge (pos i -> i+1)
      nbr_pos   (n, k_geo)            tour positions of geometric neighbours
      geo_w     (n, k_geo)            row-normalised inverse-distance weights
      cost_next (n,)                  raw length of tour edge (i -> i+1)
    """
    n = len(tour)
    k_geo = min(k_geo, knn.shape[1], n - 1)
    k_density = min(k_density, knn.shape[1], n - 1)

    cities = tour
    prev_c = np.roll(cities, 1)
    next_c = np.roll(cities, -1)

    cost_prev = pair_dist(xs, ys, is_geo, prev_c, cities)
    cost_next = pair_dist(xs, ys, is_geo, cities, next_c)
    shortcut = pair_dist(xs, ys, is_geo, prev_c, next_c)
    removal_gain = cost_prev + cost_next - shortcut

    k_used = max(k_geo, k_density)
    knn_c = knn[cities][:, :k_used]                       # (n, k_used) city ids
    d_knn = pair_dist(xs, ys, is_geo, cities[:, None], knn_c)

    min_nn = np.maximum(d_knn[:, 0], 1e-9)
    avg_nn = d_knn[:, :k_density].mean(axis=1)
    tour_ratio = 0.5 * (cost_prev + cost_next) / min_nn

    # geometry (projected frame): turn angle + deviation from prev-next segment
    px = proj_x[cities]; py = proj_y[cities]
    ax = proj_x[prev_c]; ay = proj_y[prev_c]
    bx = proj_x[next_c]; by = proj_y[next_c]
    vpx = ax - px; vpy = ay - py
    vnx = bx - px; vny = by - py
    mp = np.hypot(vpx, vpy); mn = np.hypot(vnx, vny)
    cos_angle = (vpx * vnx + vpy * vny) / np.maximum(mp * mn, 1e-9)
    # point-to-segment distance (vectorised copy of _point_to_segment_distance)
    dx = bx - ax; dy = by - ay
    seg_len_sq = dx * dx + dy * dy
    t = np.clip(((px - ax) * dx + (py - ay) * dy) / np.maximum(seg_len_sq, 1e-18),
                0.0, 1.0)
    seg_dev = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
    seg_dev = np.where(seg_len_sq < 1e-18, np.hypot(px - ax, py - ay), seg_dev)

    # kNN rank of the current tour neighbours (absent -> 1.25 = "worse than
    # the whole list"); low rank means the tour edge already links near cities.
    def _rank_of(target: np.ndarray) -> np.ndarray:
        match = knn_c[:, :k_geo] == target[:, None]
        found = match.any(axis=1)
        idx = match.argmax(axis=1)
        return np.where(found, (idx + 1.0) / max(k_geo, 1), 1.25)

    rank_prev = _rank_of(prev_c)
    rank_next = _rank_of(next_c)

    X = np.stack([cost_prev, cost_next, removal_gain, min_nn, avg_nn,
                  tour_ratio, cos_angle, seg_dev, rank_prev, rank_next], axis=1)
    X = _zscore_cols(X)

    mu_e = cost_next.mean()
    std_e = cost_next.std(ddof=1) if n > 1 else 1.0
    std_e = std_e if std_e > 1e-9 else 1.0
    edge_feats = np.stack([(cost_next - mu_e) / std_e, rank_next], axis=1)

    pos_of = np.empty(int(cities.max()) + 1, dtype=np.int64)
    pos_of[cities] = np.arange(n)
    nbr_geo = knn_c[:, :k_geo]
    nbr_pos = pos_of[nbr_geo]
    w = 1.0 / np.maximum(d_knn[:, :k_geo], 1e-9)
    geo_w = w / w.sum(axis=1, keepdims=True)

    return {
        "X": X, "edge_feats": edge_feats,
        "nbr_pos": nbr_pos, "geo_w": geo_w,
        "cost_next": cost_next,
    }


# ---------------------------------------------------------------------------
#  Weights: init / save / load
# ---------------------------------------------------------------------------
def init_weights(seed: int = 0) -> dict[str, np.ndarray]:
    """He-initialised random weights (also usable for smoke tests)."""
    rng = np.random.default_rng(seed)
    H, F, EF = HIDDEN_DIM, FEATURE_DIM, EDGE_FEATURE_DIM

    def he(shape, fan_in):
        return rng.normal(0.0, math.sqrt(2.0 / fan_in), size=shape)

    w: dict[str, np.ndarray] = {
        "W_in": he((F, H), F), "b_in": np.zeros(H),
        "w_node": he((H,), H), "b_node": np.zeros(1),
        "W_e1": he((2 * H + EF, H), 2 * H + EF), "b_e1": np.zeros(H),
        "w_edge": he((H,), H), "b_edge": np.zeros(1),
    }
    for l in range(MP_LAYERS):
        w[f"Ws{l}"] = he((H, H), 3 * H)
        w[f"Wt{l}"] = he((H, H), 3 * H)
        w[f"Wg{l}"] = he((H, H), 3 * H)
        w[f"b{l}"] = np.zeros(H)
    return w


def save_weights(path: str | Path, weights: dict[str, np.ndarray],
                 meta: Optional[dict] = None) -> None:
    payload = dict(weights)
    payload["__config__"] = np.frombuffer(
        json.dumps(MODEL_CONFIG).encode("utf-8"), dtype=np.uint8)
    if meta is not None:
        payload["__meta__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_weights(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    """Returns (weights, meta). Raises if the file's architecture config does
    not match this module's constants."""
    with np.load(path) as z:
        data = {k: z[k].copy() for k in z.files}
    cfg_raw = data.pop("__config__", None)
    meta_raw = data.pop("__meta__", None)
    if cfg_raw is not None:
        cfg = json.loads(bytes(cfg_raw.tobytes()).decode("utf-8"))
        if cfg != MODEL_CONFIG:
            raise ValueError(
                f"gnn_trained: weights config {cfg} does not match module "
                f"config {MODEL_CONFIG}; retrain or use matching code version")
    meta = (json.loads(bytes(meta_raw.tobytes()).decode("utf-8"))
            if meta_raw is not None else {})
    return data, meta


_WEIGHTS_CACHE: dict[str, tuple[dict[str, np.ndarray], dict]] = {}


def get_weights(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    key = str(Path(path).resolve())
    if key not in _WEIGHTS_CACHE:
        _WEIGHTS_CACHE[key] = load_weights(key)
    return _WEIGHTS_CACHE[key]


# ---------------------------------------------------------------------------
#  Forward pass (and training-time backward pass)
# ---------------------------------------------------------------------------
def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # numerically stable


def model_forward(w: dict[str, np.ndarray], g: dict, want_cache: bool = False):
    """Runs the model on one graph. Returns (H_final, node_logit, edge_logit
    [, cache])."""
    X = g["X"]; edge_feats = g["edge_feats"]
    nbr_pos = g["nbr_pos"]; geo_w = g["geo_w"]

    pre0 = X @ w["W_in"] + w["b_in"]
    H = _relu(pre0)
    layers = []
    for l in range(MP_LAYERS):
        T = 0.5 * (np.roll(H, 1, axis=0) + np.roll(H, -1, axis=0))
        G = np.einsum("nkh,nk->nh", H[nbr_pos], geo_w)
        pre = H @ w[f"Ws{l}"] + T @ w[f"Wt{l}"] + G @ w[f"Wg{l}"] + w[f"b{l}"]
        if want_cache:
            layers.append((H, T, G, pre))
        H = _relu(pre)

    node_logit = H @ w["w_node"] + w["b_node"][0]
    Hj = np.roll(H, -1, axis=0)
    E_in = np.concatenate([H * Hj, np.abs(H - Hj), edge_feats], axis=1)
    E_pre = E_in @ w["W_e1"] + w["b_e1"]
    E_h = _relu(E_pre)
    edge_logit = E_h @ w["w_edge"] + w["b_edge"][0]

    if not want_cache:
        return H, node_logit, edge_logit
    cache = {"pre0": pre0, "layers": layers, "H_final": H, "Hj": Hj,
             "E_in": E_in, "E_pre": E_pre, "E_h": E_h}
    return H, node_logit, edge_logit, cache


def _bce_with_logits(z: np.ndarray, y: np.ndarray) -> float:
    """Numerically stable mean binary cross-entropy with logits."""
    return float(np.mean(np.maximum(z, 0.0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def model_loss_and_grads(
    w: dict[str, np.ndarray],
    g: dict,
    y_node: np.ndarray,
    y_edge: np.ndarray,
    node_weight: float = 1.0,
    edge_weight: float = 1.0,
) -> tuple[float, dict[str, np.ndarray]]:
    """Full analytic backward pass for one graph (used only in training)."""
    H_final, node_logit, edge_logit, cache = model_forward(w, g, want_cache=True)
    n = g["X"].shape[0]
    nbr_pos = g["nbr_pos"]; geo_w = g["geo_w"]

    loss = (node_weight * _bce_with_logits(node_logit, y_node)
            + edge_weight * _bce_with_logits(edge_logit, y_edge))

    grads = {k: np.zeros_like(v) for k, v in w.items()}
    dz_node = node_weight * (_sigmoid(node_logit) - y_node) / n
    dz_edge = edge_weight * (_sigmoid(edge_logit) - y_edge) / n

    H = cache["H_final"]; Hj = cache["Hj"]
    E_in = cache["E_in"]; E_pre = cache["E_pre"]; E_h = cache["E_h"]
    Hd = HIDDEN_DIM

    # edge head
    grads["w_edge"] = E_h.T @ dz_edge
    grads["b_edge"] = np.array([dz_edge.sum()])
    dE_pre = np.outer(dz_edge, w["w_edge"]) * (E_pre > 0)
    grads["W_e1"] = E_in.T @ dE_pre
    grads["b_e1"] = dE_pre.sum(axis=0)
    dE_in = dE_pre @ w["W_e1"].T
    dProd = dE_in[:, :Hd]
    dAbs = dE_in[:, Hd:2 * Hd]
    sign = np.sign(H - Hj)
    dH = dProd * Hj + dAbs * sign
    dHj = dProd * H - dAbs * sign
    dH += np.roll(dHj, 1, axis=0)   # Hj[i] = H[i+1]

    # node head
    grads["w_node"] = H.T @ dz_node
    grads["b_node"] = np.array([dz_node.sum()])
    dH += np.outer(dz_node, w["w_node"])

    # message-passing layers (reverse)
    for l in range(MP_LAYERS - 1, -1, -1):
        H_in, T, G, pre = cache["layers"][l]
        dPre = dH * (pre > 0)
        grads[f"Ws{l}"] = H_in.T @ dPre
        grads[f"Wt{l}"] = T.T @ dPre
        grads[f"Wg{l}"] = G.T @ dPre
        grads[f"b{l}"] = dPre.sum(axis=0)
        dH_in = dPre @ w[f"Ws{l}"].T
        dT = dPre @ w[f"Wt{l}"].T
        dH_in += 0.5 * (np.roll(dT, -1, axis=0) + np.roll(dT, 1, axis=0))
        dG = dPre @ w[f"Wg{l}"].T
        np.add.at(dH_in, nbr_pos, geo_w[:, :, None] * dG[:, None, :])
        dH = dH_in

    # encoder
    dPre0 = dH * (cache["pre0"] > 0)
    grads["W_in"] = g["X"].T @ dPre0
    grads["b_in"] = dPre0.sum(axis=0)
    return loss, grads


def gradient_check(seed: int = 3, n: int = 24, eps: float = 1e-5) -> float:
    """Compares analytic vs numerical gradients on a tiny random graph.
    Returns the worst relative error (should be < 1e-4)."""
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1000, n); ys = rng.uniform(0, 1000, n)
    tour = np.arange(n); rng.shuffle(tour)
    d2 = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    np.fill_diagonal(d2, np.inf)
    knn = np.argsort(d2, axis=1)[:, :max(K_GEO, K_DENSITY)]
    g = compute_graph_tensors(tour, xs, ys, False, xs, ys, knn)
    y_node = rng.uniform(0, 1, n)
    y_edge = rng.uniform(0, 1, n)
    w = init_weights(seed)

    _, grads = model_loss_and_grads(w, g, y_node, y_edge)
    worst = 0.0
    for key, W in w.items():
        flat = W.reshape(-1)
        gflat = grads[key].reshape(-1)
        idxs = rng.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in idxs:
            old = flat[i]
            flat[i] = old + eps
            lp, _ = model_loss_and_grads(w, g, y_node, y_edge)
            flat[i] = old - eps
            lm, _ = model_loss_and_grads(w, g, y_node, y_edge)
            flat[i] = old
            num = (lp - lm) / (2 * eps)
            denom = max(abs(num), abs(gflat[i]), 1e-8)
            worst = max(worst, abs(num - gflat[i]) / denom)
    return worst


# ---------------------------------------------------------------------------
#  Trained snapshot: scores for the guided passes
# ---------------------------------------------------------------------------
def _compute_trained_snapshot(
    weights: dict[str, np.ndarray],
    tour: list[int],
    xs: np.ndarray,
    ys: np.ndarray,
    is_geo: bool,
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    knn: np.ndarray,
):
    """One consistent model evaluation of the current tour. Returns
    (city_to_pos, H_pos, city_embeddings, misplacement_ranking, edge_quality)
    with EXACTLY the same shapes/semantics the training-free snapshot feeds
    into the guided passes:
      misplacement_ranking: [(score, pos, city)] sorted desc (high = misplaced)
      edge_quality:         [(score, pos)] sorted asc (low = suspicious edge)
    """
    tour_arr = np.asarray(tour, dtype=np.int64)
    g = compute_graph_tensors(tour_arr, xs, ys, is_geo, proj_x, proj_y, knn)
    H, node_logit, edge_logit = model_forward(weights, g)

    node_prob = _sigmoid(node_logit)          # P(misplaced)
    edge_good = _sigmoid(edge_logit)          # P(backbone edge)

    n = len(tour)
    misplacement = sorted(
        ((float(node_prob[i]), i, int(tour_arr[i])) for i in range(n)),
        reverse=True,
    )
    edge_quality = sorted((float(edge_good[i]), i) for i in range(n))

    norms = np.linalg.norm(H, axis=1)
    Hn = H / np.maximum(norms, 1e-12)[:, None]
    max_city = int(tour_arr.max())
    city_emb = np.zeros((max_city + 1, H.shape[1]))
    city_emb[tour_arr] = Hn                    # unit-norm embeddings by city id

    city_to_pos = lro._rebuild_city_to_pos_map(list(tour))
    misplacement_by_city = {c: s for s, _, c in misplacement}
    return (city_to_pos, Hn, city_emb, misplacement, edge_quality,
            misplacement_by_city)


def _trained_relocate_pass(
    tour: list[int],
    current_cost: float,
    dist_matrix,
    projected,
    city_emb: np.ndarray,
    city_nearest_neighbors: list[list[int]],
    misplacement_ranking: list[tuple[float, int, int]],
    misplacement_by_city: dict[int, float],
    top_count: int,
    k_insert_candidates: int,
    city_neighbor_limit: int,
) -> tuple[list[int], float, int]:
    """Applies the single best relocate move from one trained snapshot.
    Move ACCEPTANCE is still gated by the true cost delta (same as the
    training-free pass); the model only ranks equally-improving moves."""
    n = len(tour)
    candidate_mask = [False] * len(city_nearest_neighbors)
    for score, _, city in misplacement_ranking[:top_count]:
        candidate_mask[city] = True

    sensitivities = lro.compute_sensitivities(
        tour, dist_matrix, projected, city_nearest_neighbors,
        k_nearest_edges=max(20, k_insert_candidates),
        city_neighbor_limit=city_neighbor_limit,
        candidate_city_mask=candidate_mask,
    )

    best_candidate: Optional[PointSensitivity] = None
    best_score: Optional[tuple[float, float, float, float]] = None
    for candidate in sensitivities:
        if candidate.delta_total >= -1e-9:
            continue
        insert_after = candidate.best_other_edge_index
        left_city = tour[insert_after]
        right_city = tour[(insert_after + 1) % n]
        emb_city = city_emb[candidate.city_index]
        mid = 0.5 * (city_emb[left_city] + city_emb[right_city])
        mid_norm = np.linalg.norm(mid)
        compatibility = float(emb_city @ mid / mid_norm) if mid_norm > 1e-12 else 0.0
        misplacement_score = misplacement_by_city.get(candidate.city_index, 0.0)
        score = (candidate.delta_total, -compatibility, -misplacement_score,
                 -candidate.removal_gain)
        if best_score is None or score < best_score:
            best_candidate = candidate
            best_score = score

    if best_candidate is None:
        return list(tour), current_cost, 0
    new_tour = lro.relocate_point(
        tour, best_candidate.tour_position, best_candidate.best_other_edge_index)
    new_cost = tour_cost_from_matrix(new_tour, dist_matrix)
    if new_cost < current_cost - 1e-9:
        return new_tour, new_cost, 1
    return list(tour), current_cost, 0


def _trained_two_opt_pass(
    tour: list[int],
    current_cost: float,
    dist_matrix,
    H_pos: np.ndarray,
    edge_quality: list[tuple[float, int]],
    city_nearest_neighbors: list[list[int]],
    city_to_pos: list[int],
    top_edges: int,
    k_partner: int,
) -> tuple[list[int], float, int]:
    """Applies the best improving 2-opt move among the model's most suspicious
    tour edges (lowest predicted backbone probability). Acceptance is gated by
    the true delta; embeddings break ties toward structurally coherent joins."""
    n = len(tour)
    best_move: Optional[tuple[tuple[float, float, float], int, int]] = None

    for quality, edge_pos in edge_quality[:top_edges]:
        i = edge_pos
        if i >= n - 1:
            continue
        a = tour[i]
        b = tour[(i + 1) % n]
        candidate_edges: set[int] = set()
        for endpoint in (a, b):
            for nn_city in city_nearest_neighbors[endpoint][:k_partner]:
                nn_pos = city_to_pos[nn_city]
                candidate_edges.add(nn_pos)
                candidate_edges.add((nn_pos - 1) % n)

        for j in candidate_edges:
            if j <= i + 1 or (i == 0 and j == n - 1):
                continue
            delta = lro._two_opt_delta(tour, dist_matrix, i, j)
            if delta >= -1e-9:
                continue
            post_sim = (float(H_pos[i] @ H_pos[j])
                        + float(H_pos[(i + 1) % n] @ H_pos[(j + 1) % n]))
            score = (delta, -post_sim, quality)
            if best_move is None or score < best_move[0]:
                best_move = (score, i, j)

    if best_move is None:
        return list(tour), current_cost, 0
    _, i, j = best_move
    new_tour = lro._apply_two_opt(tour, i, j)
    new_cost = tour_cost_from_matrix(new_tour, dist_matrix)
    if new_cost < current_cost - 1e-9:
        return new_tour, new_cost, 1
    return list(tour), current_cost, 0


# ---------------------------------------------------------------------------
#  Learned kicks: model-targeted perturbations (v2 skeleton)
# ---------------------------------------------------------------------------
_KICK_POOL_FRAC = 0.06    # fraction of edges eligible as kick targets
_KICK_GUIDED_PROB = 0.75  # per cut: guided pick vs uniform (keeps exploration)
_KICK_LS_ROUNDS = 6       # LS depth after a kick (mirrors the ILS descents)


def _suspicious_positions(edge_quality: list[tuple[float, int]], n: int) -> list[int]:
    """Tour positions of the edges the model considers least backbone-like."""
    m = max(12, int(n * _KICK_POOL_FRAC))
    return [pos for _, pos in edge_quality[:m]]


def entropy_edge_quality(
    edge_quality: list[tuple[float, int]],
) -> list[tuple[float, int]]:
    """Ranks tour edges from highest to lowest normalized binary entropy."""
    ranked = []
    for probability, pos in edge_quality:
        p = min(1.0, max(0.0, probability))
        entropy = 0.0
        if 0.0 < p < 1.0:
            entropy = (-p * math.log(p) - (1.0 - p) * math.log1p(-p)) / math.log(2.0)
        ranked.append((-entropy, pos))
    return sorted(ranked)


def fused_entropy_edge_quality(
    edge_quality: list[tuple[float, int]],
    entropy_weight: float = 0.25,
) -> list[tuple[float, int]]:
    """Fuses suspicious-edge and predictive-entropy ranks without scale calibration."""
    if not 0.0 <= entropy_weight <= 1.0:
        raise ValueError("entropy_weight must be between 0 and 1")
    suspicious = sorted(edge_quality)
    uncertain = entropy_edge_quality(edge_quality)
    suspicious_rank = {pos: rank for rank, (_, pos) in enumerate(suspicious)}
    entropy_rank = {pos: rank for rank, (_, pos) in enumerate(uncertain)}
    return sorted(
        ((1.0 - entropy_weight) * suspicious_rank[pos]
         + entropy_weight * entropy_rank[pos], pos)
        for _, pos in suspicious
    )


def _tour_edge_set(tour: list[int]) -> set[tuple[int, int]]:
    return {
        (min(a, b), max(a, b))
        for a, b in zip(tour, tour[1:] + tour[:1])
    }


def update_elite_archive(
    archive: list[tuple[float, list[int]]],
    tour: list[int],
    cost: float,
    limit: int = 8,
) -> list[tuple[float, list[int]]]:
    edges = _tour_edge_set(tour)
    retained = [row for row in archive if _tour_edge_set(row[1]) != edges]
    retained.append((cost, tour[:]))
    return sorted(retained, key=lambda row: row[0])[:limit]


def elite_entropy_edge_quality(
    tour: list[int],
    archive: list[tuple[float, list[int]]],
) -> list[tuple[float, int]]:
    """Ranks current edges by entropy of their presence in elite local optima."""
    if len(archive) < 2:
        return []
    edge_sets = [_tour_edge_set(row[1]) for row in archive]
    probabilities = []
    for pos, (a, b) in enumerate(zip(tour, tour[1:] + tour[:1])):
        edge = (min(a, b), max(a, b))
        probabilities.append((sum(edge in edges for edges in edge_sets) / len(edge_sets), pos))
    return entropy_edge_quality(probabilities)


def _kick_edge_quality(
    edge_quality: list[tuple[float, int]],
    strategy: str,
    entropy_weight: float = 0.25,
) -> list[tuple[float, int]]:
    if strategy == "entropy":
        return entropy_edge_quality(edge_quality)
    if strategy == "hybrid":
        return fused_entropy_edge_quality(edge_quality, entropy_weight)
    if strategy == "random":
        return []
    if strategy == "suspicious":
        return edge_quality
    raise ValueError(f"unknown kick strategy: {strategy}")


def _guided_double_bridge(
    tour: list[int],
    edge_quality: list[tuple[float, int]],
    rng,
) -> list[int]:
    """Double-bridge whose three cut points are sampled (with probability
    _KICK_GUIDED_PROB each) from the model's most suspicious tour edges, so the
    kick preferentially breaks non-backbone edges and preserves the learned
    consensus structure. Falls back to uniform cuts to keep exploration."""
    n = len(tour)
    if n < 8:
        return tour[:]
    pool = _suspicious_positions(edge_quality, n)
    cuts: set[int] = set()
    for _ in range(32):
        if len(cuts) == 3:
            break
        if pool and rng.random() < _KICK_GUIDED_PROB:
            c = pool[rng.randrange(len(pool))] + 1  # cut between pos and pos+1
        else:
            c = rng.randint(1, n - 1)
        if 1 <= c <= n - 1 and all(abs(c - o) >= 2 for o in cuts):
            cuts.add(c)
    while len(cuts) < 3:  # uniform fallback (degenerate pools)
        c = rng.randint(1, n - 1)
        if all(abs(c - o) >= 2 for o in cuts):
            cuts.add(c)
    a, b, c = sorted(cuts)
    return tour[:a] + tour[b:c] + tour[a:b] + tour[c:]


def _guided_segment_reverse(
    tour: list[int],
    edge_quality: list[tuple[float, int]],
    rng,
    min_seg_ratio: float = 0.05,
    max_seg_ratio: float = 0.25,
) -> list[int]:
    """Segment-reverse kick anchored (with probability _KICK_GUIDED_PROB) right
    after one of the model's most suspicious edges, so the soft kick breaks a
    predicted non-backbone edge instead of a random one."""
    n = len(tour)
    min_seg = max(3, int(n * min_seg_ratio))
    max_seg = max(min_seg + 1, int(n * max_seg_ratio))
    seg_len = rng.randint(min_seg, min(max_seg, n - 2))
    pool = _suspicious_positions(edge_quality, n)
    if pool and rng.random() < _KICK_GUIDED_PROB:
        p = pool[rng.randrange(len(pool))]
        start = max(0, min(p + 1, n - seg_len))
    else:
        start = rng.randint(0, n - seg_len)
    return tour[:start] + tour[start:start + seg_len][::-1] + tour[start + seg_len:]


# ---------------------------------------------------------------------------
#  Search skeleton (v2: guided intensification + learned kicks at ILS cadence)
# ---------------------------------------------------------------------------
def run_trained_gnn_local_search(
    instance: TSPInstance,
    initial_tour: list[int],
    weights: dict[str, np.ndarray],
    max_iterations: int = 300,
    max_no_improve: int = 80,
    max_time_seconds: float = 0.0,
    k_geo: int = K_GEO,
    top_misplaced: int = DEFAULT_TOP_MISPLACED,
    k_insert_candidates: int = DEFAULT_K_INSERT_CANDIDATES,
    top_edges_2opt: int = DEFAULT_TOP_EDGES_2OPT,
    seed: int = 42,
    kick_strategy: str = "suspicious",
    entropy_weight: float = 0.25,
) -> LineReassignResult:
    """Trained-GNN-Guided Local Search with selectable kick guidance.

    Intensification and exact-cost acceptance are shared by all strategies.
    Kicks target low-backbone-probability edges by default, predictive or
    elite-archive entropy, a rank fusion of learned signals, or uniform cuts.
    Model snapshots are recomputed only when the incumbent changes."""
    import random as random_module

    if kick_strategy not in ("suspicious", "entropy", "elite_entropy", "hybrid", "random"):
        raise ValueError(f"unknown kick strategy: {kick_strategy}")
    if not 0.0 <= entropy_weight <= 1.0:
        raise ValueError("entropy_weight must be between 0 and 1")
    rng = random_module.Random(seed)
    instance.validate_tour(initial_tour)
    projected = lro.compute_projected_coords(instance)
    dist_matrix = instance.distance_matrix
    n_cities = len(initial_tour)
    neighbor_limit_2opt = min(n_cities - 1, max(40, k_geo * 2))
    neighbor_limit_oropt = min(n_cities - 1, max(24, k_geo))
    relocate_k = min(n_cities - 1, max(30, k_insert_candidates))
    relocate_neighbor_limit = min(n_cities - 1, max(160, k_insert_candidates * 4))
    polish_ls_rounds = lro.DEFAULT_GNN_POLISH_LS_ROUNDS
    max_nn = min(
        n_cities - 1,
        max(k_geo, K_DENSITY, k_insert_candidates, neighbor_limit_2opt,
            neighbor_limit_oropt, relocate_neighbor_limit),
    )

    if lro.VERBOSE:
        print(f"\n{'=' * 65}")
        print(f"[{time.strftime('%H:%M:%S')}] TRAINED GNN-GUIDED LOCAL SEARCH STARTED")
        print(f"{'=' * 65}")

    t0 = time.perf_counter()
    # attribute access on lro so the sparse-kNN monkeypatch applies
    city_nearest_neighbors = lro._precompute_city_nearest_neighbors(instance, max_nn)
    knn_arr = knn_table_to_array(city_nearest_neighbors, min(max_nn, n_cities - 1))
    xs, ys, is_geo = extract_xy(instance)
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
    if lro.VERBOSE:
        sys.stdout.write("  [tGNN-LS] Phase 0: Initial hybrid local search...\n")
        sys.stdout.flush()

    working_tour, working_cost, init_accepted, _ = lro._combined_local_search(
        tour=list(initial_tour),
        current_cost=initial_cost,
        dist_matrix=dist_matrix,
        projected=projected,
        city_nearest_neighbors=city_nearest_neighbors,
        neighbor_limit_2opt=neighbor_limit_2opt,
        neighbor_limit_oropt=neighbor_limit_oropt,
        k_nearest_edges_relocate=relocate_k,
        city_neighbor_limit_relocate=relocate_neighbor_limit,
        max_ls_rounds=polish_ls_rounds,
        deadline=ls_deadline,
    )
    elapsed_init = time.perf_counter() - start_time
    if lro.VERBOSE:
        sys.stdout.write(
            f"  [tGNN-LS] Initial LS: {initial_cost:.2f} -> {working_cost:.2f} "
            f"({init_accepted} moves, {elapsed_init:.1f}s)\n")

    best_tour = list(working_tour)
    best_cost = working_cost
    elite_archive: list[tuple[float, list[int]]] = []
    if kick_strategy == "elite_entropy":
        elite_archive = update_elite_archive(elite_archive, best_tour, best_cost)

    def quality_for(tour, edge_quality):
        if kick_strategy == "elite_entropy":
            return elite_entropy_edge_quality(tour, elite_archive)
        return _kick_edge_quality(edge_quality, kick_strategy, entropy_weight)

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
    no_improve_count = 0
    restart_count = 0
    iteration = 0

    # Model snapshot cache: recomputed only when the incumbent (best) changes.
    snap = None
    snap_cost = None
    micro_pending = True

    for iteration in range(1, max_iterations + 1):
        elapsed = time.perf_counter() - start_time
        if max_time_seconds > 0 and elapsed >= max_time_seconds:
            if lro.VERBOSE:
                sys.stdout.write("\n  [tGNN-LS] Time limit reached.\n")
            break
        if no_improve_count >= max_no_improve:
            if max_time_seconds <= 0:
                if lro.VERBOSE:
                    sys.stdout.write(
                        f"\n  [tGNN-LS] Stagnation limit reached ({max_no_improve}).\n")
                break
            # Time budget remains: escalate instead of quitting — a stacked
            # (double) guided double-bridge is a much stronger diversification,
            # then keep searching until the time cap.
            escalated = _guided_double_bridge(best_tour, quality_for(best_tour, snap[4]), rng)
            escalated = _guided_double_bridge(escalated, quality_for(best_tour, snap[4]), rng)
            esc_cost = tour_cost_from_matrix(escalated, dist_matrix)
            working_tour, working_cost, _esc_moves, _ = lro._combined_local_search(
                tour=escalated, current_cost=esc_cost,
                dist_matrix=dist_matrix, projected=projected,
                city_nearest_neighbors=city_nearest_neighbors,
                neighbor_limit_2opt=neighbor_limit_2opt,
                neighbor_limit_oropt=neighbor_limit_oropt,
                k_nearest_edges_relocate=relocate_k,
                city_neighbor_limit_relocate=relocate_neighbor_limit,
                max_ls_rounds=_KICK_LS_ROUNDS, deadline=ls_deadline,
            )
            no_improve_count = 0
            restart_count += 1
            if working_cost < best_cost - 1e-9:
                best_tour = list(working_tour)
                best_cost = working_cost
                total_improvements += 1
            if kick_strategy == "elite_entropy":
                elite_archive = update_elite_archive(
                    elite_archive, working_tour, working_cost)
            continue

        if snap_cost is None or abs(snap_cost - best_cost) > 1e-9:
            snap = _compute_trained_snapshot(
                weights, best_tour, xs, ys, is_geo, proj_x, proj_y, knn_arr)
            snap_cost = best_cost
            micro_pending = True
        (city_to_pos, H_pos, city_emb, misplacement, edge_quality,
         misplacement_by_city) = snap
        kick_quality = quality_for(best_tour, edge_quality)

        round_accepted = 0
        restart_label = ""
        working_tour = list(best_tour)
        working_cost = best_cost

        # --- 1) guided intensification (skipped once exhausted on this best) ---
        if micro_pending:
            new_tour, new_cost, acc = _trained_relocate_pass(
                working_tour, working_cost, dist_matrix, projected, city_emb,
                city_nearest_neighbors, misplacement, misplacement_by_city,
                top_count=top_misplaced,
                k_insert_candidates=relocate_k,
                city_neighbor_limit=relocate_neighbor_limit,
            )
            if acc:
                working_tour, working_cost = new_tour, new_cost
                round_accepted += acc
                # positions shifted: refresh the cheap map, reuse the embeddings
                # (they only rank moves; acceptance is gated by the true delta)
                city_to_pos = lro._rebuild_city_to_pos_map(working_tour)

            new_tour, new_cost, acc = _trained_two_opt_pass(
                working_tour, working_cost, dist_matrix, H_pos, edge_quality,
                city_nearest_neighbors, city_to_pos,
                top_edges=top_edges_2opt, k_partner=neighbor_limit_2opt,
            )
            if acc:
                working_tour, working_cost = new_tour, new_cost
                round_accepted += acc

            if round_accepted:
                working_tour, working_cost, ls_accepted, _ = lro._combined_local_search(
                    tour=working_tour,
                    current_cost=working_cost,
                    dist_matrix=dist_matrix,
                    projected=projected,
                    city_nearest_neighbors=city_nearest_neighbors,
                    neighbor_limit_2opt=neighbor_limit_2opt,
                    neighbor_limit_oropt=neighbor_limit_oropt,
                    k_nearest_edges_relocate=relocate_k,
                    city_neighbor_limit_relocate=relocate_neighbor_limit,
                    max_ls_rounds=polish_ls_rounds,
                    deadline=ls_deadline,
                )
                round_accepted += ls_accepted
            else:
                micro_pending = False  # nothing left; kick from now on

        # --- 2) learned kick + deep descent when intensification found nothing
        #        (same perturb+descend cadence as the ILS baseline) ---
        if working_cost >= best_cost - 1e-9:
            if no_improve_count < 5:
                use_hard = rng.random() < 0.3
            elif no_improve_count < 15:
                use_hard = rng.random() < 0.6
            else:
                use_hard = True
            if use_hard:
                perturbed_tour = _guided_double_bridge(best_tour, kick_quality, rng)
                restart_label = "g-bridge"
            else:
                perturbed_tour = _guided_segment_reverse(best_tour, kick_quality, rng)
                restart_label = "g-segrev"
            perturbed_cost = tour_cost_from_matrix(perturbed_tour, dist_matrix)
            working_tour, working_cost, kick_moves, _ = lro._combined_local_search(
                tour=perturbed_tour,
                current_cost=perturbed_cost,
                dist_matrix=dist_matrix,
                projected=projected,
                city_nearest_neighbors=city_nearest_neighbors,
                neighbor_limit_2opt=neighbor_limit_2opt,
                neighbor_limit_oropt=neighbor_limit_oropt,
                k_nearest_edges_relocate=relocate_k,
                city_neighbor_limit_relocate=relocate_neighbor_limit,
                max_ls_rounds=_KICK_LS_ROUNDS,
                deadline=ls_deadline,
            )
            round_accepted += kick_moves
            restart_count += 1

        if kick_strategy == "elite_entropy":
            elite_archive = update_elite_archive(
                elite_archive, working_tour, working_cost)
        elapsed = time.perf_counter() - start_time
        if working_cost < best_cost - 1e-9:
            improvement = best_cost - working_cost
            best_tour = list(working_tour)
            best_cost = working_cost
            total_improvements += 1
            no_improve_count = 0
            frames.append(TourFrame(
                step=len(frames), phase="repair",
                stage_index=total_improvements,
                elapsed_time=elapsed, cost=best_cost, best_cost=best_cost,
                constructed_cities=round_accepted,
                tour=lro._frame_tour(best_tour),
            ))
            history.append(HistoryPoint(
                iteration=total_improvements + 1,
                elapsed_time=elapsed, cost=best_cost,
            ))
            if lro.VERBOSE:
                sys.stdout.write(f"\n  *** IMPROVED by {improvement:.2f} ***\n")
        else:
            no_improve_count += 1

        if lro.VERBOSE:
            sys.stdout.write(
                "\r"
                f"[tGNN-LS] Iter: {iteration}/{max_iterations} | "
                f"Moves: {round_accepted} | Cost: {working_cost:.2f} | "
                f"Best: {best_cost:.2f} | Improved: {total_improvements} | "
                f"Stall: {no_improve_count}/{max_no_improve} | "
                f"Restarts: {restart_count} | Elapsed: {elapsed:.1f}s   ")
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
        print("[tGNN-LS] COMPLETED")
        print(f"  Initial cost  : {initial_cost:.2f}")
        print(f"  Best cost     : {best_cost:.2f}")
        print(f"  Total saving  : {initial_cost - best_cost:.2f}")
        print(f"  Improvements  : {total_improvements}")
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


def build_trained_gnn_ls_result(
    instance: TSPInstance,
    initial_tour: list[int],
    max_iterations: int = 300,
    max_no_improve: int = 80,
    max_time_seconds: float = 0.0,
    seed: int = 42,
    weights_path: Optional[str | Path] = None,
) -> tuple[TSPRunResult, LineReassignResult]:
    """Dashboard-compatible result for Trained-GNN-Guided Local Search.
    `weights_path` defaults to $GNN_TRAINED_WEIGHTS or gnn_trained_weights.npz
    next to this module."""
    if weights_path is None:
        weights_path = os.environ.get(
            "GNN_TRAINED_WEIGHTS",
            str(Path(__file__).resolve().parent / "gnn_trained_weights.npz"))
    weights, _meta = get_weights(weights_path)

    result = run_trained_gnn_local_search(
        instance=instance,
        initial_tour=initial_tour,
        weights=weights,
        max_iterations=max_iterations,
        max_no_improve=max_no_improve,
        max_time_seconds=max_time_seconds,
        seed=seed,
    )
    metrics = RunMetrics(
        initialization_name="Trained GNN-Guided Local Search",
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


if __name__ == "__main__":
    err = gradient_check()
    print(f"gradient check worst relative error: {err:.2e} "
          f"({'OK' if err < 1e-4 else 'FAIL'})")
