# -*- coding: utf-8 -*-
"""Fast iterated local search (ILS) testbed for kick-targeting experiments.

Engine: 2-opt + Or-opt (segments of 1..3 cities, both orientations) with
K-nearest-neighbour candidate lists and don't-look bits (Bentley 1992;
Johnson & McGeoch 1997).  After a kick only the endpoints of the changed
edges are re-activated, so one kick costs O(window) instead of a full sweep.

Kick geometries
  * global double-bridge  : three cut positions anywhere on the tour
                            (the CRINE-ILS operator of the original study)
  * local (segment) double-bridge : the three cuts lie inside a window of
                            at most W tour positions (Applegate et al. 2003
                            style "segment" kick)
Kick targeting (the only thing that differs between strategies)
  * random     : uniform cut positions
  * gnn        : with prob. 0.75 a cut is placed on one of the m tour edges the
                 LKH-trained GNN scores lowest (m = max(12, 0.06 n))
  * knn        : same rule, but the pool is the m tour edges with the worst
                 mutual k-nearest-neighbour rank (training-free proxy)

Acceptance: strictly better tour cost (same as the original CRINE-ILS loop).
Time is wall-clock and includes every model / proxy evaluation.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

EUC, GEO = 0, 1
RRR = 6378.388
EPS = 1e-7

# ---------------------------------------------------------------------------
# distances (TSPLIB-exact, identical to gnn_trained.pair_dist)
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def dist(i, j, xs, ys, kind):
    if kind == 0:
        dx = xs[i] - xs[j]
        dy = ys[i] - ys[j]
        return math.floor(math.sqrt(dx * dx + dy * dy) + 0.5)
    if i == j:
        return 0.0
    la1 = xs[i] * math.pi / 180.0
    lo1 = ys[i] * math.pi / 180.0
    la2 = xs[j] * math.pi / 180.0
    lo2 = ys[j] * math.pi / 180.0
    q1 = math.cos(lo1 - lo2)
    q2 = math.cos(la1 - la2)
    q3 = math.cos(la1 + la2)
    v = 0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)
    if v > 1.0:
        v = 1.0
    elif v < -1.0:
        v = -1.0
    return math.floor(RRR * math.acos(v) + 1.0)


@njit(cache=True)
def tour_cost(tour, xs, ys, kind):
    n = tour.shape[0]
    s = 0.0
    for i in range(n):
        s += dist(tour[i], tour[(i + 1) % n], xs, ys, kind)
    return s


@njit(cache=True)
def knn_bruteforce(xs, ys, kind, k):
    n = xs.shape[0]
    out = np.empty((n, k), dtype=np.int32)
    d = np.empty(n)
    for i in range(n):
        for j in range(n):
            d[j] = dist(i, j, xs, ys, kind) if j != i else 1e300
        order = np.argsort(d, kind="mergesort")
        for t in range(k):
            out[i, t] = order[t]
    return out


def build_knn(xs: np.ndarray, ys: np.ndarray, kind: int, k: int) -> np.ndarray:
    n = len(xs)
    k = min(k, n - 1)
    if kind == EUC:
        from scipy.spatial import cKDTree
        tree = cKDTree(np.column_stack([xs, ys]))
        _, idx = tree.query(np.column_stack([xs, ys]), k=k + 1)
        out = np.empty((n, k), dtype=np.int32)
        for i in range(n):
            row = [int(j) for j in idx[i] if j != i][:k]
            out[i] = row
        return out
    return knn_bruteforce(xs, ys, kind, k)


# ---------------------------------------------------------------------------
# tour primitives
# ---------------------------------------------------------------------------


@njit(cache=True, inline="always")
def succ(c, tour, pos, n):
    return tour[(pos[c] + 1) % n]


@njit(cache=True, inline="always")
def pred(c, tour, pos, n):
    return tour[(pos[c] - 1 + n) % n]


@njit(cache=True)
def reverse_path(tour, pos, i, j, n):
    """Reverse the forward path of positions i..j (inclusive, cyclic).  The
    shorter of the path and its complement is reversed; both give the same
    undirected cycle."""
    L = (j - i) % n + 1
    if 2 * L > n:
        i, j = (j + 1) % n, (i - 1 + n) % n
        L = n - L
    for k in range(L // 2):
        a = (i + k) % n
        b = (j - k + n) % n
        ca = tour[a]
        cb = tour[b]
        tour[a] = cb
        pos[cb] = a
        tour[b] = ca
        pos[ca] = b


@njit(cache=True, inline="always")
def q_push(c, queue, inq, qhead_tail, n):
    if inq[c] == 0:
        inq[c] = 1
        queue[qhead_tail[1]] = c
        qhead_tail[1] = (qhead_tail[1] + 1) % n
        qhead_tail[2] += 1


@njit(cache=True)
def try_2opt(a, tour, pos, n, xs, ys, kind, nbr, queue, inq, qht):
    K = nbr.shape[1]
    for direction in range(2):
        b = succ(a, tour, pos, n) if direction == 0 else pred(a, tour, pos, n)
        dab = dist(a, b, xs, ys, kind)
        for t in range(K):
            c = nbr[a, t]
            g1 = dab - dist(a, c, xs, ys, kind)
            if g1 <= EPS:
                break
            d = succ(c, tour, pos, n) if direction == 0 else pred(c, tour, pos, n)
            if c == b or d == a:
                continue
            delta = g1 + dist(c, d, xs, ys, kind) - dist(b, d, xs, ys, kind)
            if delta > EPS:
                if direction == 0:
                    reverse_path(tour, pos, pos[b], pos[c], n)
                else:
                    reverse_path(tour, pos, pos[a], pos[d], n)
                q_push(a, queue, inq, qht, n)
                q_push(b, queue, inq, qht, n)
                q_push(c, queue, inq, qht, n)
                q_push(d, queue, inq, qht, n)
                return delta
    return 0.0


@njit(cache=True)
def apply_or_move(tour, pos, n, s1, s2, L, c, e, end_first_at_c, buf):
    """Remove forward segment s1..s2 (length L) and re-insert it on edge (c,e).
    If end_first_at_c==1 the city adjacent to c is s1, else s2."""
    p = pred(s1, tour, pos, n)
    nx = succ(s2, tour, pos, n)
    start = pos[nx]
    idx = 0
    m = n - L
    for k in range(m):
        city = tour[(start + k) % n]
        buf[idx] = city
        idx += 1
        nxt = tour[(start + k + 1) % n] if k < m - 1 else nx
        if (city == c and nxt == e) or (city == e and nxt == c):
            # insert so that the chosen end touches c
            if city == c:
                first = s1 if end_first_at_c == 1 else s2
            else:
                first = s2 if end_first_at_c == 1 else s1
            sp = pos[s1]
            if first == s1:
                for q in range(L):
                    buf[idx] = tour[(sp + q) % n]
                    idx += 1
            else:
                for q in range(L):
                    buf[idx] = tour[(sp + L - 1 - q) % n]
                    idx += 1
    for k in range(n):
        tour[k] = buf[k]
        pos[buf[k]] = k
    _ = p


@njit(cache=True)
def try_oropt(a, tour, pos, n, xs, ys, kind, nbr, queue, inq, qht, buf):
    K = nbr.shape[1]
    if n < 8:
        return 0.0
    for L in range(1, 4):
        for side in range(2):
            if L == 1 and side == 1:
                continue
            if side == 0:
                s1 = a
                s2 = tour[(pos[a] + L - 1) % n]
            else:
                s2 = a
                s1 = tour[(pos[a] - L + 1 + n) % n]
            p = pred(s1, tour, pos, n)
            nx = succ(s2, tour, pos, n)
            g0 = (dist(p, s1, xs, ys, kind) + dist(s2, nx, xs, ys, kind)
                  - dist(p, nx, xs, ys, kind))
            if g0 <= EPS:
                continue
            ps1 = pos[s1]
            for which in range(2):
                end = s1 if which == 0 else s2
                other = s2 if which == 0 else s1
                for t in range(K):
                    c = nbr[end, t]
                    dce = dist(end, c, xs, ys, kind)
                    if dce >= g0 - EPS:
                        break
                    if (pos[c] - ps1) % n < L:
                        continue
                    for dirn in range(2):
                        e = succ(c, tour, pos, n) if dirn == 0 else pred(c, tour, pos, n)
                        if (pos[e] - ps1) % n < L:
                            continue
                        added = dce + dist(other, e, xs, ys, kind) - dist(c, e, xs, ys, kind)
                        delta = g0 - added
                        if delta > EPS:
                            apply_or_move(tour, pos, n, s1, s2, L, c, e,
                                          1 if which == 0 else 0, buf)
                            q_push(p, queue, inq, qht, n)
                            q_push(nx, queue, inq, qht, n)
                            q_push(s1, queue, inq, qht, n)
                            q_push(s2, queue, inq, qht, n)
                            q_push(c, queue, inq, qht, n)
                            q_push(e, queue, inq, qht, n)
                            return delta
    return 0.0


@njit(cache=True)
def local_search(tour, pos, n, xs, ys, kind, nbr, queue, inq, qht, buf):
    """Process the don't-look-bit queue until empty; returns total gain."""
    gain = 0.0
    while qht[2] > 0:
        a = queue[qht[0]]
        qht[0] = (qht[0] + 1) % n
        qht[2] -= 1
        inq[a] = 0
        while True:
            d = try_2opt(a, tour, pos, n, xs, ys, kind, nbr, queue, inq, qht)
            if d > 0.0:
                gain += d
                continue
            d = try_oropt(a, tour, pos, n, xs, ys, kind, nbr, queue, inq, qht, buf)
            if d > 0.0:
                gain += d
                continue
            break
    return gain


@njit(cache=True)
def nn_tour(xs, ys, kind, nbr, start):
    n = xs.shape[0]
    tour = np.empty(n, dtype=np.int32)
    visited = np.zeros(n, dtype=np.uint8)
    cur = start
    visited[cur] = 1
    tour[0] = cur
    for k in range(1, n):
        nxt = -1
        for t in range(nbr.shape[1]):
            c = nbr[cur, t]
            if visited[c] == 0:
                nxt = c
                break
        if nxt < 0:
            best = 1e300
            for c in range(n):
                if visited[c] == 0:
                    dd = dist(cur, c, xs, ys, kind)
                    if dd < best:
                        best = dd
                        nxt = c
        visited[nxt] = 1
        tour[k] = nxt
        cur = nxt
    return tour


@njit(cache=True)
def full_local_search(tour, xs, ys, kind, nbr):
    n = tour.shape[0]
    pos = np.empty(n, dtype=np.int32)
    for i in range(n):
        pos[tour[i]] = i
    queue = np.empty(n, dtype=np.int32)
    inq = np.zeros(n, dtype=np.uint8)
    qht = np.zeros(3, dtype=np.int64)
    buf = np.empty(n, dtype=np.int32)
    for i in range(n):
        q_push(tour[i], queue, inq, qht, n)
    local_search(tour, pos, n, xs, ys, kind, nbr, queue, inq, qht, buf)
    return tour


@njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


# ---------------------------------------------------------------------------
# kicks
# ---------------------------------------------------------------------------


@njit(cache=True)
def pool_cut(pool_u, pool_v, pool_m, tour, pos, n):
    """Cut position (edge between position cut-1 and cut) of a random pool
    edge that is still in the tour; -1 if the sampled edge has disappeared."""
    k = np.random.randint(pool_m)
    u = pool_u[k]
    v = pool_v[k]
    if tour[(pos[u] + 1) % n] == v:
        return pos[v]
    if tour[(pos[v] + 1) % n] == u:
        return pos[u]
    return -1


@njit(cache=True)
def kick_global(tour, pos, n, xs, ys, kind, guided, pool_u, pool_v, pool_m,
                prob, queue, inq, qht, buf, stats):
    # three cuts 1 <= a < b < c <= n-1, pairwise >= 2 apart (paper operator)
    cuts = np.empty(3, dtype=np.int64)
    nc = 0
    for _ in range(32):
        if nc == 3:
            break
        c = -1
        if guided and pool_m > 0 and np.random.random() < prob:
            c = pool_cut(pool_u, pool_v, pool_m, tour, pos, n)
            stats[2] += 1
            if c < 0:
                stats[3] += 1
        if c < 1:
            c = 1 + np.random.randint(n - 1)
        ok = True
        for q in range(nc):
            if abs(c - cuts[q]) < 2:
                ok = False
        if ok:
            cuts[nc] = c
            nc += 1
    while nc < 3:
        c = 1 + np.random.randint(n - 1)
        ok = True
        for q in range(nc):
            if abs(c - cuts[q]) < 2:
                ok = False
        if ok:
            cuts[nc] = c
            nc += 1
    cuts.sort()
    a, b, c = cuts[0], cuts[1], cuts[2]
    return apply_double_bridge(tour, pos, n, xs, ys, kind, a, b - a, c - a,
                               queue, inq, qht, buf)


@njit(cache=True)
def apply_double_bridge(tour, pos, n, xs, ys, kind, a, o1, o2, queue, inq, qht, buf):
    """Segments A=[a, a+o1) and B=[a+o1, a+o2) (cyclic positions) are swapped:
    ... X A B Y ... -> ... X B A Y ...; returns the cost delta (new - old)."""
    x = tour[(a - 1 + n) % n]
    a0 = tour[a % n]
    a1 = tour[(a + o1 - 1) % n]
    b0 = tour[(a + o1) % n]
    b1 = tour[(a + o2 - 1) % n]
    y = tour[(a + o2) % n]
    old = (dist(x, a0, xs, ys, kind) + dist(a1, b0, xs, ys, kind)
           + dist(b1, y, xs, ys, kind))
    new = (dist(x, b0, xs, ys, kind) + dist(b1, a0, xs, ys, kind)
           + dist(a1, y, xs, ys, kind))
    lb = o2 - o1
    for q in range(lb):
        buf[q] = tour[(a + o1 + q) % n]
    for q in range(o1):
        buf[lb + q] = tour[(a + q) % n]
    for q in range(o2):
        p = (a + q) % n
        tour[p] = buf[q]
        pos[buf[q]] = p
    for c in (x, a0, a1, b0, b1, y):
        q_push(c, queue, inq, qht, n)
    return new - old


@njit(cache=True)
def kick_local(tour, pos, n, xs, ys, kind, guided, pool_u, pool_v, pool_m,
               prob, window, queue, inq, qht, buf, stats):
    a = -1
    if guided and pool_m > 0 and np.random.random() < prob:
        a = pool_cut(pool_u, pool_v, pool_m, tour, pos, n)
        stats[2] += 1
        if a < 0:
            stats[3] += 1
    if a < 0:
        a = np.random.randint(n)
    w = min(window, n - 3)
    # 1 <= o1 < o2 <= w with o2 - o1 >= 1
    o1 = 1 + np.random.randint(w - 1)
    o2 = o1 + 1 + np.random.randint(w - o1)
    return apply_double_bridge(tour, pos, n, xs, ys, kind, a, o1, o2,
                               queue, inq, qht, buf)


@njit(cache=True)
def run_kicks(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, n_kicks,
              geometry, guided, pool_u, pool_v, pool_m, prob, window,
              queue, inq, qht, buf, stats):
    """Run n_kicks ILS iterations with better-only acceptance.
    stats: [kicks, accepted, guided_cut_attempts, stale_cuts]."""
    for _ in range(n_kicks):
        if geometry == 0:
            delta = kick_global(tour, pos, n, xs, ys, kind, guided, pool_u, pool_v,
                                pool_m, prob, queue, inq, qht, buf, stats)
        else:
            delta = kick_local(tour, pos, n, xs, ys, kind, guided, pool_u, pool_v,
                               pool_m, prob, window, queue, inq, qht, buf, stats)
        g = local_search(tour, pos, n, xs, ys, kind, nbr, queue, inq, qht, buf)
        newc = cost + delta - g
        stats[0] += 1
        if newc < cost - EPS:
            cost = newc
            stats[1] += 1
            for i in range(n):
                btour[i] = tour[i]
                bpos[i] = pos[i]
        else:
            for i in range(n):
                tour[i] = btour[i]
                pos[i] = bpos[i]
    return cost


# ---------------------------------------------------------------------------
# pools: GNN and training-free kNN proxy
# ---------------------------------------------------------------------------


def pool_size(n: int) -> int:
    return max(12, int(n * 0.06))


def knn_rank_scores(tour: np.ndarray, knn: np.ndarray, xs, ys, kind) -> np.ndarray:
    """Suspicion score of tour edge (tour[i], tour[i+1]): mean mutual kNN rank
    (absent -> k+6), tie-broken by length relative to the endpoints' nearest
    neighbour distance.  Higher = more suspicious."""
    n = len(tour)
    k = knn.shape[1]
    u = tour
    v = np.roll(tour, -1)
    ku = knn[u]
    kv = knn[v]
    mu = ku == v[:, None]
    mv = kv == u[:, None]
    ru = np.where(mu.any(1), mu.argmax(1) + 1.0, k + 6.0)
    rv = np.where(mv.any(1), mv.argmax(1) + 1.0, k + 6.0)
    u = np.ascontiguousarray(u, dtype=np.int32)
    v = np.ascontiguousarray(v, dtype=np.int32)
    duv = _edge_lengths(u, v, xs, ys, kind)
    d1u = _edge_lengths(u, np.ascontiguousarray(ku[:, 0], dtype=np.int32), xs, ys, kind)
    d1v = _edge_lengths(v, np.ascontiguousarray(kv[:, 0], dtype=np.int32), xs, ys, kind)
    rel = duv / np.maximum(0.5 * (d1u + d1v), 1e-9)
    return 0.5 * (ru + rv) + 1e-3 * np.minimum(rel, 999.0)


@njit(cache=True)
def _edge_lengths(u, v, xs, ys, kind):
    out = np.empty(u.shape[0])
    for i in range(u.shape[0]):
        out[i] = dist(u[i], v[i], xs, ys, kind)
    return out


class GNNScorer:
    """Wraps the frozen LKH-GNN (gnn_trained.py) exactly as trained: k_geo=24
    neighbour lists, TSPLIB distances, projected frame for the geometric
    features.  Returns P(edge in near-optimal tour) per tour position."""

    def __init__(self, xs, ys, kind, weights):
        import gnn_trained as gt
        self.gt = gt
        self.xs, self.ys = xs, ys
        self.is_geo = kind == GEO
        self.w = weights
        if self.is_geo:
            lat, lon = xs, ys
            mlat = lat.mean()
            mlon = lon.mean()
            self.px = 6371.0 * np.radians(lon - mlon) * math.cos(math.radians(mlat))
            self.py = 6371.0 * np.radians(lat - mlat)
        else:
            self.px, self.py = xs, ys
        self.knn = build_knn(xs, ys, kind, min(24, len(xs) - 1)).astype(np.int64)

    def edge_prob(self, tour: np.ndarray) -> np.ndarray:
        gt = self.gt
        g = gt.compute_graph_tensors(tour.astype(np.int64), self.xs, self.ys,
                                     self.is_geo, self.px, self.py, self.knn)
        _, _, edge_logit = gt.model_forward(self.w, g)
        return gt._sigmoid(edge_logit)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

STRATEGIES = {
    # name: (geometry 0=global 1=local, pool source)
    "random_global": (0, None),
    "gnn_global": (0, "gnn"),
    "knn_global": (0, "knn"),
    "random_local": (1, None),
    "gnn_local": (1, "gnn"),
    "knn_local": (1, "knn"),
}

CHECKPOINTS = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0)


def start_tour(xs, ys, kind, nbr, seed: int) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    n = len(xs)
    rng = np.random.default_rng(seed)
    tour = nn_tour(xs, ys, kind, nbr, int(rng.integers(n)))
    tour = full_local_search(tour, xs, ys, kind, nbr)
    return tour, float(tour_cost(tour, xs, ys, kind)), time.perf_counter() - t0


def run_strategy(xs, ys, kind, nbr, init_tour, init_cost, strategy: str,
                 seed: int, budget: float, gnn: GNNScorer | None = None,
                 knn24: np.ndarray | None = None, window: int = 50,
                 prob: float = 0.75, overhead_cap: float = 0.10,
                 batch_target_s: float = 0.004) -> dict:
    geometry, source = STRATEGIES[strategy]
    n = len(xs)
    tour = init_tour.copy()
    pos = np.empty(n, dtype=np.int32)
    pos[tour] = np.arange(n, dtype=np.int32)
    btour = tour.copy()
    bpos = pos.copy()
    queue = np.empty(n, dtype=np.int32)
    inq = np.zeros(n, dtype=np.uint8)
    qht = np.zeros(3, dtype=np.int64)
    buf = np.empty(n, dtype=np.int32)
    stats = np.zeros(4, dtype=np.int64)
    seed_rng(seed)
    m = pool_size(n)
    pool_u = np.zeros(m, dtype=np.int32)
    pool_v = np.zeros(m, dtype=np.int32)
    pool_m = 0
    cost = init_cost
    checkpoints = [t for t in CHECKPOINTS if t <= budget + 1e-9]
    trace = {}
    model_time = 0.0
    refreshes = 0
    accepted_at_refresh = -1
    batch = 8
    start = time.perf_counter()
    elapsed = 0.0
    while True:
        elapsed = time.perf_counter() - start
        while checkpoints and elapsed >= checkpoints[0]:
            trace[str(checkpoints.pop(0))] = cost
        if elapsed >= budget:
            break
        if source is not None and stats[1] != accepted_at_refresh and (
                refreshes == 0 or model_time <= overhead_cap * max(elapsed, 1e-9)):
            t1 = time.perf_counter()
            if source == "gnn":
                p = gnn.edge_prob(btour)
                idx = np.argpartition(p, m - 1)[:m]
            else:
                s = knn_rank_scores(btour, knn24, xs, ys, kind)
                idx = np.argpartition(-s, m - 1)[:m]
            pool_u[:] = btour[idx]
            pool_v[:] = btour[(idx + 1) % n]
            pool_m = m
            refreshes += 1
            accepted_at_refresh = stats[1]
            model_time += time.perf_counter() - t1
        tb = time.perf_counter()
        cost = run_kicks(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, batch,
                         geometry, source is not None, pool_u, pool_v, pool_m, prob,
                         window, queue, inq, qht, buf, stats)
        dt = time.perf_counter() - tb
        if dt < batch_target_s * 0.5:
            batch = min(batch * 2, 1 << 20)
        elif dt > batch_target_s * 2 and batch > 1:
            batch = max(1, batch // 2)
    for t in checkpoints:
        trace[str(t)] = cost
    exact = float(tour_cost(btour, xs, ys, kind))
    if abs(exact - cost) > 1e-6 * max(1.0, abs(exact)):
        raise RuntimeError(f"cost drift: tracked {cost} vs exact {exact}")
    if len(np.unique(btour)) != n:
        raise RuntimeError("invalid tour")
    return {
        "strategy": strategy, "seed": seed, "cost": exact, "initial_cost": init_cost,
        "elapsed": elapsed, "kicks": int(stats[0]), "accepted": int(stats[1]),
        "guided_cuts": int(stats[2]), "stale_cuts": int(stats[3]),
        "model_time": model_time, "refreshes": refreshes, "trace": trace,
    }


def warmup() -> None:
    """Trigger JIT compilation of every code path (EUC and GEO, both kick
    geometries, guided and unguided) so that timed runs exclude compilation."""
    rng = np.random.default_rng(0)
    for kind in (EUC, GEO):
        xs = rng.uniform(30, 50, 60)
        ys = rng.uniform(0, 20, 60)
        nbr = build_knn(xs, ys, kind, 8)
        knn24 = build_knn(xs, ys, kind, 24)
        t0, c0, _ = start_tour(xs, ys, kind, nbr, 0)
        for s in ("random_global", "knn_global", "random_local", "knn_local"):
            run_strategy(xs, ys, kind, nbr, t0, c0, s, 0, 0.05, knn24=knn24)
