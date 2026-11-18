# -*- coding: utf-8 -*-
"""Online kick targeting learned from the search itself (PROTOCOL_v2, method candidate for H7).

Motivation: membership-trained scores rank *subtle* wrong edges first (objective mismatch).  Here
the score is learned during the run from the realised outcome of every kick, so its objective is
the search outcome by construction.

Model.  Each tour edge e has the 17 O(1) features of cheap_scorer.features (standardised with the
cheap model's mu/sd) plus a bias.  A double-bridge removes three edges; its outcome is
    r = -(C_after_LS - C_before) / median_edge_length      (signed, defined for EVERY kick)
and is modelled additively, r ~ w^T (x_e1 + x_e2 + x_e3).  The signed reward is used on purpose:
with acceptance rates of 0.2-0.4 % a max(0, .) reward would be zero for >99 % of kicks.  Cuts that
fall uniformly (prob 1 - 0.75, and every cut of a random kick) act as built-in exploration.
w is estimated by online ridge regression (A = lam*I + sum x x^T, b = lam*w0 + sum r x); at each
pool refresh a Thompson sample w ~ N(A^-1 b, nu^2 A^-1) scores the tour edges and the m highest
predicted-r edges form the pool (same pool rule, prob, 10 % model-time cap as every other signal).
w0 is a warm start: a ridge fit of the gain target on the GNN training data (fit_prior), or zero.

fast_ils.py is unchanged: the kick loop below reuses its compiled primitives and mirrors
fast_ils.kick_global exactly (same RNG consumption), only additionally reporting the cut edges.

    py -3.13 online_targeting.py fit_prior        # writes models/online_prior.npz
    py -3.13 online_targeting.py smoke            # small sanity run
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

import fast_ils as F
from fast_ils import EPS, apply_double_bridge, local_search, pool_cut

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "models" / "online_prior.npz"
D = 18  # 17 features + bias


@njit(cache=True)
def _kick_global_cuts(tour, pos, n, xs, ys, kind, guided, pool_u, pool_v, pool_m,
                      prob, queue, inq, qht, buf, stats, cuts):
    """Identical to fast_ils.kick_global, but leaves the sorted cut positions in `cuts`."""
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
def _ref_row(u, v, ref_tour, ref_pos, n):
    """Row of edge {u,v} in the feature matrix of the refresh tour, -1 if it is not a tour edge there."""
    i = ref_pos[u]
    if ref_tour[(i + 1) % n] == v:
        return i
    j = (i - 1 + n) % n
    if ref_tour[j] == v:
        return j
    return -1


@njit(cache=True)
def run_kicks_online(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, n_kicks,
                     guided, pool_u, pool_v, pool_m, prob, queue, inq, qht, buf, stats,
                     X, ref_tour, ref_pos, inv_scale, A, bvec, learn):
    """fast_ils.run_kicks (global geometry) plus an online ridge update per kick.
    stats: [kicks, accepted, guided_cut_attempts, stale_cuts, learned_kicks]."""
    cuts = np.empty(3, dtype=np.int64)
    rows = np.empty(3, dtype=np.int64)
    xsum = np.empty(X.shape[1])
    for _ in range(n_kicks):
        delta = _kick_global_cuts(tour, pos, n, xs, ys, kind, guided, pool_u, pool_v, pool_m,
                                  prob, queue, inq, qht, buf, stats, cuts)
        # cut position p removed edge (btour[p-1], btour[p]); btour == tour before the kick
        ok = learn
        if learn:
            for q in range(3):
                p = cuts[q]
                rows[q] = _ref_row(btour[p - 1], btour[p], ref_tour, ref_pos, n)
                if rows[q] < 0:
                    ok = False
        g = local_search(tour, pos, n, xs, ys, kind, nbr, queue, inq, qht, buf)
        newc = cost + delta - g
        stats[0] += 1
        if ok:
            r = -(newc - cost) * inv_scale
            for k in range(X.shape[1]):
                xsum[k] = X[rows[0], k] + X[rows[1], k] + X[rows[2], k]
            for k in range(X.shape[1]):
                bvec[k] += r * xsum[k]
                for l in range(X.shape[1]):
                    A[k, l] += xsum[k] * xsum[l]
            stats[4] += 1
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


class OnlinePolicy:
    """Per-run state: feature extractor (cheap_scorer), ridge statistics, Thompson sampler."""

    def __init__(self, xs, ys, kind, knn24, params, w0=None, lam=50.0, nu=0.5, seed=0):
        from cheap_scorer import neighbour_dists
        self.xs, self.ys, self.kind, self.knn = xs, ys, kind, knn24
        self.dk = neighbour_dists(knn24, xs, ys, kind)
        self.mu, self.sd = params["mu"], params["sd"]
        self.w0 = np.zeros(D) if w0 is None else np.asarray(w0, dtype=np.float64)
        self.lam, self.nu = lam, nu
        self.A = lam * np.eye(D)
        self.b = lam * self.w0
        self.rng = np.random.default_rng(seed)

    def features(self, tour):
        from cheap_scorer import features
        X = (features(tour, self.knn, self.xs, self.ys, self.kind, self.dk) - self.mu) / self.sd
        return np.ascontiguousarray(np.column_stack([X, np.ones(len(tour))]))

    def sample_w(self):
        Ainv = np.linalg.inv(self.A)
        mean = Ainv @ self.b
        if self.nu <= 0:
            return mean
        L = np.linalg.cholesky(0.5 * (Ainv + Ainv.T) + 1e-12 * np.eye(D))
        return mean + self.nu * (L @ self.rng.standard_normal(D))


def run_online(xs, ys, kind, nbr, init_tour, init_cost, seed, budget, knn24, params, w0=None,
               lam=50.0, nu=0.5, refresh_kicks=2000, prob=0.75, overhead_cap=0.10,
               batch_target_s=0.004, learn=True):
    """Mirrors fast_ils.run_strategy for geometry=global with the online pool source."""
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
    stats = np.zeros(5, dtype=np.int64)
    F.seed_rng(seed)
    m = F.pool_size(n)
    pool_u = np.zeros(m, dtype=np.int32)
    pool_v = np.zeros(m, dtype=np.int32)
    pool_m = 0
    cost = init_cost
    checkpoints = [t for t in F.CHECKPOINTS if t <= budget + 1e-9]
    trace = {}
    model_time = 0.0
    refreshes = 0
    accepted_at_refresh = -1
    kicks_at_refresh = 0
    batch = 8
    # per-instance setup (neighbour distances, reward scale) mirrors GNNScorer/CheapScorer
    # construction, which is not charged to the budget for any signal either
    pol = OnlinePolicy(xs, ys, kind, knn24, params, w0, lam, nu, seed)
    X = ref_tour = ref_pos = None
    el0 = F._edge_lengths(np.ascontiguousarray(btour), np.ascontiguousarray(np.roll(btour, -1)), xs, ys, kind)
    inv_scale = 1.0 / max(float(np.median(el0)), 1e-12)
    start = time.perf_counter()
    elapsed = 0.0
    while True:
        elapsed = time.perf_counter() - start
        while checkpoints and elapsed >= checkpoints[0]:
            trace[str(checkpoints.pop(0))] = cost
        if elapsed >= budget:
            break
        due = stats[1] != accepted_at_refresh or stats[0] - kicks_at_refresh >= refresh_kicks
        if due and (refreshes == 0 or model_time <= overhead_cap * max(elapsed, 1e-9)):
            t1 = time.perf_counter()
            if stats[1] != accepted_at_refresh or refreshes == 0:
                X = pol.features(btour)
                ref_tour = btour.copy()
                ref_pos = bpos.copy()
            w = pol.sample_w()
            s = X[:, :D - 1] @ w[:D - 1]
            idx = np.argpartition(-s, m - 1)[:m]
            pool_u[:] = btour[idx]
            pool_v[:] = btour[(idx + 1) % n]
            pool_m = m
            refreshes += 1
            accepted_at_refresh = stats[1]
            kicks_at_refresh = stats[0]
            model_time += time.perf_counter() - t1
        tb = time.perf_counter()
        cost = run_kicks_online(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, batch,
                                True, pool_u, pool_v, pool_m, prob, queue, inq, qht, buf, stats,
                                X, ref_tour, ref_pos, inv_scale, pol.A, pol.b, learn)
        dt = time.perf_counter() - tb
        if dt < batch_target_s * 0.5:
            batch = min(batch * 2, 1 << 20)
        elif dt > batch_target_s * 2 and batch > 1:
            batch = max(1, batch // 2)
    for t in checkpoints:
        trace[str(t)] = cost
    exact = float(F.tour_cost(btour, xs, ys, kind))
    if abs(exact - cost) > 1e-6 * max(1.0, abs(exact)):
        raise RuntimeError(f"cost drift: tracked {cost} vs exact {exact}")
    if len(np.unique(btour)) != n:
        raise RuntimeError("invalid tour")
    w_final = np.linalg.solve(pol.A, pol.b)
    return {
        "strategy": "online_global", "seed": seed, "cost": exact, "initial_cost": init_cost,
        "elapsed": elapsed, "kicks": int(stats[0]), "accepted": int(stats[1]),
        "guided_cuts": int(stats[2]), "stale_cuts": int(stats[3]), "learned_kicks": int(stats[4]),
        "model_time": model_time, "refreshes": refreshes, "trace": trace,
        "w_final": w_final.tolist(), "hyper": {"lam": lam, "nu": nu, "refresh_kicks": refresh_kicks,
                                              "warm": w0 is not None, "learn": learn},
    }


def warmup(params):
    rng = np.random.default_rng(0)
    for kind in (F.EUC, F.GEO):
        xs = rng.uniform(30, 50, 80)
        ys = rng.uniform(0, 20, 80)
        nbr = F.build_knn(xs, ys, kind, 8)
        knn24 = F.build_knn(xs, ys, kind, 24)
        t0, c0, _ = F.start_tour(xs, ys, kind, nbr, 0)
        run_online(xs, ys, kind, nbr, t0, c0, 0, 0.05, knn24, params)


def fit_prior():
    """Warm start w0: ridge fit of the gain target (cheap_scorer._gain_rows, same 100 training
    instances as the cheap models) on the standardised features, rescaled so that predictions are
    in reward units only up to a positive factor (only the ranking of edges matters for the pool)."""
    from cheap_scorer import _gain_rows, load_params
    p = load_params()
    files = sorted((HERE / "gnn_train_data_lkh").glob("inst_*.npz"))
    perm = np.random.default_rng(2026).permutation(len(files))
    tr = [files[i] for i in perm[20:]]
    X, y, _ = _gain_rows(tr)
    Z = np.column_stack([(X - p["mu"]) / p["sd"], np.ones(len(X))])
    t = np.log1p(y)
    w = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(D), Z.T @ t)
    np.savez(PRIOR, w0=w)
    print("prior written", PRIOR, np.round(w, 3))


def load_prior():
    return np.load(PRIOR)["w0"] if PRIOR.exists() else None


def smoke():
    from cheap_scorer import load_params
    from kick_targeting_experiment import load_xy
    params = load_params()
    warmup(params)
    xs, ys, kind = load_xy(HERE / "data_vlsi" / "dka1376.tsp")
    n = len(xs)
    nbr = F.build_knn(xs, ys, kind, 10)
    knn24 = F.build_knn(xs, ys, kind, 24)
    init, c0, _ = F.start_tour(xs, ys, kind, nbr, 1)
    import vlsi_datasets as v
    bks = float(v.CATALOG["dka1376"][1])
    for name, kw in (("online_cold", {}), ("online_warm", {"w0": load_prior()}),
                     ("online_nolearn_warm", {"w0": load_prior(), "learn": False})):
        r = run_online(xs, ys, kind, nbr, init, c0, 104729, 2.0, knn24, params, **kw)
        print(name, f"gap={100 * (r['cost'] - bks) / bks:.3f}", r["kicks"], r["accepted"], r["learned_kicks"],
              r["refreshes"], round(r["model_time"], 3))
    for s in ("random_global", "knn_global"):
        r = F.run_strategy(xs, ys, kind, nbr, init, c0, s, 104729, 2.0, knn24=knn24)
        print(s, f"gap={100 * (r['cost'] - bks) / bks:.3f}", r["kicks"], r["accepted"])


if __name__ == "__main__":
    {"fit_prior": fit_prior, "smoke": smoke}[sys.argv[1]]()
