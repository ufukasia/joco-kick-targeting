# -*- coding: utf-8 -*-
"""Online kick targeting with improvement-based credit -- exploratory block B5 (PROTOCOL_v4_exploratory).

online_targeting.py (hash-registered in PROTOCOL_v2) is not modified.  This module copies its kick
kernel and run loop verbatim and changes only the reward that enters the ridge update:

    credit = 0  signed reward  r = -(C_after - C_before) / median edge length   (original, Eq. 2)
    credit = 1  improvement credit  max(0, r): only kicks that shorten the tour are rewarded
    credit = 2  success credit  1[kick accepted]: binary success, as in adaptive operator selection

Credits 1 and 2 are the instantaneous improvement / success credits of the adaptive-operator-
selection literature (Fialho et al.).  Extreme-value credit is defined per discrete operator and has
no direct counterpart in this contextual linear model, so it is not included.

    py -3.13 online_credit.py check      # credit = 0 reproduces online_targeting exactly
"""
from __future__ import annotations

import time

import numpy as np
from numba import njit

import fast_ils as F
import online_targeting as OT
from fast_ils import EPS, local_search
from online_targeting import D, OnlinePolicy, _kick_global_cuts, _ref_row


@njit(cache=True)
def run_kicks_online_credit(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, n_kicks,
                            guided, pool_u, pool_v, pool_m, prob, queue, inq, qht, buf, stats,
                            X, ref_tour, ref_pos, inv_scale, A, bvec, learn, credit):
    """online_targeting.run_kicks_online with a selectable credit (see module docstring)."""
    cuts = np.empty(3, dtype=np.int64)
    rows = np.empty(3, dtype=np.int64)
    xsum = np.empty(X.shape[1])
    for _ in range(n_kicks):
        delta = _kick_global_cuts(tour, pos, n, xs, ys, kind, guided, pool_u, pool_v, pool_m,
                                  prob, queue, inq, qht, buf, stats, cuts)
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
            if credit == 1:
                r = max(r, 0.0)
            elif credit == 2:
                r = 1.0 if newc < cost - EPS else 0.0
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


def run_online_credit(xs, ys, kind, nbr, init_tour, init_cost, seed, budget, knn24, params, w0=None,
                      lam=50.0, nu=0.5, refresh_kicks=2000, prob=0.75, overhead_cap=0.10,
                      batch_target_s=0.004, learn=True, credit=0):
    """online_targeting.run_online, verbatim except for the credit argument of the kernel."""
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
        cost = run_kicks_online_credit(tour, pos, btour, bpos, cost, n, xs, ys, kind, nbr, batch,
                                       True, pool_u, pool_v, pool_m, prob, queue, inq, qht, buf, stats,
                                       X, ref_tour, ref_pos, inv_scale, pol.A, pol.b, learn, credit)
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
        "strategy": f"online_credit{credit}_global", "seed": seed, "cost": exact, "initial_cost": init_cost,
        "elapsed": elapsed, "kicks": int(stats[0]), "accepted": int(stats[1]),
        "guided_cuts": int(stats[2]), "stale_cuts": int(stats[3]), "learned_kicks": int(stats[4]),
        "model_time": model_time, "refreshes": refreshes, "trace": trace,
        "w_final": w_final.tolist(), "hyper": {"lam": lam, "nu": nu, "refresh_kicks": refresh_kicks,
                                              "warm": w0 is not None, "learn": learn, "credit": credit},
    }


def check():
    """Kernel equivalence: credit = 0 must reproduce online_targeting.run_kicks_online bit for bit;
    credit = 1 / 2 must leave the tour trajectory unchanged for the same pool and differ only in b."""
    from cheap_scorer import load_params
    from kick_targeting_experiment import HERE, load_xy
    params = load_params()
    xs, ys, kind = load_xy(HERE / "data_vlsi" / "dka1376.tsp")
    n = len(xs)
    nbr = F.build_knn(xs, ys, kind, 10)
    knn24 = F.build_knn(xs, ys, kind, 24)
    t0, c0, _ = F.start_tour(xs, ys, kind, nbr, 0)
    m = F.pool_size(n)
    res = {}
    for name in ("orig", 0, 1, 2):
        tour = t0.copy(); pos = np.empty(n, dtype=np.int32); pos[tour] = np.arange(n, dtype=np.int32)
        btour = tour.copy(); bpos = pos.copy()
        queue = np.empty(n, dtype=np.int32); inq = np.zeros(n, dtype=np.uint8)
        qht = np.zeros(3, dtype=np.int64); buf = np.empty(n, dtype=np.int32); stats = np.zeros(5, dtype=np.int64)
        pol = OnlinePolicy(xs, ys, kind, knn24, params, OT.load_prior(), 50.0, 0.5, 0)
        X = pol.features(btour); ref_tour = btour.copy(); ref_pos = bpos.copy()
        s = X[:, :D - 1] @ pol.sample_w()[:D - 1]
        idx = np.argpartition(-s, m - 1)[:m]
        pool_u = np.ascontiguousarray(btour[idx]); pool_v = np.ascontiguousarray(btour[(idx + 1) % n])
        F.seed_rng(11)
        args = (tour, pos, btour, bpos, c0, n, xs, ys, kind, nbr, 3000, True, pool_u, pool_v, m, 0.75,
                queue, inq, qht, buf, stats, X, ref_tour, ref_pos, 1.0, pol.A, pol.b, True)
        cost = OT.run_kicks_online(*args) if name == "orig" else run_kicks_online_credit(*args, name)
        res[name] = (cost, stats.copy(), pol.A.copy(), pol.b.copy(), btour.copy())
    o = res["orig"]
    same0 = o[0] == res[0][0] and np.array_equal(o[1], res[0][1]) and np.array_equal(o[2], res[0][2]) \
        and np.array_equal(o[3], res[0][3]) and np.array_equal(o[4], res[0][4])
    traj = all(o[0] == res[c][0] and np.array_equal(o[4], res[c][4]) for c in (1, 2))
    bdiff = all(not np.array_equal(o[3], res[c][3]) for c in (1, 2))
    print(f"credit 0 identical to original kernel: {same0}; credits 1/2 same trajectory: {traj}; "
          f"different ridge targets: {bdiff}; accepted {int(o[1][1])} of {int(o[1][0])} kicks")
    raise SystemExit(0 if (same0 and traj and bdiff) else 1)


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["check"]:
        check()
