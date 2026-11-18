# -*- coding: utf-8 -*-
"""Unit tests of the online-targeting reward and pool rule (reviewer round 1, item B3).

The tested code (online_targeting.py) is hash-registered and is used unmodified.  The per-kick
reward is recovered from the ridge statistics: each learned kick adds r * xsum to b, and the bias
component of xsum is 3 (three cut edges, bias column = 1), so  delta b[bias] = 3 r.

    py -3.13 -m pytest -q -p no:cacheprovider test_online_reward.py
"""
import numpy as np
import pytest

import fast_ils as F
import online_targeting as OT
from online_targeting import D

N = 120


def _instance(seed=3):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(0, 1000, N)
    ys = rng.uniform(0, 1000, N)
    kind = F.EUC
    nbr = F.build_knn(xs, ys, kind, 10)
    return xs, ys, kind, nbr


def _state(tour):
    n = len(tour)
    tour = np.ascontiguousarray(tour, dtype=np.int32).copy()
    pos = np.empty(n, dtype=np.int32)
    pos[tour] = np.arange(n, dtype=np.int32)
    return {"tour": tour, "pos": pos, "btour": tour.copy(), "bpos": pos.copy(),
            "queue": np.empty(n, dtype=np.int32), "inq": np.zeros(n, dtype=np.uint8),
            "qht": np.zeros(3, dtype=np.int64), "buf": np.empty(n, dtype=np.int32),
            "stats": np.zeros(5, dtype=np.int64)}


def _features(n, seed=0):
    X = np.random.default_rng(seed).standard_normal((n, D))
    X[:, D - 1] = 1.0                       # bias column, as in OnlinePolicy.features
    return np.ascontiguousarray(X)


def _one_kick_original(xs, ys, kind, nbr, tour, X, inv_scale, kseed, guided=False, pool=None, prob=0.0):
    """One kick through the unmodified run_kicks_online; returns (r_recovered, db, returned cost, A delta)."""
    n = len(tour)
    st = _state(tour)
    cost = float(F.tour_cost(st["tour"], xs, ys, kind))
    pu, pv = (pool if pool is not None else (np.zeros(1, np.int32), np.zeros(1, np.int32)))
    A = 50.0 * np.eye(D)
    b = np.zeros(D)
    F.seed_rng(kseed)
    ret = OT.run_kicks_online(st["tour"], st["pos"], st["btour"], st["bpos"], cost, n, xs, ys, kind, nbr, 1,
                              guided, pu, pv, len(pu) if pool is not None else 0, prob,
                              st["queue"], st["inq"], st["qht"], st["buf"], st["stats"],
                              X, np.ascontiguousarray(tour, dtype=np.int32).copy(),
                              _pos(tour), inv_scale, A, b, True)
    return b[D - 1] / 3.0, b.copy(), ret, A - 50.0 * np.eye(D), cost, st


def _pos(tour):
    p = np.empty(len(tour), dtype=np.int32)
    p[np.asarray(tour)] = np.arange(len(tour), dtype=np.int32)
    return p


def _one_kick_reference(xs, ys, kind, nbr, tour, kseed):
    """Independent replica of the kick: same RNG seed, same primitives, exact post-LS cost by
    recomputing the whole tour length (not the incremental bookkeeping)."""
    n = len(tour)
    st = _state(tour)
    cuts = np.empty(3, dtype=np.int64)
    F.seed_rng(kseed)
    OT._kick_global_cuts(st["tour"], st["pos"], n, xs, ys, kind, False, np.zeros(1, np.int32),
                         np.zeros(1, np.int32), 0, 0.0, st["queue"], st["inq"], st["qht"], st["buf"],
                         st["stats"], cuts)
    F.local_search(st["tour"], st["pos"], n, xs, ys, kind, nbr, st["queue"], st["inq"], st["qht"], st["buf"])
    newc = float(F.tour_cost(st["tour"], xs, ys, kind))
    t0 = np.asarray(tour)
    edges = [(t0[p - 1], t0[p]) for p in cuts]
    return newc, cuts.copy(), edges


def _check_kick(xs, ys, kind, nbr, tour, kseed, X):
    cost0 = float(F.tour_cost(np.ascontiguousarray(tour, dtype=np.int32), xs, ys, kind))
    el = F._edge_lengths(np.ascontiguousarray(tour, dtype=np.int32),
                         np.ascontiguousarray(np.roll(tour, -1), dtype=np.int32), xs, ys, kind)
    inv_scale = 1.0 / float(np.median(el))
    r, db, ret, dA, cost, _ = _one_kick_original(xs, ys, kind, nbr, tour, X, inv_scale, kseed)
    newc, cuts, edges = _one_kick_reference(xs, ys, kind, nbr, tour, kseed)
    expected = -(newc - cost0) * inv_scale
    assert r == pytest.approx(expected, rel=1e-7, abs=1e-7)
    # full b update = r * (sum of the three cut edges' feature rows); bias part = 3
    rows = [int(c) - 1 for c in cuts]           # cut at position p removes edge (tour[p-1], tour[p]) = row p-1
    xsum = X[rows].sum(0)
    assert xsum[D - 1] == 3.0
    np.testing.assert_allclose(db, r * xsum, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(dA, np.outer(xsum, xsum), rtol=1e-12, atol=1e-12)
    # accepted iff improved; returned cost tracks the accepted tour
    if newc < cost0 - F.EPS:
        assert ret == pytest.approx(newc, rel=1e-9)
    else:
        assert ret == cost0
    return newc - cost0, r


def test_improving_kick_gets_positive_reward():
    xs, ys, kind, nbr = _instance()
    tour = np.random.default_rng(1).permutation(N).astype(np.int32)   # random tour: LS after any kick improves
    X = _features(N)
    dc, r = _check_kick(xs, ys, kind, nbr, tour, 11, X)
    assert dc < 0 and r > 0


def test_worsening_kick_gets_negative_reward_and_neutral_zero():
    xs, ys, kind, nbr = _instance()
    tour = F.full_local_search(F.nn_tour(xs, ys, kind, nbr, 0), xs, ys, kind, nbr)  # local optimum
    X = _features(N)
    seen = {"worse": 0, "neutral": 0, "better": 0}
    for kseed in range(60):
        dc, r = _check_kick(xs, ys, kind, nbr, tour, kseed, X)
        if dc > 1e-9:
            assert r < 0
            seen["worse"] += 1
        elif dc < -1e-9:
            assert r > 0
            seen["better"] += 1
        else:
            assert abs(r) < 1e-9
            seen["neutral"] += 1
    assert seen["worse"] > 0, seen


def test_pool_takes_highest_predicted_reward(monkeypatch):
    """run_online's pool = the m tour edges with the largest X.w (predicted reward)."""
    from cheap_scorer import MODEL, load_params
    params = load_params(MODEL)
    xs, ys, kind, nbr = _instance(5)
    knn24 = F.build_knn(xs, ys, kind, 24)
    tour = F.full_local_search(F.nn_tour(xs, ys, kind, nbr, 0), xs, ys, kind, nbr)
    c0 = float(F.tour_cost(tour, xs, ys, kind))
    w_fixed = np.random.default_rng(7).standard_normal(D)
    monkeypatch.setattr(OT.OnlinePolicy, "sample_w", lambda self: w_fixed)
    seen = []
    orig = OT.run_kicks_online

    def spy(tour_, pos, btour, bpos, cost, n, xs_, ys_, kind_, nbr_, n_kicks, guided, pool_u, pool_v,
            pool_m, *rest):
        X, ref_tour = rest[6], rest[7]
        seen.append((pool_u.copy(), pool_v.copy(), pool_m, X.copy(), ref_tour.copy()))
        return orig(tour_, pos, btour, bpos, cost, n, xs_, ys_, kind_, nbr_, 1, guided, pool_u, pool_v,
                    pool_m, *rest)

    monkeypatch.setattr(OT, "run_kicks_online", spy)
    OT.run_online(xs, ys, kind, nbr, tour, c0, 3, 0.02, knn24, params, w0=np.zeros(D))
    assert seen
    pu, pv, pm, X, ref = seen[0]
    m = F.pool_size(N)
    assert pm == m
    pred = X @ w_fixed                               # bias adds a constant: same ranking as X[:, :17] w
    top = set(map(int, np.argsort(-pred)[:m]))
    chosen = {int(np.where(ref == u)[0][0]) for u in pu}
    assert chosen == top
    for u, v in zip(pu, pv):                         # pool entries are the edges (ref[i], ref[i+1])
        i = int(np.where(ref == u)[0][0])
        assert ref[(i + 1) % N] == v
    assert pred[list(chosen)].min() >= pred[[i for i in range(N) if i not in chosen]].max()


def test_ridge_weight_follows_reward_sign():
    """Kicks whose cut edges have high feature k and positive reward push w_k up (and vice versa)."""
    rng = np.random.default_rng(0)
    A = 50.0 * np.eye(D)
    b = np.zeros(D)
    for _ in range(2000):
        x = rng.standard_normal(D)
        x[D - 1] = 3.0
        r = 0.8 * x[2] - 0.5 * x[5] + 0.05 * rng.standard_normal()
        A += np.outer(x, x)
        b += r * x
    w = np.linalg.solve(A, b)
    assert w[2] > 0.7 and w[5] < -0.4


def test_probe_kernel_is_bit_identical_to_original():
    """analysis_online_reward.run_kicks_probe = run_kicks_online + accounting only."""
    import analysis_online_reward as AR
    xs, ys, kind, nbr = _instance(9)
    tour = F.full_local_search(F.nn_tour(xs, ys, kind, nbr, 0), xs, ys, kind, nbr)
    cost = float(F.tour_cost(tour, xs, ys, kind))
    X = _features(N, 4)
    m = F.pool_size(N)
    pu = np.ascontiguousarray(tour[:m], dtype=np.int32)
    pv = np.ascontiguousarray(tour[1:m + 1], dtype=np.int32)
    el = F._edge_lengths(tour, np.ascontiguousarray(np.roll(tour, -1)), xs, ys, kind)
    inv = 1.0 / float(np.median(el))
    for learn in (True, False):
        outs = []
        for fn in ("orig", "probe"):
            st = _state(tour)
            A = 50.0 * np.eye(D)
            b = np.arange(D, dtype=float)
            F.seed_rng(123)
            args = (st["tour"], st["pos"], st["btour"], st["bpos"], cost, N, xs, ys, kind, nbr, 3000,
                    True, pu, pv, m, 0.75, st["queue"], st["inq"], st["qht"], st["buf"], st["stats"],
                    X, tour.copy(), _pos(tour), inv, A, b, learn)
            if fn == "orig":
                c = OT.run_kicks_online(*args)
                acc = None
            else:
                acc = np.zeros(AR.NACC)
                c = AR.run_kicks_probe(*args, acc)
            outs.append((c, st["btour"].copy(), st["stats"].copy(), A.copy(), b.copy(), acc))
        (c1, t1, s1, A1, b1, _), (c2, t2, s2, A2, b2, acc) = outs
        assert c1 == c2
        np.testing.assert_array_equal(t1, t2)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(A1, A2)
        np.testing.assert_array_equal(b1, b2)
        assert acc[AR.A_KICKS] == 3000 and acc[AR.A_IMPR] == s1[1]
        if learn:   # sum of r over learnable kicks == delta b[bias] / 3
            assert acc[AR.A_SUMR_ROWOK] == pytest.approx((b1[D - 1] - (D - 1)) / 3.0, rel=1e-9, abs=1e-9)
