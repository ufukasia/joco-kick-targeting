# -*- coding: utf-8 -*-
"""Correctness tests for fast_ils (run: py -3.13 -m pytest -q test_fast_ils.py)."""
import numpy as np

import fast_ils as F


def _inst(n=300, seed=0, kind=F.EUC):
    rng = np.random.default_rng(seed)
    if kind == F.EUC:
        xs = rng.uniform(0, 1000, n)
        ys = rng.uniform(0, 1000, n)
    else:
        xs = rng.uniform(30, 50, n)
        ys = rng.uniform(-10, 30, n)
    return xs, ys


def _check_tour(t, n):
    assert sorted(t.tolist()) == list(range(n))


def test_local_search_is_valid_and_improves():
    for kind in (F.EUC, F.GEO):
        xs, ys = _inst(kind=kind)
        nbr = F.build_knn(xs, ys, kind, 10)
        t = F.nn_tour(xs, ys, kind, nbr, 0)
        c0 = F.tour_cost(t, xs, ys, kind)
        t = F.full_local_search(t.copy(), xs, ys, kind, nbr)
        _check_tour(t, len(xs))
        assert F.tour_cost(t, xs, ys, kind) < c0


def test_two_opt_local_optimum():
    xs, ys = _inst(150, 3)
    nbr = F.build_knn(xs, ys, F.EUC, 10)
    t = F.full_local_search(F.nn_tour(xs, ys, F.EUC, nbr, 0), xs, ys, F.EUC, nbr)
    n = len(t)
    d = lambda a, b: F.dist(a, b, xs, ys, 0)
    # no improving 2-opt move among neighbour-list candidates
    pos = np.empty(n, dtype=int)
    pos[t] = np.arange(n)
    for i in range(n):
        a, b = t[i], t[(i + 1) % n]
        for c in nbr[a]:
            j = pos[c]
            cc, dd = t[j], t[(j + 1) % n]
            if len({a, b, cc, dd}) < 4:
                continue
            assert d(a, b) + d(cc, dd) - d(a, cc) - d(b, dd) <= 1e-6


def test_all_strategies_consistent_costs():
    F.warmup()
    xs, ys = _inst(400, 7)
    nbr = F.build_knn(xs, ys, F.EUC, 10)
    knn24 = F.build_knn(xs, ys, F.EUC, 24)
    t0, c0, _ = F.start_tour(xs, ys, F.EUC, nbr, 1)
    import gnn_trained as gt
    w, _ = gt.get_weights("models/gnn_trained.npz")
    scorer = F.GNNScorer(xs, ys, F.EUC, w)
    for s in F.STRATEGIES:
        r = F.run_strategy(xs, ys, F.EUC, nbr, t0, c0, s, 5, 0.5,
                           gnn=scorer, knn24=knn24)
        assert r["cost"] <= c0 + 1e-9
        assert r["kicks"] > 50, (s, r["kicks"])


def test_gnn_scorer_matches_reference_snapshot():
    import gnn_trained as gt
    xs, ys = _inst(200, 11)
    w, _ = gt.get_weights("models/gnn_trained.npz")
    scorer = F.GNNScorer(xs, ys, F.EUC, w)
    nbr = F.build_knn(xs, ys, F.EUC, 10)
    t, _, _ = F.start_tour(xs, ys, F.EUC, nbr, 0)
    p = scorer.edge_prob(t)
    snap = gt._compute_trained_snapshot(w, t.tolist(), xs, ys, False, xs, ys, scorer.knn)
    ref = np.empty(len(t))
    for q, i in snap[4]:
        ref[i] = q
    assert np.allclose(p, ref)
