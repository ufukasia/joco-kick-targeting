# -*- coding: utf-8 -*-
"""NeuroLKH (Xin et al., NeurIPS 2021) edge scores as a kick-targeting signal -- step 1/2 (input graph).

The Sparse Graph Network (SGN) and its pretrained weights are used unchanged from the authors'
repository (third_party/NeuroLKH, commit recorded in the output).  This step only builds the SGN
input exactly as in their tsplib_test.py::infer_SGN_write_candidate:
    coordinates normalised to the unit square (same formula), 20 nearest neighbours from a row-wise
    np.argsort of the Euclidean distance row (their dense code argsorts the same rows), edge feature =
    normalised distance, inverse edge index (-1 if the reverse edge is absent).
The dense n x n arrays of their script are replaced by row chunks; `selftest` checks that the result
is identical to their verbatim dense code on real instances small enough for it.

    py -3.13 neurolkh_prep.py selftest
    py -3.13 neurolkh_prep.py prep --suites holdout2,vlsi_primary,synth_1000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "kick_targeting" / "neurolkh_scores"
N_EDGES = 20


def normalise(xy):
    """Verbatim from NeuroLKH tsplib_test.py (x: n x 2 array of coordinates)."""
    x = np.array(xy, dtype=np.float64)
    scale = max(x[:, 0].max() - x[:, 0].min(), x[:, 1].max() - x[:, 1].min()) * (1 + 2 * 1e-4)
    x = x - x.min(0).reshape(1, 2)
    x = x / scale
    x = x + 1e-4
    if x[:, 0].max() > x[:, 1].max():
        x[:, 1] += (1 - 1e-4 - x[:, 1].max()) / 2
    else:
        x[:, 0] += (1 - 1e-4 - x[:, 0].max()) / 2
    return x


def sparse_graph(x, chunk=512):
    n = len(x)
    edge_index = np.empty((n, N_EDGES), dtype=np.int64)
    edge_feat = np.empty((n, N_EDGES), dtype=np.float64)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        d = x[a:b, None, :] - x[None, :, :]
        d = np.sqrt((d ** 2).sum(-1))
        idx = np.argsort(d, -1)[:, 1:1 + N_EDGES]
        edge_index[a:b] = idx
        edge_feat[a:b] = np.take_along_axis(d, idx, 1)
    # inverse[i, k] = flat index of edge (j -> i) where j = edge_index[i, k], or -1
    inverse = -np.ones((n, N_EDGES), dtype=np.int64)
    pos = {}
    for j in range(n):
        for k in range(N_EDGES):
            pos[(j, int(edge_index[j, k]))] = j * N_EDGES + k
    for i in range(n):
        for k in range(N_EDGES):
            inverse[i, k] = pos.get((int(edge_index[i, k]), i), -1)
    return edge_index, edge_feat, inverse


def dense_reference(x):
    """Their dense construction, copied from tsplib_test.py (batch_size = 1)."""
    n_nodes, n_edges, batch_size = len(x), N_EDGES, 1
    x = x.reshape(1, n_nodes, 2)
    dist = x.reshape(batch_size, n_nodes, 1, 2) - x.reshape(batch_size, 1, n_nodes, 2)
    dist = np.sqrt((dist ** 2).sum(-1))
    edge_index = np.argsort(dist, -1)[:, :, 1:1 + n_edges]
    edge_feat = dist[np.arange(batch_size).reshape(-1, 1, 1), np.arange(n_nodes).reshape(1, -1, 1), edge_index]
    inverse_edge_index = -np.ones(shape=[batch_size, n_nodes, n_nodes], dtype="int")
    inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1), edge_index, np.arange(n_nodes).reshape(1, -1, 1)] = \
        np.arange(n_edges).reshape(1, 1, -1) + np.arange(n_nodes).reshape(1, -1, 1) * n_edges
    inverse_edge_index = inverse_edge_index[np.arange(batch_size).reshape(-1, 1, 1),
                                            np.arange(n_nodes).reshape(1, -1, 1), edge_index]
    return edge_index[0], edge_feat[0], inverse_edge_index[0]


def instances(suite):
    sys.path.insert(0, str(HERE))
    if suite == "holdout2":
        import holdout2_suite
        return holdout2_suite.instances()
    from kick_targeting_experiment import suite_instances
    return suite_instances(suite)


def load(path):
    from kick_targeting_experiment import load_xy
    xs, ys, kind = load_xy(Path(path))
    assert kind == 0, "NeuroLKH's SGN input is defined for EUC_2D coordinates only"
    return np.column_stack([xs, ys])


def selftest():
    from kick_targeting_experiment import suite_instances
    ok = True
    for nm, p, _ in [r for r in suite_instances("synth_1000")[:2]] + [r for r in suite_instances("vlsi_primary")[:1]]:
        x = normalise(load(p))
        a = sparse_graph(x)
        b = dense_reference(x)
        same = all(np.array_equal(u, v) for u, v in zip(a, b))
        ok &= same
        print(f"{nm} n={len(x)} identical to dense reference: {same}")
    raise SystemExit(0 if ok else 1)


def prep(suites):
    OUT.mkdir(parents=True, exist_ok=True)
    for s in suites:
        for nm, p, _ in instances(s):
            f = OUT / f"{nm}.graph.npz"
            if f.exists():
                continue
            x = normalise(load(p))
            ei, ef, inv = sparse_graph(x)
            np.savez(f, x=x.astype(np.float32), edge_index=ei.astype(np.int32), edge_feat=ef.astype(np.float32),
                     inverse=inv.astype(np.int64))
            print("graph", s, nm, len(x), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "prep"))
    ap.add_argument("--suites", default="holdout2,vlsi_primary,synth_1000")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else prep(a.suites.split(","))
