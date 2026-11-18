# -*- coding: utf-8 -*-
"""Learned-but-cheap kick-targeting scorer (exploratory, added after the primary result).

Question: is the GNN's loss to the training-free kNN proxy caused by *learning* or by
*cost*?  This scorer is trained on exactly the GNN's training data (120 synthetic
instances, n in [100, 400], LKH-3 labels, gnn_train_data_lkh/) but uses only O(1)
per-edge features built from the same 24-NN lists the kNN proxy uses, so a refresh
costs about as much as the proxy.  Model: small MLP (or logistic regression), with
the forward pass in NumPy.  The object exposes edge_prob(tour) so it can be passed to
fast_ils.run_strategy in place of GNNScorer (fast_ils.py itself is unchanged).

    py -3.13 cheap_scorer.py train      # fits, reports held-out AUC, writes models/cheap_scorer.npz
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import fast_ils as F

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "cheap_scorer.npz"
K = 24
ABSENT = 30.0


def _ranks(a, b, knn):
    """rank of b in knn[a] (1-based), ABSENT if missing."""
    m = knn[a] == b[:, None]
    return np.where(m.any(1), m.argmax(1) + 1.0, ABSENT)


def features(tour, knn, xs, ys, kind, dk=None):
    """Per tour position i, features of edge (tour[i], tour[i+1]).  dk: (n,3) distances to
    the 1st, 5th and 24th neighbour (precomputable once per instance)."""
    n = len(tour)
    t = np.ascontiguousarray(tour, dtype=np.int32)
    u = t
    v = np.roll(t, -1)
    if dk is None:
        dk = neighbour_dists(knn, xs, ys, kind)
    duv = F._edge_lengths(u, np.ascontiguousarray(v), xs, ys, kind)
    ru = _ranks(u, v, knn)
    rv = _ranks(v, u, knn)
    rmin = np.minimum(ru, rv)
    rmax = np.maximum(ru, rv)
    eps = 1e-9
    l1u = np.log((duv + eps) / (dk[u, 0] + eps))
    l1v = np.log((duv + eps) / (dk[v, 0] + eps))
    l5 = np.log((duv + eps) / (0.5 * (dk[u, 1] + dk[v, 1]) + eps))
    l24 = np.log((duv + eps) / (0.5 * (dk[u, 2] + dk[v, 2]) + eps))
    # neighbouring tour edges (prev edge ends at u, next edge starts at v)
    dprev = np.roll(duv, 1)
    dnext = np.roll(duv, -1)
    rprev = np.roll(rmin, 1)
    rnext = np.roll(rmin, -1)
    X = np.column_stack([
        np.log(ru), np.log(rv), np.log(rmin), np.log(rmax),
        (rmax >= ABSENT).astype(float), (rmin <= 1).astype(float), (rmax <= 3).astype(float),
        l1u, l1v, np.maximum(l1u, l1v), l5, l24,
        np.log((dprev + eps) / (duv + eps)), np.log((dnext + eps) / (duv + eps)),
        np.log(rprev), np.log(rnext), np.log(np.maximum(rprev, rnext)),
    ])
    return X


def neighbour_dists(knn, xs, ys, kind):
    n = knn.shape[0]
    idx = np.arange(n, dtype=np.int32)
    cols = [0, min(4, knn.shape[1] - 1), knn.shape[1] - 1]
    return np.column_stack([F._edge_lengths(idx, np.ascontiguousarray(knn[:, c], dtype=np.int32), xs, ys, kind)
                            for c in cols])


class CheapScorer:
    def __init__(self, xs, ys, kind, params, knn=None):
        self.xs, self.ys, self.kind = xs, ys, kind
        self.knn = knn if knn is not None else F.build_knn(xs, ys, kind, min(K, len(xs) - 1))
        self.dk = neighbour_dists(self.knn, xs, ys, kind)
        self.p = params

    def logit(self, tour):
        X = features(tour, self.knn, self.xs, self.ys, self.kind, self.dk)
        X = (X - self.p["mu"]) / self.p["sd"]
        if "W1" in self.p:
            h = np.maximum(X @ self.p["W1"] + self.p["b1"], 0.0)
            h = np.maximum(h @ self.p["W2"] + self.p["b2"], 0.0)
            return (h @ self.p["W3"] + self.p["b3"]).ravel()
        return X @ self.p["coef"] + self.p["intercept"]

    def edge_prob(self, tour):
        # run_strategy kicks the edges with the SMALLEST value: for the gain model that is
        # the largest predicted gain.
        if "gain" in self.p:
            return -self.logit(tour)
        return 1.0 / (1.0 + np.exp(-self.logit(tour)))


def load_params(path=None):
    path = MODEL if path is None else path
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _training_rows(files):
    Xs, ys_ = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        c = z["coords"]
        xs = np.ascontiguousarray(c[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(c[:, 1], dtype=np.float64)
        kind = F.EUC
        n = len(xs)
        knn = F.build_knn(xs, ys, kind, min(K, n - 1))
        dk = neighbour_dists(knn, xs, ys, kind)
        opt = {(min(a, b), max(a, b)) for a, b in z["edges"]}
        for snap in z["snapshots"]:
            t = np.asarray(snap, dtype=np.int32)
            X = features(t, knn, xs, ys, kind, dk)
            v = np.roll(t, -1)
            lab = np.array([1 if (min(a, b), max(a, b)) in opt else 0 for a, b in zip(t, v)])
            Xs.append(X)
            ys_.append(lab)
    return np.vstack(Xs), np.concatenate(ys_)


def train():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    files = sorted((HERE / "gnn_train_data_lkh").glob("inst_*.npz"))
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(files))
    val = [files[i] for i in perm[:20]]
    tr = [files[i] for i in perm[20:]]
    t0 = time.time()
    Xtr, ytr = _training_rows(tr)
    Xva, yva = _training_rows(val)
    print(f"rows train={len(ytr)} val={len(yva)} pos-rate={ytr.mean():.3f}  ({time.time() - t0:.0f}s)")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
    lr = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    auc_lr = roc_auc_score(yva, lr.decision_function(Zva))
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=60, early_stopping=True,
                        random_state=0).fit(Ztr, ytr)
    auc_mlp = roc_auc_score(yva, mlp.predict_proba(Zva)[:, 1])
    # proxy baseline on the same rows: mean mutual rank (higher = more suspicious => negate)
    auc_knn = roc_auc_score(yva, -(np.exp(Xva[:, 0]) + np.exp(Xva[:, 1])))
    print(f"held-out AUC (in-tour)  LR={auc_lr:.4f}  MLP={auc_mlp:.4f}  kNN-rank={auc_knn:.4f}")
    params = {"mu": mu, "sd": sd, "coef": lr.coef_.ravel(), "intercept": np.array(lr.intercept_[0])}
    if auc_mlp > auc_lr + 0.002:
        params.update({"W1": mlp.coefs_[0], "b1": mlp.intercepts_[0], "W2": mlp.coefs_[1],
                       "b2": mlp.intercepts_[1], "W3": mlp.coefs_[2], "b3": mlp.intercepts_[2]})
        chosen = "mlp"
    else:
        chosen = "lr"
    np.savez(MODEL, **params)
    info = {"chosen": chosen, "auc_lr": auc_lr, "auc_mlp": auc_mlp, "auc_knn_rank": auc_knn,
            "n_train_rows": int(len(ytr)), "n_val_rows": int(len(yva)),
            "train_files": len(tr), "val_files": len(val), "n_features": int(Xtr.shape[1])}
    (HERE / "models" / "cheap_scorer.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(info)


GAIN_MODEL = HERE / "models" / "cheap_gain_scorer.npz"


def _gain_rows(files):
    """Target = wrong(edge) * max(0, excess length) / median tour-edge length, where excess is the
    edge length minus the mean length of the optimal-tour edges at its two endpoints."""
    Xs, ys_, ws = [], [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        c = z["coords"]
        xs = np.ascontiguousarray(c[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(c[:, 1], dtype=np.float64)
        kind = F.EUC
        n = len(xs)
        knn = F.build_knn(xs, ys, kind, min(K, n - 1))
        dk = neighbour_dists(knn, xs, ys, kind)
        E = np.asarray(z["edges"])
        opt = {(min(a, b), max(a, b)) for a, b in E}
        inc = np.zeros(n)
        cnt = np.zeros(n)
        de = F._edge_lengths(np.ascontiguousarray(E[:, 0], dtype=np.int32),
                             np.ascontiguousarray(E[:, 1], dtype=np.int32), xs, ys, kind)
        np.add.at(inc, E[:, 0], de)
        np.add.at(inc, E[:, 1], de)
        np.add.at(cnt, E[:, 0], 1)
        np.add.at(cnt, E[:, 1], 1)
        dref = inc / np.maximum(cnt, 1)
        for snap in z["snapshots"]:
            t = np.asarray(snap, dtype=np.int32)
            v = np.roll(t, -1)
            X = features(t, knn, xs, ys, kind, dk)
            duv = F._edge_lengths(t, np.ascontiguousarray(v), xs, ys, kind)
            wrong = np.array([0 if (min(a, b), max(a, b)) in opt else 1 for a, b in zip(t, v)])
            exc = np.maximum(duv - 0.5 * (dref[t] + dref[v]), 0.0) / np.median(duv)
            Xs.append(X)
            ys_.append(wrong * exc)
            ws.append(wrong)
    return np.vstack(Xs), np.concatenate(ys_), np.concatenate(ws)


def train_gain():
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPRegressor
    files = sorted((HERE / "gnn_train_data_lkh").glob("inst_*.npz"))
    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(files))
    val = [files[i] for i in perm[:20]]
    tr = [files[i] for i in perm[20:]]
    Xtr, ytr, _ = _gain_rows(tr)
    Xva, yva, wva = _gain_rows(val)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zva = (Xtr - mu) / sd, (Xva - mu) / sd
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=200, early_stopping=True,
                       random_state=0).fit(Ztr, np.log1p(ytr))
    pred = mlp.predict(Zva)
    knn_s = np.exp(Xva[:, 0]) + np.exp(Xva[:, 1])
    info = {"spearman_gain": float(spearmanr(pred, yva).statistic),
            "spearman_gain_knn": float(spearmanr(knn_s, yva).statistic),
            "auc_wrong": float(roc_auc_score(wva, pred)), "auc_wrong_knn": float(roc_auc_score(wva, knn_s)),
            "n_train_rows": int(len(ytr)), "target_mean": float(ytr.mean())}
    # mean target captured by the top-6 % of each instance's edges is the pool-relevant quantity
    np.savez(GAIN_MODEL, mu=mu, sd=sd, W1=mlp.coefs_[0], b1=mlp.intercepts_[0], W2=mlp.coefs_[1],
             b2=mlp.intercepts_[1], W3=mlp.coefs_[2], b3=mlp.intercepts_[2], gain=np.array(1.0))
    (HERE / "models" / "cheap_gain_scorer.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(info)


if __name__ == "__main__":
    if sys.argv[1:] == ["train"]:
        train()
    if sys.argv[1:] == ["train_gain"]:
        train_gain()
