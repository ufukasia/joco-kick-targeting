# -*- coding: utf-8 -*-
"""Trains the GNN for Trained-GNN-Guided Local Search (pure numpy, no torch).

Reads .npz instance files and converts every snapshot into one training graph.
Backbone datasets derive node targets from tour-edge frequencies; CRINE
datasets instead supply position-aligned `node_labels` directly. The model
from gnn_trained.py is fitted with Adam on the two supervised heads.

The best-validation-loss weights are exported to --out (default
models/gnn_trained.npz) together with a training-metadata JSON that the
benchmark runner embeds in each result file for reproducibility.

Usage:
  python gnn_train.py --data gnn_train_data --epochs 40
  python gnn_train.py --check-grads          # analytic-vs-numeric gradient test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import tsplib_engine as E  # inserts the sneak-yazilim dir into sys.path  # noqa: F401
import gnn_trained as gt

HERE = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    print(msg, file=sys.__stderr__, flush=True)


# ---------------------------------------------------------------------------
#  Dataset: npz instance files -> cached training graphs
# ---------------------------------------------------------------------------
def exact_knn(xs: np.ndarray, ys: np.ndarray, k: int) -> np.ndarray:
    """Exact kNN table via a full distance matrix (fine for training sizes)."""
    n = len(xs)
    d = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, : min(k, n - 1)].astype(np.int64)


def load_graphs(data_dir: Path) -> list[dict]:
    """Returns a list of training graphs, each with inputs + soft labels."""
    files = sorted(data_dir.glob("inst_*.npz"))
    if not files:
        raise SystemExit(f"no inst_*.npz files in {data_dir}; "
                         f"run gnn_generate_data.py first")
    graphs: list[dict] = []
    k_table = max(gt.K_GEO, gt.K_DENSITY)
    for path in files:
        with np.load(path) as z:
            coords = z["coords"]
            snapshots = z["snapshots"]
            edges = z["edges"]
            freqs = z["freqs"]
            node_labels = z["node_labels"].copy() if "node_labels" in z.files else None
        if node_labels is not None and node_labels.shape != snapshots.shape:
            raise ValueError(f"node_labels/snapshots shape mismatch in {path}")
        xs = coords[:, 0].astype(np.float64)
        ys = coords[:, 1].astype(np.float64)
        knn = exact_knn(xs, ys, k_table)
        freq_of = {(int(a), int(b)): float(f)
                   for (a, b), f in zip(edges.tolist(), freqs.tolist())}

        def f_edge(a: int, b: int) -> float:
            return freq_of.get((a, b) if a < b else (b, a), 0.0)

        for snapshot_idx, tour in enumerate(snapshots):
            tour = tour.astype(np.int64)
            g = gt.compute_graph_tensors(tour, xs, ys, False, xs, ys, knn)
            nxt = np.roll(tour, -1)
            prv = np.roll(tour, 1)
            f_next = np.array([f_edge(int(a), int(b)) for a, b in zip(tour, nxt)])
            f_prev = np.array([f_edge(int(a), int(b)) for a, b in zip(prv, tour)])
            g["y_edge"] = f_next
            g["y_node"] = (node_labels[snapshot_idx] if node_labels is not None
                           else 1.0 - 0.5 * (f_prev + f_next))
            g["label_type"] = "crine_residual" if node_labels is not None else "backbone"
            g["src"] = path.name
            graphs.append(g)
    return graphs


# ---------------------------------------------------------------------------
#  Adam
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, weights: dict[str, np.ndarray], lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                 clip: float = 5.0):
        self.lr, self.b1, self.b2, self.eps, self.clip = lr, beta1, beta2, eps, clip
        self.m = {k: np.zeros_like(v) for k, v in weights.items()}
        self.v = {k: np.zeros_like(v) for k, v in weights.items()}
        self.t = 0

    def step(self, weights: dict[str, np.ndarray],
             grads: dict[str, np.ndarray]) -> None:
        gnorm = np.sqrt(sum(float((g * g).sum()) for g in grads.values()))
        scale = self.clip / gnorm if gnorm > self.clip else 1.0
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, w in weights.items():
            g = grads[k] * scale
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            w -= self.lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + self.eps)


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
def evaluate(weights: dict[str, np.ndarray], graphs: list[dict],
             node_w: float, edge_w: float) -> dict:
    """Mean loss + hard-threshold accuracy + a simple ranking score (mean edge
    label in the model's most-suspicious decile vs the overall mean; below 1.0
    means the model concentrates genuinely non-backbone edges at the top)."""
    tot_loss = 0.0
    node_acc = edge_acc = 0.0
    rank_ratio = []
    for g in graphs:
        _, node_logit, edge_logit = gt.model_forward(weights, g)
        loss = (node_w * gt._bce_with_logits(node_logit, g["y_node"])
                + edge_w * gt._bce_with_logits(edge_logit, g["y_edge"]))
        tot_loss += loss
        node_acc += float(np.mean((node_logit > 0) == (g["y_node"] > 0.5)))
        edge_acc += float(np.mean((edge_logit > 0) == (g["y_edge"] > 0.5)))
        n = len(edge_logit)
        k = max(1, n // 10)
        worst = np.argsort(edge_logit)[:k]  # lowest predicted backbone prob
        overall = float(np.mean(g["y_edge"]))
        if overall > 1e-9:
            rank_ratio.append(float(np.mean(g["y_edge"][worst])) / overall)
    m = len(graphs)
    return {
        "loss": tot_loss / m,
        "node_acc": node_acc / m,
        "edge_acc": edge_acc / m,
        "suspicious_decile_ratio": (sum(rank_ratio) / len(rank_ratio)
                                    if rank_ratio else None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="gnn_train_data")
    ap.add_argument("--out", type=str, default="models/gnn_trained.npz")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--node-weight", type=float, default=1.0)
    ap.add_argument("--edge-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="wall-clock cap; stops after the current epoch (0 = off)")
    ap.add_argument("--check-grads", action="store_true")
    args = ap.parse_args()

    if args.check_grads:
        err = gt.gradient_check()
        print(f"gradient check worst relative error: {err:.2e} "
              f"({'OK' if err < 1e-4 else 'FAIL'})")
        return

    data_dir = (HERE / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    out_path = (HERE / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    t0 = time.perf_counter()
    graphs = load_graphs(data_dir)
    _log(f"loaded {len(graphs)} graphs from {data_dir} "
         f"in {time.perf_counter() - t0:.1f}s")

    # split by INSTANCE (not snapshot) so validation instances are unseen
    srcs = sorted({g["src"] for g in graphs})
    rng = np.random.default_rng(args.seed)
    rng.shuffle(srcs)
    n_val = max(1, int(len(srcs) * args.val_frac))
    val_srcs = set(srcs[:n_val])
    train = [g for g in graphs if g["src"] not in val_srcs]
    val = [g for g in graphs if g["src"] in val_srcs]
    _log(f"train graphs: {len(train)}  val graphs: {len(val)} "
         f"({n_val}/{len(srcs)} instances held out)")

    weights = gt.init_weights(args.seed)
    opt = Adam(weights, lr=args.lr)
    best_val = float("inf")
    best_weights = {k: v.copy() for k, v in weights.items()}
    best_epoch = 0

    order = np.arange(len(train))
    t_train0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        t_ep = time.perf_counter()
        tot = 0.0
        for i in order:
            g = train[i]
            loss, grads = gt.model_loss_and_grads(
                weights, g, g["y_node"], g["y_edge"],
                node_weight=args.node_weight, edge_weight=args.edge_weight)
            opt.step(weights, grads)
            tot += loss
        vm = evaluate(weights, val, args.node_weight, args.edge_weight)
        marker = ""
        if vm["loss"] < best_val - 1e-6:
            best_val = vm["loss"]
            best_weights = {k: v.copy() for k, v in weights.items()}
            best_epoch = epoch
            marker = "  *best*"
            # crash-safe checkpoint of the running best
            gt.save_weights(out_path, best_weights,
                            meta={"checkpoint_epoch": epoch,
                                  "val_loss": round(best_val, 4)})
        _log(f"epoch {epoch:3d}/{args.epochs}  train_loss={tot / len(train):.4f}  "
             f"val_loss={vm['loss']:.4f}  node_acc={vm['node_acc']:.3f}  "
             f"edge_acc={vm['edge_acc']:.3f}  "
             f"susp_decile={vm['suspicious_decile_ratio']:.3f}  "
             f"t={time.perf_counter() - t_ep:.1f}s{marker}")
        if (args.max_hours > 0
                and time.perf_counter() - t_train0 > args.max_hours * 3600):
            _log(f"max-hours cap ({args.max_hours}h) reached at epoch {epoch}; "
                 f"stopping (best epoch so far: {best_epoch})")
            break

    final_val = evaluate(best_weights, val, args.node_weight, args.edge_weight)
    train_m = evaluate(best_weights, train, args.node_weight, args.edge_weight)
    meta = {
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "n_instances": len(srcs), "n_train_graphs": len(train),
        "n_val_graphs": len(val), "epochs": args.epochs,
        "best_epoch": best_epoch, "lr": args.lr, "seed": args.seed,
        "node_weight": args.node_weight, "edge_weight": args.edge_weight,
        "val_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in final_val.items()},
        "train_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in train_m.items()},
        "labels": "backbone-consensus edge frequencies over the best half of "
                  "independent ILS local optima (synthetic instances only)",
    }
    gt.save_weights(out_path, best_weights, meta=meta)
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"saved best weights (epoch {best_epoch}, val_loss={best_val:.4f}) "
         f"-> {out_path}")


if __name__ == "__main__":
    main()
