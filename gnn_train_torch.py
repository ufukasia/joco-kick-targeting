# -*- coding: utf-8 -*-
"""GPU trainer for the Trained GNN-Guided Local Search (PyTorch/CUDA).

Same model, same weights file, new engine: this trains EXACTLY the
architecture defined in gnn_trained.py (GraphSAGE-style message passing,
node/edge heads) but in PyTorch on the GPU, then exports the best weights
back to the pure-numpy .npz format via gnn_trained.save_weights — so the
inference path (gnn_trained.py / ngls.py, no torch dependency) is unchanged.

Data can come from gnn_generate_data.py (LKH/ILS backbone labels) or
crine_generate_data.py (counterfactual residual-improvement node labels).
Graphs are batched by concatenation (tour prev/next as explicit index arrays
instead of np.roll) so hundreds of variable-size graphs train as one gather.

`--parity` verifies the torch forward pass against gnn_trained.model_forward
in float64 (worst |Δlogit| must be ~1e-9) before you trust a training run.

Usage:
  python gnn_train_torch.py --data gnn_train_data_lkh --epochs 150
  python gnn_train_torch.py --parity
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import tsplib_engine as E  # noqa: F401  (sys.path side effect)
import gnn_trained as gt
from gnn_train import load_graphs

HERE = Path(__file__).resolve().parent


def _log(msg: str) -> None:
    print(msg, file=sys.__stderr__, flush=True)


# ---------------------------------------------------------------------------
#  Weights <-> torch parameters (same names/shapes as gnn_trained.init_weights)
# ---------------------------------------------------------------------------
def params_from_numpy(w: dict[str, np.ndarray], device, dtype) -> dict[str, torch.Tensor]:
    return {k: torch.tensor(v, device=device, dtype=dtype, requires_grad=True)
            for k, v in w.items()}


def params_to_numpy(p: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in p.items()}


# ---------------------------------------------------------------------------
#  Batching: concatenate variable-size graphs into one big gather-graph
# ---------------------------------------------------------------------------
def make_batch(graphs: list[dict], device, dtype) -> dict[str, torch.Tensor]:
    """Concatenates graphs; tour neighbours become explicit global prev/next
    index arrays (replacing np.roll) and kNN positions are offset per graph.
    `w_graph` holds 1/(n_g * n_graphs) per node so the batch loss reproduces
    the numpy trainer's mean-per-graph-then-mean-over-graphs semantics."""
    k_max = max(g["nbr_pos"].shape[1] for g in graphs)
    Xs, efs, nbrs, ws, prevs, nexts, y_ns, y_es, wgs = [], [], [], [], [], [], [], [], []
    off = 0
    for g in graphs:
        n, k = g["nbr_pos"].shape
        Xs.append(g["X"])
        efs.append(g["edge_feats"])
        nbr = g["nbr_pos"]
        gw = g["geo_w"]
        if k < k_max:  # pad with self-loops of zero weight
            pad = np.tile(np.arange(n)[:, None], (1, k_max - k))
            nbr = np.concatenate([nbr, pad], axis=1)
            gw = np.concatenate([gw, np.zeros((n, k_max - k))], axis=1)
        nbrs.append(nbr + off)
        ws.append(gw)
        idx = np.arange(n)
        prevs.append(np.roll(idx, 1) + off)
        nexts.append(np.roll(idx, -1) + off)
        y_ns.append(g["y_node"])
        y_es.append(g["y_edge"])
        wgs.append(np.full(n, 1.0 / (n * len(graphs))))
        off += n
    to = lambda a, dt=dtype: torch.tensor(np.concatenate(a), device=device, dtype=dt)
    return {
        "X": to([np.asarray(x) for x in Xs]),
        "edge_feats": to([np.asarray(e) for e in efs]),
        "nbr_pos": to(nbrs, torch.long),
        "geo_w": to(ws),
        "prev": to(prevs, torch.long),
        "next": to(nexts, torch.long),
        "y_node": to(y_ns),
        "y_edge": to(y_es),
        "w_graph": to(wgs),
    }


def torch_forward(p: dict[str, torch.Tensor], b: dict[str, torch.Tensor]):
    """Torch replica of gnn_trained.model_forward on a (batched) gather-graph."""
    H = torch.relu(b["X"] @ p["W_in"] + p["b_in"])
    for l in range(gt.MP_LAYERS):
        T = 0.5 * (H[b["prev"]] + H[b["next"]])
        G = (H[b["nbr_pos"]] * b["geo_w"].unsqueeze(-1)).sum(dim=1)
        H = torch.relu(H @ p[f"Ws{l}"] + T @ p[f"Wt{l}"] + G @ p[f"Wg{l}"] + p[f"b{l}"])
    node_logit = H @ p["w_node"] + p["b_node"][0]
    Hj = H[b["next"]]
    E_in = torch.cat([H * Hj, torch.abs(H - Hj), b["edge_feats"]], dim=1)
    E_h = torch.relu(E_in @ p["W_e1"] + p["b_e1"])
    edge_logit = E_h @ p["w_edge"] + p["b_edge"][0]
    return node_logit, edge_logit


_bce = torch.nn.functional.binary_cross_entropy_with_logits


def batch_loss(p, b, node_w: float, edge_w: float) -> torch.Tensor:
    node_logit, edge_logit = torch_forward(p, b)
    l_node = (_bce(node_logit, b["y_node"], reduction="none") * b["w_graph"]).sum()
    l_edge = (_bce(edge_logit, b["y_edge"], reduction="none") * b["w_graph"]).sum()
    return node_w * l_node + edge_w * l_edge


# ---------------------------------------------------------------------------
#  Parity: torch forward must reproduce the numpy forward bit-for-bit-ish
# ---------------------------------------------------------------------------
def parity_check(seed: int = 3) -> float:
    rng = np.random.default_rng(seed)
    worst = 0.0
    for n in (24, 120, 301):
        xs = rng.uniform(0, 1000, n); ys = rng.uniform(0, 1000, n)
        tour = np.arange(n); rng.shuffle(tour)
        d2 = np.hypot(xs[:, None] - xs[None, :], ys[:, None] - ys[None, :])
        np.fill_diagonal(d2, np.inf)
        knn = np.argsort(d2, axis=1)[:, :max(gt.K_GEO, gt.K_DENSITY)]
        g = gt.compute_graph_tensors(tour, xs, ys, False, xs, ys, knn)
        _, node_np, edge_np = gt.model_forward(gt.init_weights(seed), g)

        g["y_node"] = np.zeros(n); g["y_edge"] = np.zeros(n)
        p = params_from_numpy(gt.init_weights(seed), "cpu", torch.float64)
        b = make_batch([g], "cpu", torch.float64)
        node_t, edge_t = torch_forward(p, b)
        worst = max(worst,
                    float(np.abs(node_t.detach().numpy() - node_np).max()),
                    float(np.abs(edge_t.detach().numpy() - edge_np).max()))
    return worst


# ---------------------------------------------------------------------------
#  Metrics (torch mirror of gnn_train.evaluate)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(p, graphs: list[dict], device, dtype,
             node_w: float, edge_w: float, batch_size: int) -> dict:
    tot_loss = 0.0
    node_acc = edge_acc = 0.0
    rank_ratio = []
    for i in range(0, len(graphs), batch_size):
        chunk = graphs[i:i + batch_size]
        b = make_batch(chunk, device, dtype)
        node_logit, edge_logit = torch_forward(p, b)
        l_node = (_bce(node_logit, b["y_node"], reduction="none") * b["w_graph"]).sum()
        l_edge = (_bce(edge_logit, b["y_edge"], reduction="none") * b["w_graph"]).sum()
        tot_loss += float(node_w * l_node + edge_w * l_edge) * len(chunk)
        nl = node_logit.cpu().numpy(); el = edge_logit.cpu().numpy()
        off = 0
        for g in chunk:
            n = len(g["y_node"])
            gn = nl[off:off + n]; ge = el[off:off + n]
            node_acc += float(np.mean((gn > 0) == (g["y_node"] > 0.5)))
            edge_acc += float(np.mean((ge > 0) == (g["y_edge"] > 0.5)))
            k = max(1, n // 10)
            worst_idx = np.argsort(ge)[:k]
            overall = float(np.mean(g["y_edge"]))
            if overall > 1e-9:
                rank_ratio.append(float(np.mean(g["y_edge"][worst_idx])) / overall)
            off += n
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
    ap.add_argument("--data", type=str, default="gnn_train_data_lkh")
    ap.add_argument("--out", type=str, default="models/gnn_trained.npz")
    ap.add_argument("--init-weights", type=str, default=None,
                    help="optional compatible checkpoint for fine-tuning")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch-size", type=int, default=32, help="graphs per batch")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--node-weight", type=float, default=1.0)
    ap.add_argument("--edge-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true", help="force CPU (debug)")
    ap.add_argument("--parity", action="store_true",
                    help="check torch forward against the numpy model and exit")
    args = ap.parse_args()

    if args.parity:
        err = parity_check()
        print(f"parity check worst |logit diff| (float64): {err:.2e} "
              f"({'OK' if err < 1e-8 else 'FAIL'})")
        return

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and not args.cpu:
        _log("WARNING: CUDA not available, training on CPU")
    dtype = torch.float32
    torch.manual_seed(args.seed)

    data_dir = (HERE / args.data) if not Path(args.data).is_absolute() else Path(args.data)
    out_path = (HERE / args.out) if not Path(args.out).is_absolute() else Path(args.out)

    t0 = time.perf_counter()
    graphs = load_graphs(data_dir)
    _log(f"loaded {len(graphs)} graphs from {data_dir} "
         f"in {time.perf_counter() - t0:.1f}s  (device={device}"
         + (f", {torch.cuda.get_device_name(0)}" if device == "cuda" else "") + ")")

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

    init_path = ((HERE / args.init_weights)
                 if args.init_weights and not Path(args.init_weights).is_absolute()
                 else args.init_weights)
    initial = gt.get_weights(init_path)[0] if init_path else gt.init_weights(args.seed)
    p = params_from_numpy(initial, device, dtype)
    opt = torch.optim.Adam(p.values(), lr=args.lr)
    best_val = float("inf")
    best_weights = params_to_numpy(p)
    best_epoch = 0

    order = np.arange(len(train))
    t_train0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(order)
        t_ep = time.perf_counter()
        tot = 0.0
        for i in range(0, len(order), args.batch_size):
            chunk = [train[j] for j in order[i:i + args.batch_size]]
            b = make_batch(chunk, device, dtype)
            opt.zero_grad(set_to_none=True)
            loss = batch_loss(p, b, args.node_weight, args.edge_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(p.values(), 5.0)  # matches numpy Adam
            opt.step()
            tot += float(loss) * len(chunk)
        vm = evaluate(p, val, device, dtype, args.node_weight, args.edge_weight,
                      args.batch_size)
        marker = ""
        if vm["loss"] < best_val - 1e-6:
            best_val = vm["loss"]
            best_weights = params_to_numpy(p)
            best_epoch = epoch
            marker = "  *best*"
            gt.save_weights(out_path, best_weights,
                            meta={"checkpoint_epoch": epoch,
                                  "val_loss": round(best_val, 4)})
        susp = vm["suspicious_decile_ratio"]
        susp_text = f"{susp:.3f}" if susp is not None else "n/a"
        _log(f"epoch {epoch:3d}/{args.epochs}  train_loss={tot / len(train):.4f}  "
             f"val_loss={vm['loss']:.4f}  node_acc={vm['node_acc']:.3f}  "
             f"edge_acc={vm['edge_acc']:.3f}  "
             f"susp_decile={susp_text}  "
             f"t={time.perf_counter() - t_ep:.1f}s{marker}")

    p_best = params_from_numpy(best_weights, device, dtype)
    final_val = evaluate(p_best, val, device, dtype,
                         args.node_weight, args.edge_weight, args.batch_size)
    train_m = evaluate(p_best, train, device, dtype,
                       args.node_weight, args.edge_weight, args.batch_size)
    meta = {
        "trained": time.strftime("%Y-%m-%d %H:%M:%S"),
        "trainer": f"gnn_train_torch.py (torch {torch.__version__}, {device}"
                   + (f", {torch.cuda.get_device_name(0)}" if device == "cuda" else "")
                   + ")",
        "data_dir": str(data_dir),
        "init_weights": str(init_path) if init_path else "random",
        "n_instances": len(srcs), "n_train_graphs": len(train),
        "n_val_graphs": len(val), "epochs": args.epochs,
        "best_epoch": best_epoch, "lr": args.lr, "seed": args.seed,
        "batch_size": args.batch_size,
        "node_weight": args.node_weight, "edge_weight": args.edge_weight,
        "val_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in final_val.items()},
        "train_metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in train_m.items()},
        "labels": ("counterfactual residual improvement: true-cost gain from "
                   "single-anchor candidate expansion (crine_generate_data.py)"
                   if graphs and graphs[0].get("label_type") == "crine_residual"
                   else "backbone/edge-membership labels loaded from the dataset"),
        "train_wall_s": round(time.perf_counter() - t_train0, 1),
    }
    gt.save_weights(out_path, best_weights, meta=meta)
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"saved best weights (epoch {best_epoch}, val_loss={best_val:.4f}) "
         f"-> {out_path}")


if __name__ == "__main__":
    main()
