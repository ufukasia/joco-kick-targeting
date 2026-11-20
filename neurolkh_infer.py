# -*- coding: utf-8 -*-
"""NeuroLKH edge scores -- step 2/2 (network forward pass with the authors' pretrained SGN).

Imports SparseGCNModel from the unmodified NeuroLKH repository (third_party/NeuroLKH) and loads
pretrained/neurolkh_m.pt (trained by the authors on uniform, clustered and mixed instances; the model
they use for TSPLIB).  The forward call and the read-out (y_edges[..., 1], i.e. the predicted
probability that the edge i -> edge_index[i, k] belongs to an optimal tour) follow their
tsplib_test.py.  Output per instance: neighbour list, edge probabilities, forward wall time
(median of 3 timed passes after one warm-up, GPU synchronised, host->device transfer included).

    py -3.14 neurolkh_infer.py [--device cuda] [--model neurolkh_m.pt]
"""
from __future__ import annotations

import argparse
import hashlib
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE / "third_party" / "NeuroLKH"
OUT = HERE / "results" / "kick_targeting" / "neurolkh_scores"
sys.path.insert(0, str(REPO))
from net.sgcn_model import SparseGCNModel  # noqa: E402  (authors' code, unmodified)

N_EDGES = 20


def forward(net, g, device):
    n = g["x"].shape[0]
    node_feat = torch.as_tensor(g["x"], dtype=torch.float32, device=device).view(1, n, 2)
    edge_feat = torch.as_tensor(g["edge_feat"], dtype=torch.float32, device=device).view(1, -1, 1)
    edge_index = torch.as_tensor(g["edge_index"], dtype=torch.long, device=device).view(1, -1)
    inverse = torch.as_tensor(g["inverse"], dtype=torch.long, device=device).view(1, -1)
    with torch.no_grad():
        y_edges, _, _ = net.forward(node_feat, edge_feat, edge_index, inverse, None, None, N_EDGES)
    return y_edges[0, :, 1].view(n, N_EDGES).exp()  # stored as log-probabilities in the model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model", default="neurolkh_m.pt")
    a = ap.parse_args()
    mpath = REPO / "pretrained" / a.model
    net = SparseGCNModel()
    saved = torch.load(mpath, map_location="cpu", weights_only=False)
    net.load_state_dict(saved["model"])
    net.to(a.device)
    net.eval()
    msha = hashlib.sha256(mpath.read_bytes()).hexdigest()
    commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    tag = Path(a.model).stem
    for f in sorted(OUT.glob("*.graph.npz")):
        nm = f.name[: -len(".graph.npz")]
        out = OUT / f"{nm}.{tag}.npz"
        if out.exists():
            continue
        g = dict(np.load(f))
        prob = forward(net, g, a.device)  # warm-up
        times = []
        for _ in range(3):
            if a.device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            prob = forward(net, g, a.device)
            if a.device == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        p = prob.cpu().numpy().astype(np.float32)
        np.savez(out, edge_index=g["edge_index"], prob=p, t_forward_s=statistics.median(times),
                 device=a.device, gpu=torch.cuda.get_device_name(0) if a.device == "cuda" else "cpu",
                 model=a.model, model_sha256=msha, neurolkh_commit=commit, torch=torch.__version__)
        print(f"{nm} n={len(p)} t={statistics.median(times):.3f}s rowsum={p.sum(1).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
