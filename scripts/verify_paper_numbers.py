# -*- coding: utf-8 -*-
"""Recompute the paper's headline numbers from the raw run records and check them against the
evidence ledger; also check that every `% E…` / `% V…` comment in the paper points to a record.

    python scripts/verify_paper_numbers.py          (from the repository root; exit 0 = all match)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import v2_analysis as A  # noqa: E402

WS = ROOT / "makale_workspace" / "gnn-crine-ils-joco"
EV = {r["id"]: r for r in json.loads((WS / "evidence" / "evidence.json").read_text(encoding="utf-8"))["kayitlar"]}
EXC, RND, GNN, GAIN = "excess_global@0.06:0.5", "random_global", "gnn_global@0.04:0.5", "gain_global"
ON, LIN, LEN = "onlinewarm_global", "linprior_global", "length_global@0.06:0.5"


def mean_gap(tag, s):
    data, _ = A.per_instance(tag)
    return float(np.mean([data[k][s] for k in data]))


def win_rate(tag, a, b):
    data, _ = A.per_instance(tag)
    d = np.array([data[k][a] - data[k][b] for k in data])
    return float((d < 0).mean())


CHECKS = [  # (evidence id, description, recomputation from raw per-run JSON files)
    ("E599", "V1 online - static prior (holdout2)", lambda: A.paired("holdout2_10s", ON, LIN)["mean"]),
    ("E600", "V2 excess proxy - random", lambda: A.paired("holdout2_10s", EXC, RND)["mean"]),
    ("E601", "V3 excess proxy - GNN", lambda: A.paired("holdout2_10s", EXC, GNN)["mean"]),
    ("E602", "V4 excess proxy - gain-MLP", lambda: A.paired("holdout2_10s", EXC, GAIN)["mean"]),
    ("E608", "mean gap, random", lambda: mean_gap("holdout2_10s", RND)),
    ("E610", "mean gap, excess proxy", lambda: mean_gap("holdout2_10s", EXC)),
    ("E789", "share of instances where proxy beats GNN (41/54)", lambda: win_rate("holdout2_10s", EXC, GNN)),
    ("E965", "raw length - random (exploratory)", lambda: A.paired("holdout2_length_10s", LEN, RND)["mean"]),
    ("E972", "raw length - excess proxy (exploratory)", lambda: A.paired("holdout2_length_10s", LEN, EXC)["mean"]),
    ("E979", "excess proxy - random, length session", lambda: A.paired("holdout2_length_10s", EXC, RND)["mean"]),
    # exploratory robustness block (PROTOCOL_v3_exploratory.json)
    ("E991", "GNN - random at 0.1 s (anytime, confirmatory runs)", lambda: A.paired("holdout2_10s", GNN, RND, "0.1")["mean"]),
    ("E1047", "targeted global kick - random segment kick", lambda: A.paired("holdout2_local_10s", EXC, "random_local")["mean"]),
    ("E1051", "random segment kick - random global kick", lambda: A.paired("holdout2_local_10s", "random_local", RND)["mean"]),
    ("E1077", "excess proxy - random at 60 s", lambda: A.paired("holdout2_60s", EXC, RND)["mean"]),
    ("E1078", "excess proxy - GNN at 60 s", lambda: A.paired("holdout2_60s", EXC, GNN)["mean"]),
    ("E1091", "pool rule f=0.03, p=1 - selected rule", lambda: A.paired("holdout2_rule_10s", "excess_global@0.03:1.0", EXC)["mean"]),
    ("E1038", "Spearman: clustering index vs (excess - random)", lambda: structure_rho()),
]


def structure_rho():
    from scipy import stats
    import fast_ils as F
    import v2_targeting_experiment as V
    data, _ = A.per_instance("holdout2_10s")
    cv, d = [], []
    for nm, path, _b in V.instances("holdout2"):
        xs, ys, kind = V.load_xy(Path(path))
        knn = F.build_knn(xs, ys, kind, 2)
        idx = np.arange(len(xs), dtype=np.int32)
        d1 = F._edge_lengths(idx, np.ascontiguousarray(knn[:, 0], dtype=np.int32), xs, ys, kind)
        cv.append(d1.std() / d1.mean())
        d.append(data[nm][EXC] - data[nm][RND])
    return float(stats.spearmanr(cv, d)[0])


def main():
    bad = 0
    for rid, desc, fn in CHECKS:
        got, want = fn(), float(EV[rid]["deger"])
        ok = abs(got - want) < 1e-9
        bad += not ok
        print(f"[{'OK ' if ok else 'BAD'}] {rid} {desc}: recomputed {got:+.6f}, ledger {want:+.6f}")
    confirm = json.loads((ROOT / "results/kick_targeting/holdout2_10s_confirm.json").read_text(encoding="utf-8"))
    for h, rid in zip(confirm, ["E599", "E600", "E601", "E602"]):
        ok = h["rejected"] and abs(h["mean"] - float(EV[rid]["deger"])) < 1e-9
        bad += not ok
        print(f"[{'OK ' if ok else 'BAD'}] {h['id']} ({h['test']}): p = {h['p']:.2g}, H0 rejected = {h['rejected']}")
    ids = set()
    for tex in list((WS / "sections").glob("*.tex")) + [WS / "paper" / "main.tex"]:
        for line in tex.read_text(encoding="utf-8").splitlines():
            if "%" in line:
                ids |= set(re.findall(r"\b([EV]\d{3,4})\b", line.split("%", 1)[1]))
    missing = sorted(i for i in ids if i not in EV)
    bad += bool(missing)
    print(f"[{'OK ' if not missing else 'BAD'}] {len(ids)} evidence ids cited in the paper; missing from ledger: {missing}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
