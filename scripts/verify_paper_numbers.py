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
]


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
                ids |= set(re.findall(r"\b([EV]\d{3})\b", line.split("%", 1)[1]))
    missing = sorted(i for i in ids if i not in EV)
    bad += bool(missing)
    print(f"[{'OK ' if not missing else 'BAD'}] {len(ids)} evidence ids cited in the paper; missing from ledger: {missing}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
