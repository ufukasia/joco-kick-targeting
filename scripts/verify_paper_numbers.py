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
    # sensitivity: V1-V4 without the three TSPLIB instances used in Phase 1 (51 instances)
    ("E1100", "V1 without TSPLIB3", lambda: mean_without_tsplib(ON, LIN)),
    ("E1101", "V2 without TSPLIB3", lambda: mean_without_tsplib(EXC, RND)),
    ("E1102", "V3 without TSPLIB3", lambda: mean_without_tsplib(EXC, GNN)),
    ("E1103", "V4 without TSPLIB3", lambda: mean_without_tsplib(EXC, GAIN)),
]
TSPLIB3 = {"pcb3038", "fl3795", "fnl4461"}

# reviewer-driven exploratory block (PROTOCOL_v4_exploratory.json); ids resolved from evidence/v5_ids.json
V5_IDS = WS / "evidence" / "v5_ids.json"
if V5_IDS.exists():
    _ids = json.loads(V5_IDS.read_text(encoding="utf-8"))
    _sel = json.loads((ROOT / "results/kick_targeting/v5_selected_rules.json").read_text(encoding="utf-8"))
    _nl = "neurolkh_global@" + _sel["neurolkh"]["selected"]
    _nlf = _nl.replace("neurolkh_", "neurolkhfree_")
    _ga = GAIN if _sel["gain"]["selected"] == "0.06:0.75" else "gain_global@" + _sel["gain"]["selected"]
    for _t, _a, _b, _d in [
        ("holdout2_v5_equal_10s", EXC, _nl, "S2 excess - NeuroLKH (forward time charged)"),
        ("holdout2_v5_equal_10s", EXC, _nlf, "S2 excess - NeuroLKH (not charged)"),
        ("holdout2_v5_equal_10s", EXC, _ga, "S2 excess - tuned gain-MLP"),
        ("holdout2_v5_equal_10s", EXC, RND, "S2 excess - random"),
        ("holdout2_v5_kicks20k", EXC, _nl, "S1 excess - NeuroLKH (20 000 kicks)"),
        ("holdout2_v5_kicks20k", EXC, GNN, "S1 excess - GNN (20 000 kicks)"),
        ("holdout2_v5_kicks20k", EXC, RND, "S1 excess - random (20 000 kicks)"),
        ("holdout2_v5_credit_10s", "onlinepos_global", LIN, "S3 online max(0,r) - static prior"),
        ("holdout2_v5_credit_10s", "onlinesucc_global", LIN, "S3 online 1[accepted] - static prior"),
        ("holdout2_v5_single_10s", EXC, RND, "S4 excess - random (one worker)"),
        ("holdout2_lkh_10s", "lkh_ils_excess", "lkh_ils_random", "S5 LKH-3 targeted - random kick"),
        ("holdout2_lkh_10s", "lkh_ils_random", "lkh_default", "S5 LKH-3 ILS random - default"),
    ]:
        _k = f"v5.{_t and {'holdout2_v5_equal_10s': 'S2', 'holdout2_v5_kicks20k': 'S1', 'holdout2_v5_credit_10s': 'S3', 'holdout2_v5_single_10s': 'S4', 'holdout2_lkh_10s': 'S5'}[_t]}.{_a}-{_b}"
        if _k in _ids:
            CHECKS.append((_ids[_k], _d, (lambda t=_t, a=_a, b=_b: A.paired(t, a, b)["mean"])))


def mean_without_tsplib(a, b):
    data, _ = A.per_instance("holdout2_10s")
    d = [data[k][a] - data[k][b] for k in data if k not in TSPLIB3]
    assert len(d) == 51
    return float(np.mean(d))


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
