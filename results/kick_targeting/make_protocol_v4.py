# -*- coding: utf-8 -*-
"""Writes PROTOCOL_v4_exploratory.json (+ .sha256) BEFORE any run of blocks B1-B6.

    py -3.13 results/kick_targeting/make_protocol_v4.py && ots stamp results/kick_targeting/PROTOCOL_v4_exploratory.json
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FILES = [
    "fast_ils.py", "v2_targeting_experiment.py", "online_targeting.py", "cheap_scorer.py",
    "kick_targeting_experiment.py", "holdout2_suite.py", "v2_analysis.py",
    "v5_experiment.py", "online_credit.py", "neurolkh_prep.py", "neurolkh_infer.py", "lkh_b3_experiment.py",
    "results/kick_targeting/select_v5_rules.py", "results/kick_targeting/analysis_v5.py",
    "results/kick_targeting/analysis_v4_robustness.py", "results/kick_targeting/run_v5.ps1",
    "models/gnn_trained.npz", "models/cheap_scorer.npz", "models/cheap_gain_scorer.npz", "models/online_prior.npz",
    "data_holdout2/catalog.json",
    "third_party/NeuroLKH/pretrained/neurolkh_m.pt", "third_party/NeuroLKH/net/sgcn_model.py",
    "third_party/NeuroLKH/net/sgcn_layers.py",
    "third_party/LKH3_official/SRC/KSwapKick.c", "third_party/LKH3_targeted/SRC/KSwapKick.c",
    "third_party/LKH3_official/LKH", "third_party/LKH3_targeted/LKH",
    "results/kick_targeting/lkh_b3/tasks.txt", "results/kick_targeting/lkh_b3/cand_tasks.txt",
    "results/kick_targeting/lkh_b3/run_task.sh", "results/kick_targeting/lkh_b3/run_all.sh",
]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tree_sha(files) -> str:
    h = hashlib.sha256()
    for f in sorted(files, key=lambda x: x.name):
        h.update(f.name.encode() + b" " + sha(f).encode() + b"\n")
    return h.hexdigest()


def main():
    nl_commit = subprocess.run(["git", "-C", str(ROOT / "third_party/NeuroLKH"), "rev-parse", "HEAD"],
                               capture_output=True, text=True).stdout.strip()
    proto = {
        "title": "PROTOCOL_v4_exploratory -- reviewer-driven blocks B1-B6 (exploratory, post-confirmation)",
        "written": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": ("Exploratory.  Registered before any run of these sessions.  The confirmatory decisions "
                   "(PROTOCOL_v2, V1-V4) are not changed by anything in this block."),
        "known_before_registration": [
            "holdout2 results of random / excess / GNN / gain-MLP / online arms from the PROTOCOL_v2 and "
            "PROTOCOL_v3_exploratory sessions (published in the paper)",
            "No NeuroLKH arm, no tuned gain-MLP rule, no equal-kick session, no credit variant and no LKH run has "
            "been executed on holdout2.  The NeuroLKH forward pass was computed for all instances (inputs only, no "
            "search outcome).  The LKH patch equivalence test and the driver smoke tests used the development "
            "instance dka1376 only.",
        ],
        "competitor_code": {
            "NeuroLKH": {"repo": "https://github.com/liangxinedu/NeuroLKH", "commit": nl_commit,
                         "model": "pretrained/neurolkh_m.pt (authors' model for TSPLIB: uniform+clustered+mixed training)",
                         "use": ("SGN edge probabilities as the kick-targeting signal inside the same fast ILS; input "
                                 "graph built exactly as in the authors' tsplib_test.py (neurolkh_prep.py selftest: "
                                 "identical to their dense code); forward pass on GPU (GTX 1050), median of 3 timed "
                                 "passes; tour edge score = max(p(u->v), p(v->u)), 0 if neither is a 20-NN edge"),
                         "cost": ("neurolkh: forward time deducted from the 10 s budget; neurolkhfree: not deducted "
                                  "(treated like the per-instance neighbour lists of every signal)")},
            "LKH-3": {"version": "3.0.13", "sources": ["https://github.com/HuaYuXiao/LKH-3 (used)",
                                                          "https://github.com/Seth9976/LKH-3.0.13 (cross-check)"],
                      "note": ("official site unreachable from this machine; kick-related files byte-identical in both "
                               "copies; built with gcc 12.2 (Debian bookworm) in Docker; patch = SRC/KSwapKick.c only, "
                               "inactive without LKH_KICK_DREF; patched binary without the variable reproduced the "
                               "official binary exactly on dka1376 (300 trials, KICK_TYPE 4)")},
        },
        "sessions": {
            "G (B4 tuning)": ("v5_experiment.py --mode time --budget 10 --seeds 3 --workers 4 on vlsi_primary "
                              "(dev5_grid_vlsi_primary_10s) and synth_1000 (dev5_grid_synth1000_10s); arms "
                              "{gain, neurolkh}_global@f:p, f in {0.02,0.04,0.06,0.10}, p in {0.5,0.75,0.9}"),
            "selection": ("select_v5_rules.py: same rule as PROTOCOL_v2 development -- minimise the equal-weight "
                          "mean over both sets of the per-instance mean gap; keep 0.06/0.75 if within 0.01 pt of the best"),
            "S2 (B2+B4)": ("holdout2_v5_equal_10s: equal time 10 s, 10 seeds, 4 workers, random arm order; "
                           "random, excess@0.06:0.5, gnn@0.04:0.5, gain (default), gain@selected (if different), "
                           "neurolkh@selected, neurolkhfree@selected"),
            "S3 (B5)": ("holdout2_v5_credit_10s: equal time 10 s, 10 seeds; random, linprior, onlinewarm (signed "
                        "reward), onlinepos (max(0,r)), onlinesucc (1[accepted]); warm start w0 for all online arms"),
            "S1 (B1)": ("holdout2_v5_kicks20k: exactly 20 000 kicks per arm, pool refreshed after every accepted kick, "
                        "model time not charged, 5 seeds; random, excess@0.06:0.5, knn@0.04:0.75, gnn@0.04:0.5, "
                        "gain@selected, neurolkh@selected"),
            "S4 (B6)": ("holdout2_v5_single_10s: 1 worker, seeds 0-1, random arm order; random, excess@0.06:0.5, "
                        "gnn@0.04:0.5.  Random arm order is also used in S1-S3 (crc32 of 'instance|seed')"),
            "S5 (B3)": ("holdout2_lkh_10s: LKH-3, TIME_LIMIT 10 s search time, RUNS 1, 5 seeds, cached candidates/pi, "
                        "4 parallel processes in one container, shuffled task order; lkh_default, lkh_ils_random "
                        "(KICK_TYPE 4, KICKS 1), lkh_ils_excess (patched, f 0.06, p 0.5)"),
        },
        "runner": "results/kick_targeting/run_v5.ps1 (steps logged to v5.done)",
        "analysis": {
            "command": "py -3.13 results/kick_targeting/analysis_v5.py  (run once, after all sessions)",
            "unit": "instance (gap averaged over seeds); paired difference, bootstrap 95% CI (10 000, seed 0), "
                    "two-sided Wilcoxon, Hodges-Lehmann, Holm within family",
            "families": {
                "S2": ["excess-neurolkh", "excess-neurolkhfree", "neurolkh-random", "neurolkhfree-random",
                       "excess-gain@sel", "gain@sel-gain(default)", "excess-gnn", "excess-random"],
                "S1": ["excess-gnn", "excess-gain@sel", "excess-neurolkh", "excess-knn", "gnn-random",
                       "gain@sel-random", "neurolkh-random", "excess-random"],
                "S3": ["onlinepos-linprior", "onlinesucc-linprior", "onlinewarm-linprior", "onlinepos-onlinewarm",
                       "onlinesucc-onlinewarm", "linprior-random"],
                "S4": ["excess-random", "excess-gnn"],
                "S5": ["lkh_ils_excess-lkh_ils_random (primary)", "lkh_ils_random-lkh_default",
                       "lkh_ils_excess-lkh_default"],
            },
            "descriptive": "kick rates, acceptance, model share, elapsed time per arm; S2 kick rate by execution position",
            "non_inferiority": "read from the bootstrap CI upper bound against +0.05 pt where stated",
        },
        "expectations_not_tests": [
            "B1: if the proxy keeps its lead over GNN / gain-MLP / NeuroLKH at equal kick counts, the lead is not "
            "only a cost effect; if it vanishes, cost explains it",
            "B3: a stronger local search may repair long edges itself and shrink the effect of targeting",
        ],
        "code_and_data_sha256": {f: sha(ROOT / f) for f in FILES},
        "aggregate_sha256": {
            "neurolkh_scores/*.neurolkh_m.npz": tree_sha(list((HERE / "neurolkh_scores").glob("*.neurolkh_m.npz"))),
            "neurolkh_scores/*.graph.npz": tree_sha(list((HERE / "neurolkh_scores").glob("*.graph.npz"))),
            "lkh_b3/par/*.par": tree_sha(list((HERE / "lkh_b3" / "par").glob("*.par"))),
            "lkh_b3/dref/*.txt": tree_sha(list((HERE / "lkh_b3" / "dref").glob("*.txt"))),
        },
    }
    p = HERE / "PROTOCOL_v4_exploratory.json"
    p.write_text(json.dumps(proto, indent=1, ensure_ascii=False), encoding="utf-8")
    (HERE / "PROTOCOL_v4_exploratory.sha256").write_text(f"{sha(p)}  PROTOCOL_v4_exploratory.json\n", encoding="ascii")
    print(sha(p))


if __name__ == "__main__":
    main()
