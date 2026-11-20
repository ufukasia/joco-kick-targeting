# -*- coding: utf-8 -*-
"""Kick targeting inside LKH-3 (Lin-Kernighan-Helsgaun) -- exploratory block B3 (PROTOCOL_v4_exploratory).

Engine: LKH-3.0.13 (K. Helsgaun), built from source in a Debian container.  The source was taken
from two independent public copies whose kick-related files (KSwapKick.c, ChooseInitialTour.c,
LinKernighan.c, FindTour.c, Random.c, LKH.h) are byte-identical.  third_party/LKH3_targeted differs
from third_party/LKH3_official only in SRC/KSwapKick.c; the patch is inactive unless LKH_KICK_DREF
is set (`equivalence` checks that the patched binary then reproduces the official one exactly).

Arms (all with the patched binary, same SEED, same cached candidate set and pi values, RUNS = 1,
TIME_LIMIT = search time per run):
    lkh_default     LKH-3 defaults (each trial starts from LKH's own perturbed initial tour)
    lkh_ils_random  KICK_TYPE = 4, KICKS = 1: LKH's documented ILS mode -- every trial applies one
                    random double-bridge (4-swap) kick to the best tour, then Lin-Kernighan
    lkh_ils_excess  as lkh_ils_random, but each kick node is drawn with probability 0.5 from the
                    6 % tour edges with the largest excess length (rule selected for the fast ILS in
                    PROTOCOL_v2 development; not tuned for LKH)

    py -3.13 lkh_b3_experiment.py prep            (dref files, parameter files, task list)
    docker ... sh /work/results/kick_targeting/lkh_b3/run_all.sh   (see run_v5.ps1)
    py -3.13 lkh_b3_experiment.py collect
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B3 = HERE / "results" / "kick_targeting" / "lkh_b3"
OUT = HERE / "results" / "kick_targeting" / "holdout2_lkh_10s"
ARMS = ("lkh_default", "lkh_ils_random", "lkh_ils_excess")
SEEDS = 5
BUDGET = 10
FRAC, PROB = 0.06, 0.5
WORKERS = 4


def w(p: Path) -> str:
    """host path -> container path (/work = repository root)."""
    return "/work/" + p.resolve().relative_to(HERE).as_posix()


def instances():
    sys.path.insert(0, str(HERE))
    import holdout2_suite
    return holdout2_suite.instances()


def prep():
    import fast_ils as F
    from kick_targeting_experiment import load_xy
    from v2_targeting_experiment import ExcessProxy
    for sub in ("dref", "cand", "par", "log"):
        (B3 / sub).mkdir(parents=True, exist_ok=True)
    tasks, cand = [], []
    for nm, path, bks in instances():
        xs, ys, kind = load_xy(Path(path))
        assert kind == 0
        knn = F.build_knn(xs, ys, kind, 2)
        dref = ExcessProxy(xs, ys, kind, knn).dref
        df = B3 / "dref" / f"{nm}.txt"
        df.write_text("\n".join(f"{v:.6f}" for v in dref) + "\n", encoding="ascii")
        base = (f"PROBLEM_FILE = {w(Path(path))}\nCANDIDATE_FILE = {w(B3 / 'cand' / (nm + '.cand'))}\n"
                f"PI_FILE = {w(B3 / 'cand' / (nm + '.pi'))}\nRUNS = 1\nTRACE_LEVEL = 1\n")
        pc = B3 / "par" / f"{nm}__cand.par"
        pc.write_text(base + "MAX_TRIALS = 1\nSEED = 1\n", encoding="ascii")
        cand.append(f"cand {w(pc)} {w(B3 / 'log' / (nm + '__cand.log'))} -")
        for s in range(SEEDS):
            for arm in ARMS:
                extra = "" if arm == "lkh_default" else "KICK_TYPE = 4\nKICKS = 1\n"
                p = B3 / "par" / f"{nm}__s{s}__{arm}.par"
                p.write_text(base + f"SEED = {s + 1}\nTIME_LIMIT = {BUDGET}\n" + extra, encoding="ascii")
                tasks.append(f"{arm} {w(p)} {w(B3 / 'log' / f'{nm}__s{s}__{arm}.log')} {w(df)}")
    random.Random(20261120).shuffle(tasks)  # randomised execution order (arms interleaved)
    (B3 / "cand_tasks.txt").write_text("\n".join(cand) + "\n", encoding="ascii")
    (B3 / "tasks.txt").write_text("\n".join(tasks) + "\n", encoding="ascii")
    (B3 / "run_task.sh").write_bytes((
        "#!/bin/sh\n"
        "arm=$1; par=$2; log=$3; dref=$4\n"
        "[ -s \"$log\" ] && grep -q 'Cost.min' \"$log\" && exit 0\n"
        "BIN=/work/third_party/LKH3_targeted/LKH\n"
        "if [ \"$arm\" = lkh_ils_excess ]; then\n"
        f"  LKH_KICK_DREF=$dref LKH_KICK_FRAC={FRAC} LKH_KICK_PROB={PROB} $BIN \"$par\" > \"$log.tmp\" 2>&1\n"
        "else\n"
        "  $BIN \"$par\" > \"$log.tmp\" 2>&1\n"
        "fi\n"
        "mv \"$log.tmp\" \"$log\"\n").encode("ascii"))
    (B3 / "run_all.sh").write_bytes((
        "#!/bin/sh\nset -e\ncd /work/results/kick_targeting/lkh_b3\n"
        f"xargs -P {WORKERS} -L 1 sh run_task.sh < cand_tasks.txt\n"
        f"xargs -P {WORKERS} -L 1 sh run_task.sh < tasks.txt\n"
        "echo B3_DONE\n").encode("ascii"))
    print(len(cand), "candidate tasks,", len(tasks), "run tasks")


def parse(log: Path):
    t = log.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"Cost\.min = (-?\d+)", t)
    tr = re.search(r"Trials\.avg = ([\d.]+)", t)
    tm = re.search(r"Time\.avg = ([\d.]+)", t)
    return (int(m.group(1)) if m else None, float(tr.group(1)) if tr else None, float(tm.group(1)) if tm else None)


def collect():
    OUT.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    for nm, path, bks in instances():
        for s in range(SEEDS):
            runs = {}
            for arm in ARMS:
                c, trials, tm = parse(B3 / "log" / f"{nm}__s{s}__{arm}.log")
                if c is None:
                    break
                runs[arm] = {"cost": c, "gap_pct": 100.0 * (c - bks) / bks, "trials": trials, "time": tm}
            if len(runs) == len(ARMS):
                n = sum(1 for ln in (B3 / "dref" / f"{nm}.txt").read_text(encoding="ascii").splitlines() if ln.strip())
                (OUT / f"{nm}__s{s}.json").write_text(json.dumps(
                    {"suite": "holdout2", "instance": nm, "n": n, "bks": bks, "seed": s, "budget": BUDGET, "runs": runs,
                     "engine": "LKH-3.0.13 (KSwapKick patch, see lkh_b3_experiment.py)"}), encoding="utf-8")
                n_ok += 1
    print(n_ok, "instance-seed files written to", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "collect"))
    a = ap.parse_args()
    prep() if a.cmd == "prep" else collect()
