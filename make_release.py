# -*- coding: utf-8 -*-
"""Builds the clean reproduction package for the JoCO kick-targeting paper.

    py -3.13 make_release.py            -> ~/repos/joco-kick-targeting/

Copies only what the paper needs (code closure, data, run records, paper sources), byte-identical,
then checks the PROTOCOL_v2 hash registry against the copies.
"""
import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = Path.home() / "repos" / "joco-kick-targeting"   # outside OneDrive: git + sync do not mix
WS = "makale_workspace/gnn-crine-ils-joco"

ENTRY = [
    # stage 2/3 engine, drivers, analyses
    "fast_ils", "kick_targeting_experiment", "test_fast_ils", "v2_targeting_experiment", "holdout2_suite",
    "v2_analysis", "cheap_scorer", "online_targeting", "gnn_trained", "v3_extra_experiment", "v4_robustness_experiment",
    "analysis_online_mechanism", "analysis_online_reward", "test_online_reward", "kick_targeting_analysis",
    "v5_experiment", "online_credit", "neurolkh_prep", "neurolkh_infer", "lkh_b3_experiment",  # PROTOCOL_v4 blocks
    "kick_mechanism_analysis", "cheap_analysis", "cheap_mechanism", "cheap_targeting_experiment",
    "pool_composition", "make_synth_suite", "kick_oracle_analysis", "kick_oracle_experiment",
    # GNN / MLP training
    "gnn_generate_data", "gnn_train_torch",
    # stage 1 (withdrawn claim; reported briefly)
    "adaptive_ils", "focused_experiment", "diversification_experiment", "entropy_experiment",
    "entropy_fusion_experiment", "lkh_comparison_experiment", "prepare_tsplib_scale_suite",
    "circuit_regime_analysis", "real_regime_analysis", "paper_result_sanity", "merge_focused_experiments",
]
EXTRA_FILES = ["make_release.py", "readme_figures.py"]
DATA_DIRS = ["data", "data_holdout2", "data_synth_fast", "data_tsplib_scale", "data_vlsi", "data_world",
             "gnn_train_data_lkh"]
MODELS = ["gnn_trained", "cheap_scorer", "cheap_gain_scorer", "online_prior"]
STAGE1_RESULTS = [  # result files cited by the evidence ledger outside results/kick_targeting
    "diversification_synthetic_100", "diversification_tsplib_9x10", "diversification_validation_40",
    "diversification_world_10x10", "focused_synthetic_100", "focused_tsplib_9x10", "focused_validation_40",
    "focused_world_10x10", "focused_world_120s_preflight", "focused_world_30s_preflight",
    "lkh_comparison_19x10", "tsplib_scale_72x5", "waterloo_vlsi_current_16x5"]
STAGE1_SINGLE = ["elite_entropy_validation_40_summary", "entropy_fusion_holdout_100_summary",
                 "entropy_fusion_tsplib_9x10_summary", "entropy_fusion_validation_40_summary",
                 "entropy_holdout_100_t1_summary", "entropy_tsplib_9x10_summary", "tsplib_circuit_regimes",
                 "tsplib_scale_72x5_analysis"]
PAPER = ["paper/main.tex", "paper/main_tek.tex", "paper/main.pdf", "paper/references.bib", "paper/sn-jnl.cls",
         "paper/sn-basic.bst", "paper/tek_dosya_uret.py", "sections", "figures", "evidence/evidence.json", "evidence/v4_ids.json", "evidence/v5_ids.json",
         "evidence/scripts", "evidence/tables", "evidence/build_v3c.py", "evidence/logs"]
SKIP = {"__pycache__", ".tmp", "_onceki", "_v1_crine_taslak", "_onizleme"}


def closure():
    local = {p.stem for p in ROOT.glob("*.py")}
    seen, st = set(), [e for e in ENTRY if e in local]
    missing = [e for e in ENTRY if e not in local]
    while st:
        m = st.pop()
        if m in seen:
            continue
        seen.add(m)
        for n in ast.walk(ast.parse((ROOT / f"{m}.py").read_text(encoding="utf-8"))):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module]
            st += [x.split(".")[0] for x in mods if x.split(".")[0] in local]
    return sorted(seen), missing


def cp(src, dst):
    if src.is_dir():
        for p in src.rglob("*"):
            if (p.is_file() and not (set(p.parts) & SKIP) and p.suffix not in (".pyc", ".bak", ".tmp")
                    and not p.name.endswith(".graph.npz")):   # NeuroLKH input graphs: rebuilt by neurolkh_prep.py
                q = dst / p.relative_to(src)
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)
    elif src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        raise FileNotFoundError(src)


def main():
    if OUT.exists():
        for p in OUT.iterdir():          # keep .git if the repo was already initialised
            if p.name != ".git":
                shutil.rmtree(p) if p.is_dir() else p.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    mods, missing = closure()
    for m in mods:
        cp(ROOT / f"{m}.py", OUT / f"{m}.py")
    for f in EXTRA_FILES:
        cp(ROOT / f, OUT / f)
    for f in (ROOT / "release_meta").iterdir():   # README, LICENSE, CITATION.cff, .zenodo.json, ...
        cp(f, OUT / f.name)
    for d in DATA_DIRS:
        cp(ROOT / d, OUT / d)
    for m in MODELS:
        for ext in (".npz", ".json", ".meta.json"):
            if (ROOT / "models" / f"{m}{ext}").exists():
                cp(ROOT / "models" / f"{m}{ext}", OUT / "models" / f"{m}{ext}")
    cp(ROOT / "results" / "kick_targeting", OUT / "results" / "kick_targeting")
    for s in STAGE1_RESULTS:
        for suf in (".json", "_summary.json"):
            if (ROOT / "results" / f"{s}{suf}").exists():
                cp(ROOT / "results" / f"{s}{suf}", OUT / "results" / f"{s}{suf}")
    for s in STAGE1_SINGLE:
        cp(ROOT / "results" / f"{s}.json", OUT / "results" / f"{s}.json")
    for p in PAPER:
        cp(ROOT / WS / p, OUT / WS / p)
    # hash registry check
    proto = json.loads((OUT / "results/kick_targeting/PROTOCOL_v2.json").read_text(encoding="utf-8"))
    reg = dict(proto.get("code_and_data_sha256") or {})
    v1 = json.loads((OUT / "results/kick_targeting/PROTOCOL.json").read_text(encoding="utf-8"))
    for k, v in v1.items():
        if isinstance(v, dict) and k.endswith("sha256"):
            reg.update({f"v1:{a}": b for a, b in v.items()})
    p3 = OUT / "results/kick_targeting/PROTOCOL_v3_exploratory.json"
    if p3.exists():   # exploratory robustness block: analysis script + runner are frozen too
        reg.update({f"v3:{a}": b for a, b in json.loads(p3.read_text(encoding="utf-8"))["code_and_data_sha256"].items()})
    p4 = OUT / "results/kick_targeting/PROTOCOL_v4_exploratory.json"
    if p4.exists():   # reviewer blocks; third-party code is not redistributed (see third_party/README.md)
        reg.update({f"v4:{a}": b for a, b in json.loads(p4.read_text(encoding="utf-8"))["code_and_data_sha256"].items()
                    if not a.startswith("third_party/")})
    bad = []
    for f, h in (reg.items() if isinstance(reg, dict) else []):
        q = OUT / f.removeprefix("v1:").removeprefix("v3:").removeprefix("v4:")
        if not q.exists() or hashlib.sha256(q.read_bytes()).hexdigest() != h:
            bad.append(f)
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"modules {len(mods)}; missing entries {missing}; registry entries {len(reg)}; mismatches {bad}; "
          f"size {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
