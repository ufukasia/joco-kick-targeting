# Kick targeting in fast iterated local search — reproduction package

Code, data, preregistration files and every per-run record behind the paper

> Ufuk ASIL. *Vuruşları Hedeflemek İçin Öğrenmek Gerekir mi? Hızlı İteratif Yerel Aramada Eğitimsiz
> Kenar Uzunluğu Vekillerinin Ön-Kayıtlı Sınaması.* Submitted to *Journal of Combinatorial Optimization*.

**Question.** In a fast 2-opt/Or-opt iterated local search (ILS) for the TSP, does it pay to aim the
double-bridge kick at "suspicious" tour edges, and does that need a learned model?

**Answer (preregistered, 54 untouched instances × 10 seeds, 10 s).** Targeting helps; an untrained
O(1) edge-length proxy is enough. The excess-length proxy lowered the gap by 0.172 points versus random
kicks (p = 3.7e-7) and passed non-inferiority tests against an LKH-3-trained GNN and a gain-trained MLP
(margin 0.05). An in-run online learner raised its own kick reward while worsening the final gap
(+0.291 points). An exploratory raw-edge-length arm reached the same level as the excess proxy.

## Layout

| Path | Content |
|---|---|
| `fast_ils.py` | numba 2-opt + Or-opt ILS engine with targeted double-bridge kicks (`test_fast_ils.py`) |
| `v2_targeting_experiment.py` | Stage 3 driver (signals, pool rule `@frac:prob`, paired common random numbers) |
| `v2_analysis.py` | paired Wilcoxon / bootstrap comparisons and the fixed-sequence confirmatory test |
| `v3_extra_experiment.py`, `results/kick_targeting/analysis_length_arm.py` | exploratory raw-edge-length arm |
| `online_targeting.py`, `analysis_online_*.py`, `test_online_reward.py` | online Thompson-ridge scorer and its reward mechanism |
| `gnn_trained.py`, `cheap_scorer.py`, `models/` | trained GNN and MLP scorers (weights included) |
| `kick_targeting_experiment.py` | Stage 2 driver (PROTOCOL v1, 38 VLSI instances) |
| `holdout2_suite.py`, `data_holdout2/` | the 54-instance confirmatory set and its catalog |
| `data*/` | benchmark instances (Waterloo VLSI, National TSP, TSPLIB; synthetic suites) |
| `results/kick_targeting/` | protocols, deviation log, one JSON per (instance, seed) run, summaries |
| `results/*.json` | Stage 1 records (withdrawn earlier claim, reported briefly in the paper) |
| `makale_workspace/gnn-crine-ils-joco/` | paper sources (Springer `sn-jnl`), evidence ledger `evidence/evidence.json`, table/figure generators |
| `make_release.py` | rebuilds this package from the author's working folder |

## Setup

```bash
python -m pip install -r requirements.txt      # Python 3.13, numpy, scipy, numba
python -m pytest test_fast_ils.py test_online_reward.py
```

## Verify the preregistration

`results/kick_targeting/PROTOCOL_v2.json` was frozen before the confirmatory run. It lists the SHA-256
of every code, model and data file it depends on (`code_and_data_sha256`). Git stores all files
byte-for-byte (`.gitattributes`), so the hashes hold in a fresh clone.

```bash
cd results/kick_targeting
sha256sum PROTOCOL_v2.json            # = 96eb1d9e0f21ead9a513894359ac66cbceb39754a72b564ac667b3a369738085
pip install opentimestamps-client
ots verify PROTOCOL_v2.json.ots       # Bitcoin block 976258
```

The protocol's `external_timestamp` field mentions OSF; OSF was unavailable, so OpenTimestamps was used
instead. This change was written to `PROTOCOL_DEVIATIONS.log` before the run started (entry 2026-11-17 12:51).

## Reproduce the confirmatory result

From the stored run records (seconds):

```bash
python v2_analysis.py confirm --tag holdout2_10s --out results/kick_targeting/holdout2_10s_confirm.json
```

Re-running the experiment itself (about 4.5 h on 4 cores; timings depend on the machine):

```powershell
results/kick_targeting/run_v2_holdout2.ps1      # 54 instances x 10 seeds x 8 strategies, 10 s each
results/kick_targeting/run_v3_length.ps1        # exploratory raw-length arm
```

Because the budget is wall-clock time, a re-run on other hardware gives close but not identical gaps.

## Rebuild the paper's numbers, tables and PDF

```bash
python makale_workspace/gnn-crine-ils-joco/evidence/scripts/build_v3.py   # evidence ledger + tables + figures
cd makale_workspace/gnn-crine-ils-joco/paper && xelatex main && bibtex main && xelatex main && xelatex main
```

Every number in the text carries a `% E…` comment that points to a record in `evidence/evidence.json`.
Each record names its source file and field.

## License

Code: MIT (see `LICENSE`). Third-party benchmark instances keep their original terms.
