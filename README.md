# Do you need to learn where to kick? Kick targeting in fast ILS for the TSP

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/hero_dark.svg">
  <img alt="Final gap vs random kicks for seven kick-targeting signals on 54 held-out instances: the untrained excess-length proxy is best (−0.17 points, better on 44/54); trained GNN and MLPs give smaller gains; the online learner is worse than random (+0.21)." src="docs/hero_light.svg" width="100%">
</picture>

**In one line:** aim the double-bridge kick at long tour edges. It is free and needs no training. It does
as well as a GNN trained on LKH-3 tours, in a preregistered test on 54 held-out instances.

| | |
|---|---|
| **Takeaway for practitioners** | Rank tour edges by length (or by length minus the local neighbour scale) and draw kick cut points from the top 6%. This costs O(1) per edge and needs no model. |
| **Evidence** | Preregistered protocol, frozen with SHA-256 and timestamped on Bitcoin (OpenTimestamps, block 976258) before the run. 54 untouched instances × 10 seeds × 8 strategies. All four hypotheses confirmed. |
| **Lesson** | Learning *during* the run backfired: the learner optimised a myopic reward and drifted to harmless kicks. |
| **Paper** | Ufuk ASIL, *Vuruşları Hedeflemek İçin Öğrenmek Gerekir mi? Hızlı İteratif Yerel Aramada Eğitimsiz Kenar Uzunluğu Vekillerinin Ön-Kayıtlı Sınaması*, submitted to *Journal of Combinatorial Optimization*. |

---

## 1. What is being compared

Iterated local search (ILS) alternates a local search (here 2-opt + Or-opt, numba) with a **kick**: a
double-bridge move that cuts the tour at 4 points and reconnects it. Classic ILS picks the 4 cut points at
random. *Kick targeting* draws them, with probability `prob`, from a **pool** of the `m = max(12, frac·n)`
most suspicious tour edges. The strategies differ only in how "suspicious" is scored:

| Signal | Training | Cost per edge | Idea |
|---|---|---|---|
| Random | none | – | baseline |
| kNN rank | none | O(1) | edge to a far neighbour rank is suspicious |
| **Excess-length proxy** | none | O(1) | `d(u,v) − (dref(u)+dref(v))/2`, where `dref` is the mean distance to the 1st/2nd nearest neighbour |
| **Raw edge length** (exploratory) | none | O(1) | `d(u,v)`: guided local search's utility without penalties |
| GNN | LKH-3 tours (n 101–400) | model inference | predicts edges not in a near-optimal tour |
| MLP (membership / gain) | LKH-3 tours / kick outcomes | small MLP | per-edge features |
| Static linear prior | fitted offline | O(1) | linear score on the same features |
| Online learner | **during the run** | ridge update | Thompson-sampled ridge regression on per-kick reward |

All strategies share the start tour, the kick random numbers and the wall-clock budget (10 s).
They are therefore paired: each instance-seed pair is a matched comparison.

## 2. The trained model is not needed

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/per_instance_dark.svg">
  <img alt="Per-instance difference between the excess-length proxy and the GNN, sorted: the proxy has the lower gap on 41 of 54 instances; the largest gains are around one percentage point." src="docs/per_instance_light.svg" width="100%">
</picture>

The preregistered claim against the GNN is **non-inferiority** (margin 0.05 points). It passed with
p = 7.9e−9. That the proxy is *better* on most instances is an exploratory finding. The gains concentrate
on clustered and real instances (VLSI, national). On uniform random instances, targeting adds little,
because edge lengths there carry little information.

## 3. Even simpler: plain edge length

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/length_arm_dark.svg">
  <img alt="Exploratory arm: raw edge length lowers the gap by 0.19 points vs random, the excess proxy by 0.20; the two do not differ (+0.01, p = 0.99)." src="docs/length_arm_light.svg" width="100%">
</picture>

A follow-up arm (exploratory, run after the confirmatory test) targets the longest tour edges with no
local-scale correction. It reaches the same level as the excess proxy. This suggests the signal is the
edge length itself. Absence of a difference is not proof of equivalence: the 95% CI is [−0.03, +0.05].

## 4. Why online learning hurts

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/online_dark.svg">
  <img alt="Online learner vs static prior: kick penalty falls from 25.9 to 2.8 and neutral kicks rise from 47% to 88%, while the final gap rises from 1.01% to 1.30%." src="docs/online_light.svg" width="100%">
</picture>

The online learner is rewarded by the immediate cost change of each kick. Most kicks are repaired by local
search, so the cheapest way to raise the reward is to cut short edges that local search restores at
once. The learner does exactly that: its weights reverse relative to the prior, with cosine −0.62.
The design lesson is that a reward for sparse-improvement search should be tied to *accepted
improvements*, and that a learner's own reward must be monitored together with final quality.

## 5. How the evidence was protected

1. **Development:** all choices (signals, pool rule `frac`/`prob`, margin) were made on development
   sets only (`results/kick_targeting/dev_*`).
2. **Freeze:** `PROTOCOL_v2.json` fixes strategies, seeds, hypotheses V1–V4, test order and analysis
   command. It also records the SHA-256 of every code, model and data file it depends on
   (`code_and_data_sha256`).
3. **Timestamp:** `PROTOCOL_v2.json.ots` proves the file existed before the run (Bitcoin block 976258).
   The planned OSF upload was unavailable; the switch was logged *before* the run in
   `PROTOCOL_DEVIATIONS.log`.
4. **Confirm:** a single run on 54 never-touched instances (`holdout2_10s/`, 540 files), analysed once
   with the frozen command. Fixed-sequence testing: V1 online > static, V2 proxy < random,
   V3/V4 proxy non-inferior to GNN / gain-MLP. All rejected H0.
5. **Exploratory work** (the raw-length arm, mechanism analyses, Hodges–Lehmann estimates) is logged as
   such and never changes the confirmatory decisions.

```bash
cd results/kick_targeting
sha256sum PROTOCOL_v2.json            # 96eb1d9e0f21ead9a513894359ac66cbceb39754a72b564ac667b3a369738085
pip install opentimestamps-client && ots verify PROTOCOL_v2.json.ots
```

Git stores every file byte-for-byte (`.gitattributes`), so the registered hashes hold in a fresh clone.

## 6. Reproduce

```bash
python -m pip install -r requirements.txt               # Python 3.13, numpy, scipy, numba
python -m pytest test_fast_ils.py test_online_reward.py # 9 tests
python v2_analysis.py confirm --tag holdout2_10s        # confirmatory result from the stored runs (seconds)
python readme_figures.py                                # the figures above (writes to release_meta/docs/ in the author's tree)
```

To re-run the experiments (about 4.5 h on 4 cores), use `results/kick_targeting/run_v2_holdout2.ps1`
and `results/kick_targeting/run_v3_length.ps1`. The budget is wall-clock time, so other hardware gives
close but not identical gaps.

To rebuild the paper's numbers, tables and PDF, run
`python makale_workspace/gnn-crine-ils-joco/evidence/scripts/build_v3.py`, then
`xelatex main && bibtex main && xelatex main && xelatex main` in `makale_workspace/gnn-crine-ils-joco/paper/`.
Every number in the text carries a `% E…` comment pointing to a record in `evidence/evidence.json`,
which names its source file and field.

## Repository layout

| Path | Content |
|---|---|
| `fast_ils.py`, `test_fast_ils.py` | ILS engine with targeted double-bridge kicks |
| `v2_targeting_experiment.py`, `v2_analysis.py` | Stage 3 driver and preregistered analysis |
| `v3_extra_experiment.py`, `results/kick_targeting/analysis_length_arm.py` | raw-length arm |
| `online_targeting.py`, `analysis_online_*.py`, `test_online_reward.py` | online learner and its mechanism |
| `gnn_trained.py`, `cheap_scorer.py`, `models/`, `gnn_train_torch.py` | trained scorers and training code |
| `kick_targeting_experiment.py` | Stage 2 (PROTOCOL v1, 38 VLSI instances) |
| `holdout2_suite.py`, `data_holdout2/` | the 54-instance confirmatory set |
| `data*/` | Waterloo VLSI, National TSP and TSPLIB instances; synthetic suites |
| `results/kick_targeting/` | protocols, deviation log, one JSON per (instance, seed) run |
| `results/*.json` | Stage 1 records (an earlier, withdrawn claim; reported briefly in the paper) |
| `makale_workspace/gnn-crine-ils-joco/` | paper sources (Springer `sn-jnl`), evidence ledger, generators |
| `readme_figures.py`, `make_release.py` | README figures; rebuilds this package |

## Scope and limits

Results hold for this engine (2-opt + Or-opt), a 10 s budget and n up to about 10k. The GNN is a single
small model trained on n ≤ 400. LK/LKH-level local search, longer budgets and larger published models
are the natural next tests.

## License and citation

Code: MIT (`LICENSE`). Third-party benchmark instances keep their original terms. See
`CITATION.cff`; a Zenodo DOI is attached to each release.
