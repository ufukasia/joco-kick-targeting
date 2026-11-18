# PROTOCOL_v2 setup chain (2026-11-16, Intel i5-9300H machine).
# 1) LKH-3 references for the 42 fresh synthetic holdout2 instances (no strategy is run on holdout2).
# 2) Exploratory development run: training-free excess-length proxy vs kNN / learned scorers on the
#    development sets only (vlsi_old, synth_1000).  Baselines re-run in the same session.
Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
py -3.13 holdout2_suite.py ref --workers 4 *>> "$L\holdout2_ref.log"
"REF_DONE" | Out-File -Append "$L\v2_ref.done"
$S = "random_global,knn_global,cheap_global,gain_global,excess_global"
py -3.13 v2_targeting_experiment.py --suite vlsi_old --tag dev_excess_vlsi_old_10s --seeds 5 --budget 10 --workers 4 --strategies $S *>> "$L\dev_excess_vlsi_old_10s.log"
py -3.13 v2_targeting_experiment.py --suite synth_1000 --tag dev_excess_synth1000_10s --seeds 5 --budget 10 --workers 4 --strategies $S *>> "$L\dev_excess_synth1000_10s.log"
"DEV_DONE" | Out-File -Append "$L\v2_dev.done"
