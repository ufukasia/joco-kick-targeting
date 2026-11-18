# PROTOCOL_v2 development, step 4 (exploratory tuning): pool-rule grid per signal.
# frac {0.02,0.04,0.06,0.10} x prob {0.5,0.75,0.9} for knn / excess / gnn (+ random baseline),
# 3 seeds, 10 s, 4 workers, on vlsi_primary and synth_1000.  Selection rule fixed beforehand in
# PROTOCOL_DEVIATIONS.log.
Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
$list = @("random_global")
foreach ($sig in "knn", "excess", "gnn") {
  foreach ($f in "0.02", "0.04", "0.06", "0.10") {
    foreach ($p in "0.5", "0.75", "0.9") { $list += "${sig}_global@${f}:${p}" }
  }
}
$S = $list -join ","
py -3.13 v2_targeting_experiment.py --suite vlsi_primary --tag dev_grid_vlsi_primary_10s --seeds 3 --budget 10 --workers 4 --strategies $S *>> "$L\dev_grid_vlsi_primary_10s.log"
py -3.13 v2_targeting_experiment.py --suite synth_1000 --tag dev_grid_synth1000_10s --seeds 3 --budget 10 --workers 4 --strategies $S *>> "$L\dev_grid_synth1000_10s.log"
"GRID_DONE" | Out-File -Append "$L\v2_grid.done"
