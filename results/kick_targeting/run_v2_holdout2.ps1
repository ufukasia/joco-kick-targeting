# PROTOCOL_v2 CONFIRMATORY run on holdout2.  Requires the frozen PROTOCOL_v2.json (+ .sha256);
# the driver refuses to run otherwise.  Strategy list = PROTOCOL_v2.json "strategies".
Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
$S = "random_global,knn_global@0.04:0.75,excess_global@0.06:0.5,gnn_global@0.04:0.5,gain_global,cheap_global,linprior_global,onlinewarm_global"
py -3.13 v2_targeting_experiment.py --suite holdout2 --tag holdout2_10s --seeds 10 --budget 10 --workers 4 --strategies $S *>> "$L\holdout2_10s.log"
"HOLDOUT2_DONE" | Out-File -Append "$L\v2_holdout2.done"
