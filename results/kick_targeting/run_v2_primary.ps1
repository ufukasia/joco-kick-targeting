# PROTOCOL_v2 development, step 3 (exploratory): the primary VLSI set (already used, most
# discriminative) as an additional development set, because vlsi_old did not separate even
# knn_global from random_global.  Waits for run_v2_online.ps1.
Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
while (-not (Test-Path "$L\v2_online.done")) { Start-Sleep -Seconds 30 }
$S = "random_global,knn_global,excess_global,gain_global,linprior_global,onlinewarm_global"
py -3.13 v2_targeting_experiment.py --suite vlsi_primary --tag dev_vlsi_primary_10s --seeds 5 --budget 10 --workers 4 --strategies $S *>> "$L\dev_vlsi_primary_10s.log"
"PRIMARY_DONE" | Out-File -Append "$L\v2_primary.done"
