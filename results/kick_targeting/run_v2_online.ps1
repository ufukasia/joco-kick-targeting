# PROTOCOL_v2 development, step 2 (exploratory): online targeting variants on the development sets.
# Waits for run_v2_dev.ps1 to finish so that no two campaigns share the CPU.
Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
while (-not (Test-Path "$L\v2_dev.done")) { Start-Sleep -Seconds 30 }
$S = "random_global,knn_global,excess_global,linprior_global,online_global,onlinewarm_global"
py -3.13 v2_targeting_experiment.py --suite vlsi_old --tag dev_online_vlsi_old_10s --seeds 5 --budget 10 --workers 4 --strategies $S *>> "$L\dev_online_vlsi_old_10s.log"
py -3.13 v2_targeting_experiment.py --suite synth_1000 --tag dev_online_synth1000_10s --seeds 5 --budget 10 --workers 4 --strategies $S *>> "$L\dev_online_synth1000_10s.log"
"ONLINE_DONE" | Out-File -Append "$L\v2_online.done"
