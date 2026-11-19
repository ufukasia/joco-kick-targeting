Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
$E = "excess_global@0.06:0.5"
py -3.13 v4_robustness_experiment.py --tag holdout2_local_10s --strategies "random_global,$E,random_local,excess_local@0.06:0.5,length_local@0.06:0.5" *>> "$L\holdout2_local_10s.log"
"S1_DONE" | Out-File -Append "$L\v4_robustness.done"
py -3.13 v4_robustness_experiment.py --tag holdout2_60s --budget 60 --seeds 5 --strategies "random_global,$E,gnn_global@0.04:0.5" *>> "$L\holdout2_60s.log"
"S2_DONE" | Out-File -Append "$L\v4_robustness.done"
py -3.13 v4_robustness_experiment.py --tag holdout2_rule_10s --strategies "random_global,$E,excess_global@0.03:0.25,excess_global@0.03:1.0,excess_global@0.12:0.25,excess_global@0.12:1.0" *>> "$L\holdout2_rule_10s.log"
"S3_DONE" | Out-File -Append "$L\v4_robustness.done"
