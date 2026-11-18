Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
py -3.13 v3_extra_experiment.py --tag holdout2_length_10s *>> "$L\holdout2_length_10s.log"
"LENGTH_DONE" | Out-File -Append "$L\v3_length.done"
