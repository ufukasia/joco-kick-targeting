# PROTOCOL_v4_exploratory: runs every registered session in order; each finished step is appended to
# v5.done so that a restart skips completed steps (the drivers also skip finished instance-seed files).
#   G1, G2  pool-rule grid for gain-MLP and NeuroLKH (development sets)       -> select_v5_rules.py
#   S2      equal time 10 s, holdout2 (B2 NeuroLKH, B4 tuned gain-MLP, random arm order)
#   S3      online credit variants, holdout2 (B5)
#   S1      equal kick count (20 000), holdout2 (B1)
#   S4      one worker, random arm order (B6)
#   S5      LKH-3 inside Docker (B3)
#   then the frozen analysis, once.
Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
$D = "$L\v5.done"
function Done($s) { "$s $(Get-Date -Format s)" | Out-File -Append -Encoding ascii $D }
function Has($s) { (Test-Path $D) -and (Select-String -Path $D -Pattern "^$s " -Quiet) }
function Run($step, $log, [string[]]$a) {
  if (Has "${step}_DONE") { return }
  & py -3.13 @a *>> "$L\$log"
  if ($LASTEXITCODE -ne 0) { Done "${step}_FAIL"; exit 1 }
  Done "${step}_DONE"
}

$grid = @()
foreach ($sig in "gain", "neurolkh") {
  foreach ($f in "0.02", "0.04", "0.06", "0.10") {
    foreach ($p in "0.5", "0.75", "0.9") { $grid += "${sig}_global@${f}:${p}" }
  }
}
$G = $grid -join ","
Run "G1" "dev5_grid_vlsi_primary_10s.log" @("v5_experiment.py", "--suite", "vlsi_primary", "--tag", "dev5_grid_vlsi_primary_10s", "--mode", "time", "--budget", "10", "--seeds", "3", "--workers", "4", "--strategies", $G)
Run "G2" "dev5_grid_synth1000_10s.log" @("v5_experiment.py", "--suite", "synth_1000", "--tag", "dev5_grid_synth1000_10s", "--mode", "time", "--budget", "10", "--seeds", "3", "--workers", "4", "--strategies", $G)
Run "SEL" "v5_select.log" @("$L\select_v5_rules.py")

$sel = Get-Content "$L\v5_selected_rules.json" -Raw | ConvertFrom-Json
$GA = if ($sel.gain.selected -eq "0.06:0.75") { "gain_global" } else { "gain_global@" + $sel.gain.selected }
$NL = $sel.neurolkh.selected
$E = "excess_global@0.06:0.5"
$GN = "gnn_global@0.04:0.5"

$s2 = @("random_global", $E, $GN, "gain_global")
if ($GA -ne "gain_global") { $s2 += $GA }
$s2 += @("neurolkh_global@$NL", "neurolkhfree_global@$NL")
Run "S2" "holdout2_v5_equal_10s.log" @("v5_experiment.py", "--suite", "holdout2", "--tag", "holdout2_v5_equal_10s", "--mode", "time", "--budget", "10", "--seeds", "10", "--workers", "4", "--strategies", ($s2 -join ","))

$s3 = "random_global,linprior_global,onlinewarm_global,onlinepos_global,onlinesucc_global"
Run "S3" "holdout2_v5_credit_10s.log" @("v5_experiment.py", "--suite", "holdout2", "--tag", "holdout2_v5_credit_10s", "--mode", "time", "--budget", "10", "--seeds", "10", "--workers", "4", "--strategies", $s3)

$s1 = @("random_global", $E, "knn_global@0.04:0.75", $GN, $GA, "neurolkh_global@$NL") -join ","
Run "S1" "holdout2_v5_kicks20k.log" @("v5_experiment.py", "--suite", "holdout2", "--tag", "holdout2_v5_kicks20k", "--mode", "kicks", "--kicks", "20000", "--seeds", "5", "--workers", "4", "--strategies", $s1)

Run "S4" "holdout2_v5_single_10s.log" @("v5_experiment.py", "--suite", "holdout2", "--tag", "holdout2_v5_single_10s", "--mode", "time", "--budget", "10", "--seeds", "2", "--workers", "1", "--strategies", "random_global,$E,$GN")

if (-not (Has "S5_DONE")) {
  Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
  $up = $false
  for ($i = 0; $i -lt 60; $i++) { docker info *> $null; if ($LASTEXITCODE -eq 0) { $up = $true; break }; Start-Sleep 5 }
  if (-not $up) { Done "S5_FAIL_DOCKER"; exit 1 }
  $R = (Resolve-Path .).Path
  docker run --rm -v "${R}:/work" -w /work debian:bookworm-slim sh /work/results/kick_targeting/lkh_b3/run_all.sh *>> "$L\lkh_b3.log"
  if ($LASTEXITCODE -ne 0) { Done "S5_FAIL"; exit 1 }
  & py -3.13 lkh_b3_experiment.py collect *>> "$L\lkh_b3.log"
  Done "S5_DONE"
  docker desktop stop *> $null
}

Run "ANALYSIS" "v5_analysis.log" @("$L\analysis_v5.py")
Done "ALL_DONE"
