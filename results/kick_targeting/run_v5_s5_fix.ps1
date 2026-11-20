# S5 (B3, LKH-3) re-run after the CRLF defect of 2026-11-20 (see PROTOCOL_DEVIATIONS.log).
# tasks.txt was written with CRLF line ends; the last field (dref path) carried a trailing CR, so the patched
# binary could not open it and the targeted arm stopped at its first kick. Registered files are unchanged;
# the only difference is that CR characters are stripped when the task list is fed to xargs.
# All three arms are re-run in the registered shuffled order; old task logs are kept in quarantine.
Set-Location (Join-Path $PSScriptRoot "..\..")
$L = "results\kick_targeting"
$D = "$L\v5.done"
function Done($s) { "$s $(Get-Date -Format s)" | Out-File -Append -Encoding ascii $D }
$Q = "$L\quarantine_v5_lkh_crlf_20261120"
New-Item -ItemType Directory -Force "$Q\log", "$Q\holdout2_lkh_10s" | Out-Null
Get-ChildItem "$L\lkh_b3\log" -Filter "*__s*__lkh_*.log" | Move-Item -Destination "$Q\log"
Get-ChildItem "$L\holdout2_lkh_10s" -Filter "*.json" -ErrorAction SilentlyContinue | Move-Item -Destination "$Q\holdout2_lkh_10s"
Copy-Item "$L\v5_analysis.json" "$Q\v5_analysis_run1_lkh_invalid.json"
Copy-Item "$L\lkh_b3.log" "$Q\lkh_b3_run1.log"

Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
$up = $false
for ($i = 0; $i -lt 60; $i++) { docker info *> $null; if ($LASTEXITCODE -eq 0) { $up = $true; break }; Start-Sleep 5 }
if (-not $up) { Done "S5FIX_FAIL_DOCKER"; exit 1 }
$R = (Resolve-Path .).Path
docker run --rm -v "${R}:/work" -w /work/results/kick_targeting/lkh_b3 debian:bookworm-slim sh -c "tr -d '\r' < tasks.txt | xargs -P 4 -L 1 sh run_task.sh && echo B3_DONE" *>> "$L\lkh_b3_fix.log"
if ($LASTEXITCODE -ne 0) { Done "S5FIX_FAIL"; exit 1 }
& py -3.13 lkh_b3_experiment.py collect *>> "$L\lkh_b3_fix.log"
Done "S5FIX_DONE"
docker desktop stop *> $null
& py -3.13 "$L\analysis_v5.py" *>> "$L\v5_analysis_run2.log"
if ($LASTEXITCODE -ne 0) { Done "ANALYSIS2_FAIL"; exit 1 }
Done "ANALYSIS2_DONE"
