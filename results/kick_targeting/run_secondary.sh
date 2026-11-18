#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
L=results/kick_targeting
W=4
run() { py -3.13 kick_targeting_experiment.py run --suite $1 --tag $2 --seeds $3 --budget $4 --workers $W >> $L/$2.log 2>&1; }
run vlsi_primary vlsi_primary_10s 10 10
py -3.13 lkh_runner.py ref --workers $W >> $L/lkh_ref.log 2>&1
py -3.13 kick_mechanism_analysis.py > $L/mechanism.log 2>&1
run vlsi_old vlsi_old_10s 10 10
run synth_1000 synth_1000_10s 5 10
run tsplib72 tsplib72_10s 5 10
run synth_holdout synth_holdout_1s 5 1
py -3.13 lkh_runner.py baseline --workers 3 >> $L/lkh_baseline.log 2>&1
py -3.13 kick_targeting_analysis.py vlsi_primary_10s vlsi_old_10s tsplib72_10s synth_holdout_1s synth_1000_10s > $L/analysis_all.log 2>&1
echo SECONDARY_DONE >> $L/secondary.done
