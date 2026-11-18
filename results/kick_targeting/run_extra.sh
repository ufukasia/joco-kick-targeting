#!/usr/bin/env bash
# Robustness runs added after the primary result (NOT pre-registered; reported as exploratory).
cd "$(dirname "$0")/../.."
L=results/kick_targeting
until [ -f $L/secondary.done ]; do sleep 60; done
py -3.13 kick_targeting_experiment.py run --suite vlsi_primary --tag vlsi_primary_fresh_10s --seeds 5 --budget 10 --workers 4 --strategies random_global,gnn_global,knn_global --extra '{"overhead_cap": 1000000000.0}' >> $L/vlsi_primary_fresh_10s.log 2>&1
py -3.13 kick_targeting_experiment.py run --suite vlsi_primary --tag vlsi_primary_60s --seeds 3 --budget 60 --workers 4 --strategies random_global,gnn_global,knn_global >> $L/vlsi_primary_60s.log 2>&1
echo EXTRA_DONE >> $L/extra.done
