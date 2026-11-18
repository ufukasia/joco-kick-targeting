#!/usr/bin/env bash
# Exploratory: cheap learned scorer + large-instance scale (added 2026-11-16 after all primary/secondary runs).
cd "$(dirname "$0")/../.."
L=results/kick_targeting
until [ -f $L/cheap_mechanism_summary.json ]; do sleep 20; done
py -3.13 cheap_targeting_experiment.py --suite vlsi_primary --tag vlsi_primary_cheap_10s --seeds 5 --budget 10 --workers 4 --strategies random_global,knn_global,gnn_global,cheap_global >> $L/vlsi_primary_cheap_10s.log 2>&1
echo CHEAP_DONE >> $L/cheap.done
py -3.13 cheap_targeting_experiment.py --suite vlsi_large --tag vlsi_large_30s --seeds 3 --budget 30 --workers 4 --strategies random_global,knn_global,gnn_global,cheap_global,random_local,knn_local >> $L/vlsi_large_30s.log 2>&1
echo LARGE_DONE >> $L/large.done
