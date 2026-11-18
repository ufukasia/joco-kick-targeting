#!/usr/bin/env bash
# Exploratory: gain-trained cheap scorer (cheap_scorer.py train_gain), added 2026-11-16 after the pool-composition diagnostic.
cd "$(dirname "$0")/../.."
L=results/kick_targeting
until [ -f $L/large.done ]; do sleep 20; done
py -3.13 cheap_targeting_experiment.py --suite vlsi_primary --tag vlsi_primary_gain_10s --seeds 5 --budget 10 --workers 4 --strategies random_global,knn_global,gain_global >> $L/vlsi_primary_gain_10s.log 2>&1
echo GAIN_DONE >> $L/gain.done
