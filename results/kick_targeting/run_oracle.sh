#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
L=results/kick_targeting
until [ -f $L/extra.done ]; do sleep 60; done
py -3.13 kick_oracle_experiment.py --seeds 3 --workers 4 >> $L/vlsi_primary_oracle_10s.log 2>&1
echo ORACLE_DONE >> $L/oracle.done
