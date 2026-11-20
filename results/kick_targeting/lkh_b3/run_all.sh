#!/bin/sh
set -e
cd /work/results/kick_targeting/lkh_b3
xargs -P 4 -L 1 sh run_task.sh < cand_tasks.txt
xargs -P 4 -L 1 sh run_task.sh < tasks.txt
echo B3_DONE
