#!/bin/sh
arm=$1; par=$2; log=$3; dref=$4
[ -s "$log" ] && grep -q 'Cost.min' "$log" && exit 0
BIN=/work/third_party/LKH3_targeted/LKH
if [ "$arm" = lkh_ils_excess ]; then
  LKH_KICK_DREF=$dref LKH_KICK_FRAC=0.06 LKH_KICK_PROB=0.5 $BIN "$par" > "$log.tmp" 2>&1
else
  $BIN "$par" > "$log.tmp" 2>&1
fi
mv "$log.tmp" "$log"
