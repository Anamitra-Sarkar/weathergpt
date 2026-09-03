#!/usr/bin/env bash
# Waits for the D1 corpus to finish, then launches the three trainers that
# depend on it.  Written to backup/ rather than the scratchpad so it survives.
set -u
cd /home/anamitra/weathergpt
TARGET=${TARGET:-125}
STALL_LIMIT=${STALL_LIMIT:-12}     # ~12 minutes with no new location = done or stuck
last=-1; stalls=0
for i in $(seq 1 240); do
  n=$(timeout 90 modal volume ls weathergpt-data d1_mos 2>/dev/null | grep -c 'loc_')
  echo "$(date +%H:%M:%S) d1=$n/127"
  if [ "$n" -ge "$TARGET" ]; then echo "REACHED $n"; break; fi
  if [ "$n" -eq "$last" ]; then stalls=$((stalls+1)); else stalls=0; fi
  if [ "$stalls" -ge "$STALL_LIMIT" ]; then echo "STALLED at $n"; break; fi
  last=$n
  sleep 60
done
n=$(timeout 90 modal volume ls weathergpt-data d1_mos 2>/dev/null | grep -c 'loc_')
echo "launching trainers on $n locations"
modal run --detach modal_jobs/train_mos.py --epochs 30           > /tmp/m2.log 2>&1 &
sleep 20
modal run --detach modal_jobs/train_calibration.py --epochs 40   > /tmp/m4.log 2>&1 &
sleep 20
modal run --detach modal_jobs/train_trust_ranker.py              > /tmp/m5.log 2>&1 &
wait
echo "ALL TRAINERS EXITED"
