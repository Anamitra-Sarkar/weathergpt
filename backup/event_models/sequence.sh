#!/bin/bash
# Push the validation kernel when the truth kernel finishes, then shard 5 when validation finishes.
cd /home/anamitra/weathergpt
status() { kaggle kernels status "anamitrasarkar007/weathergpt-$1" 2>&1 | tail -1 | sed 's/.*status //'; }
push() { until kaggle kernels push -p "$1" 2>&1 | tail -1 | grep -q "successfully pushed"; do echo "$(date +%T) push $1 waiting for a slot"; sleep 45; done; echo "$(date +%T) pushed $1"; }
until s=$(status truth-collect); case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) true;; *) false;; esac; do sleep 30; done
echo "$(date +%T) truth-collect: $s"
push backup/event_models/validate_smoke
sleep 20
until s=$(status validate-smoke); case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) true;; *) false;; esac; do sleep 20; done
echo "$(date +%T) validate-smoke: $s"
push backup/event_models/gfs_s5
echo "SEQUENCE_DONE"
