#!/bin/bash
# Re-push validation (fixed) and the GEFS probe as soon as Kaggle CPU slots free up.
cd /home/anamitra/weathergpt
push() { until kaggle kernels push -p "$1" 2>&1 | tail -1 | grep -q "successfully pushed"; do echo "$(date +%T) waiting for a slot: $1"; sleep 60; done; echo "$(date +%T) pushed $1"; }
push backup/event_models/validate_smoke
push backup/event_models/probe4
echo SEQUENCE2_DONE
