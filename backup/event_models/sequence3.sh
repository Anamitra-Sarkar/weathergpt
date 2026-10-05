#!/bin/bash
# Push the four extension shards as Kaggle CPU slots free up (max 5 concurrent sessions).
cd /home/anamitra/weathergpt
for tag in E1 E2 E3 E4; do
  until kaggle kernels push -p "backup/event_models/ext_$tag" 2>&1 | tail -1 | grep -q "successfully pushed"; do
    echo "$(date +%T) ext_$tag waiting for a slot"; sleep 60
  done
  echo "$(date +%T) pushed ext_$tag"
done
echo SEQUENCE3_DONE
