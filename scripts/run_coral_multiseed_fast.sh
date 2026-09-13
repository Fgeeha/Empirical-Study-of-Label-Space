#!/bin/bash
PYTHON="/home/nkolesnikov/.cache/pypoetry/virtualenvs/plant-disease-eccv2026-vTaypfzd-py3.12/bin/python"
LOGDIR="logs/coral_multiseed"
mkdir -p "$LOGDIR"

echo "[$(date)] Starting CORAL multi-seed runs (fast)" | tee -a "$LOGDIR/progress_fast.log"

for SEED in 43 44; do
    for CFG in "Fuzzy 1.0" "hybrid 0.01" "sbert 0.01"; do
        STRATEGY=$(echo $CFG | cut -d' ' -f1)
        LAMBDA=$(echo $CFG | cut -d' ' -f2)
        LOGFILE="$LOGDIR/coral_${STRATEGY}_lambda${LAMBDA}_seed${SEED}.log"
        echo "[$(date)] strategy=$STRATEGY lambda=$LAMBDA seed=$SEED" | tee -a "$LOGDIR/progress_fast.log"
        $PYTHON scripts/6.3_train_coral.py             --strategy "$STRATEGY"             --target plantdoc             --lambda_coral "$LAMBDA"             --epochs 20             --seed "$SEED"             2>&1 | tee "$LOGFILE"
        echo "[$(date)] DONE: $STRATEGY lambda=$LAMBDA seed=$SEED" | tee -a "$LOGDIR/progress_fast.log"
    done
done

echo "[$(date)] All done" | tee -a "$LOGDIR/progress_fast.log"
