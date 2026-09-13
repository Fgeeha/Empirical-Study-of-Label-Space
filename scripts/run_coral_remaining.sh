#!/bin/bash
# Run the 5 remaining CORAL configs after Fuzzy/seed43 completes.
# Must be called only when GPU is free (after PIDs 398388 and 403129 finish).
PYTHON=~/.cache/pypoetry/virtualenvs/plant-disease-eccv2026-vTaypfzd-py3.12/bin/python
LOGDIR="logs/coral_multiseed"
mkdir -p "$LOGDIR"

echo "[$(date)] Starting remaining 5 CORAL configs" | tee -a "$LOGDIR/progress_remaining.log"

# Seed 43: hybrid and sbert (Fuzzy already running as PID 398388)
for CFG in "hybrid 0.01" "sbert 0.01"; do
    STRATEGY=$(echo $CFG | cut -d' ' -f1)
    LAMBDA=$(echo $CFG | cut -d' ' -f2)
    LOGFILE="$LOGDIR/coral_${STRATEGY}_lambda${LAMBDA}_seed43.log"
    echo "[$(date)] strategy=$STRATEGY lambda=$LAMBDA seed=43" | tee -a "$LOGDIR/progress_remaining.log"
    $PYTHON scripts/6.3_train_coral.py \
        --strategy "$STRATEGY" --target plantdoc \
        --lambda_coral "$LAMBDA" --epochs 20 --seed 43 \
        2>&1 | tee "$LOGFILE"
    echo "[$(date)] DONE: $STRATEGY lambda=$LAMBDA seed=43" | tee -a "$LOGDIR/progress_remaining.log"
done

# Seed 44: all 3 strategies
for CFG in "Fuzzy 1.0" "hybrid 0.01" "sbert 0.01"; do
    STRATEGY=$(echo $CFG | cut -d' ' -f1)
    LAMBDA=$(echo $CFG | cut -d' ' -f2)
    LOGFILE="$LOGDIR/coral_${STRATEGY}_lambda${LAMBDA}_seed44.log"
    echo "[$(date)] strategy=$STRATEGY lambda=$LAMBDA seed=44" | tee -a "$LOGDIR/progress_remaining.log"
    $PYTHON scripts/6.3_train_coral.py \
        --strategy "$STRATEGY" --target plantdoc \
        --lambda_coral "$LAMBDA" --epochs 20 --seed 44 \
        2>&1 | tee "$LOGFILE"
    echo "[$(date)] DONE: $STRATEGY lambda=$LAMBDA seed=44" | tee -a "$LOGDIR/progress_remaining.log"
done

echo "[$(date)] All remaining CORAL configs complete" | tee -a "$LOGDIR/progress_remaining.log"
