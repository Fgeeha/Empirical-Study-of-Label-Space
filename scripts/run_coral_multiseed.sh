#!/bin/bash
# Run CORAL for seeds 43 and 44 on best lambda per strategy.
# Seed 42 already exists. This adds seeds 43/44 to enable mean±std.
# Best lambdas: Fuzzy=1.0, Hybrid=0.01, SBERT=0.01 (from T2_coral_sweep)

set -e
LOGDIR="logs/coral_multiseed"
mkdir -p "$LOGDIR"

echo "[$(date)] Starting CORAL multi-seed runs" | tee "$LOGDIR/progress.log"

for SEED in 43 44; do
    for CFG in "Fuzzy 1.0" "hybrid 0.01" "sbert 0.01"; do
        STRATEGY=$(echo $CFG | cut -d' ' -f1)
        LAMBDA=$(echo $CFG | cut -d' ' -f2)
        LOGFILE="$LOGDIR/coral_${STRATEGY}_lambda${LAMBDA}_seed${SEED}.log"
        echo "[$(date)] Running: strategy=$STRATEGY lambda=$LAMBDA seed=$SEED" | tee -a "$LOGDIR/progress.log"
        poetry run python scripts/6.3_train_coral.py \
            --strategy "$STRATEGY" \
            --target plantdoc \
            --lambda_coral "$LAMBDA" \
            --epochs 20 \
            --seed "$SEED" \
            2>&1 | tee "$LOGFILE"
        echo "[$(date)] Done: $STRATEGY lambda=$LAMBDA seed=$SEED" | tee -a "$LOGDIR/progress.log"
    done
done

echo "[$(date)] All CORAL multi-seed runs complete" | tee -a "$LOGDIR/progress.log"
