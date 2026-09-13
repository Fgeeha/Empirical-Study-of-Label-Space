#!/bin/bash
# UDA benchmark runner — Wave 1: seed=42, then seeds 43+44
# All methods: source-val model selection (fair protocol)
# Usage: bash scripts/run_uda_benchmark.sh

set -e
PYTHON=~/.cache/pypoetry/virtualenvs/plant-disease-eccv2026-vTaypfzd-py3.12/bin/python
LOGDIR="logs/uda_benchmark"
mkdir -p "$LOGDIR"

run_method() {
    METHOD=$1
    SEED=$2
    LOG="$LOGDIR/${METHOD}_hybrid_plantdoc_seed${SEED}.log"
    echo "[$(date)] START: method=$METHOD seed=$SEED" | tee -a "$LOGDIR/progress.log"
    $PYTHON scripts/train_uda.py \
        --method "$METHOD" \
        --target plantdoc \
        --seed "$SEED" \
        --strategy hybrid \
        --epochs 20 \
        2>&1 | tee "$LOG"
    echo "[$(date)] DONE:  method=$METHOD seed=$SEED" | tee -a "$LOGDIR/progress.log"
}

echo "[$(date)] === UDA Benchmark Wave 1+2 ===" | tee "$LOGDIR/progress.log"

# AdaBN is fast (no training) — run all seeds at once
for SEED in 42 43 44; do
    run_method adabn $SEED
done

# Wave 1: seed=42 for training methods
for METHOD in coral dann mcc cdan; do
    run_method "$METHOD" 42
done

echo "[$(date)] === Wave 1 complete. Starting Wave 2 ===" | tee -a "$LOGDIR/progress.log"

# Wave 2: seeds 43 and 44 for all training methods
for SEED in 43 44; do
    for METHOD in coral dann mcc cdan; do
        run_method "$METHOD" $SEED
    done
done

echo "[$(date)] === All runs complete ===" | tee -a "$LOGDIR/progress.log"
