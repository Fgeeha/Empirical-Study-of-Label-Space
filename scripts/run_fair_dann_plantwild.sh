#!/bin/bash
# Re-run DANN with fair protocol (source-val selection) + checkpoint saving.
# After completion, run evaluate_plantwild.py to get PlantWild results.
PYTHON=~/.cache/pypoetry/virtualenvs/plant-disease-eccv2026-vTaypfzd-py3.12/bin/python
LOGDIR="logs/fair_dann_plantwild"
mkdir -p "$LOGDIR"

echo "[$(date)] Starting fair DANN re-run (3 seeds)" | tee "$LOGDIR/progress.log"

for SEED in 42 43 44; do
    echo "[$(date)] seed=$SEED START" | tee -a "$LOGDIR/progress.log"
    $PYTHON scripts/train_uda.py \
        --method dann --target plantdoc \
        --seed $SEED --strategy hybrid --epochs 20 \
        2>&1 | tee "$LOGDIR/dann_seed${SEED}.log"
    echo "[$(date)] seed=$SEED DONE" | tee -a "$LOGDIR/progress.log"
done

echo "[$(date)] Training complete. Evaluating on PlantWild..." | tee -a "$LOGDIR/progress.log"
$PYTHON scripts/evaluate_plantwild.py --method dann --seeds 42 43 44 \
    2>&1 | tee "$LOGDIR/plantwild_eval.log"
echo "[$(date)] ALL DONE" | tee -a "$LOGDIR/progress.log"
