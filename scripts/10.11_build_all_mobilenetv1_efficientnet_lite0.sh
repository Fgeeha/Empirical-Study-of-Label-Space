#!/usr/bin/env bash
# 10.11 Master pipeline: MobileNetV1 + EfficientNet-Lite0
#
# Runs the full pipeline for both EdgeTPU-compatible architectures:
#   1. Train MobileNetV1     (PyTorch/timm, GPU)
#   2. Export MobileNetV1    INT8 TFLite  (TF/Keras, CPU)
#   3. Train EfficientNet-Lite0  (PyTorch/timm, GPU)
#   4. Export EfficientNet-Lite0 INT8 TFLite  (TF/Keras, CPU)
#   5. Compile all models for EdgeTPU
#
# Training uses PyTorch (CUDA on GPU) -- fast.
# TFLite export uses TF/Keras (CPU only, CUDA_VISIBLE_DEVICES="" inside scripts)
# because TF 2.20 doesn't have pre-compiled kernels for RTX 5070 Ti (sm_120).
#
# Prerequisites:
#   - splits/{strategy}/pv_train.csv, pv_val.csv, pv_calib_10pct.csv
#   - edgetpu_compiler installed
#   - poetry environment with torch, timm, tensorflow>=2.20
#
# Usage:
#   bash scripts/10.11_build_all_mobilenetv1_efficientnet_lite0.sh
#   bash scripts/10.11_build_all_mobilenetv1_efficientnet_lite0.sh --force
#   bash scripts/10.11_build_all_mobilenetv1_efficientnet_lite0.sh --skip-train
#
# Estimated time:
#   Training: ~5-10 min per model on GPU (6 models total, ~30-60 min)
#   Export + EdgeTPU compile: ~10 minutes

set -euo pipefail

FORCE=false
SKIP_TRAIN=false

for arg in "$@"; do
    case "$arg" in
        --force)      FORCE=true ;;
        --skip-train) SKIP_TRAIN=true ;;
        *)            echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_DIR}"

STRATEGIES=("hybrid" "Fuzzy" "sbert")
MODELS_DIR="models"
EXPORT_DIR="export"

# Counters
TOTAL=0
OK=0
FAIL=0

log_step()  { echo -e "\n\033[1;34m>>> $*\033[0m"; }
log_info()  { echo -e "\033[32m[INFO]\033[0m $*"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $*"; }

run_step() {
    local desc="$1"
    shift
    ((TOTAL++)) || true
    log_step "${desc}"
    if "$@"; then
        ((OK++)) || true
        log_info "Done: ${desc}"
    else
        ((FAIL++)) || true
        log_error "FAILED: ${desc}"
    fi
}

# =========================================================
# Phase 1: Train MobileNetV1 (PyTorch/timm, GPU)
# =========================================================
if [ "${SKIP_TRAIN}" = false ]; then
    for strategy in "${STRATEGIES[@]}"; do
        CKPT="${MODELS_DIR}/mobilenetv1_pv_${strategy}_best.pt"
        if [ -f "${CKPT}" ] && [ "${FORCE}" = false ]; then
            log_info "Already exists: ${CKPT} -- skipping (use --force to rebuild)"
        else
            run_step "Train MobileNetV1 [${strategy}] (PyTorch GPU)" \
                poetry run python scripts/10.6_train_mobilenetv1_pv.py --strategy "${strategy}"
        fi
    done
fi

# =========================================================
# Phase 2: Export MobileNetV1 INT8 TFLite (TF CPU)
# =========================================================
for strategy in "${STRATEGIES[@]}"; do
    TFLITE="${EXPORT_DIR}/mobilenetv1_int8_ptq_${strategy}.tflite"
    if [ -f "${TFLITE}" ] && [ "${FORCE}" = false ]; then
        log_info "Already exists: ${TFLITE} -- skipping"
    else
        run_step "Export MobileNetV1 INT8 [${strategy}] (TF CPU)" \
            poetry run python scripts/10.7_export_tflite_int8_mobilenetv1.py --strategy "${strategy}"
    fi
done

# =========================================================
# Phase 3: Train EfficientNet-Lite0 (PyTorch/timm, GPU)
# =========================================================
if [ "${SKIP_TRAIN}" = false ]; then
    for strategy in "${STRATEGIES[@]}"; do
        CKPT="${MODELS_DIR}/efficientnet_lite0_pv_${strategy}_best.pt"
        if [ -f "${CKPT}" ] && [ "${FORCE}" = false ]; then
            log_info "Already exists: ${CKPT} -- skipping (use --force to rebuild)"
        else
            run_step "Train EfficientNet-Lite0 [${strategy}] (PyTorch GPU)" \
                poetry run python scripts/10.8_train_efficientnet_lite0_pv.py --strategy "${strategy}"
        fi
    done
fi

# =========================================================
# Phase 4: Export EfficientNet-Lite0 INT8 TFLite (TF CPU)
# =========================================================
for strategy in "${STRATEGIES[@]}"; do
    TFLITE="${EXPORT_DIR}/efficientnet_lite0_int8_ptq_${strategy}.tflite"
    if [ -f "${TFLITE}" ] && [ "${FORCE}" = false ]; then
        log_info "Already exists: ${TFLITE} -- skipping"
    else
        run_step "Export EfficientNet-Lite0 INT8 [${strategy}] (TF CPU)" \
            poetry run python scripts/10.9_export_tflite_int8_efficientnet_lite0.py --strategy "${strategy}"
    fi
done

# =========================================================
# Phase 5: Compile EdgeTPU
# =========================================================
run_step "EdgeTPU compilation (MobileNetV1 + EfficientNet-Lite0)" \
    bash scripts/10.10_compile_edgetpu_mobilenetv1_efficientnet_lite0.sh

# =========================================================
# Summary
# =========================================================
echo ""
echo "==========================================================="
echo "  Pipeline Summary"
echo "==========================================================="
echo "  Total steps:  ${TOTAL}"
echo "  Succeeded:    ${OK}"
echo "  Failed:       ${FAIL}"
echo ""
echo "  Output models (export/):"
echo "  -------------------------------------------------------"
for strategy in "${STRATEGIES[@]}"; do
    for arch in mobilenetv1 efficientnet_lite0; do
        TFLITE="${EXPORT_DIR}/${arch}_int8_ptq_${strategy}.tflite"
        EDGETPU="${EXPORT_DIR}/${arch}_int8_ptq_${strategy}_edgetpu.tflite"
        if [ -f "${TFLITE}" ]; then
            SIZE=$(du -h "${TFLITE}" | cut -f1)
            echo "    ${TFLITE}  (${SIZE})"
        fi
        if [ -f "${EDGETPU}" ]; then
            SIZE=$(du -h "${EDGETPU}" | cut -f1)
            echo "    ${EDGETPU}  (${SIZE})"
        fi
    done
done
echo "==========================================================="

if [ "${FAIL}" -gt 0 ]; then
    log_error "${FAIL} step(s) failed!"
    exit 1
fi

echo ""
log_info "All done."
