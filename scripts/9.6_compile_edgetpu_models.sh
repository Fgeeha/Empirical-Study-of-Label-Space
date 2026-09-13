#!/bin/bash
# =============================================================================
# EdgeTPU Model Compilation Script
# Compiles INT8 TFLite models for Google Coral EdgeTPU
#
# Prerequisites:
#   - edgetpu_compiler installed (https://coral.ai/docs/edgetpu/compiler/)
#   - INT8 quantized TFLite models in export/
#
# Usage:
#   bash scripts/9.6_compile_edgetpu_models.sh
# =============================================================================

set -euo pipefail

EXPORT_DIR="export"
OUTPUT_DIR="export/edgetpu"

echo "=============================================="
echo "EdgeTPU Model Compilation"
echo "=============================================="

# Check compiler availability
if ! command -v edgetpu_compiler &> /dev/null; then
    echo "[ERROR] edgetpu_compiler not found."
    echo "Install from: https://coral.ai/docs/edgetpu/compiler/"
    echo ""
    echo "On Debian/Ubuntu:"
    echo "  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -"
    echo "  echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list"
    echo "  sudo apt update && sudo apt install edgetpu-compiler"
    exit 1
fi

edgetpu_compiler --version

mkdir -p "${OUTPUT_DIR}"

# All EdgeTPU-compatible architectures (EfficientNet-B3 excluded: Swish + SE)
MODELS=()
for arch in mobilenetv2 mobilenetv1 efficientnet_lite0 resnet50; do
    for strat in hybrid Fuzzy sbert; do
        tflite="${EXPORT_DIR}/${arch}_int8_ptq_${strat}.tflite"
        if [ -f "$tflite" ]; then
            MODELS+=("${arch}_int8_ptq_${strat}")
        fi
    done
done

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "[WARN] No INT8 TFLite models found in ${EXPORT_DIR}/"
    echo "       Run Phase 5 (quantization) first."
    exit 0
fi

echo ""
echo "Models to compile: ${#MODELS[@]}"
for model in "${MODELS[@]}"; do
    echo "  - ${model}.tflite"
done
echo ""

COMPILED=0
FAILED=0
SKIPPED=0

for model in "${MODELS[@]}"; do
    INPUT="${EXPORT_DIR}/${model}.tflite"
    OUTPUT="${OUTPUT_DIR}/${model}_edgetpu.tflite"

    # Skip already compiled
    if [ -f "${OUTPUT}" ]; then
        SIZE_MB=$(du -h "${OUTPUT}" | cut -f1)
        echo "[SKIP] ${OUTPUT} already exists (${SIZE_MB})"
        ((SKIPPED++)) || true
        continue
    fi

    echo "----------------------------------------------"
    echo "Compiling: ${model}"
    echo "----------------------------------------------"

    if edgetpu_compiler -o "${OUTPUT_DIR}" "${INPUT}" 2>&1; then
        if [ -f "${OUTPUT}" ]; then
            SIZE_MB=$(du -h "${OUTPUT}" | cut -f1)
            echo "[OK] ${OUTPUT} (${SIZE_MB})"
            ((COMPILED++)) || true
        else
            echo "[ERROR] Compilation produced no output for ${model}"
            ((FAILED++)) || true
        fi
    else
        echo "[ERROR] edgetpu_compiler failed for ${model}"
        ((FAILED++)) || true
    fi
    echo ""
done

echo "=============================================="
echo "Summary"
echo "=============================================="
echo "Compiled: ${COMPILED}"
echo "Skipped:  ${SKIPPED}"
echo "Failed:   ${FAILED}"
echo ""

if [ $((COMPILED + SKIPPED)) -gt 0 ]; then
    echo "EdgeTPU models in: ${OUTPUT_DIR}/"
    ls -la "${OUTPUT_DIR}"/*.tflite 2>/dev/null || true
fi

# Return success if no failures
exit ${FAILED}
