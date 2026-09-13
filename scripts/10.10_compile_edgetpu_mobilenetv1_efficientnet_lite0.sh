#!/usr/bin/env bash
# 10.10 Compile INT8 TFLite models for Google Coral EdgeTPU
#
# Compiles MobileNetV1 and EfficientNet-Lite0 models for EdgeTPU.
# Both architectures are fully EdgeTPU-compatible:
#   - MobileNetV1: DepthwiseSeparableConv + ReLU6
#   - EfficientNet-Lite0: MBConv + ReLU6 (no Swish, no SE)
#
# Prerequisites:
#   - edgetpu_compiler installed
#   - export/mobilenetv1_int8_ptq_{strategy}.tflite   (from 10.7)
#   - export/efficientnet_lite0_int8_ptq_{strategy}.tflite  (from 10.9)
#
# Usage:
#   bash scripts/10.10_compile_edgetpu_mobilenetv1_efficientnet_lite0.sh

set -euo pipefail

EXPORT_DIR="export"
STRATEGIES=("hybrid" "Fuzzy" "sbert")

# All models are EdgeTPU-compatible (ReLU6 only, no Swish/SE)
ARCHITECTURES=("mobilenetv1" "efficientnet_lite0")

log_info()  { echo -e "\033[32m[INFO]\033[0m $*"; }
log_warn()  { echo -e "\033[33m[WARN]\033[0m $*"; }
log_error() { echo -e "\033[31m[ERROR]\033[0m $*"; }

# -- Check edgetpu_compiler --
if ! command -v edgetpu_compiler &>/dev/null; then
    log_error "edgetpu_compiler not found. Install:"
    echo "  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -"
    echo "  echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list"
    echo "  sudo apt update && sudo apt install edgetpu-compiler"
    exit 1
fi

log_info "edgetpu_compiler found: $(edgetpu_compiler --version 2>&1 | head -1)"

COMPILED=0
FAILED=0
SKIPPED=0

for arch in "${ARCHITECTURES[@]}"; do
    for strategy in "${STRATEGIES[@]}"; do
        INPUT="${EXPORT_DIR}/${arch}_int8_ptq_${strategy}.tflite"
        OUTPUT="${EXPORT_DIR}/${arch}_int8_ptq_${strategy}_edgetpu.tflite"

        if [ ! -f "${INPUT}" ]; then
            log_warn "Input not found, skipping: ${INPUT}"
            ((SKIPPED++)) || true
            continue
        fi

        if [ -f "${OUTPUT}" ]; then
            log_info "Already exists: ${OUTPUT} (use --force to rebuild)"
            ((SKIPPED++)) || true
            continue
        fi

        log_info "Compiling: ${INPUT}"

        # Compile into export/ directory
        if edgetpu_compiler -s -o "${EXPORT_DIR}" "${INPUT}" 2>&1; then
            if [ -f "${OUTPUT}" ]; then
                SIZE_MB=$(du -m "${OUTPUT}" | cut -f1)
                log_info "OK: ${OUTPUT} (${SIZE_MB} MB)"
                ((COMPILED++)) || true
            else
                log_error "Compiler succeeded but output not found: ${OUTPUT}"
                ((FAILED++)) || true
            fi
        else
            log_error "FAILED: ${INPUT}"
            ((FAILED++)) || true
        fi
    done
done

echo ""
echo "==========================================="
echo "  EdgeTPU Compilation Summary"
echo "==========================================="
echo "  Compiled:  ${COMPILED}"
echo "  Failed:    ${FAILED}"
echo "  Skipped:   ${SKIPPED}"
echo "==========================================="

if [ "${FAILED}" -gt 0 ]; then
    exit 1
fi
