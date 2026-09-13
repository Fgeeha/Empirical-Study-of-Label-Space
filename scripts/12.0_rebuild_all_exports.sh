#!/usr/bin/env bash
# =============================================================================
# 12.0 Rebuild all TFLite exports and EdgeTPU models
#
# Fixes critical issues:
#   1. ResNet-50 SavedModel weight transfer (T3 accuracy bug)
#   2. ResNet-50 INT8 TFLite for EdgeTPU
#   3. MobileNetV2 INT8 TFLite (backbone weights were missing)
#   4. MobileNetV1 INT8 TFLite (from 10.7)
#   5. EfficientNet-Lite0 INT8 TFLite (from 10.9)
#   6. EdgeTPU compilation for compatible architectures
#   7. Re-evaluate T3 (FP32 vs INT8)
#
# Prerequisites:
#   - export/*_weights.npz files for all architectures
#   - splits/{strategy}/pv_calib_10pct.csv
#   - splits/{strategy}/pv_test.csv
#   - splits/{strategy}/target_plantdoc_test.csv
#
# Usage:
#   bash scripts/12.0_rebuild_all_exports.sh [--strategy hybrid|Fuzzy|sbert|all]
#   bash scripts/12.0_rebuild_all_exports.sh --strategy all
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Force CPU for TensorFlow (RTX 5070 Ti compute 12.0 not supported by TF 2.20)
export CUDA_VISIBLE_DEVICES=""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${CYAN}========== $* ==========${NC}"; }

# Parse arguments
STRATEGY="all"
SKIP_EDGETPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --strategy) STRATEGY="$2"; shift 2 ;;
        --skip-edgetpu) SKIP_EDGETPU=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$STRATEGY" = "all" ]; then
    STRATEGIES=("hybrid" "Fuzzy" "sbert")
else
    STRATEGIES=("$STRATEGY")
fi

log_info "Strategies: ${STRATEGIES[*]}"
log_info "Project dir: $PROJECT_DIR"

TOTAL_OK=0
TOTAL_FAIL=0

run_step() {
    local desc="$1"
    shift
    log_info "$desc"
    if "$@" 2>&1; then
        ((TOTAL_OK++)) || true
        log_info "OK: $desc"
    else
        ((TOTAL_FAIL++)) || true
        log_error "FAILED: $desc"
    fi
}

# =============================================================================
# STEP 1: ResNet-50 SavedModel (fixes T3 accuracy bug)
# =============================================================================
log_step "STEP 1: ResNet-50 SavedModel (8.2b) -- fixes T3"

for strategy in "${STRATEGIES[@]}"; do
    NPZ="export/resnet50_${strategy}_weights.npz"
    if [ ! -f "$NPZ" ]; then
        log_warn "Weights not found: $NPZ -- skipping"
        continue
    fi
    run_step "ResNet-50 SavedModel ($strategy)" \
        poetry run python scripts/8.2b_build_tf_savedmodel.py --strategy "$strategy"
done

# =============================================================================
# STEP 2: ResNet-50 FP32 + INT8 TFLite (T3 evaluation path)
# =============================================================================
log_step "STEP 2: ResNet-50 FP32 + INT8 TFLite (8.3 + 8.4) -- for T3"

for strategy in "${STRATEGIES[@]}"; do
    SM="export/resnet50_tf_${strategy}"
    if [ ! -d "$SM" ]; then
        log_warn "SavedModel not found: $SM -- skipping"
        continue
    fi

    run_step "ResNet-50 FP32 TFLite ($strategy)" \
        poetry run python scripts/8.3_export_tflite_fp32.py --strategy "$strategy"

    run_step "ResNet-50 INT8 TFLite via PTQ ($strategy)" \
        poetry run python scripts/8.4_export_tflite_int8.py --strategy "$strategy"
done

# =============================================================================
# STEP 3: Re-evaluate T3 (FP32 vs INT8)
# =============================================================================
log_step "STEP 3: Re-evaluate T3 (8.5)"

for strategy in "${STRATEGIES[@]}"; do
    FP32="export/model_fp32_${strategy}.tflite"
    INT8="export/model_int8_ptq_${strategy}.tflite"
    if [ ! -f "$FP32" ] || [ ! -f "$INT8" ]; then
        log_warn "TFLite models not found for $strategy -- skipping T3 eval"
        continue
    fi
    run_step "T3 eval ($strategy)" \
        poetry run python scripts/8.5_eval_tflite_quant.py --strategy "$strategy"
done

# =============================================================================
# STEP 4: EdgeTPU-compatible INT8 TFLite exports
# =============================================================================
log_step "STEP 4: INT8 TFLite for EdgeTPU (all architectures)"

# 4a. ResNet-50 (from 10.4, no preprocess_input)
for strategy in "${STRATEGIES[@]}"; do
    NPZ="export/resnet50_${strategy}_weights.npz"
    if [ ! -f "$NPZ" ]; then
        log_warn "ResNet-50 weights not found: $NPZ -- skipping"
        continue
    fi
    run_step "ResNet-50 INT8 EdgeTPU ($strategy)" \
        poetry run python scripts/10.4_export_tflite_int8_resnet50.py --strategy "$strategy"
done

# 4b. MobileNetV2 (from 9.2, fixed backbone weights)
for strategy in "${STRATEGIES[@]}"; do
    NPZ="export/mobilenetv2_${strategy}_weights.npz"
    if [ ! -f "$NPZ" ]; then
        log_warn "MobileNetV2 weights not found: $NPZ -- skipping"
        continue
    fi
    run_step "MobileNetV2 INT8 EdgeTPU ($strategy)" \
        poetry run python scripts/9.2_export_tflite_int8_mobilenet.py --strategy "$strategy"
done

# 4c. MobileNetV1 (from 10.7, already correct)
for strategy in "${STRATEGIES[@]}"; do
    NPZ="export/mobilenetv1_${strategy}_weights.npz"
    if [ ! -f "$NPZ" ]; then
        log_warn "MobileNetV1 weights not found: $NPZ -- skipping"
        continue
    fi
    run_step "MobileNetV1 INT8 EdgeTPU ($strategy)" \
        poetry run python scripts/10.7_export_tflite_int8_mobilenetv1.py --strategy "$strategy"
done

# 4d. EfficientNet-Lite0 (from 10.9, already correct)
for strategy in "${STRATEGIES[@]}"; do
    NPZ="export/efficientnet_lite0_${strategy}_weights.npz"
    if [ ! -f "$NPZ" ]; then
        log_warn "EfficientNet-Lite0 weights not found: $NPZ -- skipping"
        continue
    fi
    run_step "EfficientNet-Lite0 INT8 EdgeTPU ($strategy)" \
        poetry run python scripts/10.9_export_tflite_int8_efficientnet_lite0.py --strategy "$strategy"
done

# 4e. EfficientNet-B3 (CPU only, NOT EdgeTPU-compatible)
for strategy in "${STRATEGIES[@]}"; do
    NPZ="export/efficientnet_${strategy}_weights.npz"
    if [ ! -f "$NPZ" ]; then
        log_warn "EfficientNet-B3 weights not found: $NPZ -- skipping"
        continue
    fi
    # NOTE: 10.3 weight transfer for EfficientNet-B3 has naming mismatch
    # between torchvision and tf.keras.applications. EfficientNet-B3 is NOT
    # EdgeTPU-compatible (Swish + SE), so this is low priority.
    # Run anyway to produce the file (accuracy may be degraded).
    run_step "EfficientNet-B3 INT8 CPU-only ($strategy)" \
        poetry run python scripts/10.3_export_tflite_int8_efficientnet.py --strategy "$strategy"
done

# =============================================================================
# STEP 5: EdgeTPU Compilation
# =============================================================================
if [ "$SKIP_EDGETPU" = true ]; then
    log_step "STEP 5: EdgeTPU Compilation -- SKIPPED (--skip-edgetpu)"
else
    log_step "STEP 5: EdgeTPU Compilation"

    if ! command -v edgetpu_compiler &>/dev/null; then
        log_warn "edgetpu_compiler not found. Install:"
        echo "  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -"
        echo "  echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list"
        echo "  sudo apt update && sudo apt install edgetpu-compiler"
        log_warn "Skipping EdgeTPU compilation."
    else
        log_info "edgetpu_compiler found: $(edgetpu_compiler --version 2>&1 | head -1)"
        mkdir -p export/edgetpu

        # EdgeTPU-compatible architectures (ReLU/ReLU6 only)
        EDGETPU_ARCHS=("mobilenetv2" "mobilenetv1" "efficientnet_lite0" "resnet50")

        for arch in "${EDGETPU_ARCHS[@]}"; do
            for strategy in "${STRATEGIES[@]}"; do
                INPUT="export/${arch}_int8_ptq_${strategy}.tflite"
                OUTPUT="export/edgetpu/${arch}_int8_ptq_${strategy}_edgetpu.tflite"

                if [ ! -f "$INPUT" ]; then
                    log_warn "INT8 model not found: $INPUT -- skipping"
                    continue
                fi

                log_info "Compiling: $INPUT"
                if edgetpu_compiler -s -o export/edgetpu "$INPUT" 2>&1; then
                    if [ -f "$OUTPUT" ]; then
                        SIZE=$(du -h "$OUTPUT" | cut -f1)
                        log_info "OK: $OUTPUT ($SIZE)"
                        ((TOTAL_OK++)) || true
                    else
                        log_error "Compiler OK but output missing: $OUTPUT"
                        ((TOTAL_FAIL++)) || true
                    fi
                else
                    log_error "EdgeTPU compilation failed: $INPUT"
                    ((TOTAL_FAIL++)) || true
                fi
            done
        done
    fi
fi

# =============================================================================
# SUMMARY
# =============================================================================
log_step "SUMMARY"

echo ""
echo "Results:"
echo "  Successful steps: $TOTAL_OK"
echo "  Failed steps:     $TOTAL_FAIL"
echo ""

# List generated TFLite models
echo "Generated TFLite models:"
for f in export/*.tflite; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  $f ($SIZE)"
    fi
done

echo ""
echo "EdgeTPU models:"
for f in export/edgetpu/*.tflite; do
    if [ -f "$f" ]; then
        SIZE=$(du -h "$f" | cut -f1)
        echo "  $f ($SIZE)"
    fi
done 2>/dev/null || echo "  (none)"

echo ""
echo "Updated result files:"
for f in results/T3_quant_fp32_vs_int8_*.csv; do
    if [ -f "$f" ]; then
        echo "  $f"
    fi
done

if [ "$TOTAL_FAIL" -gt 0 ]; then
    log_error "$TOTAL_FAIL steps failed. Check output above."
    exit 1
else
    log_info "All steps completed successfully!"
    exit 0
fi
