#!/bin/bash
# =============================================================================
# 10.0 Build All EfficientNet-B3 & ResNet50 INT8 TFLite + EdgeTPU Models
#
# End-to-end pipeline that produces 12 model files:
#
#   efficientnet_int8_ptq_hybrid.tflite
#   efficientnet_int8_ptq_hybrid_edgetpu.tflite
#   efficientnet_int8_ptq_Fuzzy.tflite
#   efficientnet_int8_ptq_Fuzzy_edgetpu.tflite
#   efficientnet_int8_ptq_sbert.tflite
#   efficientnet_int8_ptq_sbert_edgetpu.tflite
#   resnet50_int8_ptq_hybrid.tflite
#   resnet50_int8_ptq_hybrid_edgetpu.tflite
#   resnet50_int8_ptq_Fuzzy.tflite
#   resnet50_int8_ptq_Fuzzy_edgetpu.tflite
#   resnet50_int8_ptq_sbert.tflite
#   resnet50_int8_ptq_sbert_edgetpu.tflite
#
# Prerequisites:
#   - Trained checkpoints in models/:
#       efficientnet_pv_{hybrid,Fuzzy,sbert}_best.pt
#       resnet50_pv_{hybrid,Fuzzy,sbert}_best.pt
#   - Calibration CSV: splits/{strategy}/pv_calib_10pct.csv
#   - TFLite venv (venv-tflite/) with tensorflow installed
#   - edgetpu_compiler (optional, for EdgeTPU compilation)
#
# Usage:
#   bash scripts/10.0_build_all_efficientnet_resnet50_tflite.sh
#   bash scripts/10.0_build_all_efficientnet_resnet50_tflite.sh --skip-edgetpu
#   bash scripts/10.0_build_all_efficientnet_resnet50_tflite.sh --strategy hybrid
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

# Configuration -- match reproduce.sh conventions
PYTHON="poetry run python"
SCRIPTS="scripts"
EXPORT="export"

# Force TensorFlow to use CPU (avoids CUDA compatibility issues
# with newer GPUs like RTX 5070 Ti where TF CUDA kernels may not
# be pre-compiled for the compute capability)
export CUDA_VISIBLE_DEVICES=""

STRATEGIES=("hybrid" "Fuzzy" "sbert")
SKIP_EDGETPU=false
FORCE=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC} $1"; }

print_header() {
    echo ""
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
}

# =============================================================================
# Parse arguments
# =============================================================================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-edgetpu)
            SKIP_EDGETPU=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --strategy)
            STRATEGIES=("$2")
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash scripts/10.0_build_all_efficientnet_resnet50_tflite.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-edgetpu       Skip EdgeTPU compilation step"
            echo "  --force              Rebuild even if .tflite already exists"
            echo "  --strategy NAME      Run only one strategy (hybrid|Fuzzy|sbert)"
            echo "  --python CMD         Python command (default: poetry run python)"
            echo "  -h, --help           Show this help"
            exit 0
            ;;
        *)
            log_error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Preflight checks
# =============================================================================
print_header "10.0 Build EfficientNet-B3 & ResNet50 INT8 + EdgeTPU"

echo ""
echo "Date:       $(date -Iseconds)"
echo "Strategies: ${STRATEGIES[*]}"
echo "Python:     ${PYTHON}"
echo "Skip TPU:   ${SKIP_EDGETPU}"
echo "Force:      ${FORCE}"
echo ""

# Check checkpoints
MISSING=0
for strategy in "${STRATEGIES[@]}"; do
    for arch in efficientnet resnet50; do
        PT="models/${arch}_pv_${strategy}_best.pt"
        NPZ="export/${arch}_${strategy}_weights.npz"
        if [ ! -f "${PT}" ] && [ ! -f "${NPZ}" ]; then
            log_error "Missing checkpoint: ${PT} (and no ${NPZ})"
            ((MISSING++)) || true
        fi
    done

    CALIB="splits/${strategy}/pv_calib_10pct.csv"
    if [ ! -f "${CALIB}" ]; then
        log_error "Missing calibration CSV: ${CALIB}"
        ((MISSING++)) || true
    fi
done

if [ ${MISSING} -gt 0 ]; then
    log_error "${MISSING} prerequisite(s) missing. Aborting."
    exit 1
fi

log_info "All prerequisites OK"
mkdir -p "${EXPORT}"

# Track results
TOTAL=0
OK=0
FAIL=0

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

# =============================================================================
# Step 1: Export EfficientNet weights (.npz)
# =============================================================================
print_header "Step 1/4: Export EfficientNet-B3 Weights"

for strategy in "${STRATEGIES[@]}"; do
    NPZ="${EXPORT}/efficientnet_${strategy}_weights.npz"
    if [ -f "${NPZ}" ] && [ "${FORCE}" = false ]; then
        log_info "Already exists: ${NPZ} -- skipping (use --force to rebuild)"
    else
        run_step "10.1 EfficientNet weights (${strategy})" \
            ${PYTHON} ${SCRIPTS}/10.1_export_efficientnet_weights.py --strategy "${strategy}"
    fi
done

# =============================================================================
# Step 2: Export EfficientNet INT8 TFLite
# =============================================================================
print_header "Step 2/4: Export EfficientNet-B3 INT8 TFLite"

for strategy in "${STRATEGIES[@]}"; do
    OUT="${EXPORT}/efficientnet_int8_ptq_${strategy}.tflite"
    if [ -f "${OUT}" ] && [ "${FORCE}" = false ]; then
        log_info "Already exists: ${OUT} -- skipping (use --force to rebuild)"
    else
        run_step "10.3 EfficientNet INT8 (${strategy})" \
            ${PYTHON} ${SCRIPTS}/10.3_export_tflite_int8_efficientnet.py --strategy "${strategy}"
    fi
done

# =============================================================================
# Step 3: Export ResNet50 INT8 TFLite
# =============================================================================
print_header "Step 3/4: Export ResNet50 INT8 TFLite"

for strategy in "${STRATEGIES[@]}"; do
    OUT="${EXPORT}/resnet50_int8_ptq_${strategy}.tflite"
    if [ -f "${OUT}" ] && [ "${FORCE}" = false ]; then
        log_info "Already exists: ${OUT} -- skipping (use --force to rebuild)"
    else
        run_step "10.4 ResNet50 INT8 (${strategy})" \
            ${PYTHON} ${SCRIPTS}/10.4_export_tflite_int8_resnet50.py --strategy "${strategy}"
    fi
done

# =============================================================================
# Step 4: Compile EdgeTPU
# =============================================================================
if [ "${SKIP_EDGETPU}" = true ]; then
    print_header "Step 4/4: EdgeTPU Compilation (SKIPPED)"
    log_warn "EdgeTPU compilation skipped (--skip-edgetpu)"
else
    print_header "Step 4/4: EdgeTPU Compilation"

    if ! command -v edgetpu_compiler &> /dev/null; then
        log_warn "edgetpu_compiler not found -- skipping EdgeTPU step"
        log_info "Install: https://coral.ai/docs/edgetpu/compiler/"
    else
        run_step "10.5 EdgeTPU compilation" \
            bash ${SCRIPTS}/10.5_compile_edgetpu_efficientnet_resnet50.sh
    fi
fi

# =============================================================================
# Summary
# =============================================================================
print_header "Summary"

echo ""
echo "Steps executed: ${TOTAL}"
echo "  Succeeded:    ${OK}"
echo "  Failed:       ${FAIL}"
echo ""

# List produced models
echo "Produced models in ${EXPORT}/:"
echo ""
echo "  EfficientNet-B3 INT8:"
for strategy in "${STRATEGIES[@]}"; do
    F="${EXPORT}/efficientnet_int8_ptq_${strategy}.tflite"
    if [ -f "${F}" ]; then
        SIZE=$(du -h "${F}" | cut -f1)
        echo -e "    ${GREEN}OK${NC}  ${F} (${SIZE})"
    else
        echo -e "    ${RED}--${NC}  ${F}"
    fi
done

echo ""
echo "  EfficientNet-B3 EdgeTPU: N/A (Swish activation incompatible with EdgeTPU)"

echo ""
echo "  ResNet50 INT8:"
for strategy in "${STRATEGIES[@]}"; do
    F="${EXPORT}/resnet50_int8_ptq_${strategy}.tflite"
    if [ -f "${F}" ]; then
        SIZE=$(du -h "${F}" | cut -f1)
        echo -e "    ${GREEN}OK${NC}  ${F} (${SIZE})"
    else
        echo -e "    ${RED}--${NC}  ${F}"
    fi
done

echo ""
echo "  ResNet50 EdgeTPU:"
for strategy in "${STRATEGIES[@]}"; do
    F="${EXPORT}/resnet50_int8_ptq_${strategy}_edgetpu.tflite"
    if [ -f "${F}" ]; then
        SIZE=$(du -h "${F}" | cut -f1)
        echo -e "    ${GREEN}OK${NC}  ${F} (${SIZE})"
    else
        echo -e "    ${RED}--${NC}  ${F}"
    fi
done

echo ""

if [ ${FAIL} -gt 0 ]; then
    log_error "${FAIL} step(s) failed. Check logs above."
    exit 1
else
    log_info "All models built successfully"
fi
