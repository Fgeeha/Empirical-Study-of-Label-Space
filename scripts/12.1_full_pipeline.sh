#!/usr/bin/env bash
# =============================================================================
# 12.1 Full ECCV 2026 Pipeline: from class mapping to Edge-Bench upload
#
# Runs the complete reproducible pipeline:
#   Phase 1: Data preparation (class mapping, filtering, splits)
#   Phase 2: Training (5 architectures x 3 strategies)
#   Phase 3: Evaluation (PV baselines, T1 domain shift)
#   Phase 4: CORAL domain adaptation (lambda sweep, stats)
#   Phase 5: Analysis & visualization (t-SNE, ablation, SOTA)
#   Phase 6: Quantization & TFLite export (FP32, INT8)
#   Phase 7: EdgeTPU compilation
#   Phase 8: Edge-Bench upload & benchmarks
#
# Usage:
#   bash scripts/12.1_full_pipeline.sh                          # all phases
#   bash scripts/12.1_full_pipeline.sh --from 6                 # resume from Phase 6
#   bash scripts/12.1_full_pipeline.sh --phase 2                # run only Phase 2
#   bash scripts/12.1_full_pipeline.sh --server http://192.168.1.4:8000
#   bash scripts/12.1_full_pipeline.sh --dry-run                # print steps only
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# ========================= Config ==========================================
STRATEGIES=("hybrid" "Fuzzy" "sbert")
LAMBDAS=(0.01 0.1 1.0 10.0)
BEST_LAMBDA_HYBRID=0.1
BEST_LAMBDA_FUZZY=0.1
BEST_LAMBDA_SBERT=0.01
TARGET="plantdoc"
EDGEBENCH_SERVER="http://192.168.1.4:8000"
CORAL_EPOCHS=20
PHASE_FROM=1
PHASE_TO=8
PHASE_ONLY=""
DRY_RUN=false
LOG_FILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"

# ========================= Parse Args ======================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --from)       PHASE_FROM="$2"; shift 2 ;;
        --to)         PHASE_TO="$2"; shift 2 ;;
        --phase)      PHASE_ONLY="$2"; shift 2 ;;
        --server)     EDGEBENCH_SERVER="$2"; shift 2 ;;
        --target)     TARGET="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --log)        LOG_FILE="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--from N] [--to N] [--phase N] [--server URL] [--dry-run]"
            echo ""
            echo "Phases:"
            echo "  1  Data preparation (mapping, filtering, splits)"
            echo "  2  Training (ResNet-50, EfficientNet-B3, MobileNetV2/V1, EffLite0)"
            echo "  3  Evaluation (baselines, domain shift, fine-tuning, CORAL)"
            echo "  4  Analysis & visualization (t-SNE, ablation, SOTA)"
            echo "  5  Quantization & TFLite export (FP32, INT8, PTQ)"
            echo "  6  EdgeTPU compilation"
            echo "  7  Edge-Bench upload & benchmarks"
            echo "  8  Results collection & plots (T4, F7-F10)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$PHASE_ONLY" ]; then
    PHASE_FROM="$PHASE_ONLY"
    PHASE_TO="$PHASE_ONLY"
fi

# ========================= Logging =========================================
mkdir -p "$(dirname "$LOG_FILE")" logs
exec > >(tee -a "$LOG_FILE") 2>&1

# ========================= Helpers =========================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[0;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

step_num=0
ok_count=0
fail_count=0
skip_count=0

phase_hdr() { echo -e "\n${BOLD}${CYAN}################################################################${NC}"; echo -e "${BOLD}${CYAN}# PHASE $1: $2${NC}"; echo -e "${BOLD}${CYAN}################################################################${NC}\n"; }
step()      { ((step_num++)) || true; echo -e "${GREEN}[STEP $step_num]${NC} $*"; }
warn()      { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()      { echo -e "${RED}[FAIL]${NC} $*"; }

should_run() { [ "$PHASE_FROM" -le "$1" ] && [ "$PHASE_TO" -ge "$1" ]; }

run() {
    local desc="$1"; shift
    step "$desc"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] $*"
        ((skip_count++)) || true
        return 0
    fi
    if "$@" 2>&1; then
        ((ok_count++)) || true
    else
        fail "$desc"
        ((fail_count++)) || true
    fi
}

run_gpu() {
    # GPU scripts (training, inference) -- use default CUDA settings
    run "$@"
}

run_cpu() {
    # TF export scripts -- force CPU (RTX 5070 Ti not supported by TF 2.20)
    local desc="$1"; shift
    step "$desc"
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] CUDA_VISIBLE_DEVICES='' $*"
        ((skip_count++)) || true
        return 0
    fi
    if CUDA_VISIBLE_DEVICES="" "$@" 2>&1; then
        ((ok_count++)) || true
    else
        fail "$desc"
        ((fail_count++)) || true
    fi
}

best_lambda() {
    local s="$1"
    case "$s" in
        hybrid) echo "$BEST_LAMBDA_HYBRID" ;;
        Fuzzy)  echo "$BEST_LAMBDA_FUZZY" ;;
        sbert)  echo "$BEST_LAMBDA_SBERT" ;;
    esac
}

# ========================= Banner ==========================================
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} ECCV 2026 Full Pipeline${NC}"
echo -e "${BOLD} Plant Disease Recognition + Domain Adaptation + Edge Deploy${NC}"
echo -e "${BOLD}============================================================${NC}"
echo "  Phases:     $PHASE_FROM -> $PHASE_TO"
echo "  Strategies: ${STRATEGIES[*]}"
echo "  Target:     $TARGET"
echo "  Server:     $EDGEBENCH_SERVER"
echo "  Log:        $LOG_FILE"
echo "  Dry run:    $DRY_RUN"
echo ""

# #########################################################################
#  PHASE 1: DATA PREPARATION
# #########################################################################
if should_run 1; then
phase_hdr 1 "Data Preparation"

# 1.x Collect classes
run "Collect PV classes" \
    poetry run python scripts/1.1_collect_classes_pv.py
run "Collect PlantDoc classes" \
    poetry run python scripts/1.2_collect_classes_plantdoc.py
run "Collect FCDD classes" \
    poetry run python scripts/1.3_collect_classes_fcdd.py

# 2.x Build shared subsets (3 strategies)
run "Build shared subset (Fuzzy)" \
    poetry run python scripts/2.4_build_shared_subset.py
run "Build shared subset (SBERT)" \
    poetry run python scripts/2.5_build_shared_subset_sbert.py
run "Build shared subset (Hybrid)" \
    poetry run python scripts/2.6_build_shared_subset_hybrid.py

# 3.1 Filter datasets
for s in "${STRATEGIES[@]}"; do
    mapping="configs/class_mapping_${s}.json"
    if [ "$s" = "Fuzzy" ]; then mapping="configs/class_mapping_Fuzzy.json"; fi
    run "Filter shared subset ($s)" \
        poetry run python scripts/3.1_filter_shared_subset.py --mapping "$mapping"
done

# 4.1 Split PV
for s in "${STRATEGIES[@]}"; do
    run "Split PV dataset ($s)" \
        poetry run python scripts/4.1_split_pv_dataset.py --strategy "$s"
done

# 8.1 Calibration set
for s in "${STRATEGIES[@]}"; do
    run "Prepare calibration set ($s)" \
        poetry run python scripts/8.1_prepare_pv_calibration.py --strategy "$s"
done

# 5.1 Prepare target test
for s in "${STRATEGIES[@]}"; do
    run "Prepare target test ($s)" \
        poetry run python scripts/5.1_prepare_target_test.py --strategy "$s" --target "$TARGET"
done

# 5.5.1 Split target
for s in "${STRATEGIES[@]}"; do
    run "Split target dataset ($s)" \
        poetry run python scripts/5.5.1_split_target_dataset.py --strategy "$s" --target "$TARGET"
done

# 6.2 Prepare target unlabeled
for s in "${STRATEGIES[@]}"; do
    run "Prepare target unlabeled ($s)" \
        poetry run python scripts/6.2_prepare_target_unlabeled.py --strategy "$s" --target "$TARGET"
done

fi  # Phase 1

# #########################################################################
#  PHASE 2: TRAINING (GPU)
# #########################################################################
if should_run 2; then
phase_hdr 2 "Model Training (GPU)"

for s in "${STRATEGIES[@]}"; do
    # Baselines
    run_gpu "Train ResNet-50 ($s)" \
        poetry run python scripts/4.2_train_resnet50_pv.py --strategy "$s"
    run_gpu "Train EfficientNet-B3 ($s)" \
        poetry run python scripts/4.2b_train_efficientnet_pv.py --strategy "$s"
    run_gpu "Train MobileNetV2 baseline ($s)" \
        poetry run python scripts/4.2c_train_mobilenet_baseline_pv.py --strategy "$s"

    # Edge architectures
    run_gpu "Train MobileNetV2 edge ($s)" \
        poetry run python scripts/9.0_train_mobilenetv2_pv.py --strategy "$s"
    run_gpu "Train MobileNetV1 ($s)" \
        poetry run python scripts/10.6_train_mobilenetv1_pv.py --strategy "$s"
    run_gpu "Train EfficientNet-Lite0 ($s)" \
        poetry run python scripts/10.8_train_efficientnet_lite0_pv.py --strategy "$s"
done

# CORAL -- ResNet-50
for s in "${STRATEGIES[@]}"; do
    for l in "${LAMBDAS[@]}"; do
        run_gpu "Train CORAL ResNet-50 ($s, lambda=$l)" \
            poetry run python scripts/6.3_train_coral.py \
                --strategy "$s" --target "$TARGET" --lambda_coral "$l" --epochs "$CORAL_EPOCHS"
    done
done

# CORAL -- MobileNetV2 (best lambda only)
for s in "${STRATEGIES[@]}"; do
    l=$(best_lambda "$s")
    run_gpu "Train CORAL MobileNetV2 ($s, lambda=$l)" \
        poetry run python scripts/9.1_train_coral_mobilenetv2.py \
            --strategy "$s" --target "$TARGET" --lambda_coral "$l" --epochs "$CORAL_EPOCHS"
done

# Fine-tune upper bound
for s in "${STRATEGIES[@]}"; do
    run_gpu "Fine-tune ResNet-50 on target ($s)" \
        poetry run python scripts/5.5.2_finetune_target.py --strategy "$s" --target "$TARGET"
done

fi  # Phase 2

# #########################################################################
#  PHASE 3: EVALUATION (GPU)
# #########################################################################
if should_run 3; then
phase_hdr 3 "Evaluation"

# PV baselines
for s in "${STRATEGIES[@]}"; do
    run_gpu "Eval ResNet-50 on PV ($s)" \
        poetry run python scripts/4.3_eval_resnet50_pv.py --strategy "$s"
    run_gpu "Eval EfficientNet-B3 on PV ($s)" \
        poetry run python scripts/4.3b_eval_efficientnet_pv.py --strategy "$s"
    run_gpu "Eval MobileNetV2 on PV ($s)" \
        poetry run python scripts/4.3c_eval_mobilenet_baseline_pv.py --strategy "$s"
done

# Baseline summary
run "Build baseline summary (T-Baseline)" \
    poetry run python scripts/4.4_build_baseline_summary.py

# Confusion matrices
for s in "${STRATEGIES[@]}"; do
    run_gpu "Plot confusion matrices ($s)" \
        poetry run python scripts/4.5_plot_all_cm.py --strategy "$s"
done

# Domain shift (zero-shot)
for s in "${STRATEGIES[@]}"; do
    for m in resnet50 efficientnet mobilenet_baseline; do
        run_gpu "Infer target $m ($s)" \
            poetry run python scripts/5.2_infer_target.py --strategy "$s" --model "$m"
        run "Eval target $m ($s)" \
            poetry run python scripts/5.3_eval_target.py --strategy "$s" --model "$m"
    done
done

# T1 domain shift table
run "Build T1 domain shift" \
    poetry run python scripts/5.4_build_T1_domain_shift.py

# Fine-tuned eval
for s in "${STRATEGIES[@]}"; do
    run_gpu "Eval fine-tuned target ($s)" \
        poetry run python scripts/5.5.3_eval_finetuned_target.py --strategy "$s" --target "$TARGET"
done

# CORAL eval (all lambdas)
for s in "${STRATEGIES[@]}"; do
    for l in "${LAMBDAS[@]}"; do
        run_gpu "Eval CORAL ($s, lambda=$l)" \
            poetry run python scripts/6.4_eval_coral.py \
                --strategy "$s" --target "$TARGET" --lambda_coral "$l"
    done
done

# T2 CORAL sweep
for s in "${STRATEGIES[@]}"; do
    run "Build T2 CORAL sweep ($s)" \
        poetry run python scripts/6.5_build_T2_coral_sweep.py --strategy "$s"
done

# Lambda vs F1 plot (with bootstrap CI, significance markers)
for s in "${STRATEGIES[@]}"; do
    run "Plot lambda vs F1 ($s)" \
        poetry run python scripts/6.6_plot_lambda_vs_f1.py --strategy "$s"
done

# Statistical tests
for s in "${STRATEGIES[@]}"; do
    run "Statistical tests ($s)" \
        poetry run python scripts/6.7_statistical_tests.py --strategy "$s"
done

# Ablation table
run "Build T5 ablation" \
    poetry run python scripts/6.7_build_T5_ablation.py

fi  # Phase 3

# #########################################################################
#  PHASE 4: ANALYSIS & VISUALIZATION
# #########################################################################
if should_run 4; then
phase_hdr 4 "Analysis and Visualization"

# t-SNE
for s in "${STRATEGIES[@]}"; do
    run_gpu "Extract features baseline ($s)" \
        poetry run python scripts/7.1_extract_features_tsne.py --strategy "$s" --target "$TARGET" --mode baseline
    run_gpu "Extract features CORAL ($s)" \
        poetry run python scripts/7.1_extract_features_tsne.py --strategy "$s" --target "$TARGET" --mode coral
    l=$(best_lambda "$s")
    run "Plot t-SNE ($s)" \
        poetry run python scripts/7.6_tsne_plot.py --strategy "$s" --target "$TARGET" --lambda_coral "$l"
done

# Error shift, per-class F1
run "Error category shift" \
    poetry run python scripts/7.3_error_shift.py

for s in "${STRATEGIES[@]}"; do
    run "Per-class F1 ($s)" \
        poetry run python scripts/7.7_per_class_f1.py --strategy "$s"
done

# Tables
run "Build T7 SOTA" \
    poetry run python scripts/7.5_build_T7_sota.py
run "Build T5 ablation summary" \
    poetry run python scripts/7.8_build_T5_ablation_summary.py

# Paper-ready tables (T1, T5, T3, T4, T6, T7, p-values)
run "Format paper tables" \
    poetry run python scripts/format_paper_tables.py

fi  # Phase 4

# #########################################################################
#  PHASE 5: QUANTIZATION & TFLITE EXPORT
# #########################################################################
if should_run 5; then
phase_hdr 5 "Quantization and TFLite Export"

# Export PyTorch weights to .npz
for s in "${STRATEGIES[@]}"; do
    run "Export ResNet-50 weights ($s)" \
        poetry run python scripts/8.2a_export_torch_weights.py --strategy "$s"
    run "Export MobileNetV2 weights ($s)" \
        poetry run python scripts/9.1a_export_mobilenetv2_weights.py --strategy "$s"
    run "Export EfficientNet-B3 weights ($s)" \
        poetry run python scripts/10.1_export_efficientnet_weights.py --strategy "$s"
    # MobileNetV1 and EffLite0 auto-export in their TFLite scripts
done

# ResNet-50 SavedModel (for T3) + FP32 + INT8
for s in "${STRATEGIES[@]}"; do
    run_cpu "ResNet-50 SavedModel ($s)" \
        poetry run python scripts/8.2b_build_tf_savedmodel.py --strategy "$s"
    run_cpu "ResNet-50 FP32 TFLite ($s)" \
        poetry run python scripts/8.3_export_tflite_fp32.py --strategy "$s"
    run_cpu "ResNet-50 INT8 TFLite via SavedModel ($s)" \
        poetry run python scripts/8.4_export_tflite_int8.py --strategy "$s"
done

# T3 evaluation
for s in "${STRATEGIES[@]}"; do
    run_cpu "T3 quant eval ($s)" \
        poetry run python scripts/8.5_eval_tflite_quant.py --strategy "$s"
    run "Plot quant tradeoff ($s)" \
        poetry run python scripts/8.6_plot_quant_tradeoff.py --strategy "$s"
done

# EdgeTPU-ready INT8 TFLite (all architectures)
for s in "${STRATEGIES[@]}"; do
    run_cpu "ResNet-50 INT8 EdgeTPU ($s)" \
        poetry run python scripts/10.4_export_tflite_int8_resnet50.py --strategy "$s"
    run_cpu "MobileNetV2 INT8 EdgeTPU ($s)" \
        poetry run python scripts/9.2_export_tflite_int8_mobilenet.py --strategy "$s"
    run_cpu "MobileNetV1 INT8 EdgeTPU ($s)" \
        poetry run python scripts/10.7_export_tflite_int8_mobilenetv1.py --strategy "$s"
    run_cpu "EfficientNet-Lite0 INT8 EdgeTPU ($s)" \
        poetry run python scripts/10.9_export_tflite_int8_efficientnet_lite0.py --strategy "$s"
    run_cpu "EfficientNet-B3 INT8 CPU-only ($s)" \
        poetry run python scripts/10.3_export_tflite_int8_efficientnet.py --strategy "$s"
done

fi  # Phase 5

# #########################################################################
#  PHASE 6: EDGETPU COMPILATION
# #########################################################################
if should_run 6; then
phase_hdr 6 "EdgeTPU Compilation"

if ! command -v edgetpu_compiler &>/dev/null; then
    warn "edgetpu_compiler not found. Install:"
    echo "  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -"
    echo "  echo 'deb https://packages.cloud.google.com/apt coral-edgetpu-stable main' | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list"
    echo "  sudo apt update && sudo apt install edgetpu-compiler"
    warn "Skipping EdgeTPU compilation."
else
    run "Compile EdgeTPU models (all)" \
        bash scripts/9.6_compile_edgetpu_models.sh
fi

fi  # Phase 6

# #########################################################################
#  PHASE 7: EDGE-BENCH UPLOAD
# #########################################################################
if should_run 7; then
phase_hdr 7 "Edge-Bench: Upload Models and Run Benchmarks"

run "Upload, deploy to RPi, run benchmarks and wait" \
    poetry run python scripts/9.9_run_edgebench.py \
        --server "$EDGEBENCH_SERVER" \
        --runs 100 \
        --timeout 600 \
        --export-csv results/edgebench/raw_from_bench.csv

fi  # Phase 7

# #########################################################################
#  PHASE 8: RESULTS COLLECTION & FINAL PLOTS
# #########################################################################
if should_run 8; then
phase_hdr 8 "Results Collection and Final Plots"

run "Fetch Edge-Bench results" \
    poetry run python scripts/11.1_fetch_edgebench_results.py \
        --server "$EDGEBENCH_SERVER"

run "Build T4 from Edge-Bench" \
    poetry run python scripts/11.2_build_T4_from_edgebench.py

run "Plot Edge-Bench analysis (F7-F10)" \
    poetry run python scripts/11.3_plot_edgebench_analysis.py

fi  # Phase 8

# #########################################################################
#  SUMMARY
# #########################################################################
echo ""
echo -e "${BOLD}================================================================${NC}"
echo -e "${BOLD} PIPELINE COMPLETE${NC}"
echo -e "${BOLD}================================================================${NC}"
echo ""
echo "  Steps run:     $step_num"
echo -e "  ${GREEN}Succeeded:${NC}    $ok_count"
echo -e "  ${RED}Failed:${NC}       $fail_count"
echo -e "  ${YELLOW}Skipped:${NC}      $skip_count"
echo "  Log:           $LOG_FILE"
echo ""

# List key outputs
echo "Key result files:"
for f in \
    results/baseline_pv_metrics.csv \
    results/T1_domain_shift.csv \
    "results/T2_coral_sweep_hybrid.csv" \
    results/T3_quant_fp32_vs_int8_hybrid.csv \
    results/T5_ablation.csv \
    "results/T6_per_class_f1_hybrid.csv" \
    results/T7_sota.csv \
    results/edgebench/raw_results.csv \
    results/T4_edgebench.csv \
    results/T4_edgebench.tex
do
    if [ -f "$f" ]; then
        echo -e "  ${GREEN}[OK]${NC} $f"
    else
        echo -e "  ${YELLOW}[--]${NC} $f"
    fi
done

echo ""
echo "Key figures:"
for f in \
    figures/F2_f1_vs_lambda_hybrid.png \
    figures/F3_tsne_baseline_hybrid.png \
    figures/F3_tsne_coral_hybrid.png \
    figures/F4_quant_tradeoff_hybrid.png \
    figures/F5_latency_comparison.png \
    figures/F7_latency_comparison.pdf \
    figures/F8_throughput_comparison.pdf \
    figures/F9_speedup_heatmap.pdf \
    figures/F10_size_vs_latency.pdf
do
    if [ -f "$f" ]; then
        echo -e "  ${GREEN}[OK]${NC} $f"
    else
        echo -e "  ${YELLOW}[--]${NC} $f"
    fi
done

echo ""
echo "TFLite models:"
count_tflite=$(ls export/*.tflite 2>/dev/null | wc -l)
count_edgetpu=$(ls export/edgetpu/*.tflite 2>/dev/null | wc -l)
echo "  INT8 TFLite:  $count_tflite"
echo "  EdgeTPU:      $count_edgetpu"

echo ""
if [ "$fail_count" -gt 0 ]; then
    echo -e "${RED}$fail_count step(s) failed. Check log: $LOG_FILE${NC}"
    exit 1
else
    echo -e "${GREEN}All steps completed successfully.${NC}"
    exit 0
fi
