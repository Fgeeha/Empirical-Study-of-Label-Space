#!/bin/bash
# =============================================================================
# EdgeTPU Batch Benchmark Script
# Runs benchmarks on all MobileNetV2 models (CPU and EdgeTPU)
#
# Prerequisites:
#   - Raspberry Pi with Coral USB Accelerator
#   - tflite-runtime installed
#   - libedgetpu installed
#   - EdgeTPU-compiled models in export/edgetpu/
#
# Usage:
#   bash scripts/9.7_benchmark_edgetpu_batch.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BENCHMARK_SCRIPT="${PROJECT_ROOT}/edge-bench/scripts/benchmark_tflite.py"
EXPORT_DIR="${PROJECT_ROOT}/export"
EDGETPU_DIR="${EXPORT_DIR}/edgetpu"
RESULTS_DIR="${PROJECT_ROOT}/results/edgetpu"

RUNS=100
WARMUP=20
THREADS=4

echo "=============================================="
echo "EdgeTPU Batch Benchmark"
echo "=============================================="
echo "Date: $(date -Iseconds)"
echo "Host: $(hostname)"
echo "Platform: $(uname -m)"
echo ""

mkdir -p "${RESULTS_DIR}"

# Check benchmark script
if [ ! -f "${BENCHMARK_SCRIPT}" ]; then
    echo "[ERROR] Benchmark script not found: ${BENCHMARK_SCRIPT}"
    exit 1
fi

# Strategies
STRATEGIES=("hybrid" "Fuzzy" "sbert")

# Results aggregation file
AGG_CSV="${RESULTS_DIR}/T4_edgetpu_combined.csv"
echo "strategy,model,backend,latency_mean_ms,latency_p95_ms,fps,model_size_mb" > "${AGG_CSV}"

for strategy in "${STRATEGIES[@]}"; do
    echo ""
    echo "=============================================="
    echo "Strategy: ${strategy}"
    echo "=============================================="

    # INT8 model (CPU)
    INT8_MODEL="${EXPORT_DIR}/mobilenetv2_int8_ptq_${strategy}.tflite"

    if [ -f "${INT8_MODEL}" ]; then
        echo ""
        echo "[CPU] ${INT8_MODEL}"

        OUT_JSON="${RESULTS_DIR}/benchmark_cpu_mobilenetv2_${strategy}.json"

        python3 "${BENCHMARK_SCRIPT}" \
            --model "${INT8_MODEL}" \
            --backend cpu \
            --threads ${THREADS} \
            --warmup ${WARMUP} \
            --runs ${RUNS} \
            --output "${OUT_JSON}"

        # Extract metrics and append to CSV
        if [ -f "${OUT_JSON}" ]; then
            MEAN=$(python3 -c "import json; d=json.load(open('${OUT_JSON}')); print(d['latency']['mean_ms'])")
            P95=$(python3 -c "import json; d=json.load(open('${OUT_JSON}')); print(d['latency']['p95_ms'])")
            FPS=$(python3 -c "import json; d=json.load(open('${OUT_JSON}')); print(d['throughput']['fps'])")
            SIZE=$(python3 -c "import json; d=json.load(open('${OUT_JSON}')); print(round(d['model']['size_bytes']/1024/1024, 2))")
            echo "${strategy},mobilenetv2,cpu,${MEAN},${P95},${FPS},${SIZE}" >> "${AGG_CSV}"
            echo "[OK] CPU: ${MEAN}ms (p95: ${P95}ms), ${FPS} FPS"
        fi
    else
        echo "[WARN] INT8 model not found: ${INT8_MODEL}"
    fi

    # EdgeTPU model
    EDGETPU_MODEL="${EDGETPU_DIR}/mobilenetv2_int8_ptq_${strategy}_edgetpu.tflite"

    if [ -f "${EDGETPU_MODEL}" ]; then
        echo ""
        echo "[EdgeTPU] ${EDGETPU_MODEL}"

        OUT_JSON="${RESULTS_DIR}/benchmark_edgetpu_mobilenetv2_${strategy}.json"

        python3 "${BENCHMARK_SCRIPT}" \
            --model "${EDGETPU_MODEL}" \
            --backend edgetpu \
            --warmup ${WARMUP} \
            --runs ${RUNS} \
            --output "${OUT_JSON}"

        # Extract metrics and append to CSV
        if [ -f "${OUT_JSON}" ]; then
            MEAN=$(python3 -c "import json; d=json.load(open('${OUT_JSON}')); print(d['latency']['mean_ms'])")
            P95=$(python3 -c "import json; d=json.load(open('${OUT_JSON}')); print(d['latency']['p95_ms'])")
            FPS=$(python3 -c "import json; d=json.load(open('${OUT_JSON}')); print(d['throughput']['fps'])")
            SIZE=$(python3 -c "import json; d=json.load(open('${OUT_JSON}')); print(round(d['model']['size_bytes']/1024/1024, 2))")
            echo "${strategy},mobilenetv2,edgetpu,${MEAN},${P95},${FPS},${SIZE}" >> "${AGG_CSV}"
            echo "[OK] EdgeTPU: ${MEAN}ms (p95: ${P95}ms), ${FPS} FPS"
        fi
    else
        echo "[WARN] EdgeTPU model not found: ${EDGETPU_MODEL}"
        echo "       Run scripts/9.6_compile_edgetpu_models.sh first"
    fi
done

echo ""
echo "=============================================="
echo "Results Summary"
echo "=============================================="
echo ""
echo "Combined results: ${AGG_CSV}"
echo ""
cat "${AGG_CSV}" | column -t -s','

# Calculate speedup
echo ""
echo "----------------------------------------------"
echo "EdgeTPU Speedup vs CPU"
echo "----------------------------------------------"

python3 << 'EOF'
import pandas as pd

df = pd.read_csv("results/edgetpu/T4_edgetpu_combined.csv")

for strategy in df['strategy'].unique():
    cpu = df[(df['strategy'] == strategy) & (df['backend'] == 'cpu')]
    tpu = df[(df['strategy'] == strategy) & (df['backend'] == 'edgetpu')]

    if len(cpu) > 0 and len(tpu) > 0:
        cpu_lat = cpu['latency_mean_ms'].values[0]
        tpu_lat = tpu['latency_mean_ms'].values[0]
        speedup = cpu_lat / tpu_lat
        print(f"  {strategy}: {speedup:.1f}x speedup ({cpu_lat:.1f}ms -> {tpu_lat:.1f}ms)")
EOF

echo ""
echo "[DONE] Benchmark complete"
