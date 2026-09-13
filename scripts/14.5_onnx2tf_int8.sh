#!/usr/bin/env bash
# ONNX -> TFLite (float32 + full-integer INT8) via onnx2tf, then Edge TPU compile.
#
#   ONNX2TF_PY=<python with onnx2tf> bash scripts/14.5_onnx2tf_int8.sh resnet50 mobilenetv1 ...
#
# Inputs : export/onnx_20260912/<model>_hybrid.onnx        (from 14.4_export_onnx.py)
#          export/onnx_20260912/calib_hybrid_200_nhwc_normalized.npy  (200 calibration images,
#          ImageNet-normalized, NHWC; built from splits/hybrid/pv_calib_10pct.csv)
# Outputs: export/onnx2tf_20260912/<model>/<model>_hybrid_float32.tflite
#          export/onnx2tf_20260912/<model>/<model>_hybrid_full_integer_quant.tflite  (int8 in/out)
#          export/onnx2tf_20260912/<model>/edgetpu/*_edgetpu.tflite + compile log
# The calibration data are already normalized, so mean=0 / std=1 are passed to onnx2tf.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${ONNX2TF_PY:-python3}
IN=export/onnx_20260912
OUT=export/onnx2tf_20260912
CALIB=$IN/calib_hybrid_200_nhwc_normalized.npy
export TF_CPP_MIN_LOG_LEVEL=3
for m in "$@"; do
  echo "== $m"
  mkdir -p "$OUT/$m/edgetpu"
  # onnx2tf 1.20 downloads a pickled sample-image .npy and loads it with allow_pickle=False (crashes
  # under numpy>=1.16); the shim re-enables pickle loading for that internal call only.
  $PY -c "import functools, sys, numpy as np; np.load = functools.partial(np.load, allow_pickle=True); from onnx2tf import main; sys.argv = ['onnx2tf'] + sys.argv[1:]; main()" -i "$IN/${m}_hybrid.onnx" -o "$OUT/$m" -osd -n 2>&1 | grep -a -i "error\|Traceback" | tail -5
  # INT8 PTQ in a separate, lighter step (onnx2tf -oiqt builds six variants at once and exhausts RAM on ResNet-50)
  FQ="$OUT/$m/${m}_hybrid_full_integer_quant.tflite"
  [ -f "$FQ" ] || $PY scripts/14.5b_savedmodel_to_int8.py --saved_model "$OUT/$m" --calib "$CALIB" --out "$FQ" 2>&1 | grep -a "\[OK\]\|Error\|Traceback" | tail -3
  ls -la "$OUT/$m"/*.tflite 2>/dev/null | awk '{print $5, $9}'
  if [ -f "$FQ" ]; then
    edgetpu_compiler -s -o "$OUT/$m/edgetpu" "$FQ" > "$OUT/$m/edgetpu/compile.txt" 2>&1
    grep -a -E "Mapped to Edge TPU|CPU|Number of operations|Total number" "$OUT/$m/edgetpu/compile.txt" | head -14
  fi
done
echo "ONNX2TF DONE"
