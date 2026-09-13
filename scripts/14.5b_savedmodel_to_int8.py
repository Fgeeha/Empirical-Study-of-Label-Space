#!/usr/bin/env python3
"""Full-integer INT8 PTQ of an onnx2tf SavedModel with our own calibration images.

    python scripts/14.5b_savedmodel_to_int8.py --saved_model export/onnx2tf_20260912/resnet50 \
        --calib export/onnx_20260912/calib_hybrid_200_nhwc_normalized.npy \
        --out export/onnx2tf_20260912/resnet50/resnet50_hybrid_full_integer_quant.tflite

Lighter than onnx2tf -oiqt (which materialises six quantization variants at once).
Input/output tensors are int8 so the model can be compiled for the Edge TPU.
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--saved_model', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    calib = np.load(args.calib).astype(np.float32)

    def rep():
        for i in range(len(calib)):
            yield [calib[i : i + 1]]

    conv = tf.lite.TFLiteConverter.from_saved_model(args.saved_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    out = Path(args.out)
    out.write_bytes(conv.convert())
    print(
        f'[OK] {out} ({out.stat().st_size / 1024 / 1024:.2f} MiB, {len(calib)} calibration images)'
    )


if __name__ == '__main__':
    main()
