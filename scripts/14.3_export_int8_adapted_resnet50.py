#!/usr/bin/env python3
"""Export an adapted (UDA) ResNet-50 checkpoint to INT8 TFLite through the same
Keras-rebuild + PTQ path as scripts/10.4_export_tflite_int8_resnet50.py.

    venv-tflite/bin/python scripts/14.3_export_int8_adapted_resnet50.py \
        --weights export/resnet50_dann42_weights.npz \
        --out export/resnet50_dann42_int8_ptq_hybrid.tflite --strategy hybrid

The .npz must contain torchvision ResNet-50 state_dict keys including fc.weight /
fc.bias (see 14.2_eval_pytorch_reference.py for how the DANN checkpoint, stored
as backbone + classifier, is reassembled). Calibration: first --num_calib rows
of splits/{strategy}/pv_calib_10pct.csv with ImageNet normalization, as in 10.4.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import tensorflow as tf

HERE = Path(__file__).resolve().parent
SPLITS = Path('splits')


def load_10_4():
    spec = importlib.util.spec_from_file_location(
        'export_resnet50', HERE / '10.4_export_tflite_int8_resnet50.py'
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--strategy', default='hybrid')
    ap.add_argument('--num_calib', type=int, default=200)
    args = ap.parse_args()
    m = load_10_4()

    w = np.load(args.weights)
    num_classes = w['fc.weight'].shape[0]
    inputs = tf.keras.Input(batch_shape=(1, 224, 224, 3), name='input')
    base = tf.keras.applications.ResNet50(
        include_top=False, weights=None, input_tensor=inputs, pooling='avg'
    )
    loaded = m.transfer_resnet50_weights(w, base)
    total = len([layer for layer in base.layers if layer.get_weights()])
    print(f'[INFO] transferred {loaded}/{total} backbone layers')
    if loaded < total * 0.9:
        raise RuntimeError('weight transfer failed')
    dense = tf.keras.layers.Dense(num_classes, activation=None, name='logits')
    model = tf.keras.Model(inputs, dense(base.output))
    dense.set_weights([w['fc.weight'].T, w['fc.bias']])

    concrete_fn = tf.function(lambda x: model(x, training=False)).get_concrete_function(
        tf.TensorSpec([1, 224, 224, 3], tf.float32)
    )
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = m.representative_dataset(
        SPLITS / args.strategy / 'pv_calib_10pct.csv', num_samples=args.num_calib
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    out = Path(args.out)
    out.write_bytes(converter.convert())
    print(f'[OK] {out} ({out.stat().st_size / 1024 / 1024:.2f} MiB)')


if __name__ == '__main__':
    main()
