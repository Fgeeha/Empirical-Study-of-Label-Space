"""
10.7 Export INT8 PTQ TFLite for MobileNetV1 (PyTorch weights -> TF Keras).

Loads PyTorch weights (.npz) trained by 10.6, builds a matching custom Keras
MobileNetV1 model, transfers weights with proper transpositions, then converts
to fully-quantized INT8 TFLite for EdgeTPU.

Architecture: 13 DepthwiseSeparableConv blocks + ReLU6, no SE.
Keras layer names match timm state_dict keys (dots -> underscores) for
systematic weight mapping.

Prerequisites:
    - models/mobilenetv1_pv_{strategy}_best.pt  (from 10.6)
    - splits/{strategy}/pv_calib_10pct.csv       (from 8.1)

Output:
    export/mobilenetv1_int8_ptq_{strategy}.tflite

Usage:
    poetry run python scripts/10.7_export_tflite_int8_mobilenetv1.py --strategy hybrid
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

EXPORT = Path('export')
SPLITS = Path('splits')
MODELS = Path('models')
EXPORT.mkdir(exist_ok=True)

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MobileNetV1 block config derived from timm `mobilenetv1_100` state_dict.
# Each entry: (stage, sublayer, in_channels, out_channels, dw_stride)
MOBILENETV1_BLOCKS = [
    (0, 0, 32, 64, 1),
    (1, 0, 64, 128, 2),
    (1, 1, 128, 128, 1),
    (2, 0, 128, 256, 2),
    (2, 1, 256, 256, 1),
    (3, 0, 256, 512, 2),
    (3, 1, 512, 512, 1),
    (3, 2, 512, 512, 1),
    (3, 3, 512, 512, 1),
    (3, 4, 512, 512, 1),
    (3, 5, 512, 512, 1),
    (4, 0, 512, 1024, 2),
    (4, 1, 1024, 1024, 1),
]


def build_mobilenetv1_keras(num_classes):
    """Build custom Keras MobileNetV1 with layer names matching timm state_dict.

    Layer naming convention:
        PyTorch key "blocks.3.2.conv_dw.weight" -> Keras layer name "blocks_3_2_conv_dw"
        (dots replaced with underscores in the module path).
    """
    inputs = tf.keras.Input(batch_shape=(1, IMG_SIZE, IMG_SIZE, 3), name='input')

    # Stem
    x = tf.keras.layers.Conv2D(
        32, 3, strides=2, padding='same', use_bias=False, name='conv_stem'
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name='bn1')(x)
    x = tf.keras.layers.ReLU(max_value=6)(x)

    # 13 DepthwiseSeparableConv blocks
    for stage, sub, _in_ch, out_ch, stride in MOBILENETV1_BLOCKS:
        p = f'blocks_{stage}_{sub}'
        x = tf.keras.layers.DepthwiseConv2D(
            3, strides=stride, padding='same', use_bias=False, name=f'{p}_conv_dw'
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f'{p}_bn1')(x)
        x = tf.keras.layers.ReLU(max_value=6)(x)
        x = tf.keras.layers.Conv2D(out_ch, 1, use_bias=False, name=f'{p}_conv_pw')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{p}_bn2')(x)
        x = tf.keras.layers.ReLU(max_value=6)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, name='classifier')(x)

    return tf.keras.Model(inputs, outputs, name='mobilenetv1')


def _set_layer_weights(layer, params):
    """Set weights on a single Keras layer from PyTorch params.

    Returns True if weights were set successfully.
    """
    if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
        kernel = params['weight'].transpose(2, 3, 0, 1)
        layer.set_weights([kernel])
    elif isinstance(layer, tf.keras.layers.Conv2D):
        kernel = params['weight'].transpose(2, 3, 1, 0)
        layer.set_weights([kernel])
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.set_weights(
            [
                params['weight'],
                params['bias'],
                params['running_mean'],
                params['running_var'],
            ]
        )
    elif isinstance(layer, tf.keras.layers.Dense):
        layer.set_weights([params['weight'].T, params['bias']])
    else:
        return False
    return True


def transfer_weights(npz_weights, model):
    """Transfer PyTorch weights from .npz to Keras model.

    Weight transpositions:
        Conv2D:          PyTorch [out, in, H, W]  -> TF [H, W, in, out]
        DepthwiseConv2D: PyTorch [ch, 1, H, W]    -> TF [H, W, ch, 1]
        Dense:           PyTorch [out, in]         -> TF [in, out]
        BatchNorm:       weight->gamma, bias->beta, running_mean, running_var
    """
    layer_dict = {layer.name: layer for layer in model.layers}

    # Group weights by layer prefix
    groups = defaultdict(dict)
    for key in npz_weights.files:
        if 'num_batches_tracked' in key:
            continue
        parts = key.rsplit('.', 1)
        if len(parts) != 2:
            continue
        prefix, param = parts
        tf_name = prefix.replace('.', '_')
        groups[tf_name][param] = npz_weights[key]

    loaded = 0
    for tf_name, params in groups.items():
        if tf_name not in layer_dict:
            print(f'[WARN] Keras layer not found: {tf_name}')
            continue

        try:
            if _set_layer_weights(layer_dict[tf_name], params):
                loaded += 1
        except Exception as e:
            print(f'[WARN] Failed to set weights for {tf_name}: {e}')

    return loaded


def load_image(path, size=IMG_SIZE):
    img = Image.open(path).convert('RGB').resize((size, size))
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr


def representative_dataset(csv_path, num_samples=200):
    df = pd.read_csv(csv_path)
    paths = df['path'].tolist()[:num_samples]

    def gen():
        for p in paths:
            img = load_image(p)
            img = np.expand_dims(img, axis=0)
            yield [img]

    return gen


def _export_weights_if_missing(strategy):
    """Auto-export PyTorch .pt checkpoint to .npz."""
    weights_npz = EXPORT / f'mobilenetv1_{strategy}_weights.npz'
    if weights_npz.exists():
        return weights_npz

    model_pt = MODELS / f'mobilenetv1_pv_{strategy}_best.pt'
    if not model_pt.exists():
        raise FileNotFoundError(
            f'PyTorch checkpoint not found: {model_pt}\n'
            f'Run 10.6 first to train the model.'
        )

    print(f'[INFO] Exporting PyTorch weights to .npz: {model_pt} -> {weights_npz}')
    import torch

    ckpt = torch.load(model_pt, map_location='cpu', weights_only=True)
    weights = {k: v.detach().cpu().numpy() for k, v in ckpt.items()}
    np.savez(weights_npz, **weights)
    print(f'[OK] Exported {len(weights)} weight tensors -> {weights_npz}')
    return weights_npz


def main():
    parser = argparse.ArgumentParser(
        description='Export INT8 PTQ TFLite for MobileNetV1'
    )
    parser.add_argument(
        '--strategy', required=True, choices=['hybrid', 'Fuzzy', 'sbert']
    )
    parser.add_argument(
        '--num_calib',
        type=int,
        default=200,
        help='Number of calibration samples (default: 200)',
    )
    args = parser.parse_args()
    strategy = args.strategy

    # -- Paths --
    weights_npz = _export_weights_if_missing(strategy)
    calib_csv = SPLITS / strategy / 'pv_calib_10pct.csv'
    out_tflite = EXPORT / f'mobilenetv1_int8_ptq_{strategy}.tflite'

    if not calib_csv.exists():
        raise FileNotFoundError(f'Missing calibration CSV: {calib_csv}')

    # -- Load weights & detect num_classes --
    w = np.load(weights_npz)
    num_classes = w['classifier.weight'].shape[0]

    print(f'[INFO] MobileNetV1 INT8 export for strategy={strategy}')
    print(f'       classes={num_classes}, weights={weights_npz}')

    # -- Build Keras model & transfer weights --
    model = build_mobilenetv1_keras(num_classes)
    loaded = transfer_weights(w, model)
    total_layers = len([layer for layer in model.layers if layer.get_weights()])
    print(f'[INFO] Transferred weights for {loaded}/{total_layers} layers')

    if loaded < total_layers:
        print(
            f'[WARN] Some layers did not receive weights ({total_layers - loaded} missed)'
        )

    model.trainable = False

    # -- TFLite INT8 PTQ --
    @tf.function(
        input_signature=[tf.TensorSpec([1, IMG_SIZE, IMG_SIZE, 3], tf.float32)]
    )
    def serving_fn(x):
        return model(x, training=False)

    concrete_fn = serving_fn.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(
        calib_csv, num_samples=args.num_calib
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    print(
        f'[INFO] Converting to INT8 TFLite ({args.num_calib} calibration samples) ...'
    )
    tflite_model = converter.convert()
    out_tflite.write_bytes(tflite_model)

    size_mb = out_tflite.stat().st_size / 1024 / 1024
    print(f'[OK] INT8 PTQ TFLite exported -> {out_tflite} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
