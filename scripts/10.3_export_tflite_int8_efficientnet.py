"""
10.3 Export INT8 PTQ TFLite for EfficientNet-B3 (self-contained).

Loads weights from export/efficientnet_{strategy}_weights.npz, builds a
TF Keras EfficientNetB3 model in memory, and converts directly to INT8 TFLite.

No intermediate SavedModel needed -- mirrors the approach in 9.2 (MobileNetV2).

Prerequisites:
    - export/efficientnet_{strategy}_weights.npz  (from 10.1)
    - splits/{strategy}/pv_calib_10pct.csv          (from 8.1)

Output files:
    export/efficientnet_int8_ptq_{strategy}.tflite

Usage:
    poetry run python scripts/10.3_export_tflite_int8_efficientnet.py --strategy hybrid
    poetry run python scripts/10.3_export_tflite_int8_efficientnet.py --strategy Fuzzy
    poetry run python scripts/10.3_export_tflite_int8_efficientnet.py --strategy sbert
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

EXPORT = Path('export')
SPLITS = Path('splits')
MODELS = Path('models')
EXPORT.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)

# ImageNet normalization (same as PyTorch training transforms)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path, size=IMG_SIZE):
    """Load and normalize image (matches PyTorch T.ToTensor + T.Normalize)."""
    img = Image.open(path).convert('RGB').resize(size)
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
    """Auto-export PyTorch weights to .npz if not yet done (step 10.1)."""
    weights_npz = EXPORT / f'efficientnet_{strategy}_weights.npz'
    if weights_npz.exists():
        return weights_npz

    model_pt = MODELS / f'efficientnet_pv_{strategy}_best.pt'
    if not model_pt.exists():
        raise FileNotFoundError(
            f'Neither weights .npz nor PyTorch checkpoint found:\n'
            f'  .npz:  {weights_npz}\n'
            f'  .pt:   {model_pt}\n'
            f'Run 10.1 first, or place the checkpoint in models/'
        )

    print(f'[INFO] Weights .npz not found, exporting from {model_pt} ...')

    import torch
    from torchvision import models

    model = models.efficientnet_b3(weights=None)
    ckpt = torch.load(model_pt, map_location='cpu')

    num_classes = ckpt['classifier.1.weight'].shape[0]
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(ckpt)
    model.eval()

    weights = {}
    for k, v in model.state_dict().items():
        weights[k] = v.detach().cpu().numpy()

    np.savez(weights_npz, **weights)
    print(f'[OK] Weights auto-exported -> {weights_npz}')

    return weights_npz


def main():
    parser = argparse.ArgumentParser(
        description='Export INT8 PTQ TFLite for EfficientNet-B3 (self-contained)'
    )
    parser.add_argument(
        '--strategy',
        required=True,
        choices=['hybrid', 'Fuzzy', 'sbert'],
    )
    parser.add_argument(
        '--num_calib',
        type=int,
        default=200,
        help='Number of calibration samples (default: 200)',
    )
    args = parser.parse_args()

    strategy = args.strategy

    # ----------------------------
    # Paths
    # ----------------------------
    weights_npz = _export_weights_if_missing(strategy)
    calib_csv = SPLITS / strategy / 'pv_calib_10pct.csv'
    out_tflite = EXPORT / f'efficientnet_int8_ptq_{strategy}.tflite'

    if not calib_csv.exists():
        raise FileNotFoundError(f'Missing calibration CSV: {calib_csv}')

    # ----------------------------
    # Load weights
    # ----------------------------
    w = np.load(weights_npz)

    assert 'classifier.1.weight' in w, (
        f'Expected classifier.1.weight, got: {list(w.keys())[:10]}'
    )
    assert 'classifier.1.bias' in w, 'Expected classifier.1.bias'

    num_classes = w['classifier.1.weight'].shape[0]

    print(f'[INFO] Starting PTQ INT8 conversion for EfficientNet-B3 ({strategy})')
    print(f'       Weights: {weights_npz}')
    print(f'       Classes: {num_classes}')
    print(f'       Calibration: {calib_csv}')

    # ----------------------------
    # Build TF Keras EfficientNetB3
    # - batch_size=1: required for EdgeTPU compiler (static tensors)
    # - NO preprocess_input: avoids ops unsupported by EdgeTPU;
    #   normalization is done in calibration data instead
    # ----------------------------
    inputs = tf.keras.Input(batch_shape=(1, 224, 224, 3), name='input')

    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling='avg',
    )

    # Load backbone weights (best-effort matching)
    loaded_layers = 0
    for layer in base.layers:
        w_key = layer.name + '.weight'
        b_key = layer.name + '.bias'
        if w_key in w:
            try:
                if hasattr(layer, 'use_bias') and layer.use_bias:
                    layer.set_weights([w[w_key], w[b_key]])
                else:
                    layer.set_weights([w[w_key]])
                loaded_layers += 1
            except Exception as e:
                print(f'[WARN] Could not set weights for {layer.name}: {e}')

    print(f'[INFO] Loaded weights for {loaded_layers} backbone layers')

    # Classification head
    dense = tf.keras.layers.Dense(num_classes, activation=None, name='logits')
    outputs = dense(base.output)

    model = tf.keras.Model(inputs, outputs)

    # Set classifier weights: PyTorch [out, in] -> TF [in, out]
    dense_kernel = w['classifier.1.weight'].T
    dense_bias = w['classifier.1.bias']
    dense.set_weights([dense_kernel, dense_bias])

    model.trainable = False

    # ----------------------------
    # TFLite INT8 PTQ
    # Use from_concrete_functions to produce a clean TFLite graph
    # without DELEGATE nodes (Keras 3 / TF 2.20 inserts them
    # via from_keras_model, breaking EdgeTPU compilation)
    # ----------------------------
    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32)])
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

    tflite_model = converter.convert()
    out_tflite.write_bytes(tflite_model)

    size_mb = out_tflite.stat().st_size / 1024 / 1024
    print(f'[OK] INT8 PTQ TFLite exported -> {out_tflite} ({size_mb:.1f} MB)')


if __name__ == '__main__':
    main()
