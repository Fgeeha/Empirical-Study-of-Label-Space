"""
10.4 Export INT8 PTQ TFLite for ResNet50 (self-contained).

Loads weights from export/resnet50_{strategy}_weights.npz, builds a
TF Keras ResNet50 model with correct PyTorch -> Keras weight mapping,
and converts directly to INT8 TFLite.

No intermediate SavedModel needed.
No preprocess_input (avoids strided_slice/reverse ops for EdgeTPU).
Static batch_size=1 for EdgeTPU compiler.

Prerequisites:
    - export/resnet50_{strategy}_weights.npz  (from 8.2a_export_torch_weights.py)
    - splits/{strategy}/pv_calib_10pct.csv     (from 8.1_prepare_pv_calibration.py)

Output files:
    export/resnet50_int8_ptq_{strategy}.tflite

Usage:
    CUDA_VISIBLE_DEVICES="" poetry run python scripts/10.4_export_tflite_int8_resnet50.py --strategy hybrid
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

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

# ResNet-50 bottleneck blocks per layer
BLOCKS_PER_LAYER = [3, 4, 6, 3]


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


def build_resnet50_mapping():
    """Build mapping: PyTorch state_dict prefix -> (Keras layer name, type).

    PyTorch: conv1, bn1, layer{L}.{B}.conv{K}, layer{L}.{B}.bn{K},
             layer{L}.0.downsample.{0,1}, fc
    Keras:   conv1_conv, conv1_bn, conv{L+1}_block{B+1}_{K}_conv/bn,
             conv{L+1}_block{B+1}_0_conv/bn
    """
    mapping = {}

    mapping['conv1'] = ('conv1_conv', 'conv')
    mapping['bn1'] = ('conv1_bn', 'bn')

    for layer_idx, num_blocks in enumerate(BLOCKS_PER_LAYER):
        for block_idx in range(num_blocks):
            pt = f'layer{layer_idx + 1}.{block_idx}'
            ks = f'conv{layer_idx + 2}_block{block_idx + 1}'

            for conv_idx in range(1, 4):
                mapping[f'{pt}.conv{conv_idx}'] = (
                    f'{ks}_{conv_idx}_conv',
                    'conv',
                )
                mapping[f'{pt}.bn{conv_idx}'] = (f'{ks}_{conv_idx}_bn', 'bn')

            if block_idx == 0:
                mapping[f'{pt}.downsample.0'] = (f'{ks}_0_conv', 'conv')
                mapping[f'{pt}.downsample.1'] = (f'{ks}_0_bn', 'bn')

    return mapping


def transfer_resnet50_weights(npz_weights, keras_model):
    """Transfer PyTorch ResNet-50 weights to Keras model."""
    mapping = build_resnet50_mapping()
    layer_dict = {layer.name: layer for layer in keras_model.layers}

    loaded = 0

    for pt_prefix, (keras_name, layer_type) in mapping.items():
        if keras_name not in layer_dict:
            continue

        layer = layer_dict[keras_name]

        try:
            if layer_type == 'conv':
                w_key = f'{pt_prefix}.weight'
                if w_key not in npz_weights:
                    continue
                # PyTorch [out, in, H, W] -> Keras [H, W, in, out]
                kernel = npz_weights[w_key].transpose(2, 3, 1, 0)
                # Keras ResNet50 uses use_bias=True; PyTorch has no bias
                bias = np.zeros(kernel.shape[-1], dtype=np.float32)
                layer.set_weights([kernel, bias])
                loaded += 1

            elif layer_type == 'bn':
                gamma = npz_weights[f'{pt_prefix}.weight']
                beta = npz_weights[f'{pt_prefix}.bias']
                mean = npz_weights[f'{pt_prefix}.running_mean']
                var = npz_weights[f'{pt_prefix}.running_var']
                layer.set_weights([gamma, beta, mean, var])
                loaded += 1

        except Exception as e:
            print(f'[WARN] Failed: {keras_name}: {e}')

    return loaded


def _export_weights_if_missing(strategy):
    """Auto-export PyTorch weights to .npz if not yet done (step 8.2a)."""
    weights_npz = EXPORT / f'resnet50_{strategy}_weights.npz'
    if weights_npz.exists():
        return weights_npz

    model_pt = MODELS / f'resnet50_pv_{strategy}_best.pt'
    if not model_pt.exists():
        raise FileNotFoundError(
            f'Neither weights .npz nor PyTorch checkpoint found:\n'
            f'  .npz:  {weights_npz}\n'
            f'  .pt:   {model_pt}\n'
            f'Run 8.2a first, or place the checkpoint in models/'
        )

    print(f'[INFO] Weights .npz not found, exporting from {model_pt} ...')

    import torch
    from torchvision import models

    model = models.resnet50(weights=None)
    ckpt = torch.load(model_pt, map_location='cpu')

    num_classes = ckpt['fc.weight'].shape[0]
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
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
        description='Export INT8 PTQ TFLite for ResNet50 (self-contained)'
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
    out_tflite = EXPORT / f'resnet50_int8_ptq_{strategy}.tflite'

    if not calib_csv.exists():
        raise FileNotFoundError(f'Missing calibration CSV: {calib_csv}')

    # ----------------------------
    # Load weights
    # ----------------------------
    w = np.load(weights_npz)

    assert 'fc.weight' in w, f'Expected fc.weight in .npz, got: {list(w.keys())[:10]}'
    assert 'fc.bias' in w, 'Expected fc.bias in .npz'

    num_classes = w['fc.weight'].shape[0]

    print(f'[INFO] Starting PTQ INT8 conversion for ResNet50 ({strategy})')
    print(f'       Weights: {weights_npz}')
    print(f'       Classes: {num_classes}')
    print(f'       Calibration: {calib_csv}')

    # ----------------------------
    # Build TF Keras ResNet50
    # - batch_size=1: required for EdgeTPU compiler (static tensors)
    # - NO preprocess_input: avoids strided_slice/reverse ops
    #   normalization is done in calibration data instead
    # ----------------------------
    inputs = tf.keras.Input(batch_shape=(1, 224, 224, 3), name='input')

    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling='avg',
    )

    # Transfer backbone weights with correct mapping
    loaded = transfer_resnet50_weights(w, base)
    total = len([layer for layer in base.layers if layer.get_weights()])
    print(f'[INFO] Transferred {loaded}/{total} backbone layers')

    if loaded < total * 0.9:
        raise RuntimeError(
            f'Weight transfer failed: only {loaded}/{total} layers matched'
        )

    # Classification head
    dense = tf.keras.layers.Dense(num_classes, activation=None, name='logits')
    outputs = dense(base.output)

    model = tf.keras.Model(inputs, outputs)

    # Set classifier weights: PyTorch [out, in] -> TF [in, out]
    dense_kernel = w['fc.weight'].T
    dense_bias = w['fc.bias']
    dense.set_weights([dense_kernel, dense_bias])

    model.trainable = False

    # ----------------------------
    # TFLite INT8 PTQ
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
