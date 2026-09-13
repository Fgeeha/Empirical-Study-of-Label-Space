"""
8.2b Build TF Keras SavedModel for ResNet-50 from PyTorch weights.

Performs proper PyTorch -> Keras weight transfer with layer name mapping,
embeds ImageNet normalization into the graph, and exports SavedModel
for downstream FP32/INT8 TFLite conversion (scripts 8.3, 8.4).

Prerequisites:
    - export/resnet50_{strategy}_weights.npz  (from 8.2a)

Output:
    export/resnet50_tf_{strategy}/   (TF SavedModel)

Usage:
    CUDA_VISIBLE_DEVICES="" poetry run python scripts/8.2b_build_tf_savedmodel.py --strategy hybrid
"""

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

EXPORT = Path('export')

# ResNet-50 bottleneck blocks per layer
BLOCKS_PER_LAYER = [3, 4, 6, 3]

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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


def main():
    parser = argparse.ArgumentParser(
        description='Build TF SavedModel for ResNet-50 from PyTorch weights'
    )
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    weights_npz = EXPORT / f'resnet50_{args.strategy}_weights.npz'
    if not weights_npz.exists():
        raise FileNotFoundError(f'Missing weights: {weights_npz}')

    w = np.load(weights_npz)
    num_classes = w['fc.weight'].shape[0]

    print(f'[INFO] Building TF SavedModel for ResNet-50 ({args.strategy})')
    print(f'       Weights: {weights_npz}')
    print(f'       Classes: {num_classes}')

    # Build Keras ResNet-50 with ImageNet normalization embedded
    raw_inputs = tf.keras.Input(shape=(224, 224, 3), name='input')

    # Normalize: [0,1] float -> ImageNet mean/std
    x = tf.keras.layers.Rescaling(
        scale=1.0 / np.array(IMAGENET_STD, dtype=np.float32),
        offset=-np.array(IMAGENET_MEAN, dtype=np.float32)
        / np.array(IMAGENET_STD, dtype=np.float32),
        name='imagenet_norm',
    )(raw_inputs)

    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling='avg',
    )

    # Transfer backbone weights
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
    model = tf.keras.Model(raw_inputs, outputs)

    # Set classifier weights: PyTorch [out, in] -> TF [in, out]
    dense.set_weights([w['fc.weight'].T, w['fc.bias']])

    # Export SavedModel
    out = EXPORT / f'resnet50_tf_{args.strategy}'
    # Keras 3 / TF 2.20: model.save() requires .keras extension.
    # Use model.export() for SavedModel format (consumed by 8.3/8.4).
    model.export(str(out))
    print(f'[OK] TF SavedModel exported -> {out}')


if __name__ == '__main__':
    main()
