"""
10.2 Build TF SavedModel for EfficientNet-B3 from exported weights.

Reads export/efficientnet_{strategy}_weights.npz and constructs a
tf.keras EfficientNetB3 model, then saves as TF SavedModel.

Usage:
    poetry run python scripts/10.2_build_tf_savedmodel_efficientnet.py --strategy hybrid
    poetry run python scripts/10.2_build_tf_savedmodel_efficientnet.py --strategy Fuzzy
    poetry run python scripts/10.2_build_tf_savedmodel_efficientnet.py --strategy sbert
"""

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

EXPORT = Path('export')


def main():
    parser = argparse.ArgumentParser(
        description='Build TF SavedModel for EfficientNet-B3'
    )
    parser.add_argument(
        '--strategy',
        required=True,
        choices=['hybrid', 'Fuzzy', 'sbert'],
    )
    args = parser.parse_args()

    weights_path = EXPORT / f'efficientnet_{args.strategy}_weights.npz'
    if not weights_path.exists():
        raise FileNotFoundError(f'Missing weights file: {weights_path}')

    w = np.load(weights_path)

    # -------------------------------------------------
    # Build EfficientNetB3
    # -------------------------------------------------
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')

    # EfficientNet preprocessing (scales to [0, 1] internally)
    x = tf.keras.applications.efficientnet.preprocess_input(inputs)

    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling='avg',
    )

    # -------------------------------------------------
    # Load backbone weights (best-effort matching)
    # -------------------------------------------------
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

    print(f'[INFO] Loaded weights for {loaded_layers} layers')

    # -------------------------------------------------
    # Classification head
    # -------------------------------------------------
    num_classes = w['classifier.1.weight'].shape[0]

    logits = tf.keras.layers.Dense(
        num_classes,
        activation=None,
        name='logits',
    )(base.output)

    model = tf.keras.Model(inputs, logits)

    # Set classifier weights
    dense_kernel = w['classifier.1.weight'].T  # PyTorch [out, in] -> TF [in, out]
    dense_bias = w['classifier.1.bias']
    model.get_layer('logits').set_weights([dense_kernel, dense_bias])

    # -------------------------------------------------
    # Save TF SavedModel
    # -------------------------------------------------
    out_dir = EXPORT / f'efficientnet_tf_{args.strategy}'
    model.save(out_dir)

    print(f'[OK] TF SavedModel exported -> {out_dir}')
    print(f'     num_classes = {num_classes}')


if __name__ == '__main__':
    main()
