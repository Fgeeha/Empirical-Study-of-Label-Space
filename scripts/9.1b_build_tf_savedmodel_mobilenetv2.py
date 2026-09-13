import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

EXPORT = Path('export')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strategy',
        required=True,
        choices=['hybrid', 'Fuzzy', 'sbert'],
    )
    args = parser.parse_args()

    weights_path = EXPORT / f'mobilenetv2_{args.strategy}_weights.npz'
    if not weights_path.exists():
        raise FileNotFoundError(f'Missing weights file: {weights_path}')

    w = np.load(weights_path)

    # -------------------------------------------------
    # Build MobileNetV2 (EdgeTPU-friendly)
    # -------------------------------------------------
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling='avg',
    )

    # -------------------------------------------------
    # Load backbone weights
    # -------------------------------------------------
    for layer in base.layers:
        w_key = layer.name + '.weight'
        b_key = layer.name + '.bias'

        if w_key in w:
            if layer.use_bias:
                layer.set_weights([w[w_key], w[b_key]])
            else:
                layer.set_weights([w[w_key]])

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

    # -------------------------------------------------
    # Save TF SavedModel
    # -------------------------------------------------
    out_dir = EXPORT / f'mobilenetv2_tf_{args.strategy}'
    model.save(out_dir)

    print(f'[OK] TF SavedModel exported → {out_dir}')


if __name__ == '__main__':
    main()
