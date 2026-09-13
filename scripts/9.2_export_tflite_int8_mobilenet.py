import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

EXPORT = Path('export')
SPLITS = Path('splits')
EXPORT.mkdir(exist_ok=True)


def load_image(path, size=(224, 224)):
    img = Image.open(path).convert('RGB').resize(size)
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    strategy = args.strategy

    # ----------------------------
    # Correct paths ✅
    # ----------------------------
    weights_npz = EXPORT / f'mobilenetv2_{strategy}_weights.npz'
    calib_csv = SPLITS / strategy / 'pv_calib_10pct.csv'
    out_tflite = EXPORT / f'mobilenetv2_int8_ptq_{strategy}.tflite'

    if not weights_npz.exists():
        raise FileNotFoundError(f'Missing weights: {weights_npz}')

    if not calib_csv.exists():
        raise FileNotFoundError(f'Missing calibration CSV: {calib_csv}')

    # ----------------------------
    # Load weights
    # ----------------------------
    w = np.load(weights_npz)

    assert 'classifier.1.weight' in w, w.keys()
    assert 'classifier.1.bias' in w, w.keys()

    num_classes = w['classifier.1.weight'].shape[0]

    # ----------------------------
    # Build TF MobileNetV2
    # ----------------------------
    inputs = tf.keras.Input(shape=(224, 224, 3), name='input')
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights=None,
        input_tensor=x,
        pooling='avg',
    )

    dense = tf.keras.layers.Dense(num_classes, activation=None)
    outputs = dense(base.output)

    model = tf.keras.Model(inputs, outputs)

    # Load classifier weights
    dense_kernel = w['classifier.1.weight'].T
    dense_bias = w['classifier.1.bias']
    dense.set_weights([dense_kernel, dense_bias])

    model.trainable = False

    # ----------------------------
    # TFLite INT8 PTQ
    # ----------------------------
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.representative_dataset = representative_dataset(calib_csv)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_tflite.write_bytes(tflite_model)

    print(f'[OK] INT8 TFLite exported → {out_tflite}')


if __name__ == '__main__':
    main()
