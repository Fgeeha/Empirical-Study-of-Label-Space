import argparse
from pathlib import Path

import tensorflow as tf

EXPORT = Path('export')
EXPORT.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    strategy = args.strategy
    saved_model_dir = EXPORT / f'resnet50_tf_{strategy}'
    out_tflite = EXPORT / f'model_fp32_{strategy}.tflite'

    if not saved_model_dir.exists():
        raise FileNotFoundError(f'Missing SavedModel: {saved_model_dir}')

    print(f'[INFO] Loading SavedModel from {saved_model_dir}')
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    # FP32 — без оптимизаций
    converter.optimizations = []

    tflite_model = converter.convert()

    out_tflite.write_bytes(tflite_model)
    print(f'[OK] FP32 TFLite exported → {out_tflite}')


if __name__ == '__main__':
    main()
