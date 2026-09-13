import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

EXPORT = Path('export')
SPLITS = Path('splits')

EXPORT.mkdir(exist_ok=True)

IMG_SIZE = (224, 224)


def representative_dataset(csv_path: Path):
    df = pd.read_csv(csv_path)

    def gen():
        for _, row in df.iterrows():
            img = Image.open(row['path']).convert('RGB')
            img = img.resize(IMG_SIZE)
            x = np.asarray(img).astype('float32')
            x = x / 255.0
            x = np.expand_dims(x, axis=0)
            yield [x]

    return gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    strategy = args.strategy

    saved_model_dir = EXPORT / f'resnet50_tf_{strategy}'
    calib_csv = SPLITS / 'pv_calib_10pct.csv'
    out_tflite = EXPORT / f'model_int8_ptq_{strategy}.tflite'

    if not saved_model_dir.exists():
        raise FileNotFoundError(f'Missing SavedModel: {saved_model_dir}')
    if not calib_csv.exists():
        raise FileNotFoundError(f'Missing calibration CSV: {calib_csv}')

    print('[INFO] Starting PTQ INT8 conversion')
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset(calib_csv)

    # Full INT8
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    out_tflite.write_bytes(tflite_model)

    print(f'[OK] INT8 PTQ TFLite exported → {out_tflite}')


if __name__ == '__main__':
    main()
