import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

# =========================
# Paths
# =========================
SPLITS = Path('splits')
EXPORT = Path('export')
RESULTS = Path('results')

RESULTS.mkdir(exist_ok=True)


# =========================
# Utils
# =========================
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(path):
    """Load and normalize image to match PyTorch training preprocessing.

    Pipeline: resize(224) -> /255 -> ImageNet normalize (mean/std).
    The TFLite model was exported WITHOUT an embedded normalization layer
    (to preserve EdgeTPU compatibility), so normalization must be applied here.
    Bug fixed: original script only divided by 255, causing ~0% accuracy.
    """
    img = Image.open(path).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img).astype('float32') / 255.0
    arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD  # ImageNet normalize
    return arr[None, ...]  # (1,224,224,3)


def eval_tflite(model_path, csv_path):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]['dtype']
    quant = input_details[0].get('quantization')
    input_scale = quant[0] if quant and len(quant) >= 1 else 1.0
    input_zero = quant[1] if quant and len(quant) >= 2 else 0

    df = pd.read_csv(csv_path)

    preds, targets = [], []
    times = []

    # warmup
    img = load_image(df.iloc[0]['path'])
    if input_dtype == np.int8:
        img = np.round(img / input_scale + input_zero).astype(np.int8)
        img = np.clip(img, -128, 127)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    for _, row in df.iterrows():
        img = load_image(row['path'])

        if input_dtype == np.int8:
            img = np.round(img / input_scale + input_zero).astype(np.int8)
            img = np.clip(img, -128, 127)

        t0 = time.time()
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        t1 = time.time()

        out = interpreter.get_tensor(output_details[0]['index'])
        preds.append(int(out.argmax(1)[0]))
        targets.append(int(row['label_id']))
        times.append((t1 - t0) * 1000.0)

    return {
        'accuracy': accuracy_score(targets, preds),
        'macro_f1': f1_score(targets, preds, average='macro', zero_division=0),
        'latency_ms': float(np.mean(times)),
        'size_mb': model_path.stat().st_size / (1024 * 1024),
    }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--target', default='plantdoc')
    args = parser.parse_args()

    strategy = args.strategy
    target = args.target

    pv_csv = SPLITS / strategy / 'pv_test.csv'
    tgt_csv = SPLITS / strategy / f'target_{target}_test.csv'

    fp32_model = EXPORT / f'model_fp32_{strategy}.tflite'
    int8_model = EXPORT / f'model_int8_ptq_{strategy}.tflite'

    if not fp32_model.exists():
        raise FileNotFoundError(fp32_model)
    if not int8_model.exists():
        raise FileNotFoundError(int8_model)

    rows = []

    for domain, csv in [
        ('pv', pv_csv),
        ('target', tgt_csv),
    ]:
        print(f'[INFO] Evaluating {domain.upper()}')

        fp32 = eval_tflite(fp32_model, csv)
        int8 = eval_tflite(int8_model, csv)

        rows.append(
            {
                'domain': domain,
                'model': 'fp32',
                **fp32,
            }
        )
        rows.append(
            {
                'domain': domain,
                'model': 'int8',
                **int8,
            }
        )

    out_csv = RESULTS / f'T3_quant_fp32_vs_int8_{strategy}.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f'[OK] Quantization eval saved → {out_csv}')


if __name__ == '__main__':
    main()
