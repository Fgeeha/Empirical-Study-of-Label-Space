#!/usr/bin/env python3
"""Evaluate exported TFLite models (FP32 / INT8) on the PlantVillage and PlantDoc
test splits and compare with the PyTorch reference metrics.

Run with the TFLite environment:
    venv-tflite/bin/python scripts/14.1_eval_tflite_accuracy.py --strategy hybrid

Preprocessing is chosen from the model's input quantization: models whose input
scale is 1/255 take [0, 1] pixels (ImageNet normalization is embedded in the
graph, 8.x/9.x pipeline); all other models take ImageNet-normalized input
(10.x pipeline). Output: results/int8_accuracy/{model}_{split}.json with
accuracy, macro-F1, n, latency on this host, and per-sample predictions.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import tensorflow as tf

EXPORT = Path('export')
SPLITS = Path('splits')
OUT = Path('results/int8_accuracy')
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_image(path: str, normalize: bool) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert('RGB').resize((224, 224)), np.float32)
    arr = arr / 255.0
    if normalize:
        arr = (arr - MEAN) / STD
    return arr[None]


def evaluate(model_path: Path, csv_path: Path, threads: int) -> dict:
    it = tf.lite.Interpreter(model_path=str(model_path), num_threads=threads)
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    in_scale, in_zp = inp['quantization']
    normalize = not (inp['dtype'] == np.int8 and abs(in_scale - 1 / 255) < 1e-6)
    df = pd.read_csv(csv_path)
    y_true, y_pred, times = [], [], []
    for _, row in df.iterrows():
        x = load_image(row['path'], normalize)
        if inp['dtype'] == np.int8:
            x = np.clip(np.round(x / in_scale + in_zp), -128, 127).astype(np.int8)
        t0 = time.perf_counter()
        it.set_tensor(inp['index'], x)
        it.invoke()
        logits = it.get_tensor(out['index'])[0]
        times.append(time.perf_counter() - t0)
        y_true.append(int(row['label_id']))
        y_pred.append(int(np.argmax(logits)))
    return {
        'model': model_path.name,
        'split': csv_path.name,
        'input_dtype': inp['dtype'].__name__,
        'input_quant': [float(in_scale), int(in_zp)],
        'normalized_input': normalize,
        'n': len(df),
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'host_latency_ms_mean': 1000 * float(np.mean(times)),
        'threads': threads,
        'y_true': y_true,
        'y_pred': y_pred,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--strategy', default='hybrid')
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument(
        '--models', nargs='*', help='TFLite files; default: all for strategy'
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    models = (
        [Path(m) for m in args.models]
        if args.models
        else sorted(
            p
            for p in EXPORT.glob(f'*_{args.strategy}.tflite')
            if 'edgetpu' not in p.name
        )
    )
    splits = {
        'pv_test': SPLITS / args.strategy / 'pv_test.csv',
        'plantdoc_test': SPLITS / args.strategy / 'target_plantdoc_test.csv',
    }
    summary = []
    for m in models:
        for split_name, csv_path in splits.items():
            res = evaluate(m, csv_path, args.threads)
            tag = (
                m.stem
                if m.parent.resolve() == EXPORT.resolve()
                else f'{m.parent.name}__{m.stem}'
            )
            out_path = OUT / f'{tag}__{split_name}.json'
            out_path.write_text(json.dumps(res, indent=1))
            res['tag'] = tag
            row = {
                k: res[k]
                for k in (
                    'tag',
                    'model',
                    'split',
                    'n',
                    'accuracy',
                    'macro_f1',
                    'normalized_input',
                    'host_latency_ms_mean',
                )
            }
            summary.append(row)
            print(
                f'{res["model"]:40s} {split_name:14s} n={res["n"]:5d} acc={res["accuracy"]:.4f} F1={res["macro_f1"]:.4f} norm={res["normalized_input"]} {res["host_latency_ms_mean"]:.1f} ms'
            )
    out_csv = OUT / f'summary_{args.strategy}_{int(time.time())}.csv'
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print('saved', out_csv)


if __name__ == '__main__':
    main()
