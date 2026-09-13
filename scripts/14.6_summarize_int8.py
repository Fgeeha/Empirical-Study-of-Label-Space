#!/usr/bin/env python3
"""Summarize results/int8_accuracy/*.json into one table: PyTorch FP32 reference vs
each TFLite variant (legacy 2026-02-08 files, 10.x re-export from current
checkpoints, onnx2tf float32 / full-integer INT8), on PlantVillage and PlantDoc
test splits. Writes results/int8_accuracy/INT8_ACCURACY_SUMMARY.md and .csv.

    python3 scripts/14.6_summarize_int8.py
"""

import csv
import json
from pathlib import Path

R = Path('results/int8_accuracy')
ARCHS = [
    ('resnet50', 'ResNet-50', 'resnet50'),
    ('mobilenetv1', 'MobileNetV1', 'mobilenetv1'),
    ('mobilenetv2', 'MobileNetV2', 'mobilenetv2'),
    ('efficientnet_lite0', 'EfficientNet-Lite0', 'efficientnet_lite0'),
    ('efficientnet', 'EfficientNet-B3', 'efficientnet_b3'),
    ('dann42', 'ResNet-50 DANN seed 42', 'dann42'),
]
VARIANTS = [
    ('legacy Keras-rebuild INT8 (2026-02-08 file)', '{a}_int8_ptq_hybrid'),
    (
        'Keras-rebuild INT8, current checkpoint (10.x scripts)',
        'reexport_20260912__{a}_int8_ptq_hybrid',
    ),
    ('onnx2tf float32', '{o}__{o}_hybrid_float32'),
    ('onnx2tf INT8 full-integer', '{o}__{o}_hybrid_full_integer_quant'),
]
SPLITS = [('pv_test', 'PlantVillage test'), ('plantdoc_test', 'PlantDoc test')]


def load(tag: str, split: str) -> dict | None:
    p = R / f'{tag}__{split}.json'
    return json.load(open(p)) if p.exists() else None


def main() -> None:
    rows = []
    for a, name, pt in ARCHS:
        onnx_name = pt  # onnx exports are named by the 14.2 model key
        for split, split_name in SPLITS:
            ref = load(f'pytorch_{pt}_hybrid_pil', split) or load(
                f'pytorch_{pt}_hybrid', split
            )
            if ref is None:
                continue
            tv = load(f'pytorch_{pt}_hybrid', split)
            if tv is not None and ref is not tv:
                rows.append(
                    {
                        'arch': name,
                        'split': split_name,
                        'variant': 'PyTorch FP32, torchvision Resize (canonical eval transform)',
                        'f1': tv['macro_f1'],
                        'acc': tv['accuracy'],
                        'd_f1': tv['macro_f1'] - ref['macro_f1'],
                        'agree': sum(
                            x == y for x, y in zip(tv['y_pred'], ref['y_pred'])
                        )
                        / tv['n'],
                        'n': tv['n'],
                    }
                )
            rows.append(
                {
                    'arch': name,
                    'split': split_name,
                    'variant': 'PyTorch FP32, PIL preprocessing (reference for TFLite rows)',
                    'f1': ref['macro_f1'],
                    'acc': ref['accuracy'],
                    'd_f1': 0.0,
                    'agree': 1.0,
                    'n': ref['n'],
                }
            )
            for vname, pat in VARIANTS:
                tag = pat.format(
                    a=a if a != 'dann42' else 'resnet50_dann42', o=onnx_name
                )
                r = load(tag, split)
                if r is None:
                    continue
                agree = sum(x == y for x, y in zip(r['y_pred'], ref['y_pred'])) / r['n']
                label = (
                    vname
                    if a != 'dann42' or 'legacy' not in vname
                    else 'Keras-rebuild INT8 (10.4 path via 14.3, 2026-09-12)'
                )
                rows.append(
                    {
                        'arch': name,
                        'split': split_name,
                        'variant': label,
                        'f1': r['macro_f1'],
                        'acc': r['accuracy'],
                        'd_f1': r['macro_f1'] - ref['macro_f1'],
                        'agree': agree,
                        'n': r['n'],
                    }
                )
    with open(R / 'INT8_ACCURACY_SUMMARY.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    lines = [
        '# TFLite conversion and INT8 accuracy audit (Hybrid 18-class models)',
        '',
        'Host evaluation (x86 CPU, TFLite 2.15 interpreter). All TFLite rows and the PIL reference share one preprocessing: PIL resize to 224x224 (bicubic) -> /255 -> ImageNet normalization (or [0,1] for graphs with embedded normalization). The canonical PyTorch evaluation transform (torchvision Resize, bilinear with antialias) is shown for comparison; the 1-2 pp gap between the two PyTorch rows is preprocessing, not conversion. `agree` = top-1 agreement with the PIL reference on the same images.',
        '',
    ]
    for _a, name, _ in ARCHS:
        sub = [r for r in rows if r['arch'] == name]
        if not sub:
            continue
        lines += [
            f'## {name}',
            '',
            '| Split | Variant | macro-F1 | acc | Δ F1 vs PyTorch | agree |',
            '|---|---|---:|---:|---:|---:|',
        ]
        for r in sub:
            lines.append(
                f'| {r["split"]} | {r["variant"]} | {r["f1"]:.4f} | {r["acc"]:.4f} | {r["d_f1"]:+.4f} | {r["agree"]:.3f} |'
            )
        lines.append('')
    (R / 'INT8_ACCURACY_SUMMARY.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
