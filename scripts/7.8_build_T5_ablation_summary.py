#!/usr/bin/env python3
"""
Build comprehensive ablation study table (T5).

Combines results from:
- λ sweep (CORAL)
- Calibration size
- Backbone architecture
- Alignment strategy
"""

import csv
import json
from pathlib import Path
from typing import Any

import pandas as pd

CONFIGS = Path('configs')
RESULTS = Path('results')
OUT = RESULTS / 'T5_ablation_comprehensive.csv'


def get_n_aligned_classes(strategy: str) -> int:
    label_map = CONFIGS / f'label_map_{strategy}.csv'
    if not label_map.exists():
        return 0
    with label_map.open(encoding='utf-8') as f:
        return sum(1 for _ in csv.DictReader(f))


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def add_lambda_sweep_rows(rows: list[dict[str, Any]]) -> None:
    print('[INFO] Loading λ sweep results...')
    for strategy in ['Fuzzy', 'hybrid', 'sbert']:
        t2_csv = RESULTS / f'T2_coral_sweep_{strategy}.csv'

        if not t2_csv.exists():
            print(f'[WARN] Missing T2 for {strategy}')
            continue

        n_aligned = get_n_aligned_classes(strategy)
        df = pd.read_csv(t2_csv)
        for _, row in df.iterrows():
            rows.append(
                {
                    'ablation_type': 'lambda_sweep',
                    'strategy': strategy,
                    'n_aligned_classes': n_aligned,
                    'parameter': f'λ={row.get("lambda_coral", row.get("lambda", "?"))}',
                    'target_accuracy': row.get('target_accuracy', ''),
                    'target_f1_macro': row.get('target_macro_f1', ''),
                    'notes': '',
                }
            )


def add_calib_size_rows(rows: list[dict[str, Any]]) -> None:
    print('[INFO] Loading calibration size ablation...')
    calib_csv = RESULTS / 'T5_calib_size_ablation.csv'

    if not calib_csv.exists():
        return

    n_aligned = get_n_aligned_classes('hybrid')
    df = pd.read_csv(calib_csv)
    for _, row in df.iterrows():
        rows.append(
            {
                'ablation_type': 'calib_size',
                'strategy': 'hybrid',
                'n_aligned_classes': n_aligned,
                'parameter': f'n={row["calib_size"]}',
                'target_accuracy': row.get('target_accuracy', ''),
                'target_f1_macro': row.get('target_f1_macro', ''),
                'notes': f'{row.get("model_size_mb", 0):.1f} MB',
            }
        )


def add_backbone_rows(rows: list[dict[str, Any]]) -> None:
    print('[INFO] Loading backbone ablation...')
    strategy = 'hybrid'
    n_aligned = get_n_aligned_classes(strategy)
    for model in ['resnet50', 'efficientnet', 'mobilenet_baseline']:
        json_path = RESULTS / f'metrics_target_{model}_{strategy}.json'
        data = load_json(json_path)
        if data is None:
            continue

        rows.append(
            {
                'ablation_type': 'backbone',
                'strategy': strategy,
                'n_aligned_classes': n_aligned,
                'parameter': model.replace('_baseline', ''),
                'target_accuracy': data.get('accuracy_target', ''),
                'target_f1_macro': data.get('macro_f1_target', ''),
                'notes': 'zero-shot',
            }
        )


def add_alignment_rows(rows: list[dict[str, Any]]) -> None:
    print('[INFO] Loading alignment strategy ablation...')
    for strategy in ['Fuzzy', 'hybrid', 'sbert']:
        json_path = RESULTS / f'metrics_target_resnet50_{strategy}.json'
        data = load_json(json_path)
        if data is None:
            continue

        rows.append(
            {
                'ablation_type': 'alignment',
                'strategy': strategy,
                'n_aligned_classes': get_n_aligned_classes(strategy),
                'parameter': strategy,
                'target_accuracy': data.get('accuracy_target', ''),
                'target_f1_macro': data.get('macro_f1_target', ''),
                'notes': 'ResNet-50',
            }
        )


def save_table(rows: list[dict[str, Any]], out_path: Path) -> None:
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_path, index=False)


def main() -> None:
    rows: list[dict[str, Any]] = []

    add_lambda_sweep_rows(rows)
    add_calib_size_rows(rows)
    add_backbone_rows(rows)
    add_alignment_rows(rows)

    save_table(rows, OUT)

    print(f'[OK] Comprehensive ablation table saved → {OUT}')
    print(f'     Total entries: {len(rows)}')


if __name__ == '__main__':
    main()
