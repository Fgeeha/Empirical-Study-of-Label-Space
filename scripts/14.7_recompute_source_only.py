#!/usr/bin/env python3
"""Re-evaluated source-only baseline (archived checkpoints, canonical torchvision
transform, scripts/14.2) as the reference for Table 1, the shared-subset control
(E3) and the paired bootstrap (collect_uda_results.py).

Reads results/int8_accuracy/pytorch_resnet50_{strategy}__{pv_test,plantdoc_test}.json
and writes
  results/preds_target_resnet50_{strategy}_reeval.csv   (path, y_true, y_pred)
  results/final/T1_domain_shift_reeval.csv              (same columns as T1_domain_shift.csv)

    python scripts/14.7_recompute_source_only.py
"""

import csv
import json
from pathlib import Path

import pandas as pd

RESULTS = Path('results')
SPLITS = Path('splits')
STRATEGIES = ['Fuzzy', 'hybrid', 'sbert']


def main() -> None:
    rows = []
    for s in STRATEGIES:
        pv = json.load(open(RESULTS / f'int8_accuracy/pytorch_resnet50_{s}__pv_test.json'))
        pd_ = json.load(
            open(RESULTS / f'int8_accuracy/pytorch_resnet50_{s}__plantdoc_test.json')
        )
        split = pd.read_csv(SPLITS / s / 'target_plantdoc_test.csv')
        assert len(split) == pd_['n'] and split.label_id.tolist() == pd_['y_true']
        out = RESULTS / f'preds_target_resnet50_{s}_reeval.csv'
        pd.DataFrame(
            {'path': split.path, 'y_true': pd_['y_true'], 'y_pred': pd_['y_pred']}
        ).to_csv(out, index=False)
        rows.append(
            {
                'model': 'resnet50',
                'strategy': s,
                'pv_accuracy': round(pv['accuracy'], 4),
                'pv_macro_f1': round(pv['macro_f1'], 4),
                'target_accuracy': round(pd_['accuracy'], 4),
                'target_macro_f1': round(pd_['macro_f1'], 4),
                'delta_f1': round(pv['macro_f1'] - pd_['macro_f1'], 4),
            }
        )
        print(f'{s:7s} PV {pv["macro_f1"]:.4f}  PlantDoc {pd_["macro_f1"]:.4f}  -> {out}')
    out = RESULTS / 'final/T1_domain_shift_reeval.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'[OK] {out}')


if __name__ == '__main__':
    main()
