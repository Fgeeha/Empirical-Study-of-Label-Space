import argparse
import csv
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

RESULTS = Path('results')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--model', default='resnet50')
    parser.add_argument('--lambda_coral', type=float, default=0.1)
    args = parser.parse_args()

    strategy = args.strategy
    model = args.model
    lam = args.lambda_coral

    # ----------------------------
    # Load predictions
    # ----------------------------
    baseline_preds = RESULTS / f'preds_target_{model}_{strategy}.csv'
    coral_preds = RESULTS / f'preds_target_coral_lambda_{lam}_{strategy}.csv'

    if not baseline_preds.exists():
        raise FileNotFoundError(f'Missing baseline preds: {baseline_preds}')
    if not coral_preds.exists():
        raise FileNotFoundError(f'Missing CORAL preds: {coral_preds}')

    df_base = pd.read_csv(baseline_preds)
    df_coral = pd.read_csv(coral_preds)

    # ----------------------------
    # Per-class F1
    # ----------------------------
    report_base = classification_report(
        df_base['y_true'],
        df_base['y_pred'],
        output_dict=True,
        zero_division=0,
    )

    report_coral = classification_report(
        df_coral['y_true'],
        df_coral['y_pred'],
        output_dict=True,
        zero_division=0,
    )

    # ----------------------------
    # Build comparison table
    # ----------------------------
    rows = [
        [
            'class_id',
            'baseline_f1',
            'coral_f1',
            'delta_f1',
            'support',
        ]
    ]

    for key in report_base.keys():
        if key in ['accuracy', 'macro avg', 'weighted avg']:
            continue

        try:
            class_id = int(key)
        except ValueError:
            continue

        base_f1 = report_base[key]['f1-score']
        coral_f1 = report_coral[key]['f1-score']
        support = int(report_base[key]['support'])

        rows.append(
            [
                class_id,
                base_f1,
                coral_f1,
                coral_f1 - base_f1,
                support,
            ]
        )

    # ----------------------------
    # Save
    # ----------------------------
    out_csv = RESULTS / f'T6_per_class_f1_{strategy}.csv'

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'[OK] Per-class F1 table saved → {out_csv}')


if __name__ == '__main__':
    main()
