import argparse
import csv
import json
from pathlib import Path

RESULTS = Path('results')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    strategy = args.strategy

    rows = []
    for p in sorted(RESULTS.glob(f'coral_lambda_*_{strategy}_metrics.json')):
        data = json.loads(p.read_text(encoding='utf-8'))

        rows.append(
            {
                'lambda': data['lambda'],
                'pv_accuracy': data['pv_accuracy'],
                'pv_macro_f1': data['pv_macro_f1'],
                'target_accuracy': data['target_accuracy'],
                'target_macro_f1': data['target_macro_f1'],
                'delta_f1': data['delta_f1'],
            }
        )

    if not rows:
        raise RuntimeError(
            f"No CORAL metrics found for strategy='{strategy}'. "
            'Check filenames in results/.'
        )

    out_csv = RESULTS / f'T2_coral_sweep_{strategy}.csv'

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'lambda',
                'pv_accuracy',
                'pv_macro_f1',
                'target_accuracy',
                'target_macro_f1',
                'delta_f1',
            ],
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x['lambda']))

    print(f'[OK] CORAL sweep table saved → {out_csv}')


if __name__ == '__main__':
    main()
