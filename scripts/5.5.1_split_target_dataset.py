import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random

import pandas as pd

SEED = 42
SPLIT = {'train': 0.8, 'val': 0.2}

SPLITS = Path('splits')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--target', choices=['plantdoc', 'fcdd'], default='plantdoc')
    args = parser.parse_args()

    random.seed(SEED)
    strategy = args.strategy
    target = args.target

    src_csv = SPLITS / strategy / f'target_{target}_test.csv'
    out_dir = SPLITS / strategy
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_csv)

    by_label = defaultdict(list)
    for _, row in df.iterrows():
        by_label[row['label_id']].append(row)

    train_rows, val_rows = [], []

    for _label_id, rows in by_label.items():
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n * SPLIT['train'])
        train_rows.extend(rows[:n_train])
        val_rows.extend(rows[n_train:])

    for name, rows in [('train', train_rows), ('val', val_rows)]:
        out_csv = out_dir / f'target_{target}_{name}.csv'
        with out_csv.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['path', 'label_id'])
            for r in rows:
                writer.writerow([r['path'], r['label_id']])

        print(f'[OK] {name} split: {out_csv} ({len(rows)})')


if __name__ == '__main__':
    main()
