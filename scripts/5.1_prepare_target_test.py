import argparse
import csv
from pathlib import Path

import pandas as pd

DATA = Path('data')
CONFIGS = Path('configs')
SPLITS = Path('splits')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument(
        '--target',
        choices=['plantdoc', 'fcdd'],
        default='plantdoc',
    )
    args = parser.parse_args()

    strategy = args.strategy
    target = args.target

    label_map = pd.read_csv(CONFIGS / f'label_map_{strategy}.csv')

    label_dict = dict(
        zip(
            label_map['canonical_name'],
            label_map['label_id'],
            strict=True,
        )
    )

    if target == 'plantdoc':
        root = DATA / 'shared_subset' / strategy / 'target'
    else:
        root = DATA / 'shared_subset' / strategy / 'fcdd'

    out_dir = SPLITS / strategy
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'target_{target}_test.csv'

    rows = []

    for cls_name, label_id in label_dict.items():
        cls_dir = root / cls_name
        if not cls_dir.exists():
            continue

        for img in cls_dir.iterdir():
            if img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                rows.append([img.as_posix(), label_id])

    with out_csv.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'label_id'])
        writer.writerows(rows)

    print(f'[OK] Target test prepared: {out_csv} ({len(rows)} samples)')


if __name__ == '__main__':
    main()
