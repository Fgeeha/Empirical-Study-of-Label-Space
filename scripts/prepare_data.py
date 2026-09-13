#!/usr/bin/env python
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--configs', type=Path, required=True)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def load_mapping(path: Path):
    with open(path) as f:
        return json.load(f)


def scan_dataset(root: Path):
    """
    Expect structure:
      dataset_raw/
        class_name/
          img1.jpg
          img2.jpg
    """
    samples = []
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in cls_dir.iterdir():
            if img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                samples.append((str(img), cls_dir.name))
    return samples


def write_csv(path: Path, rows, header):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    args = parse_args()
    random.seed(args.seed)

    data_root = args.data_root
    mapping = load_mapping(args.configs)

    datasets = {
        'PlantVillage': data_root / 'PlantVillage_raw',
        'PlantDoc': data_root / 'PlantDoc_raw',
        'FCDD': data_root / 'FCDD_raw',
    }

    all_samples = []
    class_set = set()

    for name, root in datasets.items():
        samples = scan_dataset(root)
        for path, cls in samples:
            if cls not in mapping:
                continue  # ignore unmapped classes
            target_cls = mapping[cls]
            all_samples.append((path, name, target_cls))
            class_set.add(target_cls)

    classes = sorted(class_set)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # ===== artifacts/label_map.csv =====
    artifacts = Path('artifacts')
    artifacts.mkdir(exist_ok=True)

    write_csv(
        artifacts / 'label_map.csv',
        [(c, class_to_idx[c]) for c in classes],
        ['class_name', 'class_id'],
    )

    # ===== splits =====
    splits_dir = Path('splits')
    splits_dir.mkdir(exist_ok=True)

    by_dataset = defaultdict(list)
    for row in all_samples:
        by_dataset[row[1]].append(row)

    train, val, test = [], [], []

    for _dataset, rows in by_dataset.items():
        random.shuffle(rows)
        n = len(rows)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train += rows[:n_train]
        val += rows[n_train : n_train + n_val]
        test += rows[n_train + n_val :]

    def encode(rows):
        return [(p, d, c, class_to_idx[c]) for p, d, c in rows]

    write_csv(
        splits_dir / 'train.csv',
        encode(train),
        ['path', 'dataset', 'class_name', 'class_id'],
    )
    write_csv(
        splits_dir / 'val.csv',
        encode(val),
        ['path', 'dataset', 'class_name', 'class_id'],
    )
    write_csv(
        splits_dir / 'test.csv',
        encode(test),
        ['path', 'dataset', 'class_name', 'class_id'],
    )

    # ===== PV calibration (10%) =====
    pv_rows = [r for r in train if r[1] == 'PlantVillage']
    random.shuffle(pv_rows)
    k = max(1, int(0.1 * len(pv_rows)))

    write_csv(
        splits_dir / 'pv_calib_10pct.csv',
        encode(pv_rows[:k]),
        ['path', 'dataset', 'class_name', 'class_id'],
    )

    print('[OK] Data preparation finished')
    print(f'Classes: {len(classes)}')
    print(f'Train/Val/Test: {len(train)}/{len(val)}/{len(test)}')


if __name__ == '__main__':
    main()
