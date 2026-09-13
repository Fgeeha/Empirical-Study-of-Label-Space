import argparse
from collections import defaultdict
import csv
from pathlib import Path
import random

SEED = 42
SPLITS = {'train': 0.7, 'val': 0.15, 'test': 0.15}

DATA = Path('data')
CONFIGS = Path('configs')
SPLITS_ROOT = Path('splits')


def load_label_map(path: Path):
    mapping = {}
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['canonical_name']] = int(row['label_id'])
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    strategy = args.strategy
    random.seed(SEED)

    PV_ROOT = DATA / 'shared_subset' / strategy / 'pv'
    LABEL_MAP = CONFIGS / f'label_map_{strategy}.csv'
    OUT_DIR = SPLITS_ROOT / strategy
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(LABEL_MAP)
    samples = defaultdict(list)

    for cls_name, label_id in label_map.items():
        cls_dir = PV_ROOT / cls_name
        for img in cls_dir.iterdir():
            if img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                samples[label_id].append(img)

    writers = {}
    files = {}
    for split in SPLITS:
        f = (OUT_DIR / f'pv_{split}.csv').open('w', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['path', 'label_id'])
        writers[split] = writer
        files[split] = f

    for label_id, paths in samples.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = int(n * SPLITS['train'])
        n_val = int(n * SPLITS['val'])

        split_data = {
            'train': paths[:n_train],
            'val': paths[n_train : n_train + n_val],
            'test': paths[n_train + n_val :],
        }

        for split, imgs in split_data.items():
            for img in imgs:
                writers[split].writerow([img.as_posix(), label_id])

    for f in files.values():
        f.close()

    print(f"[OK] PV split created for strategy='{strategy}'")


if __name__ == '__main__':
    main()
