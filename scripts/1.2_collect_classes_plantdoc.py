from pathlib import Path

ROOT = Path('data/PlantDoc_raw')
OUTPUT = Path('artifacts/classes_target.txt')


def main():
    classes = set()

    for split in ['train', 'test']:
        split_dir = ROOT / split
        if not split_dir.exists():
            continue

        for p in split_dir.iterdir():
            if p.is_dir():
                classes.add(p.name)

    classes = sorted(classes)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open('w', encoding='utf-8') as f:
        for cls in classes:
            f.write(cls + '\n')

    print(f'[OK] PlantDoc: {len(classes)} classes saved')


if __name__ == '__main__':
    main()
