from pathlib import Path

ROOT = Path('data/FCDD_raw')
OUTPUT = Path('artifacts/classes_fcdd.txt')


def main():
    classes = []

    for crop_dir in ROOT.iterdir():
        if not crop_dir.is_dir():
            continue

        crop = crop_dir.name.lower()

        for disease_dir in crop_dir.iterdir():
            if disease_dir.is_dir():
                classes.append(f'{crop} {disease_dir.name}')

    classes = sorted(classes)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open('w', encoding='utf-8') as f:
        for cls in classes:
            f.write(cls + '\n')

    print(f'[OK] FCDD: {len(classes)} classes saved')


if __name__ == '__main__':
    main()
