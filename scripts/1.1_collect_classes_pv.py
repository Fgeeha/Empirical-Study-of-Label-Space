from pathlib import Path

PV_ROOT = Path('data/PlantVillage_raw/segmented')
OUTPUT = Path('artifacts/classes_pv.txt')


def main():
    classes = sorted(p.name for p in PV_ROOT.iterdir() if p.is_dir())

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open('w', encoding='utf-8') as f:
        for cls in classes:
            f.write(f'{cls}\n')

    print(f'[OK] Saved {len(classes)} classes to {OUTPUT}')


if __name__ == '__main__':
    main()
