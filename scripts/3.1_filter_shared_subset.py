import argparse
import csv
import json
from pathlib import Path
import shutil

# =====================================================
# Base paths
# =====================================================
DATA = Path('data')
CONFIGS = Path('configs')
ARTIFACTS = Path('artifacts')

PV_ROOT = DATA / 'PlantVillage_raw/segmented'
TARGET_ROOT = DATA / 'PlantDoc_raw/train'

REPORT = ARTIFACTS / 'mapping_report.md'


# =====================================================
# Helpers
# =====================================================
def load_mapping(path: Path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mapping',
        required=True,
        help='Path to class_mapping_*.json',
    )
    args = parser.parse_args()

    mapping_path = Path(args.mapping)
    strategy = mapping_path.stem.replace('class_mapping_', '')

    mapping = load_mapping(mapping_path)
    canonical_classes = sorted(mapping.keys())

    # strategy-aware output paths
    OUT_ROOT = DATA / 'shared_subset' / strategy
    OUT_PV = OUT_ROOT / 'pv'
    OUT_TARGET = OUT_ROOT / 'target'

    LABEL_MAP = CONFIGS / f'label_map_{strategy}.csv'

    ensure_clean_dir(OUT_PV)
    ensure_clean_dir(OUT_TARGET)

    # -------------------------------------------------
    # Build label_map.csv
    # -------------------------------------------------
    with LABEL_MAP.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['label_id', 'canonical_name', 'pv_name', 'target_name'])

        for idx, canonical in enumerate(canonical_classes):
            writer.writerow(
                [
                    idx,
                    canonical,
                    mapping[canonical],
                    canonical,
                ]
            )

    print(f'[OK] label_map_{strategy}.csv created ({len(canonical_classes)} classes)')

    # -------------------------------------------------
    # Filter PlantVillage
    # -------------------------------------------------
    for canonical, pv_class in mapping.items():
        src = PV_ROOT / pv_class
        dst = OUT_PV / canonical

        if not src.exists():
            print(f'[WARN] PV class not found: {src}')
            continue

        shutil.copytree(src, dst)

    print('[OK] PlantVillage filtered')

    # -------------------------------------------------
    # Filter Target (PlantDoc)
    # -------------------------------------------------
    for canonical in canonical_classes:
        src = TARGET_ROOT / canonical
        dst = OUT_TARGET / canonical

        if not src.exists():
            print(f'[WARN] Target class not found: {src}')
            continue

        shutil.copytree(src, dst)

    print('[OK] Target dataset filtered')

    # -------------------------------------------------
    # Append report
    # -------------------------------------------------
    with REPORT.open('a', encoding='utf-8') as f:
        f.write(f'\n\n## Shared subset filtering ({strategy})\n\n')
        f.write(f'- Alignment strategy: `{strategy}`\n')
        f.write(f'- Final shared classes (K): {len(canonical_classes)}\n')
        f.write(f'- Output directory: `{OUT_ROOT}`\n')
        f.write(f'- Label map: `{LABEL_MAP}`\n')

    print('[OK] mapping_report.md updated')


if __name__ == '__main__':
    main()
