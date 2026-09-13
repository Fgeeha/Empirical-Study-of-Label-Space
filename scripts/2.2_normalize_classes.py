import csv
from pathlib import Path
import sys

from utils.text_normalization import normalize_class_name


def main(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open(encoding='utf-8') as f:
        classes = [line.strip() for line in f if line.strip()]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['original', 'normalized'])

        for cls in classes:
            writer.writerow([cls, normalize_class_name(cls)])

    print(f'[OK] Normalized {len(classes)} → {output_path}')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python normalize_classes.py <input.txt> <output.csv>')
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])
