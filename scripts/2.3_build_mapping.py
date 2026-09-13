import csv
import json
from pathlib import Path
import sys

from rapidfuzz import fuzz, process

ARTIFACTS = Path('artifacts')

PV = ARTIFACTS / 'classes_pv_normalized.csv'
PD = ARTIFACTS / 'classes_target_normalized.csv'
FCDD = ARTIFACTS / 'classes_fcdd_normalized.csv'

OUTPUT_JSON = ARTIFACTS / 'class_mapping.json'
OUTPUT_MD = ARTIFACTS / 'mapping_report.md'

THRESHOLD = 75


def ensure_exists(path: Path):
    if not path.exists():
        print(f'[ERROR] Required file not found: {path}')
        print('Did you forget to run normalization?')
        sys.exit(1)


def load_csv(path: Path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def best_match(query, choices):
    match, score, _ = process.extractOne(
        query,
        choices,
        scorer=fuzz.token_sort_ratio,
    )
    return match, score


def main():
    for p in (PV, PD, FCDD):
        ensure_exists(p)

    pv = load_csv(PV)
    pd = load_csv(PD)
    fcdd = load_csv(FCDD)

    pd_norm = [x['normalized'] for x in pd]
    fcdd_norm = [x['normalized'] for x in fcdd]

    rows = []

    for cls in pv:
        pv_norm = cls['normalized']

        pd_match, pd_score = best_match(pv_norm, pd_norm)
        fcdd_match, fcdd_score = best_match(pv_norm, fcdd_norm)

        rows.append(
            {
                'pv_original': cls['original'],
                'pv_normalized': pv_norm,
                'plantdoc_match': pd_match if pd_score >= THRESHOLD else None,
                'plantdoc_score': pd_score,
                'fcdd_match': fcdd_match if fcdd_score >= THRESHOLD else None,
                'fcdd_score': fcdd_score,
            }
        )

    write_json(rows)
    write_report(rows)

    print(f'[OK] Mapping completed: {len(rows)} PV classes')


def write_json(rows):
    OUTPUT_JSON.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )


def write_report(rows):
    lines = [
        '# Class Mapping Report',
        '',
        'Automatic fuzzy matching between PlantVillage (PV), PlantDoc and FCDD.',
        '',
        '| PV class | PlantDoc match | score | FCDD match | score |',
        '|---------|----------------|-------|------------|-------|',
    ]

    for r in rows:
        lines.append(
            f'| {r["pv_original"]} | '
            f'{r["plantdoc_match"] or "—"} | {r["plantdoc_score"]} | '
            f'{r["fcdd_match"] or "—"} | {r["fcdd_score"]} |'
        )

    OUTPUT_MD.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
