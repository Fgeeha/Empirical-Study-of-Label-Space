import csv
import json
from pathlib import Path
import re

from rapidfuzz import fuzz

# ============================================================
# Paths
# ============================================================
ARTIFACTS = Path('artifacts')
CONFIGS = Path('configs')

PV_CSV = ARTIFACTS / 'classes_pv_normalized.csv'
TARGET_TXT = ARTIFACTS / 'classes_target.txt'

OUTPUT_MAPPING = CONFIGS / 'class_mapping_Fuzzy.json'
OUTPUT_REPORT = ARTIFACTS / 'mapping_report_Fuzzy.md'

FUZZY_THRESHOLD = 80

# Crop aliases between datasets
CROP_ALIASES = {
    'bell': 'pepper,_bell',
    'bell_pepper': 'pepper,_bell',
    'pepper': 'pepper,_bell',
    'corn': 'corn_(maize)',
    'maize': 'corn_(maize)',
    'soyabean': 'soybean',
}


# ============================================================
# IO helpers
# ============================================================
def load_csv(path: Path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_txt(path: Path):
    with path.open(encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


# ============================================================
# Parsing
# ============================================================
def parse_pv(name: str):
    # PV format: Crop___Disease
    if '___' not in name:
        return None, None
    crop, disease = name.split('___', 1)
    return crop.lower(), disease.lower()


def parse_target(name: str):
    """
    Examples:
    - Tomato_leaf_late_blight
    - Apple_Scab_Leaf
    - Tomato_leaf
    """
    tokens = name.lower().split('_')
    if not tokens:
        return None, None

    crop_raw = tokens[0]
    crop = CROP_ALIASES.get(crop_raw, crop_raw)

    rest = tokens[1:]
    rest = [t for t in rest if t != 'leaf']

    if not rest:
        return crop, None

    disease = ' '.join(rest)
    return crop, disease


# ============================================================
# Canonical disease tokenization  🔥 КЛЮЧЕВОЕ МЕСТО
# ============================================================
def disease_tokens(name: str):
    """
    Canonical tokenization for disease names.
    Works identically for PV and Target.
    """
    if not name:
        return []

    name = name.lower()

    # unify separators
    name = re.sub(r'[_,()\-]+', ' ', name)

    STOPWORDS = {
        'leaf',
        'disease',
        'including',
        'sour',
    }

    tokens = [t for t in name.split() if t not in STOPWORDS]

    return tokens


def is_valid_disease(disease: str | None):
    if not disease:
        return False

    invalid_keywords = [
        'healthy',
        'spider',
        'mites',
        'pest',
    ]

    return not any(k in disease for k in invalid_keywords)


def build_pv_index(pv_rows):
    index = []

    for row in pv_rows:
        crop, disease_raw = parse_pv(row['original'])
        if not crop or not disease_raw:
            continue

        tokens = disease_tokens(disease_raw)
        if not tokens:
            continue

        index.append(
            {
                'original': row['original'],
                'crop': crop,
                'tokens': tokens,
            }
        )

    return index


def group_by_crop(pv_index):
    pv_by_crop = {}

    for record in pv_index:
        pv_by_crop.setdefault(record['crop'], []).append(record)

    return pv_by_crop


def find_best_candidate(target_tokens, candidates):
    best_score = 0.0
    best_candidate = None

    for candidate in candidates:
        score = fuzz.token_set_ratio(
            ' '.join(target_tokens),
            ' '.join(candidate['tokens']),
        )

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate, best_score


def build_mapping(target_classes, pv_by_crop):
    mapping = {}
    manual = []
    excluded = []

    for target in target_classes:
        crop, disease_raw = parse_target(target)

        if not is_valid_disease(disease_raw):
            excluded.append((target, 'no or invalid disease'))
            continue

        target_tokens = disease_tokens(disease_raw)
        if not target_tokens:
            excluded.append((target, 'no or invalid disease'))
            continue

        candidates = pv_by_crop.get(crop)
        if not candidates:
            excluded.append((target, 'crop not present in PlantVillage'))
            continue

        best_candidate, best_score = find_best_candidate(
            target_tokens,
            candidates,
        )

        if best_score >= FUZZY_THRESHOLD and best_candidate:
            mapping[target] = best_candidate['original']
        else:
            manual.append((target, f'low fuzzy score ({best_score:.1f})'))

    return mapping, manual, excluded


# ============================================================
# Main
# ============================================================
def main():
    pv_rows = load_csv(PV_CSV)
    target_classes = load_txt(TARGET_TXT)

    pv_index = build_pv_index(pv_rows)
    pv_by_crop = group_by_crop(pv_index)

    mapping, manual, excluded = build_mapping(
        target_classes,
        pv_by_crop,
    )

    write_outputs(mapping, manual, excluded)
    print(f'[OK] Shared subset built: {len(mapping)} classes')


# ============================================================
# Outputs
# ============================================================
def write_outputs(mapping, manual, excluded):
    CONFIGS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)

    OUTPUT_MAPPING.write_text(
        json.dumps(mapping, indent=2, ensure_ascii=False),
        encoding='utf-8',
    )

    lines = [
        '# Class Mapping Report',
        '',
        '## Mapping strategy',
        '- Target → PlantVillage mapping',
        '- Canonical disease tokenization',
        '- Token-set similarity (threshold = 80%)',
        '- Manual verification for ambiguous cases',
        '',
        '## Coverage summary',
        f'- Mapped classes: {len(mapping)}',
        f'- Manual review required: {len(manual)}',
        f'- Excluded classes: {len(excluded)}',
        '',
        '## Excluded classes',
        '',
        '| Class | Reason |',
        '|------|--------|',
    ]

    for cls, reason in excluded:
        lines.append(f'| {cls} | {reason} |')

    lines += [
        '',
        '## Manual review required',
        '',
        '| Class | Reason |',
        '|------|--------|',
    ]

    for cls, reason in manual:
        lines.append(f'| {cls} | {reason} |')

    OUTPUT_REPORT.write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
