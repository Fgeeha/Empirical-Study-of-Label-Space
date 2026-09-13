import csv
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Paths
# ============================================================
ARTIFACTS = Path('artifacts')
CONFIGS = Path('configs')

PV_CSV = ARTIFACTS / 'classes_pv_normalized.csv'
TARGET_TXT = ARTIFACTS / 'classes_target.txt'

OUTPUT_MAPPING = CONFIGS / 'class_mapping_sbert.json'
OUTPUT_REPORT = ARTIFACTS / 'mapping_report_sbert.md'

SIM_THRESHOLD = 0.75  # cosine similarity


# ============================================================
# Helpers
# ============================================================
def load_csv(path: Path):
    with path.open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_txt(path: Path):
    with path.open(encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def parse_pv(name: str):
    if '___' not in name:
        return None, None
    crop, disease = name.split('___', 1)
    return crop.lower(), disease.lower()


def parse_target(name: str):
    tokens = name.lower().split('_')
    if not tokens:
        return None, None

    crop = tokens[0]
    rest = [t for t in tokens[1:] if t != 'leaf']

    if not rest:
        return crop, None

    disease = ' '.join(rest)
    return crop, disease


def is_valid_disease(disease):
    if not disease:
        return False

    invalid = ['healthy', 'spider', 'mites', 'pest']
    return not any(x in disease for x in invalid)


# ============================================================
# Main
# ============================================================
def main():
    print('[INFO] Loading SBERT model...')
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        device='cpu',
    )

    pv_rows = load_csv(PV_CSV)
    target_classes = load_txt(TARGET_TXT)

    # ---------- Build PV texts ----------
    pv_items = []
    for r in pv_rows:
        crop, disease = parse_pv(r['original'])
        if not crop or not disease:
            continue

        text = f'{crop} {disease}'
        pv_items.append(
            {
                'original': r['original'],
                'crop': crop,
                'text': text,
            }
        )

    pv_texts = [x['text'] for x in pv_items]
    pv_emb = model.encode(pv_texts, normalize_embeddings=True)

    mapping = {}
    manual = []
    excluded = []

    # ---------- Target → PV ----------
    for t in target_classes:
        crop, disease = parse_target(t)

        if not is_valid_disease(disease):
            excluded.append((t, 'no or invalid disease'))
            continue

        target_text = f'{crop} {disease}'
        t_emb = model.encode([target_text], normalize_embeddings=True)

        sims = cosine_similarity(t_emb, pv_emb)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= SIM_THRESHOLD:
            mapping[t] = pv_items[best_idx]['original']
        else:
            manual.append((t, f'low cosine similarity ({best_score:.3f})'))

    write_outputs(mapping, manual, excluded)
    print(f'[OK] SBERT shared subset built: {len(mapping)} classes')


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
        '# Class Mapping Report (SBERT)',
        '',
        '## Mapping strategy',
        '- Target → PlantVillage mapping',
        '- Sentence embeddings (all-MiniLM-L6-v2)',
        '- Cosine similarity threshold = 0.75',
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
