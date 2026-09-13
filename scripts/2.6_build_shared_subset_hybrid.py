import csv
import json
from pathlib import Path
import re

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# Paths
# ============================================================
ARTIFACTS = Path('artifacts')
CONFIGS = Path('configs')

PV_CSV = ARTIFACTS / 'classes_pv_normalized.csv'
TARGET_TXT = ARTIFACTS / 'classes_target.txt'

OUTPUT_MAPPING = CONFIGS / 'class_mapping_hybrid.json'
OUTPUT_REPORT = ARTIFACTS / 'mapping_report_hybrid.md'

FUZZY_THRESHOLD = 80
SBERT_THRESHOLD = 0.75

# ============================================================
# Crop aliases
# ============================================================
CROP_ALIASES = {
    'bell': 'pepper,_bell',
    'bell_pepper': 'pepper,_bell',
    'pepper': 'pepper,_bell',
    'corn': 'corn_(maize)',
    'maize': 'corn_(maize)',
    'soyabean': 'soybean',
}


# ============================================================
# IO
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
    if '___' not in name:
        return None, None
    crop, disease = name.split('___', 1)
    return crop.lower(), disease.lower()


def parse_target(name: str):
    tokens = name.lower().split('_')
    if not tokens:
        return None, None

    crop_raw = tokens[0]
    crop = CROP_ALIASES.get(crop_raw, crop_raw)

    rest = [t for t in tokens[1:] if t != 'leaf']
    if not rest:
        return crop, None

    return crop, ' '.join(rest)


# ============================================================
# Canonical tokenization
# ============================================================
def disease_tokens(name: str):
    if not name:
        return []

    name = name.lower()
    name = re.sub(r'[_,()\-]+', ' ', name)

    STOPWORDS = {'leaf', 'disease', 'including', 'sour'}
    return [t for t in name.split() if t not in STOPWORDS]


def is_valid_disease(disease: str | None):
    if not disease:
        return False
    invalid = {'healthy', 'spider', 'mites', 'pest'}
    return not any(k in disease for k in invalid)


# ============================================================
# Main
# ============================================================
def load_sbert() -> SentenceTransformer:
    print('[INFO] Loading SBERT (CPU)...')
    return SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2',
        device='cpu',
    )


def build_pv_index(
    pv_rows: list[dict],
    model: SentenceTransformer,
) -> tuple[list[dict], dict[str, list[dict]], np.ndarray]:
    pv_items: list[dict] = []

    for r in pv_rows:
        crop, disease = parse_pv(r['original'])
        if not crop or not disease:
            continue

        tokens = disease_tokens(disease)
        if not tokens:
            continue

        pv_items.append(
            {
                'original': r['original'],
                'crop': crop,
                'tokens': tokens,
                'text': f'{crop} {" ".join(tokens)}',
            }
        )

    pv_by_crop: dict[str, list[dict]] = {}
    for r in pv_items:
        pv_by_crop.setdefault(r['crop'], []).append(r)

    pv_texts = [r['text'] for r in pv_items]
    pv_emb = model.encode(pv_texts, normalize_embeddings=True)

    return pv_items, pv_by_crop, pv_emb


def match_target_to_pv(
    target: str,
    pv_items: list[dict],
    pv_by_crop: dict[str, list[dict]],
    pv_emb: np.ndarray,
    model: SentenceTransformer,
) -> tuple[str | None, str | None]:
    crop, disease_raw = parse_target(target)

    if not is_valid_disease(disease_raw):
        return None, 'no or invalid disease'

    target_tokens = disease_tokens(disease_raw)
    if not target_tokens:
        return None, 'no or invalid disease'

    candidates = pv_by_crop.get(crop)
    if not candidates:
        return None, 'crop not present in PlantVillage'

    # ---------- 1) FUZZY ----------
    best_fuzzy = 0
    best_idx = None

    for i, c in enumerate(candidates):
        score = fuzz.token_set_ratio(
            ' '.join(target_tokens),
            ' '.join(c['tokens']),
        )
        if score > best_fuzzy:
            best_fuzzy = score
            best_idx = i

    if best_fuzzy >= FUZZY_THRESHOLD:
        return candidates[best_idx]['original'], None

    # ---------- 2) SBERT ----------
    target_text = f'{crop} {" ".join(target_tokens)}'
    t_emb = model.encode([target_text], normalize_embeddings=True)

    cand_indices = [pv_items.index(c) for c in candidates]
    sims = cosine_similarity(t_emb, pv_emb[cand_indices])[0]

    sbert_idx = int(np.argmax(sims))
    sbert_score = float(sims[sbert_idx])

    if sbert_score >= SBERT_THRESHOLD:
        return candidates[sbert_idx]['original'], None

    return None, f'fuzzy={best_fuzzy:.1f}, sbert={sbert_score:.3f}'


def main() -> None:
    model = load_sbert()

    pv_rows = load_csv(PV_CSV)
    target_classes = load_txt(TARGET_TXT)

    pv_items, pv_by_crop, pv_emb = build_pv_index(pv_rows, model)

    mapping: dict[str, str] = {}
    manual: list[tuple[str, str]] = []
    excluded: list[tuple[str, str]] = []

    for t in target_classes:
        result, reason = match_target_to_pv(
            t,
            pv_items,
            pv_by_crop,
            pv_emb,
            model,
        )

        if result:
            mapping[t] = result
        elif reason and 'fuzzy=' in reason:
            manual.append((t, reason))
        else:
            excluded.append((t, reason))

    write_outputs(mapping, manual, excluded)
    print(f'[OK] Hybrid shared subset built: {len(mapping)} classes')


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
        '# Class Mapping Report (Hybrid)',
        '',
        '## Mapping strategy',
        '- Target → PlantVillage mapping',
        '- Primary: token-set fuzzy matching',
        '- Fallback: SBERT semantic similarity',
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
