"""
E3 — Table 1 on shared 13-class subset.

Addresses: reviewer fPLF concern that Table 1 is biased because each strategy
evaluates on a different number of classes (16/15/18), making macro-F1
averages incomparable.

Approach: filter existing prediction CSVs to only classes present in ALL
three strategies (13 shared classes), recompute macro-F1 and accuracy.
No retraining. Pure metric recomputation on existing preds.

Shared classes (13): verified by config intersection in feasible_experiments.md
"""

import datetime
import json
from pathlib import Path
import subprocess
import sys as _sys

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

RESULTS = Path('results')
CONFIGS = Path('configs')
OUT_DIR = Path('experiments/E3_shared_subset')
LOG_DIR = Path('experiments/logs/E3')
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# 1. Load class mappings and find shared class NAMES
# ----------------------------------------------------------


def load_mapping(strategy: str) -> dict:
    path = CONFIGS / f'class_mapping_{strategy}.json'
    with open(path) as f:
        return json.load(f)


fuzzy_map = load_mapping('Fuzzy')
sbert_map = load_mapping('sbert')
hybrid_map = load_mapping('hybrid')

shared_pv_names = set(fuzzy_map.keys()) & set(sbert_map.keys()) & set(hybrid_map.keys())
print(f'Shared PV class names across all 3 strategies: {len(shared_pv_names)}')
for n in sorted(shared_pv_names):
    print(f'  {n}')


# ----------------------------------------------------------
# 2. For each strategy, find label_ids of shared classes
# ----------------------------------------------------------


def load_label_map(strategy: str) -> pd.DataFrame:
    """Returns DataFrame with columns: pv_class_name, label_id"""
    path = Path('configs') / f'label_map_{strategy}.csv'
    df = pd.read_csv(path)
    return df


def get_shared_label_ids(strategy: str, shared_names: set) -> dict:
    """Returns {pv_class_name: label_id} for shared classes in this strategy."""
    df = load_label_map(strategy)
    # label_map columns: try to auto-detect
    print(f'\n  label_map_{strategy}.csv columns: {list(df.columns)}')
    print(df.head(3).to_string())
    # Find column with class names and label_id
    result = {}
    for _, row in df.iterrows():
        for col in df.columns:
            val = str(row[col])
            if val in shared_names:
                # find label_id column
                id_col = [
                    c for c in df.columns if 'label' in c.lower() or 'id' in c.lower()
                ]
                if id_col:
                    result[val] = int(row[id_col[0]])
                break
    return result


# ----------------------------------------------------------
# 3. Filter predictions and compute metrics
# ----------------------------------------------------------


def compute_metrics_shared(
    strategy: str,
    shared_pv_names: set,
    label: str,
    pred_csv: Path,
    true_col: str = 'true',
    pred_col: str = 'pred',
) -> dict | None:
    if not pred_csv.exists():
        print(f'  [MISSING] {pred_csv}')
        return None

    df = pd.read_csv(pred_csv)
    print(f'\n  {label}: {pred_csv.name} — {len(df)} rows, cols={list(df.columns)}')

    # Auto-detect columns
    cols = list(df.columns)
    if true_col not in cols or pred_col not in cols:
        # try alternatives
        for tc in ['true', 'label', 'y_true', 'label_id']:
            if tc in cols:
                true_col = tc
                break
        for pc in ['pred', 'predicted', 'y_pred']:
            if pc in cols:
                pred_col = pc
                break

    if true_col not in cols or pred_col not in cols:
        print(f'  [ERROR] Could not find true/pred columns. Got: {cols}')
        return None

    # Load label map for this strategy to get shared label_ids
    lmap = load_label_map(strategy)
    lmap_cols = list(lmap.columns)
    print(f'  label_map cols: {lmap_cols}')

    # Find the column containing PV class names in label_map
    name_col = None
    for c in lmap_cols:
        sample_vals = lmap[c].astype(str).tolist()[:5]
        if any(v in shared_pv_names for v in sample_vals):
            name_col = c
            break

    if name_col is None:
        # Try checking if any column's values overlap with shared_pv_names
        for c in lmap_cols:
            overlap = set(lmap[c].astype(str)) & shared_pv_names
            if overlap:
                name_col = c
                print(f'  Found name col: {c} (overlap={len(overlap)})')
                break

    if name_col is None:
        print(
            f'  [ERROR] Cannot find class-name column in label_map. Cols: {lmap_cols}'
        )
        return None

    # label_id column
    id_col = None
    for c in lmap_cols:
        if c != name_col and lmap[c].dtype in [int, float] or 'id' in c.lower():
            try:
                _ = lmap[c].astype(int)
                id_col = c
                break
            except (ValueError, TypeError):
                pass

    if id_col is None:
        # fallback: non-name numeric column
        for c in lmap_cols:
            if c != name_col:
                try:
                    _ = lmap[c].astype(int)
                    id_col = c
                    break
                except Exception:
                    pass

    if id_col is None:
        print('  [ERROR] Cannot find label_id column in label_map')
        return None

    shared_ids = set(
        lmap[lmap[name_col].astype(str).isin(shared_pv_names)][id_col].astype(int)
    )
    print(f'  Shared label_ids ({len(shared_ids)}): {sorted(shared_ids)}')

    # Filter predictions to shared classes only
    mask = df[true_col].isin(shared_ids)
    df_shared = df[mask].copy()
    print(f'  Filtered: {len(df)} → {len(df_shared)} samples in shared classes')

    if len(df_shared) == 0:
        print('  [ERROR] No samples after filtering')
        return None

    y_true = df_shared[true_col].values
    y_pred = df_shared[pred_col].values

    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    n_classes = len(shared_ids)
    n_samples = len(df_shared)

    return {
        'strategy': strategy,
        'label': label,
        'n_shared_classes': n_classes,
        'n_samples': n_samples,
        'macro_f1': round(f1, 4),
        'accuracy': round(acc, 4),
    }


# ----------------------------------------------------------
# 4. Run for all strategies × {baseline, best CORAL}
# ----------------------------------------------------------

rows = []


REEVAL = '--reeval' in _sys.argv  # re-evaluated source-only predictions (scripts/14.7)
_SFX = '_reeval' if REEVAL else ''
configs = [
    # (strategy, label, pred_csv_path)
    (
        'Fuzzy',
        'Baseline (zero-shot)',
        RESULTS / f'preds_target_resnet50_Fuzzy{_SFX}.csv',
    ),
    ('Fuzzy', 'CORAL λ=1.0', RESULTS / 'preds_target_coral_lambda_1.0_Fuzzy.csv'),
    (
        'hybrid',
        'Baseline (zero-shot)',
        RESULTS / f'preds_target_resnet50_hybrid{_SFX}.csv',
    ),
    ('hybrid', 'CORAL λ=0.1', RESULTS / 'preds_target_coral_lambda_0.1_hybrid.csv'),
    (
        'sbert',
        'Baseline (zero-shot)',
        RESULTS / f'preds_target_resnet50_sbert{_SFX}.csv',
    ),
    ('sbert', 'CORAL λ=0.01', RESULTS / 'preds_target_coral_lambda_0.01_sbert.csv'),
]

for strategy, label, pred_path in configs:
    r = compute_metrics_shared(strategy, shared_pv_names, label, pred_path)
    if r:
        rows.append(r)

df_out = pd.DataFrame(rows)
print('\n\n=== E3 RESULTS: Table 1 on Shared 13-Class Subset ===')
print(df_out.to_string(index=False))

# Save
out_csv = OUT_DIR / f'table1_shared13{_SFX}.csv'
df_out.to_csv(out_csv, index=False)
print(f'\n[SAVED] {out_csv}')

# Also save manifest
manifest = {
    'experiment': 'E3_shared_subset',
    'description': 'Recompute Table 1 metrics on 13 classes shared by all 3 strategies',
    'shared_classes': sorted(shared_pv_names),
    'n_shared_classes': len(shared_pv_names),
    'git_hash': subprocess.getoutput('git rev-parse --short HEAD'),
    'timestamp': datetime.datetime.now().isoformat(),
    'no_retraining': True,
}
manifest['reeval_source_only'] = REEVAL
(OUT_DIR / f'manifest{_SFX}.json').write_text(json.dumps(manifest, indent=2))

# Write log
log_lines = [
    f'E3 run at {manifest["timestamp"]}',
    f'Git: {manifest["git_hash"]}',
    f'Shared classes: {manifest["n_shared_classes"]}',
    df_out.to_string(),
]
(LOG_DIR / f'E3_run{_SFX}.log').write_text('\n'.join(log_lines))
print('[SAVED] manifest.json, E3_run.log')
