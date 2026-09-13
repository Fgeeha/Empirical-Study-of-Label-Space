"""Collect UDA benchmark results and compute statistics.

Reads all raw/<method>_hybrid_plantdoc_seed*.json files,
computes 3-seed mean±std, paired bootstrap vs source-only and DANN,
and writes uda_summary.csv and uda_summary.json.

Usage:
  python scripts/collect_uda_results.py
"""

import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import f1_score

RESULTS = Path('results') / 'uda_benchmark'
RAW = RESULTS / 'raw'
PREDS = RESULTS / 'predictions'
AGG = RESULTS / 'aggregated'
BOOT = RESULTS / 'bootstrap'

AGG.mkdir(exist_ok=True)
BOOT.mkdir(exist_ok=True)

METHODS = ['source_only', 'adabn', 'coral', 'dann', 'mcc', 'cdan']
SEEDS = [42, 43, 44]
TARGET = 'plantdoc'
STRATEGY = 'hybrid'

# Source-only baseline. Default: the archived Hybrid checkpoint re-evaluated with
# the canonical transform (scripts/14.2 -> 14.7). `--recorded` reproduces the
# original-run figures (0.156 / 0.191, results/preds_target_resnet50_hybrid.csv).
RECORDED = '--recorded' in sys.argv
if RECORDED:
    SOURCE_ONLY = {'macro_f1': 0.156, 'accuracy': 0.191}
    SRC_PREDS = Path('results') / 'preds_target_resnet50_hybrid.csv'
else:
    _src = json.loads(
        (
            Path('results')
            / 'int8_accuracy/pytorch_resnet50_hybrid__plantdoc_test.json'
        ).read_text()
    )
    SOURCE_ONLY = {'macro_f1': _src['macro_f1'], 'accuracy': _src['accuracy']}
    SRC_PREDS = Path('results') / 'preds_target_resnet50_hybrid_reeval.csv'


def load_raw(method: str, seed: int) -> dict | None:
    path = RAW / f'{method}_{STRATEGY}_{TARGET}_seed{seed}.json'
    if not path.exists():
        return None
    return json.loads(path.read_text())


def bootstrap_f1_diff(
    preds_a: list,
    labels_a: list,
    preds_b: list,
    labels_b: list,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for macro-F1 difference (B - A).

    Returns: (mean_diff, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(labels_a)
    assert len(labels_b) == n

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        ya = [labels_a[i] for i in idx]
        pa = [preds_a[i] for i in idx]
        yb = [labels_b[i] for i in idx]
        pb = [preds_b[i] for i in idx]
        f1_a = f1_score(ya, pa, average='macro', zero_division=0)
        f1_b = f1_score(yb, pb, average='macro', zero_division=0)
        diffs.append(f1_b - f1_a)

    diffs = np.array(diffs)
    return (
        float(diffs.mean()),
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def load_preds(method: str, seed: int) -> tuple[list, list] | None:
    """Load (labels, preds) from predictions CSV."""
    import pandas as pd

    path = PREDS / f'{method}_{STRATEGY}_{TARGET}_seed{seed}.csv'
    if not path.exists():
        return None
    df = pd.read_csv(str(path))
    return df['y_true'].tolist(), df['y_pred'].tolist()


def main():
    # Collect per-method, per-seed results
    results = {}
    for method in METHODS:
        if method == 'source_only':
            continue
        seeds_data = {}
        for seed in SEEDS:
            raw = load_raw(method, seed)
            if raw:
                seeds_data[seed] = raw['target_metrics']
        results[method] = seeds_data

    # Compute mean±std per method
    summary = {}
    print(f'{"Method":15} {"F1 mean":>8} {"F1 std":>7} {"Acc mean":>9} {"n_seeds":>7}')
    print('-' * 50)

    for method, seeds_data in results.items():
        if not seeds_data:
            print(f'{method:15} MISSING')
            continue
        f1_vals = [d['macro_f1'] for d in seeds_data.values()]
        acc_vals = [d['accuracy'] for d in seeds_data.values()]
        summary[method] = {
            'macro_f1_mean': float(np.mean(f1_vals)),
            'macro_f1_std': float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
            'accuracy_mean': float(np.mean(acc_vals)),
            'accuracy_std': float(np.std(acc_vals, ddof=1))
            if len(acc_vals) > 1
            else 0.0,
            'n_seeds': len(f1_vals),
            'per_seed': {str(k): v for k, v in seeds_data.items()},
        }
        print(
            f'{method:15} {np.mean(f1_vals):>8.4f} {np.std(f1_vals, ddof=1) if len(f1_vals) > 1 else 0:>7.4f}'
            f' {np.mean(acc_vals):>9.4f} {len(f1_vals):>7}'
        )

    # Bootstrap comparison vs source-only
    print('\n=== Bootstrap CI for Δ macro-F1 vs source-only ===')

    bootstrap_results = {}
    for method, data in summary.items():
        if data['n_seeds'] < 2:
            continue

        # Use seed=42 predictions for bootstrap comparison
        method_data = load_preds(method, 42)
        if method_data is None:
            continue
        labels, method_preds = method_data

        # Bootstrap vs source-only: we need source-only preds
        # Use the source target resnet50 predictions from original eval
        src_pred_path = SRC_PREDS
        if src_pred_path.exists():
            import pandas as pd

            src_df = pd.read_csv(str(src_pred_path))
            src_labels = src_df['y_true'].tolist()
            src_preds = src_df['y_pred'].tolist()

            # Align sample count (both should be same test set)
            n = min(len(src_labels), len(labels))
            diff_mean, ci_lo, ci_hi = bootstrap_f1_diff(
                src_preds[:n],
                src_labels[:n],
                method_preds[:n],
                labels[:n],
            )
            bootstrap_results[method] = {
                'vs_source_only': {
                    'mean_diff': diff_mean,
                    'ci_lower': ci_lo,
                    'ci_upper': ci_hi,
                    'significant': ci_lo > 0,
                }
            }
            print(
                f'  {method:10}: Δ={diff_mean:+.4f} CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]'
                f'  {"SIGNIFICANT" if ci_lo > 0 else "not sig."}'
            )

    # Save bootstrap results
    (BOOT / 'bootstrap_vs_source.json').write_text(
        json.dumps(bootstrap_results, indent=2)
    )

    # Write CSV summary
    fieldnames = [
        'method',
        'macro_f1_mean',
        'macro_f1_std',
        'accuracy_mean',
        'accuracy_std',
        'n_seeds',
        'delta_vs_source_only',
        'bootstrap_sig',
    ]
    rows = []
    src_f1 = SOURCE_ONLY['macro_f1']
    for method, data in summary.items():
        delta = data['macro_f1_mean'] - src_f1
        boot = bootstrap_results.get(method, {}).get('vs_source_only', {})
        rows.append(
            {
                'method': method,
                'macro_f1_mean': round(data['macro_f1_mean'], 4),
                'macro_f1_std': round(data['macro_f1_std'], 4),
                'accuracy_mean': round(data['accuracy_mean'], 4),
                'accuracy_std': round(data['accuracy_std'], 4),
                'n_seeds': data['n_seeds'],
                'delta_vs_source_only': round(delta, 4),
                'bootstrap_sig': boot.get('significant', None),
            }
        )

    rows.sort(key=lambda r: -r['macro_f1_mean'])

    out_csv = AGG / 'uda_summary.csv'
    with open(str(out_csv), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nSaved {out_csv}')

    out_json = AGG / 'uda_summary.json'
    out_json.write_text(
        json.dumps({'source_only': SOURCE_ONLY, 'methods': summary}, indent=2)
    )
    print(f'Saved {out_json}')


if __name__ == '__main__':
    main()
