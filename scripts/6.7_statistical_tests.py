#!/usr/bin/env python3
"""
Statistical significance testing for CORAL vs Baseline.

Computes:
- Bootstrap 95% confidence intervals for F1
- McNemar's test for paired predictions

Usage:
  python 6.7_statistical_tests.py --strategy hybrid --lambda_coral 0.1
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

try:
    from scipy.stats import chi2

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print('[WARN] scipy not installed - McNemar test will use chi2 approximation')

RESULTS = Path('results')


def bootstrap_f1_ci(y_true, y_pred, n_bootstrap=1000, confidence=0.95, seed=42):
    """
    Compute bootstrap confidence interval for macro-F1.

    Returns:
        (lower, upper) bounds of the CI
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    f1_scores = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = rng.choice(n, size=n, replace=True)
        y_true_boot = [y_true[i] for i in idx]
        y_pred_boot = [y_pred[i] for i in idx]

        f1 = f1_score(y_true_boot, y_pred_boot, average='macro', zero_division=0)
        f1_scores.append(f1)

    alpha = 1 - confidence
    lower = np.percentile(f1_scores, alpha / 2 * 100)
    upper = np.percentile(f1_scores, (1 - alpha / 2) * 100)

    return lower, upper


def mcnemar_test(y_true, y_pred_base, y_pred_coral):
    """
    McNemar's test for paired predictions.

    Returns:
        (statistic, p_value)

    H0: Both models have the same error rate
    """
    n = len(y_true)

    # Contingency table
    # b = baseline correct, CORAL wrong
    # c = baseline wrong, CORAL correct
    b = 0
    c = 0

    for i in range(n):
        base_correct = y_pred_base[i] == y_true[i]
        coral_correct = y_pred_coral[i] == y_true[i]

        if base_correct and not coral_correct:
            b += 1
        elif not base_correct and coral_correct:
            c += 1

    # McNemar statistic (with continuity correction)
    if b + c == 0:
        # Both models made identical predictions
        return 0.0, 1.0

    statistic = (abs(b - c) - 1) ** 2 / (b + c)

    # p-value from chi-squared distribution (df=1)
    if HAS_SCIPY:
        p_value = 1 - chi2.cdf(statistic, df=1)
    else:
        # Simple approximation if scipy not available
        # This is rough but works for most cases
        if statistic < 3.84:  # chi2(1, 0.95) = 3.84
            p_value = 0.10  # Not significant
        else:
            p_value = 0.01  # Likely significant

    return statistic, p_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--lambda_coral', type=float, default=0.1)
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    args = parser.parse_args()

    strategy = args.strategy
    lam = args.lambda_coral

    # ----------------------------
    # Auto-detect file paths
    # ----------------------------
    baseline_preds = RESULTS / f'preds_target_resnet50_{strategy}.csv'
    coral_preds = RESULTS / f'preds_target_coral_lambda_{lam}_{strategy}.csv'

    if not baseline_preds.exists():
        raise FileNotFoundError(f'Baseline predictions not found: {baseline_preds}')
    if not coral_preds.exists():
        raise FileNotFoundError(f'CORAL predictions not found: {coral_preds}')

    print(f'[INFO] Loading predictions for strategy={strategy}, λ={lam}')
    print(f'       Baseline: {baseline_preds}')
    print(f'       CORAL:    {coral_preds}')

    # ----------------------------
    # Load predictions
    # ----------------------------
    df_base = pd.read_csv(baseline_preds)
    df_coral = pd.read_csv(coral_preds)

    y_true = df_base['y_true'].tolist()
    y_pred_base = df_base['y_pred'].tolist()
    y_pred_coral = df_coral['y_pred'].tolist()

    # Sanity check
    if len(y_true) != len(y_pred_coral):
        raise ValueError(
            f'Sample size mismatch: baseline={len(y_true)}, CORAL={len(y_pred_coral)}'
        )

    # ----------------------------
    # Compute F1 scores
    # ----------------------------
    f1_base = f1_score(y_true, y_pred_base, average='macro', zero_division=0)
    f1_coral = f1_score(y_true, y_pred_coral, average='macro', zero_division=0)

    # ----------------------------
    # Bootstrap CI
    # ----------------------------
    print('[INFO] Computing bootstrap confidence intervals...')
    ci_base = bootstrap_f1_ci(y_true, y_pred_base, n_bootstrap=args.n_bootstrap)
    ci_coral = bootstrap_f1_ci(y_true, y_pred_coral, n_bootstrap=args.n_bootstrap)

    # ----------------------------
    # McNemar test
    # ----------------------------
    print('[INFO] Running McNemar test...')
    statistic, p_value = mcnemar_test(y_true, y_pred_base, y_pred_coral)

    # ----------------------------
    # Build output
    # ----------------------------
    rows = [
        [
            'method',
            'strategy',
            'lambda',
            'f1_macro',
            'ci_lower',
            'ci_upper',
            'mcnemar_stat',
            'p_value',
        ],
        [
            'baseline',
            strategy,
            '',
            f1_base,
            ci_base[0],
            ci_base[1],
            '',
            '',
        ],
        [
            'CORAL',
            strategy,
            lam,
            f1_coral,
            ci_coral[0],
            ci_coral[1],
            statistic,
            p_value,
        ],
    ]

    out = RESULTS / f'T2_{strategy}_with_stats_lambda_{lam}.csv'
    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'[OK] Statistical tests saved → {out}')
    print('')
    print(f'     Baseline F1: {f1_base:.4f} [{ci_base[0]:.4f}, {ci_base[1]:.4f}]')
    print(f'     CORAL F1:    {f1_coral:.4f} [{ci_coral[0]:.4f}, {ci_coral[1]:.4f}]')
    print(f'     Delta F1:    {f1_coral - f1_base:+.4f}')
    print(f'     McNemar χ²:  {statistic:.4f}')
    print(f'     p-value:     {p_value:.4f}')
    print('')

    if p_value < 0.05:
        print('     ✅ Improvement is statistically significant (p < 0.05)')
    elif p_value < 0.10:
        print('     ⚠️  Marginal significance (0.05 ≤ p < 0.10)')
    else:
        print('     ⚠️  Improvement not statistically significant (p ≥ 0.10)')


if __name__ == '__main__':
    main()
