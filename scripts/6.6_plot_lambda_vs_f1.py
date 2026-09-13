#!/usr/bin/env python3
"""
Plot F1 vs log(λ) with error bars (bootstrap CI) and significance markers.

X: log(λ)
Y: macro-F1
Error bars: bootstrap 95% CI
* marks points where CORAL significantly improves over baseline (McNemar p < 0.05)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RESULTS = Path('results')
FIGURES = Path('figures')

FIGURES.mkdir(exist_ok=True)


def bootstrap_f1_ci(y_true, y_pred, n_bootstrap=1000, confidence=0.95, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    f1_scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        f1_scores.append(f1_score(yt, yp, average='macro', zero_division=0))
    alpha = 1 - confidence
    lower = np.percentile(f1_scores, alpha / 2 * 100)
    upper = np.percentile(f1_scores, (1 - alpha / 2) * 100)
    return lower, upper


def mcnemar_pvalue(y_true, y_pred_base, y_pred_coral):
    b = c = 0
    for i in range(len(y_true)):
        base_ok = y_pred_base[i] == y_true[i]
        coral_ok = y_pred_coral[i] == y_true[i]
        if base_ok and not coral_ok:
            b += 1
        elif not base_ok and coral_ok:
            c += 1
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    try:
        from scipy.stats import chi2

        return 1 - chi2.cdf(stat, df=1)
    except ImportError:
        return 0.01 if stat >= 3.84 else 0.10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--n_bootstrap', type=int, default=1000)
    args = parser.parse_args()

    strategy = args.strategy

    csv_path = RESULTS / f'T2_coral_sweep_{strategy}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path).sort_values('lambda')
    lambdas = df['lambda'].tolist()

    baseline_preds = RESULTS / f'preds_target_resnet50_{strategy}.csv'
    if not baseline_preds.exists():
        raise FileNotFoundError(baseline_preds)

    df_base = pd.read_csv(baseline_preds)
    y_true = df_base['y_true'].tolist()
    y_pred_base = df_base['y_pred'].tolist()

    # Bootstrap CI for each lambda
    ci_lower, ci_upper = [], []
    significant = []

    for lam in lambdas:
        coral_preds = RESULTS / f'preds_target_coral_lambda_{lam}_{strategy}.csv'
        if not coral_preds.exists():
            ci_lower.append(np.nan)
            ci_upper.append(np.nan)
            significant.append(False)
            continue

        df_c = pd.read_csv(coral_preds)
        y_pred_coral = df_c['y_pred'].tolist()

        lo, hi = bootstrap_f1_ci(y_true, y_pred_coral, n_bootstrap=args.n_bootstrap)
        ci_lower.append(lo)
        ci_upper.append(hi)

        p = mcnemar_pvalue(y_true, y_pred_base, y_pred_coral)
        significant.append(p < 0.05)

    # Baseline CI
    base_lo, base_hi = bootstrap_f1_ci(
        y_true, y_pred_base, n_bootstrap=args.n_bootstrap
    )
    f1_base = f1_score(y_true, y_pred_base, average='macro', zero_division=0)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))

    yerr_lo = [
        df['target_macro_f1'].iloc[i] - ci_lower[i] for i in range(len(ci_lower))
    ]
    yerr_hi = [
        ci_upper[i] - df['target_macro_f1'].iloc[i] for i in range(len(ci_upper))
    ]
    yerr = [yerr_lo, yerr_hi]

    ax.errorbar(
        df['lambda'],
        df['target_macro_f1'],
        yerr=yerr,
        marker='o',
        capsize=4,
        capthick=1.5,
        linewidth=2,
        label='CORAL',
    )

    # Significance markers
    for i, (lam, sig) in enumerate(zip(df['lambda'], significant)):
        if sig:
            ax.annotate(
                '*',
                (lam, df['target_macro_f1'].iloc[i] + yerr_hi[i] + 0.01),
                ha='center',
                fontsize=14,
            )

    # Baseline line with CI
    ax.axhline(f1_base, color='gray', linestyle='--', alpha=0.8, label='Baseline')
    ax.axhspan(base_lo, base_hi, alpha=0.15, color='gray')

    ax.set_xscale('log')
    ax.set_xlabel('CORAL λ (log scale)')
    ax.set_ylabel('Target Macro-F1')
    ax.set_title(f'CORAL λ Sweep ({strategy})')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right')

    out = FIGURES / f'F2_f1_vs_lambda_{strategy}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

    print(f'[OK] CORAL sweep figure saved → {out}')


if __name__ == '__main__':
    main()
