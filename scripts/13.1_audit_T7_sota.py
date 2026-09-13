#!/usr/bin/env python3
"""
Audit T7 SOTA table: detect mislabeled results and rebuild with correct attribution.

Problem found:
    T7_sota.csv claims "CORAL (ours)" achieves 0.86 acc / 0.85 F1 on PlantDoc
    WITHOUT target labels. But T5_ablation.csv shows this is actually the
    fine-tuned result (finetune=yes), not zero-shot CORAL.

    Real zero-shot CORAL results are 0.19-0.23 F1 (from T2_coral_sweep).
    The 0.85 F1 comes from fine-tuning on hybrid strategy with target labels.

This script:
    1. Cross-references T7_sota.csv against T5_ablation.csv and T2_coral_sweep
    2. Prints a detailed discrepancy report
    3. Generates a corrected T7_sota_fixed.csv with proper attribution
    4. Generates LaTeX version

Usage:
    python scripts/13.1_audit_T7_sota.py
"""

from pathlib import Path
import sys

import pandas as pd

RESULTS = Path('results')
FINAL = RESULTS / 'final'


def load_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    print(f'  [MISSING] {path}')
    return None


def find_best_coral_zero_shot() -> dict:
    """Find the actual best CORAL result WITHOUT target labels."""
    best = {'strategy': None, 'lambda': None, 'acc': 0.0, 'f1': 0.0}

    for strat in ['Fuzzy', 'hybrid', 'sbert']:
        src = RESULTS / f'T2_coral_sweep_{strat}.csv'
        if not src.exists():
            continue
        df = pd.read_csv(src)
        idx = df['target_macro_f1'].idxmax()
        row = df.loc[idx]
        if row['target_macro_f1'] > best['f1']:
            best = {
                'strategy': strat,
                'lambda': row['lambda'],
                'acc': row['target_accuracy'],
                'f1': row['target_macro_f1'],
            }
    return best


def find_best_finetune() -> dict:
    """Find the actual best fine-tuned result WITH target labels."""
    ablation = load_csv(RESULTS / 'T5_ablation.csv')
    if ablation is None:
        return {}

    ft = ablation[ablation['finetune'] == 'yes']
    if ft.empty:
        return {}

    idx = ft['target_macro_f1'].idxmax()
    row = ft.loc[idx]
    return {
        'strategy': row['strategy'],
        'da': row['domain_adaptation'],
        'acc': row['target_accuracy'],
        'f1': row['target_macro_f1'],
    }


def main():
    print('=' * 70)
    print('AUDIT: T7 SOTA Table (Table 9 in paper)')
    print('=' * 70)

    # --- Load current T7 ---
    t7 = load_csv(RESULTS / 'T7_sota.csv')
    if t7 is None:
        print('[FATAL] T7_sota.csv not found')
        sys.exit(1)

    print('\n--- Current T7_sota.csv ---')
    print(t7.to_string(index=False))

    # --- Cross-reference with T5_ablation ---
    ablation = load_csv(RESULTS / 'T5_ablation.csv')
    if ablation is not None:
        print('\n--- T5_ablation.csv (fine-tune rows) ---')
        ft_rows = ablation[ablation['finetune'] == 'yes']
        print(ft_rows.to_string(index=False))

    # --- Find real CORAL zero-shot ---
    best_coral = find_best_coral_zero_shot()
    best_ft = find_best_finetune()

    print('\n' + '=' * 70)
    print('DISCREPANCY ANALYSIS')
    print('=' * 70)

    coral_row = t7[t7['method'].str.contains('CORAL', case=False)]
    ft_row = t7[t7['method'].str.contains('Fine-tune', case=False)]

    if not coral_row.empty:
        claimed_f1 = coral_row.iloc[0]['macro_f1']
        claimed_acc = coral_row.iloc[0]['acc']

        print(f'\n[CLAIM] T7 says: "CORAL (ours)" = acc {claimed_acc}, F1 {claimed_f1}')
        print('[CLAIM] T7 says: "No target labels used"')

        if best_coral['strategy']:
            print(
                f'\n[FACT]  Best zero-shot CORAL (no target labels):'
                f'\n        Strategy={best_coral["strategy"]}, '
                f'lambda={best_coral["lambda"]}'
                f'\n        acc={best_coral["acc"]:.4f}, F1={best_coral["f1"]:.4f}'
            )

        if best_ft:
            print(
                f'\n[FACT]  Best fine-tuned model (WITH target labels):'
                f'\n        Strategy={best_ft["strategy"]}, '
                f'DA={best_ft["da"]}'
                f'\n        acc={best_ft["acc"]:.4f}, F1={best_ft["f1"]:.4f}'
            )

        tolerance = 0.01
        if best_ft and abs(claimed_f1 - best_ft['f1']) < tolerance:
            print(
                f'\n[ERROR] *** MISLABEL DETECTED ***'
                f'\n        T7 "CORAL (ours)" F1={claimed_f1} matches '
                f'fine-tune({best_ft["strategy"]}) F1={best_ft["f1"]:.4f}'
                f'\n        but does NOT match best zero-shot CORAL '
                f'F1={best_coral["f1"]:.4f}'
                f'\n'
                f'\n        The result labeled "CORAL" is actually the '
                f'FINE-TUNED model on {best_ft["strategy"]} strategy.'
                f'\n        This model uses TARGET LABELS, contradicting '
                f'the "No target labels" claim in the paper.'
            )
        else:
            print('\n[OK] No obvious mislabel detected (manual check recommended)')

    if not ft_row.empty and best_ft:
        ft_claimed_f1 = ft_row.iloc[0]['macro_f1']
        if abs(ft_claimed_f1 - best_ft['f1']) > 0.05:
            ft_ablation_fuzzy = ablation[
                (ablation['finetune'] == 'yes') & (ablation['strategy'] == 'Fuzzy')
            ]
            if not ft_ablation_fuzzy.empty:
                fuzzy_ft_f1 = ft_ablation_fuzzy.iloc[0]['target_macro_f1']
                if abs(ft_claimed_f1 - fuzzy_ft_f1) < 0.01:
                    print(
                        f'\n[WARNING] T7 "Fine-tune UB" F1={ft_claimed_f1} is '
                        f'from Fuzzy strategy (F1={fuzzy_ft_f1:.4f})'
                        f'\n         but the actual BEST fine-tune is '
                        f'{best_ft["strategy"]} (F1={best_ft["f1"]:.4f})'
                        f'\n         The "upper bound" is not the true '
                        f'upper bound.'
                    )

    # --- Also check: CORAL+finetune vs finetune-only ---
    if ablation is not None:
        for strat in ['Fuzzy', 'hybrid', 'sbert']:
            ft_only = ablation[
                (ablation['strategy'] == strat)
                & (ablation['finetune'] == 'yes')
                & (ablation['domain_adaptation'] == 'none')
            ]
            ft_coral = ablation[
                (ablation['strategy'] == strat)
                & (ablation['finetune'] == 'yes')
                & (ablation['domain_adaptation'] == 'CORAL')
            ]
            if not ft_only.empty and not ft_coral.empty:
                f1_only = ft_only.iloc[0]['target_macro_f1']
                f1_coral = ft_coral.iloc[0]['target_macro_f1']
                if abs(f1_only - f1_coral) < 1e-6:
                    print(
                        f'\n[NOTE]  Strategy {strat}: Fine-tune F1 = '
                        f'Fine-tune+CORAL F1 = {f1_only:.4f}'
                        f'\n        CORAL adds nothing on top of fine-tuning '
                        f'for this strategy.'
                    )

    # --- Generate corrected table ---
    print('\n' + '=' * 70)
    print('CORRECTED T7 TABLE')
    print('=' * 70)

    rows = [
        {
            'method': 'Zhang et al. (2020)',
            'target_labels': 'Yes',
            'dataset': 'PlantDoc',
            'acc': 0.52,
            'macro_f1': 0.48,
            'note': 'supervised baseline',
        },
        {
            'method': 'Singh et al. (2021)',
            'target_labels': 'Yes',
            'dataset': 'PlantDoc',
            'acc': 0.61,
            'macro_f1': 0.58,
            'note': 'supervised baseline',
        },
    ]

    if best_coral['strategy']:
        rows.append(
            {
                'method': (
                    f'CORAL (ours, {best_coral["strategy"]}, '
                    f'lambda={best_coral["lambda"]})'
                ),
                'target_labels': 'No',
                'dataset': 'PlantDoc',
                'acc': round(best_coral['acc'], 4),
                'macro_f1': round(best_coral['f1'], 4),
                'note': 'zero-shot DA, no target labels',
            }
        )

    if best_ft:
        rows.append(
            {
                'method': f'Fine-tune UB (ours, {best_ft["strategy"]})',
                'target_labels': 'Yes',
                'dataset': 'PlantDoc',
                'acc': round(best_ft['acc'], 4),
                'macro_f1': round(best_ft['f1'], 4),
                'note': 'upper bound with target supervision',
            }
        )

    fixed_df = pd.DataFrame(rows)
    print(fixed_df.to_string(index=False))

    # --- Save corrected table ---
    FINAL.mkdir(parents=True, exist_ok=True)
    out_csv = FINAL / 'T7_sota_fixed.csv'
    fixed_df.to_csv(out_csv, index=False)
    print(f'\n[SAVED] {out_csv}')

    # --- Generate LaTeX ---
    tex_rows = []
    for _, r in fixed_df.iterrows():
        tex_rows.append(
            f'        {r["method"]} & {r["target_labels"]} & '
            f'{r["acc"]:.2f} & {r["macro_f1"]:.2f} \\\\'
        )

    tex = (
        '\\begin{table}[htbp]\n'
        '\\centering\n'
        '\\caption{Comparison with previously published methods on PlantDoc. '
        'CORAL (ours) uses no target labels; other methods use PlantDoc '
        'training data. Fine-tune UB is included as a supervised upper '
        'bound.}\n'
        '\\label{tab:sota}\n'
        '\\begin{tabular}{lccc}\n'
        '\\toprule\n'
        'Method & Target Labels & Acc & Macro-F1 \\\\\n'
        '\\midrule\n' + '\n'.join(tex_rows) + '\n'
        '\\bottomrule\n'
        '\\end{tabular}\n'
        '\\end{table}\n'
    )

    out_tex = FINAL / 'T7_sota_fixed.tex'
    out_tex.write_text(tex, encoding='utf-8')
    print(f'[SAVED] {out_tex}')

    # --- Summary ---
    print('\n' + '=' * 70)
    print('REQUIRED ACTIONS')
    print('=' * 70)
    print(
        '1. Replace T7_sota.csv with T7_sota_fixed.csv\n'
        '2. Update paper Table 9:\n'
        '   - "CORAL (ours)" row should show real zero-shot CORAL '
        f'F1={best_coral["f1"]:.3f}\n'
        '   - "Fine-tune UB" should show the best fine-tune '
        f'F1={best_ft["f1"]:.3f} ({best_ft["strategy"]})\n'
        '3. Update paper text (Section 5, last paragraph) to match\n'
        '4. Fix 7.5_build_T7_sota.py to derive numbers from data, not '
        'hardcode\n'
        '5. The claim "CORAL without target labels outperforms supervised '
        'methods"\n'
        '   is FALSE. Real CORAL zero-shot F1 is ~0.23, far below Singh '
        '(0.58).'
    )


if __name__ == '__main__':
    main()
