#!/usr/bin/env python3
"""
Format paper-ready tables and sentences for ECCV 2026 article.

Collects all tables into results/final/ for submission (1 file per table):
- T1_domain_shift.csv   (domain shift PV vs PlantDoc)
- T2_coral_sweep.csv    (CORAL lambda sweep, all strategies)
- T2_stats.csv          (bootstrap CI + McNemar p-value, all strategies)
- T3_quant.csv          (FP32 vs INT8 latency + size, all strategies)
- T4_edge.csv           (CPU vs EdgeTPU latency + speedup)
- T4_system.csv         (RPI system metrics: temp, RAM, CPU%)
- T5_alignment.csv      (Strategy, #Aligned classes, Macro-F1)
- T6_summary.csv        (improved/failed/rare counts, all strategies)
- T6_main_3_examples.csv (top 3 +delta F1 classes)
- T7_sota.csv           (SOTA comparison)
- pvalues.txt           (ready sentences for main text)

Usage:
    python scripts/format_paper_tables.py
"""

import csv
from pathlib import Path

import pandas as pd

RESULTS = Path('results')
CONFIGS = Path('configs')
OUT = RESULTS / 'final'

# ---------------------------------------------------------------------------
# LaTeX table captions and labels
# ---------------------------------------------------------------------------
TEX_META = {
    'T1_domain_shift': {
        'caption': (
            'Domain shift evaluation. Models trained on PlantVillage (PV) '
            'and evaluated on PlantDoc without adaptation.'
        ),
        'label': 'tab:domain_shift',
    },
    'T2_coral_sweep': {
        'caption': (
            'CORAL hyperparameter sweep ($\\lambda \\in \\{0.01, 0.1, 1.0, 10.0\\}$) '
            'across three label-alignment strategies.'
        ),
        'label': 'tab:coral_sweep',
    },
    'T2_stats': {
        'caption': (
            'Statistical significance of CORAL adaptation '
            '(best $\\lambda$ per strategy). '
            'Bootstrap 95\\% CI and McNemar test.'
        ),
        'label': 'tab:coral_stats',
    },
    'T3_quant': {
        'caption': (
            'Quantization impact: FP32 vs.\\ INT8 post-training quantization '
            '(ResNet-50, TFLite).'
        ),
        'label': 'tab:quantization',
    },
    'T4_edge': {
        'caption': (
            'Edge deployment latency on Raspberry Pi~4 with Coral USB Accelerator. '
            'All models use INT8 PTQ.'
        ),
        'label': 'tab:edge_latency',
    },
    'T4_system': {
        'caption': (
            'System resource usage during inference on '
            'Raspberry Pi~4 (CPU\\%, RAM, temperature).'
        ),
        'label': 'tab:edge_system',
    },
    'T5_alignment': {
        'caption': 'Label-alignment strategy ablation.',
        'label': 'tab:alignment',
    },
    'T6_summary': {
        'caption': (
            'Per-class F1 summary: number of improved, failed, and rare classes '
            'after CORAL adaptation.'
        ),
        'label': 'tab:perclass_summary',
    },
    'T6_main_3_examples': {
        'caption': (
            'Top-3 classes with largest F1 improvement after CORAL (hybrid strategy).'
        ),
        'label': 'tab:perclass_top3',
    },
    'T7_sota': {
        'caption': 'Comparison with prior work on PlantDoc.',
        'label': 'tab:sota',
    },
}


def _col_format(n_cols: int) -> str:
    """Generate default column format: l for first col, c for rest."""
    return 'l' + 'c' * (n_cols - 1)


def save_tex(df: pd.DataFrame, name: str) -> None:
    """Save a DataFrame as a LaTeX table next to the CSV."""
    meta = TEX_META.get(name, {})
    caption = meta.get('caption', name.replace('_', ' '))
    label = meta.get('label', f'tab:{name.lower()}')
    col_fmt = _col_format(len(df.columns))

    tex_path = OUT / f'{name}.tex'
    # Replace NaN and empty strings with -- before rendering
    clean = df.fillna('--').replace('', '--')
    latex = clean.to_latex(
        index=False,
        escape=True,
        column_format=col_fmt,
        caption=caption,
        label=label,
        position='htbp',
    )
    tex_path.write_text(latex, encoding='utf-8')
    print(f'[TEX] {tex_path}')


def get_n_aligned(strategy: str) -> int:
    p = CONFIGS / f'label_map_{strategy}.csv'
    if not p.exists():
        return 0
    with p.open(encoding='utf-8') as f:
        return sum(1 for _ in csv.DictReader(f))


def format_T1():
    src = RESULTS / 'T1_domain_shift.csv'
    if not src.exists():
        print('[SKIP] T1_domain_shift.csv not found')
        return
    df = pd.read_csv(src)
    out_path = OUT / 'T1_domain_shift.csv'
    OUT.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False, float_format='%.4f')
    print(f'[OK] {out_path}')

    # LaTeX: round for readability
    tex_df = df.copy()
    for c in tex_df.select_dtypes('number').columns:
        tex_df[c] = tex_df[c].apply(lambda x: f'{x:.3f}')
    save_tex(tex_df, 'T1_domain_shift')


def format_T2():
    """T2: CORAL lambda sweep -- all strategies combined into one table."""
    rows = []
    for strat in ['Fuzzy', 'hybrid', 'sbert']:
        src = RESULTS / f'T2_coral_sweep_{strat}.csv'
        if not src.exists():
            continue
        df = pd.read_csv(src)
        for _, r in df.iterrows():
            rows.append(
                {
                    'Strategy': strat,
                    'Lambda': r['lambda'],
                    'PV Acc': f'{r["pv_accuracy"]:.3f}',
                    'PV F1': f'{r["pv_macro_f1"]:.3f}',
                    'Target Acc': f'{r["target_accuracy"]:.3f}',
                    'Target F1': f'{r["target_macro_f1"]:.3f}',
                    'Delta F1': f'{r["delta_f1"]:.3f}',
                }
            )
    if not rows:
        print('[SKIP] T2 coral sweep CSVs not found')
        return
    OUT.mkdir(exist_ok=True)
    sweep_df = pd.DataFrame(rows)
    out_path = OUT / 'T2_coral_sweep.csv'
    sweep_df.to_csv(out_path, index=False)
    print(f'[OK] {out_path}')
    save_tex(sweep_df, 'T2_coral_sweep')

    # Single combined stats file (CI + p-value for best lambda per strategy)
    stat_frames = []
    for strat in ['Fuzzy', 'hybrid', 'sbert']:
        best_lam = 0.1 if strat != 'sbert' else 0.01
        stat_src = RESULTS / f'T2_{strat}_with_stats_lambda_{best_lam}.csv'
        if not stat_src.exists():
            stat_src = RESULTS / f'T2_{strat}_with_stats.csv'
        if not stat_src.exists():
            continue
        stat_df = pd.read_csv(stat_src)
        if 'strategy' not in stat_df.columns:
            stat_df.insert(1, 'strategy', strat)
        if 'lambda' not in stat_df.columns:
            stat_df.insert(2, 'lambda', '')
            coral_mask = stat_df['method'] == 'CORAL'
            stat_df.loc[coral_mask, 'lambda'] = str(best_lam)
        stat_frames.append(stat_df)
    if stat_frames:
        combined_stats = pd.concat(stat_frames, ignore_index=True)
        stat_out = OUT / 'T2_stats.csv'
        combined_stats.to_csv(stat_out, index=False)
        print(f'[OK] {stat_out}')
        # LaTeX: round floats
        tex_stats = combined_stats.copy()
        for c in ['f1_macro', 'ci_lower', 'ci_upper']:
            if c in tex_stats.columns:
                tex_stats[c] = tex_stats[c].apply(lambda x: f'{x:.3f}')
        if 'lambda' in tex_stats.columns:
            tex_stats['lambda'] = tex_stats['lambda'].apply(
                lambda x: (f'{float(x):g}' if pd.notna(x) and x != '' else '--')
            )
        if 'mcnemar_stat' in tex_stats.columns:
            tex_stats['mcnemar_stat'] = tex_stats['mcnemar_stat'].apply(
                lambda x: f'{x:.2f}' if pd.notna(x) and x != '' else '--'
            )
        if 'p_value' in tex_stats.columns:
            tex_stats['p_value'] = tex_stats['p_value'].apply(
                lambda x: (f'{float(x):.2e}' if pd.notna(x) and x != '' else '--')
            )
        save_tex(tex_stats, 'T2_stats')


def format_T5_alignment():
    src = RESULTS / 'T5_ablation_comprehensive.csv'
    if not src.exists():
        print('[SKIP] T5_ablation_comprehensive.csv not found')
        return
    df = pd.read_csv(src)
    align = df[df['ablation_type'] == 'alignment'].copy()
    align = align[['strategy', 'n_aligned_classes', 'target_f1_macro']]
    align.columns = ['Strategy', '#Aligned classes', 'Macro-F1']
    align['Strategy'] = align['Strategy'].str.capitalize()
    align['Macro-F1'] = align['Macro-F1'].apply(lambda x: f'{x:.3f}')
    out_path = OUT / 'T5_alignment.csv'
    OUT.mkdir(exist_ok=True)
    align.to_csv(out_path, index=False)
    print(f'[OK] {out_path}')
    save_tex(align, 'T5_alignment')


def format_T3():
    """T3: FP32 vs INT8 latency + size, all strategies in one file."""
    all_rows = []
    for strat in ['Fuzzy', 'hybrid', 'sbert']:
        src = RESULTS / f'T3_quant_fp32_vs_int8_{strat}.csv'
        if not src.exists():
            continue
        df = pd.read_csv(src)
        target = df[df['domain'] == 'target']
        if target.empty:
            target = df
        fp32 = target[target['model'] == 'fp32']
        int8 = target[target['model'] == 'int8']
        if fp32.empty or int8.empty:
            continue
        all_rows.append(
            {
                'Strategy': strat,
                'Model': 'FP32',
                'Latency (ms)': f'{fp32["latency_ms"].iloc[0]:.1f}',
                'Size (MB)': f'{fp32["size_mb"].iloc[0]:.1f}',
            }
        )
        all_rows.append(
            {
                'Strategy': strat,
                'Model': 'INT8',
                'Latency (ms)': f'{int8["latency_ms"].iloc[0]:.1f}',
                'Size (MB)': f'{int8["size_mb"].iloc[0]:.1f}',
            }
        )
    if not all_rows:
        print('[SKIP] T3 quant CSVs not found')
        return
    OUT.mkdir(exist_ok=True)
    quant_df = pd.DataFrame(all_rows)
    out_path = OUT / 'T3_quant.csv'
    quant_df.to_csv(out_path, index=False)
    print(f'[OK] {out_path}')
    save_tex(quant_df, 'T3_quant')


def format_T4_edge():
    src = RESULTS / 'T4_edgebench.csv'
    if not src.exists():
        print('[SKIP] T4_edgebench.csv not found')
        return
    df = pd.read_csv(src)
    # Pivot: model x strategy, CPU vs EdgeTPU, speedup
    rows = []
    for (arch, strat), g in df.groupby(['architecture', 'strategy']):
        cpu = g[g['backend'] == 'cpu']
        edgetpu = g[g['backend'] == 'edgetpu']
        if cpu.empty or edgetpu.empty:
            continue
        cpu_ms = cpu['latency_mean'].iloc[0]
        et_ms = edgetpu['latency_mean'].iloc[0]
        speedup = cpu_ms / et_ms if et_ms > 0 else 0
        arch_display = (
            cpu['arch_display'].iloc[0] if 'arch_display' in cpu.columns else arch
        )
        rows.append(
            {
                'Model': arch_display,
                'Strategy': strat,
                'CPU (ms)': f'{cpu_ms:.1f}',
                'EdgeTPU (ms)': f'{et_ms:.1f}',
                'Speedup': f'{speedup:.1f}x',
            }
        )
    out_path = OUT / 'T4_edge.csv'
    OUT.mkdir(exist_ok=True)
    edge_df = pd.DataFrame(rows)
    edge_df.to_csv(out_path, index=False)
    print(f'[OK] {out_path}')
    save_tex(edge_df, 'T4_edge')


def format_T4_system():
    """T4 supplement: RPI system metrics (CPU%, RAM MB, temperature)."""
    src = RESULTS / 'T4_edgebench.csv'
    if not src.exists():
        print('[SKIP] T4_edgebench.csv not found')
        return
    df = pd.read_csv(src)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            {
                'Model': r.get('arch_display', r['architecture']),
                'Strategy': r['strategy'],
                'Backend': r['backend'].upper(),
                'Latency (ms)': f'{r["latency_mean"]:.1f}',
                'FPS': f'{r["fps"]:.1f}',
                'CPU (%)': f'{r["cpu_percent"]:.1f}',
                'RAM (MB)': f'{r["memory_mb"]:.1f}',
                'Temp (C)': f'{r["cpu_temp"]:.1f}',
                'Size (MB)': f'{r["model_size_mb"]:.2f}',
                'Params': r.get('params', ''),
            }
        )
    if not rows:
        return
    OUT.mkdir(exist_ok=True)
    sys_df = pd.DataFrame(rows)
    out_path = OUT / 'T4_system.csv'
    sys_df.to_csv(out_path, index=False)
    print(f'[OK] {out_path}')
    save_tex(sys_df, 'T4_system')


def format_T6_summary():
    """T6: per-class F1 summary, all strategies in one file."""
    all_rows = []
    for strat in ['Fuzzy', 'hybrid', 'sbert']:
        src = RESULTS / f'T6_per_class_f1_{strat}.csv'
        if not src.exists():
            continue
        df = pd.read_csv(src)
        improved = int((df['delta_f1'] > 0).sum())
        failed = int(((df['baseline_f1'] < 0.1) & (df['coral_f1'] < 0.1)).sum())
        rare = int((df['support'] < 50).sum())
        all_rows.append(
            {
                'Strategy': strat,
                'N classes': len(df),
                'Improved': improved,
                'Failed both': failed,
                'Rare (<50)': rare,
            }
        )
    if not all_rows:
        print('[SKIP] T6 per-class CSVs not found')
        return
    OUT.mkdir(exist_ok=True)
    summary_df = pd.DataFrame(all_rows)
    out_path = OUT / 'T6_summary.csv'
    summary_df.to_csv(out_path, index=False)
    print(f'[OK] {out_path}')
    save_tex(summary_df, 'T6_summary')


def format_T6_main_paper():
    """Top 3 classes with largest delta_f1 (suppl. has full table)."""
    label_map_path = CONFIGS / 'label_map_hybrid.csv'
    if not label_map_path.exists():
        return
    labels = pd.read_csv(label_map_path)
    id_to_name = dict(zip(labels['label_id'], labels['canonical_name'], strict=False))

    src = RESULTS / 'T6_per_class_f1_hybrid.csv'
    if not src.exists():
        return
    df = pd.read_csv(src)
    top3 = df.nlargest(3, 'delta_f1')[
        ['class_id', 'baseline_f1', 'coral_f1', 'delta_f1']
    ]
    top3['class_name'] = top3['class_id'].map(id_to_name)
    top3 = top3[['class_id', 'class_name', 'baseline_f1', 'coral_f1', 'delta_f1']]
    top3['baseline_f1'] = top3['baseline_f1'].apply(lambda x: f'{x:.2f}')
    top3['coral_f1'] = top3['coral_f1'].apply(lambda x: f'{x:.2f}')
    top3['delta_f1'] = top3['delta_f1'].apply(lambda x: f'{x:+.2f}')
    out_path = OUT / 'T6_main_3_examples.csv'
    OUT.mkdir(exist_ok=True)
    top3.to_csv(out_path, index=False)
    print(f'[OK] {out_path} (main paper: 3 classes with largest +delta F1)')
    save_tex(top3, 'T6_main_3_examples')


def format_T7():
    src = RESULTS / 'T7_sota.csv'
    if not src.exists():
        print('[SKIP] T7_sota.csv not found')
        return
    df = pd.read_csv(src)
    out_path = OUT / 'T7_sota.csv'
    OUT.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False, float_format='%.2f')
    print(f'[OK] {out_path}')
    tex_df = df.copy()
    for c in tex_df.select_dtypes('number').columns:
        tex_df[c] = tex_df[c].apply(lambda x: f'{x:.2f}')
    save_tex(tex_df, 'T7_sota')


def format_pvalues():
    lines = []
    for strat in ['Fuzzy', 'hybrid', 'sbert']:
        best_lam = 0.1 if strat != 'sbert' else 0.01
        p1 = RESULTS / f'T2_{strat}_with_stats_lambda_{best_lam}.csv'
        if not p1.exists():
            p1 = RESULTS / f'T2_{strat}_with_stats.csv'
        if not p1.exists():
            continue
        df = pd.read_csv(p1)
        base = df[df['method'] == 'baseline']
        coral = df[df['method'] == 'CORAL']
        if base.empty or coral.empty:
            base = df.iloc[0]
            coral = df.iloc[1]
        else:
            base = base.iloc[0]
            coral = coral.iloc[0]
        f1_base = float(base['f1_macro'])
        f1_coral = float(coral['f1_macro'])
        pval = coral.get('p_value', float('nan'))
        try:
            pval = float(pval)
            p_str = f'p = {pval:.2e}' if pval < 0.001 else f'p = {pval:.4f}'
        except (TypeError, ValueError):
            p_str = str(pval)
        lines.append(
            f'CORAL (lambda={best_lam}) [{strat}]: '
            f'macro-F1 {f1_base:.2f} -> {f1_coral:.2f} '
            f'({p_str}, McNemar).'
        )
    out_path = OUT / 'pvalues.txt'
    OUT.mkdir(exist_ok=True)
    out_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'[OK] {out_path}')


def main():
    print('=' * 60)
    print('Collecting all paper tables into results/final/')
    print('=' * 60)

    OUT.mkdir(parents=True, exist_ok=True)

    format_T1()
    format_T2()
    format_T3()
    format_T4_edge()
    format_T4_system()
    format_T5_alignment()
    format_T6_summary()
    format_T6_main_paper()
    format_T7()
    format_pvalues()

    # Summary
    files = sorted(OUT.glob('*'))
    print(f'\n{"=" * 60}')
    print(f'Final tables: {len(files)} files in {OUT}/')
    print('=' * 60)
    for f in files:
        print(f'  {f.name}')
    print('\nDone.')


if __name__ == '__main__':
    main()
