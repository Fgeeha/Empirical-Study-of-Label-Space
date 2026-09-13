#!/usr/bin/env python3
"""
Step 11.3: Generate analysis figures from Edge-Bench results.

Creates publication-quality figures for the ECCV paper:
  - F7: Latency comparison (grouped bar chart, CPU vs EdgeTPU per architecture)
  - F8: Throughput (FPS) comparison
  - F9: EdgeTPU speedup heatmap (architecture x strategy)
  - F10: Model size vs latency scatter (Pareto front)

Input:
    - results/edgebench/raw_results.csv   (from 11.1)
    - results/T4_speedup.csv              (from 11.2)

Output:
    - figures/F7_latency_comparison.pdf
    - figures/F8_throughput_comparison.pdf
    - figures/F9_speedup_heatmap.pdf
    - figures/F10_size_vs_latency.pdf

Usage:
    python scripts/11.3_plot_edgebench_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INPUT_CSV = Path('results/edgebench/raw_results.csv')
SPEEDUP_CSV = Path('results/T4_speedup.csv')
FIGURES_DIR = Path('figures')

# Consistent styling
ARCH_DISPLAY = {
    'mobilenetv2': 'MobileNetV2',
    'mobilenetv1': 'MobileNetV1',
    'efficientnet_lite0': 'EffNet-Lite0',
    'efficientnet': 'EffNet-B3',
    'resnet50': 'ResNet-50',
}

ARCH_ORDER = [
    'mobilenetv2',
    'mobilenetv1',
    'efficientnet_lite0',
    'resnet50',
    'efficientnet',
]
STRATEGY_ORDER = ['hybrid', 'Fuzzy', 'sbert']

PALETTE = {
    'cpu': '#8b949e',
    'edgetpu': '#58a6ff',
}

STRATEGY_PALETTE = {
    'hybrid': '#58a6ff',
    'Fuzzy': '#3fb950',
    'sbert': '#a78bfa',
}


def setup_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update(
        {
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--',
        }
    )
    sns.set_palette('muted')


def load_data() -> pd.DataFrame:
    """Load the raw results CSV."""
    if not INPUT_CSV.exists():
        print(f'ERROR: {INPUT_CSV} not found. Run 11.1 first.')
        raise SystemExit(1)

    df = pd.read_csv(INPUT_CSV)

    # Average duplicates
    df = (
        df.groupby(['architecture', 'strategy', 'backend'])
        .agg(
            latency_mean_ms=('latency_mean_ms', 'mean'),
            latency_p95_ms=('latency_p95_ms', 'mean'),
            fps=('fps', 'mean'),
            model_size_mb=('model_size_mb', 'first'),
            cpu_percent_mean=('cpu_percent_mean', 'mean'),
            memory_mb_mean=('memory_mb_mean', 'mean'),
        )
        .reset_index()
    )

    df['arch_display'] = df['architecture'].map(ARCH_DISPLAY)
    return df


def plot_latency_comparison(df: pd.DataFrame):
    """F7: Grouped bar chart comparing CPU vs EdgeTPU latency per arch."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for i, strategy in enumerate(STRATEGY_ORDER):
        ax = axes[i]
        sdata = df[df['strategy'] == strategy].copy()

        # Prepare data for grouped bars
        archs = [a for a in ARCH_ORDER if a in sdata['architecture'].values]
        arch_labels = [ARCH_DISPLAY[a] for a in archs]
        x = np.arange(len(archs))
        width = 0.35

        cpu_vals = []
        tpu_vals = []
        for arch in archs:
            cpu_row = sdata[
                (sdata['architecture'] == arch) & (sdata['backend'] == 'cpu')
            ]
            tpu_row = sdata[
                (sdata['architecture'] == arch) & (sdata['backend'] == 'edgetpu')
            ]
            cpu_vals.append(cpu_row['latency_mean_ms'].values[0] if len(cpu_row) else 0)
            tpu_vals.append(tpu_row['latency_mean_ms'].values[0] if len(tpu_row) else 0)

        bars_cpu = ax.bar(
            x - width / 2,
            cpu_vals,
            width,
            label='CPU',
            color=PALETTE['cpu'],
            edgecolor='white',
            linewidth=0.5,
        )
        bars_tpu = ax.bar(
            x + width / 2,
            tpu_vals,
            width,
            label='EdgeTPU',
            color=PALETTE['edgetpu'],
            edgecolor='white',
            linewidth=0.5,
        )

        # Value labels
        for bar in bars_cpu:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{bar.get_height():.0f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=PALETTE['cpu'],
                )
        for bar in bars_tpu:
            if bar.get_height() > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f'{bar.get_height():.0f}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color=PALETTE['edgetpu'],
                )

        ax.set_title(f'Strategy: {strategy}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(arch_labels, rotation=30, ha='right')
        if i == 0:
            ax.set_ylabel('Latency (ms)')
        ax.legend(loc='upper right', framealpha=0.9)

    fig.suptitle(
        'Inference Latency: CPU vs EdgeTPU (Raspberry Pi 4, INT8)',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()

    out = FIGURES_DIR / 'F7_latency_comparison.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)


def plot_throughput_comparison(df: pd.DataFrame):
    """F8: FPS comparison across architectures and backends."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for i, strategy in enumerate(STRATEGY_ORDER):
        ax = axes[i]
        sdata = df[df['strategy'] == strategy].copy()

        archs = [a for a in ARCH_ORDER if a in sdata['architecture'].values]
        arch_labels = [ARCH_DISPLAY[a] for a in archs]
        x = np.arange(len(archs))
        width = 0.35

        cpu_vals = []
        tpu_vals = []
        for arch in archs:
            cpu_row = sdata[
                (sdata['architecture'] == arch) & (sdata['backend'] == 'cpu')
            ]
            tpu_row = sdata[
                (sdata['architecture'] == arch) & (sdata['backend'] == 'edgetpu')
            ]
            cpu_vals.append(cpu_row['fps'].values[0] if len(cpu_row) else 0)
            tpu_vals.append(tpu_row['fps'].values[0] if len(tpu_row) else 0)

        ax.bar(
            x - width / 2,
            cpu_vals,
            width,
            label='CPU',
            color=PALETTE['cpu'],
            edgecolor='white',
        )
        ax.bar(
            x + width / 2,
            tpu_vals,
            width,
            label='EdgeTPU',
            color=PALETTE['edgetpu'],
            edgecolor='white',
        )

        ax.set_title(f'Strategy: {strategy}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(arch_labels, rotation=30, ha='right')
        if i == 0:
            ax.set_ylabel('Throughput (FPS)')
        ax.legend(loc='upper left', framealpha=0.9)

    fig.suptitle(
        'Throughput: CPU vs EdgeTPU (Raspberry Pi 4, INT8)',
        fontsize=14,
        fontweight='bold',
        y=1.02,
    )
    fig.tight_layout()

    out = FIGURES_DIR / 'F8_throughput_comparison.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)


def plot_speedup_heatmap(df: pd.DataFrame):
    """F9: Heatmap of EdgeTPU speedup over CPU."""
    speedup_path = SPEEDUP_CSV
    if not speedup_path.exists():
        print(f'WARN: {speedup_path} not found, computing from raw data...')
        # Compute inline
        records = []
        for arch in ARCH_ORDER:
            for strategy in STRATEGY_ORDER:
                cpu = df[
                    (df['architecture'] == arch)
                    & (df['strategy'] == strategy)
                    & (df['backend'] == 'cpu')
                ]
                tpu = df[
                    (df['architecture'] == arch)
                    & (df['strategy'] == strategy)
                    & (df['backend'] == 'edgetpu')
                ]
                if cpu.empty or tpu.empty:
                    continue
                speedup = (
                    cpu['latency_mean_ms'].values[0] / tpu['latency_mean_ms'].values[0]
                )
                records.append(
                    {
                        'architecture': arch,
                        'arch_display': ARCH_DISPLAY.get(arch, arch),
                        'strategy': strategy,
                        'speedup_x': round(speedup, 2),
                    }
                )
        speedup_df = pd.DataFrame(records)
    else:
        speedup_df = pd.read_csv(speedup_path)

    if speedup_df.empty:
        print('WARN: No speedup data available, skipping heatmap.')
        return

    # Pivot for heatmap
    pivot = speedup_df.pivot_table(
        index='arch_display', columns='strategy', values='speedup_x'
    )

    # Reorder
    arch_order_display = [
        ARCH_DISPLAY[a] for a in ARCH_ORDER if ARCH_DISPLAY[a] in pivot.index
    ]
    strategy_cols = [s for s in STRATEGY_ORDER if s in pivot.columns]
    pivot = pivot.loc[arch_order_display, strategy_cols]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='YlOrRd',
        linewidths=1,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Speedup (x)'},
        vmin=1.0,
    )
    ax.set_title(
        'EdgeTPU Speedup over CPU (x)',
        fontsize=14,
        fontweight='bold',
        pad=16,
    )
    ax.set_ylabel('Architecture')
    ax.set_xlabel('Strategy')

    fig.tight_layout()

    out = FIGURES_DIR / 'F9_speedup_heatmap.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)


def plot_size_vs_latency(df: pd.DataFrame):
    """F10: Model size vs latency scatter (Pareto-style)."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for backend in ['cpu', 'edgetpu']:
        bdata = df[df['backend'] == backend]
        if bdata.empty:
            continue

        marker = 'o' if backend == 'cpu' else '^'
        label_suffix = 'CPU' if backend == 'cpu' else 'EdgeTPU'
        color = PALETTE[backend]

        for _, row in bdata.iterrows():
            ax.scatter(
                row['model_size_mb'],
                row['latency_mean_ms'],
                s=100,
                c=color,
                marker=marker,
                edgecolors='white',
                linewidths=0.5,
                alpha=0.8,
                zorder=5,
            )

        # Add one legend entry per backend
        ax.scatter(
            [],
            [],
            s=100,
            c=color,
            marker=marker,
            label=label_suffix,
            edgecolors='white',
        )

    # Annotate architecture names (deduplicated)
    annotated = set()
    for _, row in df.iterrows():
        key = (row['architecture'], row['backend'])
        if key not in annotated:
            ax.annotate(
                ARCH_DISPLAY.get(row['architecture'], row['architecture']),
                (row['model_size_mb'], row['latency_mean_ms']),
                fontsize=8,
                textcoords='offset points',
                xytext=(8, 4),
                alpha=0.7,
            )
            annotated.add(key)

    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title(
        'Model Size vs Inference Latency (RPi4 + Coral, INT8)',
        fontsize=14,
        fontweight='bold',
    )
    ax.legend(framealpha=0.9)

    fig.tight_layout()

    out = FIGURES_DIR / 'F10_size_vs_latency.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close(fig)


def main():
    print('=' * 60)
    print('Step 11.3: Generate Edge-Bench Analysis Figures')
    print('=' * 60)

    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f'Data: {len(df)} rows')
    print(f'Architectures: {sorted(df["architecture"].unique())}')
    print(f'Strategies:    {sorted(df["strategy"].unique())}')
    print(f'Backends:      {sorted(df["backend"].unique())}')
    print()

    print('Generating figures...')
    plot_latency_comparison(df)
    plot_throughput_comparison(df)
    plot_speedup_heatmap(df)
    plot_size_vs_latency(df)

    print(f'\nAll figures saved to {FIGURES_DIR}/')


if __name__ == '__main__':
    main()
