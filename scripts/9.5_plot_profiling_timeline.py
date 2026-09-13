#!/usr/bin/env python3
"""
Plot inference metrics comparison (Figure F6).

Creates a comprehensive visualization of latency, FPS, CPU, and RAM
across different models.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path('results')
FIGURES = Path('figures')

FIGURES.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--benchmark_csv',
        default='results/T4_edge_benchmarking_cpu_combined.csv',
        help='Combined benchmark results CSV',
    )
    parser.add_argument('--output', help='Output figure path')
    args = parser.parse_args()

    bench_csv = Path(args.benchmark_csv)

    if not bench_csv.exists():
        print(f'[WARN] Benchmark CSV not found: {bench_csv}')
        print('[INFO] Looking for individual benchmark files...')

        # Try to find individual files
        individual_files = list(RESULTS.glob('T4_cpu_benchmark_*.csv'))

        if not individual_files:
            print('[ERROR] No benchmark files found!')
            print('[NOTE] Run: bash run_cpu_benchmarks.sh')
            return

        # Combine them
        dfs = []
        for f in individual_files:
            df = pd.read_csv(f)
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # Save combined
        bench_csv = RESULTS / 'T4_edge_benchmarking_cpu_combined.csv'
        df.to_csv(bench_csv, index=False)
        print(f'[INFO] Combined results saved to {bench_csv}')
    else:
        df = pd.read_csv(bench_csv)

    # ----------------------------
    # Create comprehensive figure
    # ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    models = df['model'].tolist()
    colors = ['steelblue', 'coral', 'lightgreen'][: len(models)]

    # ----------------------------
    # Subplot 1: Latency (mean, median, p95)
    # ----------------------------
    ax = axes[0, 0]
    x = np.arange(len(models))
    width = 0.25

    ax.bar(
        x - width, df['latency_mean_ms'], width, label='Mean', color=colors, alpha=0.7
    )
    ax.bar(x, df['latency_median_ms'], width, label='Median', color=colors, alpha=0.5)
    ax.bar(x + width, df['latency_p95_ms'], width, label='P95', color=colors, alpha=0.3)

    ax.set_xlabel('Model')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # ----------------------------
    # Subplot 2: FPS
    # ----------------------------
    ax = axes[0, 1]
    bars = ax.barh(models, df['fps'], color=colors, alpha=0.7)

    # Add value labels
    for _i, (bar, fps) in enumerate(zip(bars, df['fps'])):
        width = bar.get_width()
        ax.text(
            width + 2,
            bar.get_y() + bar.get_height() / 2,
            f'{fps:.1f}',
            ha='left',
            va='center',
            fontsize=10,
        )

    ax.set_xlabel('Frames Per Second (FPS)')
    ax.set_title('Throughput Comparison')
    ax.grid(axis='x', alpha=0.3)

    # ----------------------------
    # Subplot 3: CPU Usage (if available)
    # ----------------------------
    ax = axes[1, 0]

    if 'cpu_mean_pct' in df.columns and df['cpu_mean_pct'].notna().any():
        cpu_data = df['cpu_mean_pct'].fillna(0)
        ax.bar(models, cpu_data, color=colors, alpha=0.7)
        ax.set_ylabel('CPU Usage (%)')
        ax.set_title('Average CPU Utilization')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            'CPU data not available\n(install psutil)',
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=12,
            color='gray',
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # ----------------------------
    # Subplot 4: Model Size vs Latency
    # ----------------------------
    ax = axes[1, 1]

    # Model sizes (approximate, MB)
    model_sizes = {
        'ResNet': 90,
        'EfficientNet': 35,
        'MobileNetV2': 9,
    }

    sizes = [model_sizes.get(m, 50) for m in models]
    latencies = df['latency_mean_ms'].tolist()

    ax.scatter(
        sizes, latencies, s=200, c=colors, alpha=0.6, edgecolors='black', linewidth=2
    )

    # Add labels
    for i, (size, lat, model) in enumerate(zip(sizes, latencies, models)):
        ax.annotate(
            model,
            (size, lat),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            bbox={'boxstyle': 'round,pad=0.3', 'facecolor': colors[i], 'alpha': 0.3},
        )

    ax.set_xlabel('Model Size (MB)')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Size vs Latency Trade-off')
    ax.grid(alpha=0.3)

    # Add Pareto frontier line (optional)
    sorted_pairs = sorted(zip(sizes, latencies))
    ax.plot(
        [p[0] for p in sorted_pairs],
        [p[1] for p in sorted_pairs],
        'k--',
        alpha=0.3,
        linewidth=1,
    )

    plt.suptitle(
        'Edge Deployment Performance Metrics (CPU)',
        fontsize=16,
        fontweight='bold',
        y=0.995,
    )
    plt.tight_layout()

    out_path = args.output or FIGURES / 'F6_deployment_metrics.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'[OK] Deployment metrics saved → {out_path}')


if __name__ == '__main__':
    main()
