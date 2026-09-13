#!/usr/bin/env python3
"""
Plot latency comparison across models (Figure F5).

Works with combined benchmark results from CPU benchmarking.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
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
    # Plot latency comparison
    # ----------------------------
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract model names and latencies
    models = []
    latencies = []
    fps_values = []

    for _, row in df.iterrows():
        model_name = row['model']
        models.append(model_name)
        latencies.append(row['latency_mean_ms'])
        fps_values.append(row['fps'])

    # Create bar plot
    x = range(len(models))
    bars = ax.bar(
        x,
        latencies,
        color=['steelblue', 'coral', 'lightgreen'][: len(models)],
        alpha=0.7,
    )

    # Add FPS labels on top of bars
    for _i, (bar, fps) in enumerate(zip(bars, fps_values, strict=False)):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            f'{fps:.1f} FPS',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold',
        )

    ax.set_xlabel('Model Architecture', fontsize=12)
    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Inference Latency Comparison (CPU)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add p95 error bars
    if 'latency_p95_ms' in df.columns:
        p95_errors = df['latency_p95_ms'] - df['latency_mean_ms']
        ax.errorbar(
            x,
            latencies,
            yerr=p95_errors,
            fmt='none',
            ecolor='black',
            capsize=5,
            alpha=0.5,
        )

    plt.tight_layout()

    out_path = args.output or FIGURES / 'F5_latency_comparison.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'[OK] Latency comparison saved → {out_path}')

    # Print summary
    print('\nSummary:')
    print(
        f'  Fastest model: {models[latencies.index(min(latencies))]} ({min(latencies):.1f} ms)'
    )
    print(
        f'  Slowest model: {models[latencies.index(max(latencies))]} ({max(latencies):.1f} ms)'
    )
    print(f'  Speedup: {max(latencies) / min(latencies):.2f}x')


if __name__ == '__main__':
    main()
