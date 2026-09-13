#!/usr/bin/env python3
"""
Build Table T4: Edge Deployment Performance

Aggregates benchmark results from CPU (x86, ARM) and EdgeTPU into a unified table
suitable for the ECCV paper.

Input:
    - results/edgetpu/benchmark_*.json (from 9.7_benchmark_edgetpu_batch.sh)
    - results/T4_edge_benchmarking_cpu_combined.csv (from 9.3)

Output:
    - results/T4_edgetpu_final.csv
    - results/T4_edgetpu_final.tex (LaTeX)

Usage:
    python scripts/9.8_build_T4_edgetpu.py
"""

import json
from pathlib import Path

import pandas as pd

RESULTS = Path('results')
EDGETPU_DIR = RESULTS / 'edgetpu'
OUTPUT_CSV = RESULTS / 'T4_edgetpu_final.csv'
OUTPUT_TEX = RESULTS / 'T4_edgetpu_final.tex'


def load_json_benchmarks(directory: Path) -> list[dict]:
    """Load all JSON benchmark results from directory."""
    records = []

    for json_file in directory.glob('benchmark_*.json'):
        with open(json_file) as f:
            data = json.load(f)

        if data.get('status') != 'completed':
            print(f'[WARN] Skipping failed benchmark: {json_file}')
            continue

        # Extract strategy from filename
        # benchmark_cpu_mobilenetv2_hybrid.json -> hybrid
        # benchmark_edgetpu_mobilenetv2_Fuzzy.json -> Fuzzy
        name = json_file.stem
        parts = name.split('_')
        backend = parts[1]  # cpu or edgetpu
        strategy = parts[-1]  # hybrid, Fuzzy, sbert

        record = {
            'strategy': strategy,
            'model': 'MobileNetV2',
            'backend': backend.upper() if backend == 'cpu' else 'EdgeTPU',
            'latency_mean_ms': data['latency']['mean_ms'],
            'latency_p50_ms': data['latency']['p50_ms'],
            'latency_p95_ms': data['latency']['p95_ms'],
            'latency_p99_ms': data['latency']['p99_ms'],
            'fps': data['throughput']['fps'],
            'model_size_mb': round(data['model']['size_bytes'] / 1024 / 1024, 2),
            'cold_start_ms': data['cold_start'].get('model_load_ms'),
            'first_inference_ms': data['cold_start'].get('first_inference_ms'),
            'platform': data['system'].get('platform', 'unknown'),
        }

        records.append(record)

    return records


def calculate_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speedup of EdgeTPU vs CPU."""
    speedups = []

    for strategy in df['strategy'].unique():
        cpu_row = df[(df['strategy'] == strategy) & (df['backend'] == 'CPU')]
        tpu_row = df[(df['strategy'] == strategy) & (df['backend'] == 'EdgeTPU')]

        if len(cpu_row) > 0 and len(tpu_row) > 0:
            cpu_lat = cpu_row['latency_mean_ms'].values[0]
            tpu_lat = tpu_row['latency_mean_ms'].values[0]
            speedup = cpu_lat / tpu_lat

            speedups.append(
                {
                    'strategy': strategy,
                    'cpu_latency_ms': cpu_lat,
                    'edgetpu_latency_ms': tpu_lat,
                    'speedup': round(speedup, 1),
                }
            )

    return pd.DataFrame(speedups)


def generate_latex(df: pd.DataFrame, output_path: Path):
    """Generate LaTeX table."""
    # Pivot for better presentation
    pivot = df.pivot_table(
        index=['strategy', 'model'],
        columns='backend',
        values=['latency_mean_ms', 'fps'],
        aggfunc='first',
    )

    # Flatten column names
    pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
    pivot = pivot.reset_index()

    # Format LaTeX
    latex = pivot.to_latex(
        index=False,
        float_format='%.1f',
        caption='Edge Deployment Performance (MobileNetV2 INT8)',
        label='tab:edge_performance',
        column_format='ll' + 'r' * (len(pivot.columns) - 2),
    )

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'[OK] LaTeX table saved: {output_path}')


def main():
    print('=' * 60)
    print('Building Table T4: Edge Deployment Performance')
    print('=' * 60)

    # Check for EdgeTPU results
    if not EDGETPU_DIR.exists():
        print(f'[WARN] EdgeTPU results directory not found: {EDGETPU_DIR}')
        print('       Run 9.7_benchmark_edgetpu_batch.sh on Raspberry Pi first.')

        # Create placeholder with expected structure
        placeholder = pd.DataFrame(
            {
                'strategy': ['hybrid', 'Fuzzy', 'sbert'] * 2,
                'model': ['MobileNetV2'] * 6,
                'backend': ['CPU'] * 3 + ['EdgeTPU'] * 3,
                'latency_mean_ms': [None] * 6,
                'fps': [None] * 6,
                'model_size_mb': [None] * 6,
            }
        )
        placeholder.to_csv(OUTPUT_CSV, index=False)
        print(f'[OK] Placeholder created: {OUTPUT_CSV}')
        return

    # Load JSON benchmarks
    records = load_json_benchmarks(EDGETPU_DIR)

    if not records:
        print('[WARN] No benchmark results found.')
        print('       Run 9.7_benchmark_edgetpu_batch.sh on Raspberry Pi first.')
        return

    df = pd.DataFrame(records)

    # Sort by strategy and backend
    df = df.sort_values(['strategy', 'backend'])

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f'\n[OK] Results saved: {OUTPUT_CSV}')

    # Calculate speedup
    print('\n' + '=' * 60)
    print('EdgeTPU Speedup vs CPU')
    print('=' * 60)

    speedup_df = calculate_speedup(df)
    if len(speedup_df) > 0:
        print(speedup_df.to_string(index=False))
    else:
        print('[INFO] Speedup calculation requires both CPU and EdgeTPU results.')

    # Generate LaTeX
    generate_latex(df, OUTPUT_TEX)

    # Summary table
    print('\n' + '=' * 60)
    print('Summary (for paper Table T4)')
    print('=' * 60)

    summary = df[
        [
            'strategy',
            'backend',
            'latency_mean_ms',
            'latency_p95_ms',
            'fps',
            'model_size_mb',
        ]
    ]
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
