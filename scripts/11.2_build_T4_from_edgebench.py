#!/usr/bin/env python3
"""
Step 11.2: Build Table T4 (Edge Deployment Performance) from Edge-Bench results.

Reads the flattened CSV produced by 11.1 and builds:
  - T4 main table: 5 architectures x 3 strategies x CPU/EdgeTPU
  - Speedup analysis: EdgeTPU vs CPU per architecture
  - LaTeX output ready for the ECCV paper

Input:
    - results/edgebench/raw_results.csv  (from 11.1)

Output:
    - results/T4_edgebench.csv           (full table)
    - results/T4_edgebench.tex           (LaTeX for paper)
    - results/T4_speedup.csv             (EdgeTPU speedup per arch)

Usage:
    python scripts/11.2_build_T4_from_edgebench.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT_CSV = Path('results/edgebench/raw_results.csv')
OUTPUT_DIR = Path('results')

# Display names for the paper
ARCH_DISPLAY = {
    'mobilenetv2': 'MobileNetV2',
    'mobilenetv1': 'MobileNetV1',
    'efficientnet_lite0': 'EffNet-Lite0',
    'efficientnet': 'EffNet-B3',
    'resnet50': 'ResNet-50',
}

# Ordering for presentation
ARCH_ORDER = [
    'mobilenetv2',
    'mobilenetv1',
    'efficientnet_lite0',
    'resnet50',
    'efficientnet',
]
STRATEGY_ORDER = ['hybrid', 'Fuzzy', 'sbert']
BACKEND_ORDER = ['cpu', 'edgetpu']

# Parameter counts for context
ARCH_PARAMS = {
    'mobilenetv2': '2.2M',
    'mobilenetv1': '3.2M',
    'efficientnet_lite0': '3.5M',
    'efficientnet': '10.8M',
    'resnet50': '23.5M',
}


def load_data() -> pd.DataFrame:
    """Load and validate the raw results CSV."""
    if not INPUT_CSV.exists():
        print(f'ERROR: {INPUT_CSV} not found.')
        print('       Run 11.1_fetch_edgebench_results.py first.')
        raise SystemExit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f'Loaded {len(df)} results from {INPUT_CSV}')

    # Validate expected columns
    required = ['architecture', 'strategy', 'backend', 'latency_mean_ms', 'fps']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f'ERROR: Missing columns: {missing}')
        raise SystemExit(1)

    return df


def build_main_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build the main T4 table with one row per arch/strategy/backend."""
    # Group by arch/strategy/backend and take means (in case of duplicates)
    agg = (
        df.groupby(['architecture', 'strategy', 'backend'])
        .agg(
            latency_mean=('latency_mean_ms', 'mean'),
            latency_std=('latency_std_ms', 'mean'),
            latency_p95=('latency_p95_ms', 'mean'),
            fps=('fps', 'mean'),
            model_load_ms=('model_load_ms', 'mean'),
            model_size_mb=('model_size_mb', 'first'),
            cpu_percent=('cpu_percent_mean', 'mean'),
            memory_mb=('memory_mb_mean', 'mean'),
            cpu_temp=('cpu_temp_celsius', 'mean'),
            n_runs=('experiment_id', 'count'),
        )
        .reset_index()
    )

    # Add display names and params
    agg['arch_display'] = agg['architecture'].map(ARCH_DISPLAY)
    agg['params'] = agg['architecture'].map(ARCH_PARAMS)

    # Sort
    agg['_arch_order'] = agg['architecture'].map(
        {a: i for i, a in enumerate(ARCH_ORDER)}
    )
    agg['_strategy_order'] = agg['strategy'].map(
        {s: i for i, s in enumerate(STRATEGY_ORDER)}
    )
    agg['_backend_order'] = agg['backend'].map(
        {b: i for i, b in enumerate(BACKEND_ORDER)}
    )
    agg = agg.sort_values(['_arch_order', '_strategy_order', '_backend_order']).drop(
        columns=['_arch_order', '_strategy_order', '_backend_order']
    )

    return agg


def compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    """Compute EdgeTPU speedup over CPU for each arch/strategy pair."""
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

            cpu_lat = cpu['latency_mean'].values[0]
            tpu_lat = tpu['latency_mean'].values[0]
            cpu_fps = cpu['fps'].values[0]
            tpu_fps = tpu['fps'].values[0]

            records.append(
                {
                    'architecture': arch,
                    'arch_display': ARCH_DISPLAY.get(arch, arch),
                    'strategy': strategy,
                    'cpu_latency_ms': round(cpu_lat, 2),
                    'edgetpu_latency_ms': round(tpu_lat, 2),
                    'speedup_x': round(cpu_lat / tpu_lat, 2) if tpu_lat > 0 else None,
                    'cpu_fps': round(cpu_fps, 1),
                    'edgetpu_fps': round(tpu_fps, 1),
                    'fps_gain_x': (
                        round(tpu_fps / cpu_fps, 2) if cpu_fps > 0 else None
                    ),
                }
            )

    return pd.DataFrame(records)


def _build_latex_rows(table: pd.DataFrame) -> list[dict]:
    """Build row dicts for the LaTeX table."""
    rows = []
    for arch in ARCH_ORDER:
        arch_data = table[table['architecture'] == arch]
        if arch_data.empty:
            continue

        for strategy in STRATEGY_ORDER:
            row = {
                'Architecture': ARCH_DISPLAY.get(arch, arch),
                'Params': ARCH_PARAMS.get(arch, ''),
                'Strategy': strategy,
            }

            for backend in BACKEND_ORDER:
                subset = arch_data[
                    (arch_data['strategy'] == strategy)
                    & (arch_data['backend'] == backend)
                ]
                bk = 'CPU' if backend == 'cpu' else 'TPU'

                if subset.empty:
                    row[f'Lat. {bk} (ms)'] = '--'
                    row[f'FPS {bk}'] = '--'
                else:
                    r = subset.iloc[0]
                    row[f'Lat. {bk} (ms)'] = f'{r["latency_mean"]:.1f}'
                    row[f'FPS {bk}'] = f'{r["fps"]:.0f}'

            rows.append(row)
    return rows


def _add_midrules(latex: str) -> str:
    """Insert midrules between different architectures in LaTeX."""
    lines = latex.split('\n')
    processed = []
    prev_arch = None
    for line in lines:
        for arch_display in ARCH_DISPLAY.values():
            if arch_display in line and prev_arch and prev_arch != arch_display:
                processed.append('\\midrule')
                break
        processed.append(line)
        for arch_display in ARCH_DISPLAY.values():
            if arch_display in line:
                prev_arch = arch_display
                break
    return '\n'.join(processed)


def generate_latex(table: pd.DataFrame, output_path: Path) -> None:
    """Generate a LaTeX table for the ECCV paper."""
    rows = _build_latex_rows(table)
    latex_df = pd.DataFrame(rows)

    latex = latex_df.to_latex(
        index=False,
        escape=True,
        column_format='llc' + 'rr' * 2,
        caption=(
            'Edge deployment performance on Raspberry Pi 4 + Coral USB.'
            ' All models use INT8 post-training quantization.'
            ' Latency = mean inference time (ms); FPS = throughput.'
        ),
        label='tab:edge_performance',
    )

    latex = _add_midrules(latex)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)

    print(f'LaTeX saved: {output_path}')


def main() -> None:
    print('=' * 60)
    print('Step 11.2: Build Table T4 from Edge-Bench')
    print('=' * 60)

    df = load_data()

    # Build main table
    table = build_main_table(df)

    # Save main table
    csv_path = OUTPUT_DIR / 'T4_edgebench.csv'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    print(f'\nMain table saved: {csv_path}')

    # Print main table
    print(f'\n{"=" * 60}')
    print('Table T4: Edge Deployment Performance')
    print(f'{"=" * 60}')
    display_cols = [
        'arch_display',
        'strategy',
        'backend',
        'latency_mean',
        'latency_p95',
        'fps',
        'model_size_mb',
    ]
    available = [c for c in display_cols if c in table.columns]
    print(table[available].to_string(index=False))

    # Compute speedup
    speedup = compute_speedup(table)
    if not speedup.empty:
        speedup_path = OUTPUT_DIR / 'T4_speedup.csv'
        speedup.to_csv(speedup_path, index=False)
        print(f'\nSpeedup table saved: {speedup_path}')

        print(f'\n{"=" * 60}')
        print('EdgeTPU Speedup vs CPU')
        print(f'{"=" * 60}')
        print(
            speedup[
                [
                    'arch_display',
                    'strategy',
                    'cpu_latency_ms',
                    'edgetpu_latency_ms',
                    'speedup_x',
                    'cpu_fps',
                    'edgetpu_fps',
                ]
            ].to_string(index=False)
        )

        # Average speedup per architecture
        print(f'\n{"=" * 60}')
        print('Average Speedup by Architecture')
        print(f'{"=" * 60}')
        avg_speedup = (
            speedup.groupby('arch_display')
            .agg(
                mean_speedup=('speedup_x', 'mean'),
                mean_fps_gain=('fps_gain_x', 'mean'),
            )
            .round(2)
        )
        print(avg_speedup.to_string())

    # Generate LaTeX
    tex_path = OUTPUT_DIR / 'T4_edgebench.tex'
    generate_latex(table, tex_path)

    # Coverage check
    print(f'\n{"=" * 60}')
    print('Coverage Matrix')
    print(f'{"=" * 60}')
    pivot = table.pivot_table(
        index='architecture',
        columns=['strategy', 'backend'],
        values='latency_mean',
        aggfunc='first',
    )
    print(pivot.round(1).to_string())

    expected = len(ARCH_ORDER) * len(STRATEGY_ORDER) * len(BACKEND_ORDER)
    # EfficientNet-B3 has no EdgeTPU
    expected -= len(STRATEGY_ORDER)  # 3 missing edgetpu for efficientnet
    actual = len(table)
    print(f'\nExpected: {expected} combinations, Got: {actual}')
    if actual < expected:
        print(f'  Missing: {expected - actual} experiments')


if __name__ == '__main__':
    main()
