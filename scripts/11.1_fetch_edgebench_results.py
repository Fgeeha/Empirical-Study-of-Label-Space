#!/usr/bin/env python3
"""
Step 11.1: Fetch experiment results from Edge-Bench server.

Downloads completed benchmark results via the Edge-Bench REST API
and saves them locally as CSV and JSON for offline analysis.

Output:
    - results/edgebench/raw_results.json   (full metrics, all experiments)
    - results/edgebench/raw_results.csv    (flattened, one row per experiment)

Usage:
    python scripts/11.1_fetch_edgebench_results.py
    python scripts/11.1_fetch_edgebench_results.py --server http://192.168.1.4:8000
    python scripts/11.1_fetch_edgebench_results.py --server http://192.168.1.4:8000 --only-completed
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import requests

DEFAULT_SERVER = 'http://192.168.1.4:8000'
OUTPUT_DIR = Path('results/edgebench')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fetch results from Edge-Bench server')
    p.add_argument('--server', default=DEFAULT_SERVER, help='Edge-Bench server URL')
    p.add_argument(
        '--only-completed',
        action='store_true',
        help='Only fetch completed experiments',
    )
    p.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DIR,
        help='Output directory',
    )
    p.add_argument('--limit', type=int, default=1000, help='Max results to fetch')
    return p.parse_args()


def check_server(server: str) -> bool:
    """Verify server is reachable."""
    try:
        resp = requests.get(f'{server}/api/devices', timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def fetch_results_json(server: str, limit: int = 1000) -> list[dict]:
    """Fetch all results from the API with full metrics."""
    resp = requests.get(f'{server}/api/results', params={'limit': limit}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_experiments(server: str, limit: int = 1000) -> list[dict]:
    """Fetch all experiments."""
    resp = requests.get(
        f'{server}/api/experiments', params={'limit': limit}, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def flatten_result(result: dict) -> dict:
    """Flatten a single result record into a flat dict for CSV."""
    metrics = result.get('metrics') or {}
    params = result.get('params') or metrics.get('params', {})
    latency = metrics.get('latency', {})
    throughput = metrics.get('throughput', {})
    cold_start = metrics.get('cold_start', {})
    system = metrics.get('system', {})
    model_info = metrics.get('model', {})
    device_info = metrics.get('device_info', {})

    # Parse architecture and strategy from model name
    # e.g. mobilenetv2_int8_ptq_hybrid.tflite -> arch=mobilenetv2, strategy=hybrid
    model_name = result.get('model_name', '')
    arch, strategy = _parse_model_name(model_name)

    return {
        'experiment_id': result.get('experiment_id', ''),
        'experiment_name': result.get('experiment_name', ''),
        'model_name': model_name,
        'architecture': arch,
        'strategy': strategy,
        'backend': params.get('backend', 'cpu'),
        'device_name': result.get('device_name', ''),
        # Latency
        'latency_mean_ms': latency.get('mean_ms'),
        'latency_std_ms': latency.get('std_ms'),
        'latency_min_ms': latency.get('min_ms'),
        'latency_max_ms': latency.get('max_ms'),
        'latency_p50_ms': latency.get('p50_ms'),
        'latency_p90_ms': latency.get('p90_ms'),
        'latency_p95_ms': latency.get('p95_ms'),
        'latency_p99_ms': latency.get('p99_ms'),
        # Throughput
        'fps': throughput.get('fps'),
        # Cold start
        'model_load_ms': cold_start.get('model_load_ms'),
        'first_inference_ms': cold_start.get('first_inference_ms'),
        # System
        'cpu_percent_mean': system.get('cpu_percent_mean'),
        'cpu_percent_max': system.get('cpu_percent_max'),
        'memory_mb_mean': system.get('memory_mb_mean'),
        'cpu_temp_celsius': system.get('cpu_temp_celsius'),
        # Model info
        'model_size_bytes': model_info.get('size_bytes'),
        'model_size_mb': (
            round(model_info['size_bytes'] / 1024 / 1024, 2)
            if model_info.get('size_bytes')
            else None
        ),
        'quantization': model_info.get('quantization'),
        # Device
        'hostname': device_info.get('hostname'),
        'tpu_detected': device_info.get('tpu_detected'),
        # Params
        'benchmark_runs': params.get('benchmark_runs'),
        'warmup_runs': params.get('warmup_runs'),
        'num_threads': params.get('num_threads'),
        # Meta
        'duration_seconds': metrics.get('duration_seconds'),
        'timestamp': metrics.get('timestamp'),
    }


def _parse_model_name(model_name: str) -> tuple[str, str]:
    """Extract architecture and strategy from model filename.

    Examples:
        mobilenetv2_int8_ptq_hybrid.tflite -> ('mobilenetv2', 'hybrid')
        efficientnet_lite0_int8_ptq_Fuzzy_edgetpu.tflite -> ('efficientnet_lite0', 'Fuzzy')
        resnet50_int8_ptq_sbert.tflite -> ('resnet50', 'sbert')
    """
    name = model_name.replace('.tflite', '').replace('_edgetpu', '')

    # Known architectures (order matters: longer first)
    archs = [
        'efficientnet_lite0',
        'efficientnet',
        'mobilenetv2',
        'mobilenetv1',
        'resnet50',
    ]

    for arch in archs:
        if name.startswith(arch):
            # Remove arch prefix and _int8_ptq_ to get strategy
            rest = name[len(arch) :]
            rest = rest.replace('_int8_ptq_', '')
            if rest.startswith('_'):
                rest = rest[1:]
            strategy = rest if rest else 'unknown'
            return arch, strategy

    return model_name, 'unknown'


def main() -> None:
    args = parse_args()
    server = args.server.rstrip('/')
    output_dir = args.output_dir

    print('=' * 60)
    print('Step 11.1: Fetch Edge-Bench Results')
    print('=' * 60)
    print(f'Server:  {server}')
    print(f'Output:  {output_dir}')
    print()

    # Check server
    if not check_server(server):
        print(f'ERROR: Cannot connect to {server}')
        print('       Make sure the edge-bench server is running.')
        raise SystemExit(1)
    print('Server: OK')

    # Fetch results
    print('Fetching results...')
    results = fetch_results_json(server, limit=args.limit)
    print(f'  Total results: {len(results)}')

    if not results:
        print('No results found. Run experiments first.')
        raise SystemExit(0)

    # Filter completed only if requested
    if args.only_completed:
        results = [r for r in results if r.get('metrics')]
        print(f'  With metrics (completed): {len(results)}')

    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw JSON
    json_path = output_dir / 'raw_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f'  Saved: {json_path}')

    # Flatten and save CSV
    flat_records = []
    for r in results:
        if r.get('metrics'):  # Only include results with actual data
            flat_records.append(flatten_result(r))

    if flat_records:
        import pandas as pd

        df = pd.DataFrame(flat_records)

        # Sort for readability
        df = df.sort_values(['architecture', 'strategy', 'backend'], ignore_index=True)

        csv_path = output_dir / 'raw_results.csv'
        df.to_csv(csv_path, index=False)
        print(f'  Saved: {csv_path}')

        # Print summary
        print(f'\n{"=" * 60}')
        print('Summary')
        print(f'{"=" * 60}')
        print(f'  Architectures: {sorted(df["architecture"].unique())}')
        print(f'  Strategies:    {sorted(df["strategy"].unique())}')
        print(f'  Backends:      {sorted(df["backend"].unique())}')
        print(f'  Devices:       {sorted(df["device_name"].dropna().unique())}')
        print(f'  Total rows:    {len(df)}')
        print()

        # Compact table
        summary = df.groupby(['architecture', 'backend']).agg(
            count=('experiment_id', 'count'),
            avg_latency=('latency_mean_ms', 'mean'),
            avg_fps=('fps', 'mean'),
        )
        print(summary.round(2).to_string())
    else:
        print('  WARNING: No completed results with metrics found.')

    # Also fetch experiment list for cross-reference
    print('\nFetching experiment list...')
    experiments = fetch_experiments(server, limit=args.limit)
    exp_path = output_dir / 'experiments.json'
    with open(exp_path, 'w') as f:
        json.dump(experiments, f, indent=2, ensure_ascii=False, default=str)
    print(f'  Saved: {exp_path}')

    # Stats
    status_counts = {}
    for e in experiments:
        s = e.get('status', 'unknown')
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f'  Experiments by status: {status_counts}')

    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    print(f'\nDone at {ts}')


if __name__ == '__main__':
    main()
