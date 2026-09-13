#!/usr/bin/env python3
"""Build the re-measured edge table (Table 5 of the MDPI manuscript) from the
per-run JSON files produced on the Raspberry Pi by scripts/14.8_rpi_latency_reeval.py
(fetched into results/edgebench/reeval_20260913/runs/).

Outputs:
  results/final/T4_system_reeval.csv   (same columns as T4_system.csv, Hybrid only)
  results/final/T5_reeval.md           (mean / std / p95 / FPS / size per model and backend)

    python3 scripts/14.9_build_T5_reeval.py
"""

import csv
import json
from pathlib import Path

RUNS = Path('results/edgebench/reeval_20260913/runs')
OUT_CSV = Path('results/final/T4_system_reeval.csv')
OUT_MD = Path('results/final/T5_reeval.md')
MODELS = [
    ('mobilenetv1_hybrid_full_integer_quant', 'MobileNetV1', '3.2M'),
    ('mobilenetv2_hybrid_full_integer_quant', 'MobileNetV2', '2.2M'),
    ('efficientnet_lite0_int8_ptq_hybrid', 'EffNet-Lite0', '3.5M'),
    ('resnet50_hybrid_full_integer_quant', 'ResNet-50', '23.5M'),
    ('dann42_hybrid_full_integer_quant', 'ResNet-50 DANN', '23.5M'),
    ('efficientnet_b3_hybrid_full_integer_quant', 'EffNet-B3', '10.8M'),
]


def main() -> None:
    rows, md = [], []
    md.append('| Model | Backend | mean, ms | std, ms | p95, ms | FPS | size, MiB | runs | temp, °C |')
    md.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for key, name, params in MODELS:
        for backend in ('cpu', 'edgetpu'):
            p = RUNS / f'{key}__{backend}.json'
            if not p.exists():
                continue
            r = json.loads(p.read_text())[0]
            if r.get('status') != 'completed':
                md.append(f'| {name} | {backend} | failed: {r.get("error")} | | | | | | |')
                continue
            lat, sy = r['latency'], r['system']
            size_mib = r['model']['size_bytes'] / 2**20
            rows.append(
                {
                    'Model': name,
                    'Strategy': 'hybrid',
                    'Backend': backend.upper(),
                    'Latency (ms)': round(lat['mean_ms'], 1),
                    'FPS': round(r['throughput']['fps'], 1),
                    'CPU (%)': sy.get('cpu_percent_mean', ''),
                    'RAM (MB)': sy.get('memory_mb_mean', ''),
                    'Temp (C)': round(sy['cpu_temp_celsius'], 1),
                    'Size (MB)': round(size_mib, 2),
                    'Params': params,
                }
            )
            md.append(
                f'| {name} | {backend} | {lat["mean_ms"]:.2f} | {lat["std_ms"]:.3f} | {lat["p95_ms"]:.2f} | '
                f'{r["throughput"]["fps"]:.1f} | {size_mib:.2f} | {r["params"]["benchmark_runs"]} | {sy["cpu_temp_celsius"]} |'
            )
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    OUT_MD.write_text('\n'.join(md) + '\n')
    print('\n'.join(md))
    print(f'[OK] {OUT_CSV}, {OUT_MD}')


if __name__ == '__main__':
    main()
