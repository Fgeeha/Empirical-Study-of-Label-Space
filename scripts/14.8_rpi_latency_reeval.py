#!/usr/bin/env python3
"""Latency re-measurement of the verified INT8 models on the Raspberry Pi 4 +
Coral USB Edge TPU, same protocol as T4 (edge-bench agent benchmark_tflite.py:
20 warm-up + 200 timed invocations, batch 1, 4 threads, fixed synthetic input,
seed 42). Runs on the device next to edge-bench-agent/benchmark_tflite.py.

    python3 14.8_rpi_latency_reeval.py --models-dir ~/eccv_reeval_20260913 --out results.json

Files: <name>.tflite (CPU) and <name>_edgetpu.tflite (Edge TPU) per model.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path.home() / 'edge-bench-agent'))
from benchmark_tflite import run_benchmark  # noqa: E402

MODELS = [
    'mobilenetv1_hybrid_full_integer_quant',
    'mobilenetv2_hybrid_full_integer_quant',
    'efficientnet_lite0_int8_ptq_hybrid',
    'resnet50_hybrid_full_integer_quant',
    'dann42_hybrid_full_integer_quant',
    'efficientnet_b3_hybrid_full_integer_quant',
]


class Args:
    pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--models-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--runs', type=int, default=200)
    ap.add_argument('--models', help='comma-separated subset of MODELS')
    ap.add_argument('--backend', choices=['cpu', 'edgetpu', 'both'], default='both')
    a = ap.parse_args()
    models = a.models.split(',') if a.models else MODELS
    d = Path(a.models_dir).expanduser()
    out = []
    for name in models:
        for backend, fname in (('cpu', f'{name}.tflite'), ('edgetpu', f'{name}_edgetpu.tflite')):
            if a.backend != 'both' and backend != a.backend:
                continue
            path = d / fname
            if not path.exists():
                print(f'[SKIP] {fname}')
                continue
            args = Args()
            args.model = str(path)
            args.backend = backend
            args.threads = 4
            args.warmup = a.warmup
            args.runs = a.runs
            args.seed = 42
            r = run_benchmark(args)
            r['reeval'] = {'model_name': name, 'backend': backend, 'date': datetime.now(UTC).isoformat()}
            if r.get('status') == 'completed':
                lat = r['latency']
                print(f'{name:44s} {backend:7s} mean {lat["mean_ms"]:8.2f} ms  std {lat["std_ms"]:.3f}  p95 {lat["p95_ms"]:.2f}  fps {r["throughput"]["fps"]:.1f}')
            else:
                print(f'{name:44s} {backend:7s} FAILED {r.get("error")}')
            out.append(r)
            Path(a.out).write_text(json.dumps(out, indent=1))
    print('[OK]', a.out)


if __name__ == '__main__':
    main()
