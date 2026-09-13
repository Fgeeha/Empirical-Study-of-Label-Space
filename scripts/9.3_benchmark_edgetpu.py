#!/usr/bin/env python3
"""
CPU-only benchmarking using PyTorch.

This is a fallback when TFLite/EdgeTPU is not available.
Measures inference performance of PyTorch models on CPU.

Usage:
  python 9.3_benchmark_cpu_pytorch.py --strategy hybrid --model mobilenet
"""

import argparse
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import models, transforms

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print('[WARN] psutil not installed - CPU/RAM monitoring disabled')

SPLITS = Path('splits')
MODELS = Path('models')
RESULTS = Path('results')

RESULTS.mkdir(exist_ok=True)


class CSVDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.tf = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        return self.tf(img), int(row['label_id'])


def get_temperature():
    """Read SoC temperature (Raspberry Pi only)."""
    try:
        temp_file = Path('/sys/class/thermal/thermal_zone0/temp')
        if temp_file.exists():
            temp = int(temp_file.read_text()) / 1000.0
            return temp
    except Exception:
        pass
    return None


def load_model(model_name, num_classes, ckpt_path):
    """Load PyTorch model."""
    if model_name == 'resnet50':
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet':
        model = models.efficientnet_b3()
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    elif model_name == 'mobilenet':
        model = models.mobilenet_v2()
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f'Unknown model: {model_name}')

    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    return model.eval()


def benchmark_pytorch(model, dataset, n_runs=100, warmup=10):
    """Run CPU benchmark on PyTorch model."""

    # Force CPU
    device = torch.device('cpu')
    model = model.to(device)

    # Limit samples
    if len(dataset) > n_runs:
        indices = np.random.RandomState(42).choice(len(dataset), n_runs, replace=False)
    else:
        indices = range(len(dataset))

    # Warmup
    print(f'[INFO] Warmup: {warmup} iterations')
    for i in range(min(warmup, len(indices))):
        idx = indices[i]
        x, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model(x)

    # Benchmark
    print(f'[INFO] Running {len(indices)} inferences...')

    latencies = []
    cpu_usage = []
    ram_usage = []
    temps = []

    for idx in indices:
        x, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)

        # CPU/RAM before
        if HAS_PSUTIL:
            cpu_before = psutil.cpu_percent(interval=None)
            ram_before = psutil.virtual_memory().percent

        # Inference
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000.0)  # ms

        # CPU/RAM after
        if HAS_PSUTIL:
            cpu_after = psutil.cpu_percent(interval=None)
            ram_after = psutil.virtual_memory().percent
            cpu_usage.append((cpu_before + cpu_after) / 2)
            ram_usage.append((ram_before + ram_after) / 2)

        # Temperature
        temp = get_temperature()
        if temp is not None:
            temps.append(temp)

    # Compute stats
    latencies = np.array(latencies)

    stats = {
        'device': 'cpu_pytorch',
        'model': model.__class__.__name__,
        'n_samples': len(latencies),
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_median_ms': float(np.median(latencies)),
        'latency_p95_ms': float(np.percentile(latencies, 95)),
        'latency_p99_ms': float(np.percentile(latencies, 99)),
        'fps': float(1000.0 / np.mean(latencies)),
        'cpu_mean_pct': float(np.mean(cpu_usage)) if cpu_usage else None,
        'ram_mean_pct': float(np.mean(ram_usage)) if ram_usage else None,
        'temp_mean_c': float(np.mean(temps)) if temps else None,
        'temp_max_c': float(np.max(temps)) if temps else None,
        'platform': platform.machine(),
    }

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument(
        '--model', choices=['resnet50', 'efficientnet', 'mobilenet'], required=True
    )
    parser.add_argument('--target', default='plantdoc')
    parser.add_argument('--n_runs', type=int, default=100)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    strategy = args.strategy
    model_name = args.model

    # Adjust model name for checkpoint lookup
    if model_name == 'mobilenet':
        ckpt_name = f'mobilenet_baseline_pv_{strategy}_best.pt'
    else:
        ckpt_name = f'{model_name}_pv_{strategy}_best.pt'

    ckpt_path = MODELS / ckpt_name
    test_csv = SPLITS / strategy / f'target_{args.target}_test.csv'

    if not ckpt_path.exists():
        raise FileNotFoundError(f'Model checkpoint not found: {ckpt_path}')
    if not test_csv.exists():
        raise FileNotFoundError(f'Test CSV not found: {test_csv}')

    print(f'[INFO] Benchmarking: {model_name} on {strategy}')
    print(f'       Platform: {platform.machine()}')
    print('       Device: CPU (PyTorch)')

    # Load data
    dataset = CSVDataset(test_csv)
    num_classes = dataset.df['label_id'].nunique()

    # Load model
    model = load_model(model_name, num_classes, ckpt_path)

    # Benchmark
    stats = benchmark_pytorch(model, dataset, n_runs=args.n_runs)

    # Save
    df = pd.DataFrame([stats])
    df.to_csv(args.output, index=False)

    print(f'\n[OK] Benchmark results saved → {args.output}')
    print(f'     Latency (mean): {stats["latency_mean_ms"]:.2f} ms')
    print(f'     FPS: {stats["fps"]:.1f}')
    if stats['cpu_mean_pct']:
        print(f'     CPU: {stats["cpu_mean_pct"]:.1f}%')
    if stats['temp_mean_c']:
        print(f'     Temp: {stats["temp_mean_c"]:.1f}°C')


if __name__ == '__main__':
    main()
