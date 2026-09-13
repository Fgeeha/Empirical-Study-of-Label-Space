"""Evaluate fair-protocol UDA checkpoints on PlantWild v2.

Uses checkpoints from models/uda_fair/{method}_{strategy}_seed{seed}.pt
(saved by the updated train_uda.py with source-val selection).

Usage:
  python scripts/evaluate_plantwild.py --method cdan --seeds 42 43 44
  python scripts/evaluate_plantwild.py --method dann  --seeds 42 43 44
  python scripts/evaluate_plantwild.py --method coral --seeds 42 43 44
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from experiments.E5b_plantwild.run_eval import (  # noqa: E402
    PathDataset,
    evaluate,
    extract_images,
)

MODELS_FAIR = ROOT / 'models' / 'uda_fair'
RESULTS_DIR = ROOT / 'results' / 'uda_benchmark' / 'plantwild'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_fair_checkpoint(ckpt_path: Path, device: str):
    """Load a fair-protocol checkpoint into ResNet-50."""
    import torch.nn as nn
    from torchvision import models

    model = models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 18)
    state = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    if 'backbone' in state and 'classifier' in state:
        merged = {
            **state['backbone'],
            'fc.weight': state['classifier']['weight'],
            'fc.bias': state['classifier']['bias'],
        }
        model.load_state_dict(merged)
    elif 'backbone' in state:
        # Some methods save backbone separately without classifier key
        backbone_state = state['backbone']
        # Rebuild: backbone has Identity fc, we need to restore proper fc
        model_bb = models.resnet50()
        model_bb.fc = nn.Identity()
        model_bb.load_state_dict(backbone_state)
        # Classifier not found — skip (would need to reconstruct from classifier key)
        raise ValueError(f'Checkpoint {ckpt_path} missing classifier key')
    else:
        model.load_state_dict(state)
    return model.eval().to(device)


def run_inference_model(model, records, device):
    dataset = PathDataset(records)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)
    model = model.to(device)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    import numpy as np

    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True)
    parser.add_argument('--strategy', default='hybrid')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(
        f'[INFO] method={args.method} strategy={args.strategy} seeds={args.seeds} device={device}'
    )

    records = extract_images()

    results = {}
    for seed in args.seeds:
        ckpt = MODELS_FAIR / f'{args.method}_{args.strategy}_seed{seed}.pt'
        if not ckpt.exists():
            print(f'  MISSING: {ckpt}')
            continue
        print(f'\n  Evaluating seed={seed} ({ckpt.name})...')
        try:
            model = load_fair_checkpoint(ckpt, device)
        except ValueError as e:
            print(f'  SKIP: {e}')
            continue
        preds, labels = run_inference_model(model, records, device)
        metrics = evaluate(preds, labels)
        results[seed] = metrics
        print(
            f'  seed={seed}: macro_f1={metrics["macro_f1"]:.4f}  acc={metrics["accuracy"]:.4f}'
        )

    if not results:
        print(
            'No checkpoints found. Run train_uda.py first (updated script saves checkpoints).'
        )
        return

    f1_vals = [v['macro_f1'] for v in results.values()]
    mean_f1 = float(np.mean(f1_vals))
    std_f1 = float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0

    print(f'\n=== {args.method} on PlantWild v2 (fair protocol) ===')
    for seed, m in results.items():
        print(f'  seed={seed}: F1={m["macro_f1"]:.4f}')
    print(f'  mean±std: {mean_f1:.4f}±{std_f1:.4f}  (n={len(f1_vals)})')

    out = RESULTS_DIR / f'{args.method}_{args.strategy}_plantwild.json'
    out.write_text(
        json.dumps(
            {
                'method': args.method,
                'strategy': args.strategy,
                'protocol': 'fair_source_val_selection',
                'n_classes': 17,
                'n_samples': len(records),
                'per_seed': {str(k): v for k, v in results.items()},
                'macro_f1_mean': mean_f1,
                'macro_f1_std': std_f1,
                'n_seeds': len(f1_vals),
            },
            indent=2,
        )
    )
    print(f'\nSaved {out}')


if __name__ == '__main__':
    main()
