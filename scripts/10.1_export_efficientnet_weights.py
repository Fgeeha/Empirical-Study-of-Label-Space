"""
10.1 Export EfficientNet-B3 PyTorch weights to NumPy (.npz).

Reads the trained checkpoint from models/efficientnet_pv_{strategy}_best.pt
and saves all state_dict tensors as export/efficientnet_{strategy}_weights.npz

Usage:
    poetry run python scripts/10.1_export_efficientnet_weights.py --strategy hybrid
    poetry run python scripts/10.1_export_efficientnet_weights.py --strategy Fuzzy
    poetry run python scripts/10.1_export_efficientnet_weights.py --strategy sbert
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import models

MODELS = Path('models')
EXPORT = Path('export')
EXPORT.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Export EfficientNet-B3 PyTorch weights to .npz'
    )
    parser.add_argument(
        '--strategy',
        required=True,
        choices=['hybrid', 'Fuzzy', 'sbert'],
        help='Label alignment strategy',
    )
    args = parser.parse_args()

    strategy = args.strategy

    model_path = MODELS / f'efficientnet_pv_{strategy}_best.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'Missing model checkpoint: {model_path}')

    # --- Load Torch model ---
    ckpt = torch.load(model_path, map_location='cpu')

    num_classes = ckpt['classifier.1.weight'].shape[0]

    model = models.efficientnet_b3(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    model.load_state_dict(ckpt)
    model.eval()

    # --- Export weights ---
    weights = {}
    for k, v in model.state_dict().items():
        weights[k] = v.detach().cpu().numpy()

    out = EXPORT / f'efficientnet_{strategy}_weights.npz'
    np.savez(out, **weights)

    print(f'[OK] EfficientNet-B3 Torch weights exported -> {out}')
    print(f'     num_classes = {num_classes}')
    print(f'     keys = {len(weights)}')


if __name__ == '__main__':
    main()
