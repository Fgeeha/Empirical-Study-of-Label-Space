import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision import models

MODELS = Path('models')
EXPORT = Path('export')
EXPORT.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strategy',
        required=True,
        choices=['hybrid', 'Fuzzy', 'sbert'],
        help='Label alignment strategy',
    )
    args = parser.parse_args()

    strategy = args.strategy

    model_path = MODELS / f'mobilenetv2_pv_{strategy}_best.pt'
    if not model_path.exists():
        raise FileNotFoundError(f'Missing model checkpoint: {model_path}')

    # --- Load Torch model ---
    ckpt = torch.load(model_path, map_location='cpu')

    num_classes = ckpt['classifier.1.weight'].shape[0]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(
        model.classifier[1].in_features,
        num_classes,
    )

    model.load_state_dict(ckpt)
    model.eval()

    # --- Export weights ---
    weights = {}
    for k, v in model.state_dict().items():
        weights[k] = v.detach().cpu().numpy()

    out = EXPORT / f'mobilenetv2_{strategy}_weights.npz'
    np.savez(out, **weights)

    print(f'[OK] MobileNetV2 Torch weights exported → {out}')


if __name__ == '__main__':
    main()
