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
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    strategy = args.strategy
    model_path = MODELS / f'resnet50_pv_{strategy}_best.pt'

    model = models.resnet50(weights=None)
    num_classes = torch.load(model_path, map_location='cpu')['fc.weight'].shape[0]
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    weights = {}
    for k, v in model.state_dict().items():
        weights[k] = v.numpy()

    out = EXPORT / f'resnet50_{strategy}_weights.npz'
    np.savez(out, **weights)

    print(f'[OK] Torch weights exported → {out}')


if __name__ == '__main__':
    main()
