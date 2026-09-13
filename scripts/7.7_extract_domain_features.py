#!/usr/bin/env python3
"""Penultimate-layer features of the source-only, DANN and CDAN+E (fair protocol, seed 42)
ResNet-50 on a balanced sample of PlantVillage test and PlantDoc test images, plus a
domain-separability score (5-fold cross-validated 5-NN accuracy of the domain label
in the 2048-d feature space; 0.5 = domains indistinguishable, 1.0 = perfectly separable).
Feeds paper/mdpi_ai/figures/make_figure3.py.

    python scripts/7.7_extract_domain_features.py --n 500
Output: results/tsne_domain/{model}.npz and results/tsne_domain/summary.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import torch
from torchvision import models

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'ref', HERE / '14.2_eval_pytorch_reference.py'
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
OUT = Path('results/tsne_domain')


def build_fair(method: str, seed: int, num_classes: int) -> torch.nn.Module:
    m = models.resnet50()
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    ck = torch.load(
        ref.MODELS / 'uda_fair' / f'{method}_hybrid_seed{seed}.pt',
        map_location='cpu',
        weights_only=False,
    )
    sd = {k: v for k, v in ck['backbone'].items() if not k.startswith('fc.')}
    clf = ck['classifier']
    sd['fc.weight'] = clf[[k for k in clf if k.endswith('weight')][0]]
    sd['fc.bias'] = clf[[k for k in clf if k.endswith('bias')][0]]
    m.load_state_dict(sd)
    return m.eval()


@torch.no_grad()
def features(m: torch.nn.Module, paths: list[str], device: str) -> np.ndarray:
    m.fc = torch.nn.Identity()
    out = []
    for i in range(0, len(paths), 64):
        x = torch.stack(
            [ref.TF(Image.open(p).convert('RGB')) for p in paths[i : i + 64]]
        )
        out.append(m(x.to(device)).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500, help='images per domain')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(args.seed)
    src = (
        pd.read_csv(ref.SPLITS / 'hybrid' / 'pv_test.csv')
        .sample(args.n, random_state=rng)
        .reset_index(drop=True)
    )
    tgt = (
        pd.read_csv(ref.SPLITS / 'hybrid' / 'target_plantdoc_test.csv')
        .sample(args.n, random_state=rng)
        .reset_index(drop=True)
    )
    summary = {'n_per_domain': args.n, 'seed': args.seed, 'knn_k': 5, 'models': {}}
    for name in ('source_only', 'dann42', 'cdan42'):
        if name == 'source_only':
            m = ref.build('resnet50', 'hybrid', 18)
        else:
            m = build_fair(name[:-2], 42, 18)
        m = m.to(device)
        xs, xt = (
            features(m, src.path.tolist(), device),
            features(m, tgt.path.tolist(), device),
        )
        X = np.concatenate([xs, xt])
        d = np.r_[np.zeros(len(xs)), np.ones(len(xt))]
        acc = cross_val_score(
            KNeighborsClassifier(5),
            X,
            d,
            cv=StratifiedKFold(5, shuffle=True, random_state=args.seed),
        ).mean()
        np.savez(
            OUT / f'{name}.npz',
            X_src=xs,
            X_tgt=xt,
            y_src=src.label_id.values,
            y_tgt=tgt.label_id.values,
        )
        summary['models'][name] = {'knn_domain_acc': float(acc)}
        print(f'{name:12s} 5-NN domain accuracy = {acc:.3f}')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('[OK]', OUT)


if __name__ == '__main__':
    main()
