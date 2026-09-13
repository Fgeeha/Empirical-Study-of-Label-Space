#!/usr/bin/env python3
"""Figure 3 for the MDPI AI manuscript: t-SNE of ResNet-50 penultimate features on
the PlantDoc evaluation set (Hybrid), source-only vs CORAL λ = 0.1, 600 dpi.

    python paper/mdpi_ai/figures/make_figure3.py  (needs matplotlib, scikit-learn)
Inputs: results/tsne_features_{baseline,coral}_hybrid.npz (scripts/7.1).
"""

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

RESULTS = Path('results')
OUT = Path('paper/mdpi_ai/figures')
DPI = 600


def embed(name: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(RESULTS / f'tsne_features_{name}_hybrid.npz')
    z = TSNE(
        n_components=2, perplexity=30, init='pca', learning_rate='auto', random_state=42
    ).fit_transform(d['X'])
    return z, d['y']


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, title in zip(
        axes, ('baseline', 'coral'), ('(a) Source-only', '(b) CORAL, λ = 0.1')
    ):
        z, y = embed(name)
        ax.scatter(z[:, 0], z[:, 1], c=y, cmap='tab20', s=5)
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(OUT / 'figure3_tsne_hybrid.png', dpi=DPI)
    print('[OK] figure3_tsne_hybrid.png')


if __name__ == '__main__':
    main()
