#!/usr/bin/env python3
"""Figure 3 for the MDPI AI manuscript: does adaptation bring PlantDoc features onto the
PlantVillage manifold? Joint two-dimensional t-SNE of source (PlantVillage test) and
target (PlantDoc test) penultimate features, 500 images per domain, for the source-only
model and the fair-protocol DANN model (seed 42); panel titles carry the 5-NN domain
separability from results/tsne_domain/summary.json.

    python paper/mdpi_ai/figures/make_figure3.py   (needs matplotlib, scikit-learn)
Inputs: results/tsne_domain/{source_only,dann42}.npz (scripts/7.7).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

RES = Path('results/tsne_domain')
OUT = Path('paper/mdpi_ai/figures')
DPI = 600
PANELS = [
    ('source_only', '(a) Source-only ResNet-50'),
    ('dann42', '(b) DANN, fair protocol (seed 42)'),
]


def main() -> None:
    summ = json.load(open(RES / 'summary.json'))['models']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, (name, title) in zip(axes, PANELS):
        d = np.load(RES / f'{name}.npz')
        X = np.concatenate([d['X_src'], d['X_tgt']])
        z = TSNE(
            n_components=2,
            perplexity=30,
            init='pca',
            learning_rate='auto',
            random_state=42,
        ).fit_transform(X)
        n = len(d['X_src'])
        ax.scatter(
            z[:n, 0],
            z[:n, 1],
            s=9,
            c='#9aa5b1',
            label='PlantVillage (source), n = 500',
            linewidths=0,
        )
        ax.scatter(
            z[n:, 0],
            z[n:, 1],
            s=9,
            c='#d1495b',
            label='PlantDoc (target), n = 500',
            linewidths=0,
        )
        acc = summ[name]['knn_domain_acc']
        ax.set_title(f'{title}\n5-NN domain separability = {acc:.2f}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color('#888')
    axes[0].legend(loc='lower left', fontsize=8, frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / 'figure3_tsne_hybrid.png', dpi=DPI, metadata={'Software': None})
    print(
        '[OK] figure3_tsne_hybrid.png',
        {k: round(v['knn_domain_acc'], 3) for k, v in summ.items()},
    )


if __name__ == '__main__':
    main()
