import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# =========================
# Paths
# =========================
RESULTS = Path('results')
FIGURES = Path('figures')
FIGURES.mkdir(exist_ok=True)


def plot_tsne(X, y, title, out_path):
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        init='pca',
        learning_rate='auto',
        random_state=42,
    )
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=y,
        cmap='tab20',
        s=8,
        alpha=0.7,
    )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--target', default='plantdoc')
    parser.add_argument('--lambda_coral', type=float, default=0.1)
    args = parser.parse_args()

    strategy = args.strategy
    lam = args.lambda_coral

    baseline_npz = RESULTS / f'tsne_features_baseline_{strategy}.npz'
    coral_npz = RESULTS / f'tsne_features_coral_{strategy}.npz'

    if not baseline_npz.exists():
        raise FileNotFoundError(f'Missing baseline features: {baseline_npz}')
    if not coral_npz.exists():
        raise FileNotFoundError(f'Missing CORAL features: {coral_npz}')

    base = np.load(baseline_npz)
    coral = np.load(coral_npz)

    plot_tsne(
        base['X'],
        base['y'],
        title=f'Baseline (PV → Target, {strategy})',
        out_path=FIGURES / f'F3_tsne_baseline_{strategy}.png',
    )

    plot_tsne(
        coral['X'],
        coral['y'],
        title=f'CORAL λ={lam} ({strategy})',
        out_path=FIGURES / f'F3_tsne_coral_{strategy}.png',
    )

    print('[OK] t-SNE figures saved:')
    print(f' - figures/F3_tsne_baseline_{strategy}.png')
    print(f' - figures/F3_tsne_coral_{strategy}.png')


if __name__ == '__main__':
    main()
