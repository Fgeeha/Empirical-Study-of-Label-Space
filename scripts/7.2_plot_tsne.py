import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

RESULTS = Path('results')
FIGURES = Path('figures')
FIGURES.mkdir(exist_ok=True)


def plot(name):
    fpath = RESULTS / f'tsne_features_{name}.npz'
    if not fpath.exists():
        print(f'[WARN] File not found: {fpath}')
        return False

    d = np.load(fpath)
    X, y = d['X'], d['y']

    Z = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

    plt.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab20', s=5)
    plt.title(name)
    plt.axis('off')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    strategy = args.strategy

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    has_baseline = plot(f'baseline_{strategy}')

    plt.subplot(1, 2, 2)
    has_coral = plot(f'coral_{strategy}')

    if has_baseline or has_coral:
        plt.tight_layout()
        out = FIGURES / f'F3_tsne_{strategy}.png'
        plt.savefig(out, dpi=300)
        print(f'[OK] F3 saved: {out}')
    else:
        print('[WARN] No t-SNE features found, skipping plot')

    plt.close()


if __name__ == '__main__':
    main()
