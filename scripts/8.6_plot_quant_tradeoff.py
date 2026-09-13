import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path('results')
FIGURES = Path('figures')
FIGURES.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    args = parser.parse_args()

    strategy = args.strategy

    csv_path = RESULTS / f'T3_quant_fp32_vs_int8_{strategy}.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f'Missing {csv_path}. Run 8.5 first.')

    df = pd.read_csv(csv_path)

    # T3: latency + size only (accuracy excluded if eval pipeline unreliable)
    df_tgt = df[df['domain'] == 'target']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # -----------------------------------
    # Latency vs Model Size (FP32 vs INT8)
    # -----------------------------------
    for _, r in df_tgt.iterrows():
        axes[0].scatter(
            r['size_mb'],
            r['latency_ms'],
            label=r['model'],
            s=80,
        )
        axes[0].annotate(
            r['model'].upper(),
            (r['size_mb'], r['latency_ms']),
            textcoords='offset points',
            xytext=(5, 5),
            fontsize=9,
        )

    axes[0].set_xlabel('Model size (MB)')
    axes[0].set_ylabel('Latency (ms, CPU)')
    axes[0].set_title('Latency vs Model Size')
    axes[0].grid(True)

    # -----------------------------------
    # Bar: Latency comparison
    # -----------------------------------
    models = df_tgt['model'].tolist()
    latencies = df_tgt['latency_ms'].tolist()
    sizes = df_tgt['size_mb'].tolist()
    x = range(len(models))
    axes[1].bar([i - 0.2 for i in x], latencies, 0.4, label='Latency (ms)')
    ax2 = axes[1].twinx()
    ax2.bar(
        [i + 0.2 for i in x], sizes, 0.4, label='Size (MB)', color='orange', alpha=0.7
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in models])
    axes[1].set_ylabel('Latency (ms)')
    ax2.set_ylabel('Size (MB)')
    axes[1].set_title('FP32 vs INT8: Latency and Size')
    axes[1].grid(True, axis='y')

    plt.suptitle(
        f'Quantization: Latency and Size (Strategy: {strategy})',
        fontsize=12,
    )

    out = FIGURES / f'F4_quant_tradeoff_{strategy}.png'
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()

    print(f'[OK] Figure saved → {out}')


if __name__ == '__main__':
    main()
