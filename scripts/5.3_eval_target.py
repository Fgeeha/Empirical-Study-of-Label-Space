import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

RESULTS = Path('results')
FIGURES = Path('figures')

FIGURES.mkdir(exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()

    df = pd.read_csv(RESULTS / f'preds_target_{args.model}_{args.strategy}.csv')

    acc = accuracy_score(df['y_true'], df['y_pred'])
    f1 = f1_score(df['y_true'], df['y_pred'], average='macro')

    metrics = {
        'accuracy_target': acc,
        'macro_f1_target': f1,
    }

    out_json = RESULTS / f'metrics_target_{args.model}_{args.strategy}.json'
    out_json.write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(df['y_true'], df['y_pred'])
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title(f'{args.model} → Target ({args.strategy})')
    plt.savefig(FIGURES / f'cm_target_{args.model}_{args.strategy}.png')
    plt.close()

    print(f'[OK] Target metrics computed for {args.model}')


if __name__ == '__main__':
    main()
