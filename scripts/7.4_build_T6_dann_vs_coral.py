import json
from pathlib import Path

import pandas as pd

RESULTS = Path('results')
OUT = RESULTS / 'T6_dann_vs_coral.csv'


def load_metrics(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    rows = []

    # -------------------------
    # Baseline
    # -------------------------
    base = load_metrics(RESULTS / 'metrics_target_resnet50_hybrid.json')
    if base:
        rows.append(
            {
                'method': 'Baseline (ResNet-50)',
                'target_accuracy': base.get('accuracy_target'),
                'target_macro_f1': base.get('macro_f1_target'),
            }
        )

    # -------------------------
    # CORAL (best λ = 0.1)
    # -------------------------
    coral = load_metrics(RESULTS / 'coral_lambda_0.1_hybrid_metrics.json')
    if coral:
        rows.append(
            {
                'method': 'CORAL (λ=0.1)',
                'target_accuracy': coral.get('target_accuracy'),
                'target_macro_f1': coral.get('target_macro_f1'),
            }
        )

    # -------------------------
    # DANN (3-seed mean ± std from E2)
    # -------------------------
    dann_path = Path('experiments/E2_dann/dann_results.json')
    dann = load_metrics(dann_path)
    if dann:
        mean_f1 = dann.get('target_macro_f1_mean')
        std_f1 = dann.get('target_macro_f1_std')
        rows.append(
            {
                'method': 'DANN (mean±std, n=3)',
                'target_accuracy': dann.get('target_accuracy_mean'),
                'target_macro_f1': f'{mean_f1:.4f}±{std_f1:.4f}'
                if (mean_f1 and std_f1)
                else mean_f1,
            }
        )
    else:
        rows.append(
            {
                'method': 'DANN',
                'target_accuracy': 'N/A',
                'target_macro_f1': 'N/A',
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)

    print(f'[OK] T6 table saved → {OUT}')


if __name__ == '__main__':
    main()
