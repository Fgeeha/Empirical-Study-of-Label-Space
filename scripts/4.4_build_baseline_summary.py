import csv
import json
from pathlib import Path

RESULTS = Path('results')
OUT = RESULTS / 'baseline_pv_metrics.csv'

STRATEGIES = ['Fuzzy', 'hybrid', 'sbert']
MODELS = ['resnet50', 'efficientnet', 'mobilenet_baseline']


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def main():
    rows = [
        [
            'model',
            'strategy',
            'accuracy',
            'precision_macro',
            'recall_macro',
            'f1_macro',
        ]
    ]

    for strategy in STRATEGIES:
        for model in MODELS:
            # Handle different naming conventions
            if model == 'mobilenet_baseline':
                json_path = RESULTS / f'{model}_pv_metrics_{strategy}.json'
            else:
                json_path = RESULTS / f'{model}_pv_metrics_{strategy}.json'

            data = load_json(json_path)

            if data is None:
                print(f'[WARN] Missing: {json_path}')
                continue

            rows.append(
                [
                    model.replace('_baseline', ''),  # Clean name for table
                    strategy,
                    data.get('accuracy', 0.0),
                    data.get('precision_macro', 0.0),
                    data.get('recall_macro', 0.0),
                    data.get('f1_macro', 0.0),
                ]
            )

    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'[OK] Baseline summary saved → {OUT}')


if __name__ == '__main__':
    main()
