import csv
import json
from pathlib import Path

RESULTS = Path('results')
OUT = RESULTS / 'T1_domain_shift.csv'

STRATEGIES = ['Fuzzy', 'hybrid', 'sbert']
MODELS = ['resnet50', 'efficientnet', 'mobilenet_baseline']


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def extract_metrics(data: dict):
    """
    Universal extractor for accuracy + macro-F1.

    Supported formats:
    - PV old: accuracy, macro_f1
    - PV new: accuracy, f1_macro
    - Target: accuracy_target, macro_f1_target
    """
    # accuracy
    if 'accuracy' in data:
        acc = data['accuracy']
    elif 'accuracy_target' in data:
        acc = data['accuracy_target']
    else:
        raise KeyError(f'No accuracy key in {list(data.keys())}')

    # macro-F1
    if 'macro_f1' in data:
        f1 = data['macro_f1']
    elif 'f1_macro' in data:
        f1 = data['f1_macro']
    elif 'macro_f1_target' in data:
        f1 = data['macro_f1_target']
    else:
        raise KeyError(f'No macro-F1 key in {list(data.keys())}')

    return acc, f1


def main():
    rows = [
        [
            'model',
            'strategy',
            'pv_accuracy',
            'pv_macro_f1',
            'target_accuracy',
            'target_macro_f1',
            'delta_f1',
        ]
    ]

    for model in MODELS:
        for strategy in STRATEGIES:
            pv_path = RESULTS / f'{model}_pv_metrics_{strategy}.json'
            tgt_path = RESULTS / f'metrics_target_{model}_{strategy}.json'

            if not pv_path.exists():
                print(f'[WARN] Missing PV metrics: {pv_path}')
                continue

            if not tgt_path.exists():
                print(f'[WARN] Missing target metrics: {tgt_path}')
                continue

            pv = load_json(pv_path)
            tgt = load_json(tgt_path)

            pv_acc, pv_f1 = extract_metrics(pv)
            tgt_acc, tgt_f1 = extract_metrics(tgt)

            rows.append(
                [
                    model,
                    strategy,
                    pv_acc,
                    pv_f1,
                    tgt_acc,
                    tgt_f1,
                    pv_f1 - tgt_f1,
                ]
            )

    with OUT.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f'[OK] Domain shift table saved → {OUT}')


if __name__ == '__main__':
    main()
