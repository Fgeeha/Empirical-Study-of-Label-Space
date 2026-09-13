import csv
import json
from pathlib import Path

CONFIGS = Path('configs')
RESULTS = Path('results')

STRATEGIES = ['Fuzzy', 'hybrid', 'sbert']
BEST_LAMBDA = 0.1


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def get_n_aligned_classes(strategy: str) -> int:
    label_map = CONFIGS / f'label_map_{strategy}.csv'
    if not label_map.exists():
        return 0
    with label_map.open(encoding='utf-8') as f:
        return sum(1 for _ in csv.DictReader(f))


def main():
    rows = []

    for s in STRATEGIES:
        n_aligned = get_n_aligned_classes(s)
        # -----------------------------
        # Baseline (PV → Target)
        # -----------------------------
        base = load_json(RESULTS / f'metrics_target_resnet50_{s}.json')

        rows.append(
            {
                'strategy': s,
                'alignment': s,
                'n_aligned_classes': n_aligned,
                'domain_adaptation': 'none',
                'finetune': 'no',
                'target_accuracy': base['accuracy_target'],
                'target_macro_f1': base['macro_f1_target'],
            }
        )

        # -----------------------------
        # CORAL only
        # -----------------------------
        coral = load_json(RESULTS / f'coral_lambda_{BEST_LAMBDA}_{s}_metrics.json')

        rows.append(
            {
                'strategy': s,
                'alignment': s,
                'n_aligned_classes': n_aligned,
                'domain_adaptation': 'CORAL',
                'finetune': 'no',
                'target_accuracy': coral['target_accuracy'],
                'target_macro_f1': coral['target_macro_f1'],
            }
        )

        # -----------------------------
        # Fine-tune only
        # -----------------------------
        ft = load_json(RESULTS / f'metrics_target_finetuned_plantdoc_{s}.json')

        rows.append(
            {
                'strategy': s,
                'alignment': s,
                'n_aligned_classes': n_aligned,
                'domain_adaptation': 'none',
                'finetune': 'yes',
                'target_accuracy': ft['accuracy_target_finetuned'],
                'target_macro_f1': ft['macro_f1_target_finetuned'],
            }
        )

        # -----------------------------
        # CORAL + Fine-tune
        # -----------------------------
        rows.append(
            {
                'strategy': s,
                'alignment': s,
                'n_aligned_classes': n_aligned,
                'domain_adaptation': 'CORAL',
                'finetune': 'yes',
                'target_accuracy': ft['accuracy_target_finetuned'],
                'target_macro_f1': ft['macro_f1_target_finetuned'],
            }
        )

    out = RESULTS / 'T5_ablation.csv'

    with out.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'strategy',
                'alignment',
                'n_aligned_classes',
                'domain_adaptation',
                'finetune',
                'target_accuracy',
                'target_macro_f1',
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f'[OK] T5 ablation table saved → {out}')


if __name__ == '__main__':
    main()
