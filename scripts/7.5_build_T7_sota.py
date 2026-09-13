"""Build Table 7: comparison of methods on PlantDoc target domain.

All UDA methods use only unlabeled PlantDoc images (transductive unsupervised DA).
Supervised fine-tune is NOT comparable to UDA methods — listed separately as upper bound.
Literature baselines (Zhang 2020, Singh 2021) are supervised on PlantDoc training labels.
Numbers must be verified against cited papers before submission.
"""

import csv
import json
from pathlib import Path

RESULTS = Path('results')
EXPERIMENTS = Path('experiments')


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    rows = []

    # -----------------------------------------------------------------------
    # Literature baselines (supervised on PlantDoc, for reference only)
    # NOTE: These values are UNVERIFIED — marked TODO in result_registry.csv
    # Must be confirmed against the actual papers before submission.
    # -----------------------------------------------------------------------
    rows.append(
        {
            'method': 'Zhang et al. 2020 [TODO: verify]',
            'protocol': 'supervised (PlantDoc labels)',
            'dataset': 'PlantDoc',
            'class_space': 'original (27)',
            'acc': 0.52,
            'macro_f1': 0.48,
            'seeds': 'N/A',
            'notes': 'UNVERIFIED — hardcoded; must check against paper',
        }
    )
    rows.append(
        {
            'method': 'Singh et al. 2021 [TODO: verify]',
            'protocol': 'supervised (PlantDoc labels)',
            'dataset': 'PlantDoc',
            'class_space': 'original (27)',
            'acc': 0.61,
            'macro_f1': 0.58,
            'seeds': 'N/A',
            'notes': 'UNVERIFIED — hardcoded; must check against paper',
        }
    )

    # -----------------------------------------------------------------------
    # Source-only baseline (no adaptation, no target labels used)
    # -----------------------------------------------------------------------
    baseline = load_json(RESULTS / 'metrics_target_resnet50_hybrid.json')
    if baseline:
        rows.append(
            {
                'method': 'Source-only (ours)',
                'protocol': 'UDA — no adaptation',
                'dataset': 'PlantDoc',
                'class_space': 'Hybrid (18)',
                'acc': round(baseline.get('accuracy_target', 0), 4),
                'macro_f1': round(baseline.get('macro_f1_target', 0), 4),
                'seeds': '42',
                'notes': 'ResNet-50 trained on PV only; evaluated on PlantDoc test',
            }
        )
    else:
        # Fallback to T1 value
        rows.append(
            {
                'method': 'Source-only (ours)',
                'protocol': 'UDA — no adaptation',
                'dataset': 'PlantDoc',
                'class_space': 'Hybrid (18)',
                'acc': 0.191,
                'macro_f1': 0.156,
                'seeds': '42',
                'notes': 'from T1_domain_shift.csv',
            }
        )

    # -----------------------------------------------------------------------
    # CORAL best result (UDA, no target labels)
    # Best lambda per strategy: Fuzzy λ=1.0, Hybrid λ=0.01, SBERT λ=0.01
    # Using Hybrid λ=0.01 as main reported result (consistent with DANN strategy)
    # -----------------------------------------------------------------------
    coral = load_json(RESULTS / 'coral_lambda_0.01_hybrid_metrics.json')
    if coral:
        rows.append(
            {
                'method': 'CORAL λ=0.01 (ours)',
                'protocol': 'UDA — unsupervised, no target labels',
                'dataset': 'PlantDoc',
                'class_space': 'Hybrid (18)',
                'acc': round(coral.get('target_accuracy', 0), 4),
                'macro_f1': round(coral.get('target_macro_f1', 0), 4),
                'seeds': '42',
                'notes': 'best CORAL lambda for Hybrid; seed 42 only (TODO: add seeds 43,44)',
            }
        )

    # Also report Fuzzy strategy best CORAL (highest absolute F1)
    coral_fuzzy = load_json(RESULTS / 'coral_lambda_1.0_Fuzzy_metrics.json')
    if coral_fuzzy:
        rows.append(
            {
                'method': 'CORAL λ=1.0 / Fuzzy (ours)',
                'protocol': 'UDA — unsupervised, no target labels',
                'dataset': 'PlantDoc',
                'class_space': 'Fuzzy (16)',
                'acc': round(coral_fuzzy.get('target_accuracy', 0), 4),
                'macro_f1': round(coral_fuzzy.get('target_macro_f1', 0), 4),
                'seeds': '42',
                'notes': 'best CORAL result overall (Fuzzy strategy); seed 42 only',
            }
        )

    # -----------------------------------------------------------------------
    # DANN (UDA, no target labels, 3 seeds)
    # -----------------------------------------------------------------------
    dann = load_json(EXPERIMENTS / 'E2_dann' / 'dann_results.json')
    if dann:
        mean_f1 = dann.get('target_macro_f1_mean')
        std_f1 = dann.get('target_macro_f1_std')
        rows.append(
            {
                'method': 'DANN (ours)',
                'protocol': 'UDA — unsupervised, no target labels',
                'dataset': 'PlantDoc',
                'class_space': 'Hybrid (18)',
                'acc': round(dann.get('target_accuracy_mean', 0), 4),
                'macro_f1': f'{mean_f1:.4f}±{std_f1:.4f}'
                if (mean_f1 and std_f1)
                else str(mean_f1),
                'seeds': '42,43,44',
                'notes': 'strongest UDA result; mean±std across 3 seeds',
            }
        )

    # -----------------------------------------------------------------------
    # Supervised fine-tune upper bound (NOT comparable to UDA — different protocol)
    # -----------------------------------------------------------------------
    ft = load_json(RESULTS / 'metrics_target_finetuned_plantdoc_hybrid.json')
    if ft:
        rows.append(
            {
                'method': 'Supervised fine-tune (ours) [upper bound]',
                'protocol': 'SUPERVISED — uses PlantDoc training labels',
                'dataset': 'PlantDoc',
                'class_space': 'Hybrid (18)',
                'acc': round(ft.get('accuracy_target_finetuned', 0), 4),
                'macro_f1': round(ft.get('macro_f1_target_finetuned', 0), 4),
                'seeds': '42',
                'notes': 'NOT comparable to UDA methods; shows upper bound with target labels',
            }
        )

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    out = RESULTS / 'T7_sota.csv'
    fieldnames = [
        'method',
        'protocol',
        'dataset',
        'class_space',
        'acc',
        'macro_f1',
        'seeds',
        'notes',
    ]
    with open(out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'[OK] T7 SOTA table saved → {out}')
    print()
    print(
        'IMPORTANT: Literature baseline numbers (Zhang 2020, Singh 2021) are UNVERIFIED.'
    )
    print('Must check against actual papers before submission.')
    print(
        'Do NOT report CORAL UDA F1=0.85 — that was a supervised fine-tune mislabeled as CORAL.'
    )


if __name__ == '__main__':
    main()
