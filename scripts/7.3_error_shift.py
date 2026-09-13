from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

RESULTS = Path('results')
OUT = RESULTS / 'T4_error_category_shift_hybrid.csv'


def load_preds(path: Path):
    df = pd.read_csv(path)
    return df['y_true'].values, df['y_pred'].values


def per_class_f1(y_true, y_pred):
    labels = sorted(set(y_true))
    rows = []

    for c in labels:
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        f1 = f1_score(yt, yp, zero_division=0)
        rows.append((c, f1))

    return dict(rows)


def main():
    baseline_path = RESULTS / 'preds_target_resnet50_hybrid.csv'
    coral_path = RESULTS / 'preds_target_coral_lambda_0.1_hybrid.csv'

    if not baseline_path.exists():
        raise FileNotFoundError(baseline_path)
    if not coral_path.exists():
        raise FileNotFoundError(coral_path)

    yb_t, yb_p = load_preds(baseline_path)
    yc_t, yc_p = load_preds(coral_path)

    f1_base = per_class_f1(yb_t, yb_p)
    f1_coral = per_class_f1(yc_t, yc_p)

    rows = []
    for cls in sorted(f1_base.keys()):
        rows.append(
            {
                'class_id': cls,
                'f1_baseline': f1_base.get(cls, 0.0),
                'f1_coral': f1_coral.get(cls, 0.0),
                'delta_f1': f1_coral.get(cls, 0.0) - f1_base.get(cls, 0.0),
            }
        )

    df = pd.DataFrame(rows).sort_values('delta_f1', ascending=False)
    df.to_csv(OUT, index=False)

    print(f'[OK] Error shift table saved → {OUT}')


if __name__ == '__main__':
    main()
