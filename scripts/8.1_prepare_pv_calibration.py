import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

SPLITS = Path('splits')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--strategy',
        default='hybrid',
        help='Which PV train split to use (default: hybrid)',
    )
    parser.add_argument('--ratio', type=float, default=0.1)
    args = parser.parse_args()

    src_csv = SPLITS / args.strategy / 'pv_train.csv'
    out_csv = SPLITS / 'pv_calib_10pct.csv'

    if not src_csv.exists():
        raise FileNotFoundError(f'Missing source CSV: {src_csv}')

    df = pd.read_csv(src_csv)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=args.ratio,
        random_state=42,
    )

    _, calib_idx = next(splitter.split(df['path'], df['label_id']))

    calib_df = df.iloc[calib_idx].reset_index(drop=True)
    calib_df.to_csv(out_csv, index=False)

    print(
        f'[OK] Calibration split saved → {out_csv} '
        f'({len(calib_df)} samples, {args.ratio * 100:.0f}%)'
    )


if __name__ == '__main__':
    main()
