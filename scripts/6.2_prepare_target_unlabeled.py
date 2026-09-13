import argparse
from pathlib import Path

import pandas as pd

SPLITS = Path('splits')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', required=True)
    parser.add_argument('--target', choices=['plantdoc', 'fcdd'], required=True)
    args = parser.parse_args()

    in_csv = SPLITS / args.strategy / f'target_{args.target}_train.csv'
    out_csv = SPLITS / args.strategy / f'target_{args.target}_unlabeled.csv'

    df = pd.read_csv(in_csv)

    # drop labels (UDA setting)
    df = df[['path']]

    df.to_csv(out_csv, index=False)
    print(f'[OK] Unlabeled target saved → {out_csv}')


if __name__ == '__main__':
    main()
