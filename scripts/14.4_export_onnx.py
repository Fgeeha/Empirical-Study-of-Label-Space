#!/usr/bin/env python3
"""Export a PyTorch checkpoint (source-only or fair-protocol DANN) to ONNX for the
ONNX -> TFLite (onnx2tf) INT8 path used by scripts/14.5_onnx2tf_int8.sh.

    python scripts/14.4_export_onnx.py --model resnet50 --strategy hybrid --out export/onnx/resnet50_hybrid.onnx

Model construction is shared with 14.2_eval_pytorch_reference.py (same
checkpoints, same architectures). Input: NCHW float32 [1,3,224,224], ImageNet-normalized.
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    'ref', HERE / '14.2_eval_pytorch_reference.py'
)
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--strategy', default='hybrid')
    ap.add_argument('--out', required=True)
    ap.add_argument('--opset', type=int, default=17)
    ap.add_argument(
        '--zero-pad-maxpool',
        action='store_true',
        help='ResNet: replace MaxPool2d(3,2,pad=1) by ZeroPad2d(1)+MaxPool2d(3,2,0). '
        'Equivalent after ReLU (inputs >= 0); avoids the -inf PadV2 op that breaks '
        'INT8 PTQ and the Edge TPU compiler',
    )
    args = ap.parse_args()
    num_classes = pd.read_csv(
        ref.SPLITS / args.strategy / 'pv_test.csv'
    ).label_id.nunique()
    m = ref.build(args.model, args.strategy, num_classes).eval()
    if args.zero_pad_maxpool:
        mp = m.maxpool
        m.maxpool = torch.nn.Sequential(
            torch.nn.ZeroPad2d(mp.padding),
            torch.nn.MaxPool2d(mp.kernel_size, mp.stride, 0),
        )
        m.eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        m,
        torch.zeros(1, 3, 224, 224),
        str(out),
        input_names=['input'],
        output_names=['logits'],
        opset_version=args.opset,
        dynamo=False,
    )
    print(f'[OK] {out} ({out.stat().st_size / 1024 / 1024:.1f} MiB)')


if __name__ == '__main__':
    main()
