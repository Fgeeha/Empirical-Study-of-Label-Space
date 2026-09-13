#!/usr/bin/env python3
"""PyTorch (FP32) reference accuracy for the edge architectures, to be compared
with the INT8 TFLite results of scripts/14.1_eval_tflite_accuracy.py.

    python scripts/14.2_eval_pytorch_reference.py --model resnet50 --strategy hybrid

Model definitions and checkpoint names follow the training scripts
(4.2, 4.2b, 9.0, 10.6, 10.8). Evaluation transform = Resize(224) -> ToTensor
-> ImageNet Normalize, as in 4.3_eval_resnet50_pv.py. Output:
results/int8_accuracy/pytorch_{model}_{strategy}__{split}.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
import torch
from torchvision import models, transforms

MODELS = Path('models')
SPLITS = Path('splits')
OUT = Path('results/int8_accuracy')
TF = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def build(model: str, strategy: str, num_classes: int) -> torch.nn.Module:
    if model == 'resnet50':
        m = models.resnet50()
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        ckpt = MODELS / f'resnet50_pv_{strategy}_best.pt'
    elif model == 'mobilenetv2':
        m = models.mobilenet_v2()
        m.classifier[1] = torch.nn.Linear(m.last_channel, num_classes)
        ckpt = MODELS / f'mobilenetv2_pv_{strategy}_best.pt'
    elif model == 'efficientnet_b3':
        m = models.efficientnet_b3()
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
        ckpt = MODELS / f'efficientnet_pv_{strategy}_best.pt'
    elif model == 'mobilenetv1':
        import timm

        m = timm.create_model(
            'mobilenetv1_100', pretrained=False, num_classes=num_classes
        )
        ckpt = MODELS / f'mobilenetv1_pv_{strategy}_best.pt'
    elif model == 'efficientnet_lite0':
        import timm

        m = timm.create_model(
            'tf_efficientnet_lite0', pretrained=False, num_classes=num_classes
        )
        ckpt = MODELS / f'efficientnet_lite0_pv_{strategy}_best.pt'
    elif model.startswith('dann'):
        # fair-protocol DANN checkpoint: {'backbone': resnet50 w/o fc, 'classifier': Linear}
        seed = model[4:]
        m = models.resnet50()
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        ck = torch.load(
            MODELS / 'uda_fair' / f'dann_{strategy}_seed{seed}.pt',
            map_location='cpu',
            weights_only=False,
        )
        sd = {k: v for k, v in ck['backbone'].items() if not k.startswith('fc.')}
        clf = ck['classifier']
        sd['fc.weight'] = clf[[k for k in clf if k.endswith('weight')][0]]
        sd['fc.bias'] = clf[[k for k in clf if k.endswith('bias')][0]]
        m.load_state_dict(sd)
        return m.eval()
    else:
        raise ValueError(model)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    m.load_state_dict(sd)
    return m.eval()


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def pil_tensor(path: str) -> torch.Tensor:
    arr = (
        np.asarray(Image.open(path).convert('RGB').resize((224, 224)), np.float32)
        / 255.0
    )
    return torch.from_numpy(((arr - MEAN) / STD).transpose(2, 0, 1).copy())


@torch.no_grad()
def evaluate(
    m: torch.nn.Module, csv_path: Path, device: str, batch: int = 64, pil: bool = False
) -> dict:
    df = pd.read_csv(csv_path)
    preds = []
    load = pil_tensor if pil else (lambda p: TF(Image.open(p).convert('RGB')))
    for i in range(0, len(df), batch):
        x = torch.stack([load(p) for p in df.path.iloc[i : i + batch]])
        preds.extend(m(x.to(device)).argmax(1).cpu().tolist())
    y = df.label_id.tolist()
    return {
        'split': csv_path.name,
        'n': len(df),
        'accuracy': accuracy_score(y, preds),
        'macro_f1': f1_score(y, preds, average='macro', zero_division=0),
        'y_true': y,
        'y_pred': preds,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--strategy', default='hybrid')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument(
        '--pil',
        action='store_true',
        help='PIL resize + numpy normalize (identical to 14.1 TFLite eval) instead of torchvision Resize',
    )
    ap.add_argument(
        '--dump-npz',
        help='write the state_dict as .npz (input for the 9.x/10.x TFLite export scripts) and exit',
    )
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    num_classes = pd.read_csv(SPLITS / args.strategy / 'pv_test.csv').label_id.nunique()
    m = build(args.model, args.strategy, num_classes)
    if args.dump_npz:
        np.savez(
            args.dump_npz,
            **{k: v.detach().cpu().numpy() for k, v in m.state_dict().items()},
        )
        print(f'[OK] {args.dump_npz} ({len(m.state_dict())} tensors)')
        return
    m = m.to(args.device)
    for split, csv_path in (
        ('pv_test', SPLITS / args.strategy / 'pv_test.csv'),
        ('plantdoc_test', SPLITS / args.strategy / 'target_plantdoc_test.csv'),
    ):
        res = evaluate(m, csv_path, args.device, pil=args.pil)
        res.update(
            {
                'model': args.model,
                'strategy': args.strategy,
                'framework': 'pytorch_fp32',
                'preprocessing': 'pil' if args.pil else 'torchvision',
            }
        )
        suffix = '_pil' if args.pil else ''
        (
            OUT / f'pytorch_{args.model}_{args.strategy}{suffix}__{split}.json'
        ).write_text(json.dumps(res, indent=1))
        print(
            f'{args.model:18s} {split:14s} n={res["n"]:5d} acc={res["accuracy"]:.4f} F1={res["macro_f1"]:.4f}'
        )


if __name__ == '__main__':
    main()
