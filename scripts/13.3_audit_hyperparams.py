#!/usr/bin/env python3
"""
Audit hyperparameter consistency: compare YAML config vs actual training
code vs paper text.

Detects mismatches that would cause a reproducibility failure or
reviewer concern.

Checks:
    1. configs/train_resnet50.yaml vs scripts/4.2_train_resnet50_pv.py
    2. configs/train_resnet50.yaml vs scripts/6.3_train_coral.py
    3. configs/train_resnet50.yaml vs scripts/5.5.2_finetune_target.py
    4. All of the above vs paper claims (hardcoded from the .docm)

Usage:
    python scripts/13.3_audit_hyperparams.py
"""

from pathlib import Path
import re
import sys

import yaml

CONFIG = Path('configs/train_resnet50.yaml')
SCRIPTS_DIR = Path('scripts')


PAPER_CLAIMS = {
    'section': '3.2 Domain Adaptation with CORAL',
    'optimizer': 'SGD',
    'momentum': 0.9,
    'lr': 0.001,
    'batch_size': 64,
    'epochs': 50,
    'scheduler': 'cosine annealing',
    'backbone': 'ResNet-50 pretrained on ImageNet',
    'feature_dim': 2048,
    'source': (
        'Paper Section 3.2: "All runs use SGD with momentum 0.9, '
        'a learning rate of 0.001, and cosine annealing over 50 epochs." '
        'and "The batch size is 64 in all cases"'
    ),
}


def load_yaml_config() -> dict:
    if not CONFIG.exists():
        print(f'[FATAL] Config not found: {CONFIG}')
        sys.exit(1)
    with CONFIG.open() as f:
        return yaml.safe_load(f)


def extract_from_script(script_path: Path) -> dict:
    """Extract hyperparameters from a training script via AST + regex."""
    if not script_path.exists():
        return {'error': f'File not found: {script_path}'}

    code = script_path.read_text(encoding='utf-8')
    info: dict = {'file': str(script_path)}

    if 'AdamW' in code:
        info['optimizer'] = 'AdamW'
    elif 'Adam(' in code:
        info['optimizer'] = 'Adam'
    elif 'SGD' in code:
        info['optimizer'] = 'SGD'
    else:
        info['optimizer'] = 'UNKNOWN'

    m = re.search(r'momentum\s*=\s*([\d.]+)', code)
    if m:
        info['momentum'] = float(m.group(1))

    lr_match = re.search(
        r"lr\s*=\s*cfg\['training'\]\['lr'\](?:\s*\*\s*([\d.]+))?", code
    )
    if lr_match:
        info['lr_source'] = "cfg['training']['lr']"
        if lr_match.group(1):
            info['lr_multiplier'] = float(lr_match.group(1))
        else:
            info['lr_multiplier'] = 1.0
    else:
        lr_direct = re.search(r'lr\s*=\s*([\d.e-]+)', code)
        if lr_direct:
            info['lr_source'] = 'hardcoded'
            info['lr_value'] = float(lr_direct.group(1))

    bs_match = re.search(r"batch_size\s*=\s*cfg\['training'\]\['batch_size'\]", code)
    if bs_match:
        info['batch_size_source'] = "cfg['training']['batch_size']"
    else:
        bs_direct = re.search(r'batch_size\s*=\s*(\d+)', code)
        if bs_direct:
            info['batch_size_source'] = 'hardcoded'
            info['batch_size_value'] = int(bs_direct.group(1))

    ep_cfg = re.search(r"range\(cfg\['training'\]\['epochs'\]\)", code)
    ep_arg = re.search(r'range\(args\.epochs\)', code)
    ep_default = re.search(r'default\s*=\s*(\d+).*epochs', code)

    if ep_cfg:
        info['epochs_source'] = "cfg['training']['epochs']"
    elif ep_arg:
        info['epochs_source'] = 'args.epochs'
        if ep_default:
            info['epochs_default'] = int(ep_default.group(1))
    else:
        info['epochs_source'] = 'UNKNOWN'

    if 'torch.manual_seed' in code:
        info['seed_set'] = True
        seed_match = re.search(r"manual_seed\(cfg\['seed'\]\)", code)
        info['seed_source'] = "cfg['seed']" if seed_match else 'other'
    else:
        info['seed_set'] = False

    if 'cudnn.deterministic' in code:
        info['cudnn_deterministic'] = True
    else:
        info['cudnn_deterministic'] = False

    if 'PYTHONHASHSEED' in code:
        info['pythonhashseed'] = True
    else:
        info['pythonhashseed'] = False

    scheduler_match = re.search(
        r'(CosineAnnealingLR|StepLR|ReduceLROnPlateau|OneCycleLR)', code
    )
    if scheduler_match:
        info['scheduler'] = scheduler_match.group(1)
    elif 'cosine' in code.lower():
        info['scheduler'] = 'cosine (mentioned but no LR scheduler object found)'
    else:
        info['scheduler'] = 'NONE'

    if 'early_stopping' in code or 'patience' in code:
        info['early_stopping'] = True
    else:
        info['early_stopping'] = False

    return info


def compare(name: str, yaml_val, code_val, paper_val, notes: str = ''):
    """Print a comparison row with mismatch detection."""
    yaml_s = str(yaml_val) if yaml_val is not None else '---'
    code_s = str(code_val) if code_val is not None else '---'
    paper_s = str(paper_val) if paper_val is not None else '---'

    mismatches = []
    if yaml_val is not None and paper_val is not None:
        if str(yaml_val).lower() != str(paper_val).lower():
            mismatches.append('YAML!=PAPER')
    if yaml_val is not None and code_val is not None:
        if str(yaml_val).lower() != str(code_val).lower():
            mismatches.append('YAML!=CODE')
    if code_val is not None and paper_val is not None:
        if str(code_val).lower() != str(paper_val).lower():
            mismatches.append('CODE!=PAPER')

    status = 'MISMATCH' if mismatches else 'OK'
    mismatch_str = ', '.join(mismatches) if mismatches else ''

    print(
        f'  {name:<25s} '
        f'YAML={yaml_s:<12s} '
        f'CODE={code_s:<20s} '
        f'PAPER={paper_s:<15s} '
        f'[{status}] {mismatch_str}'
    )
    if notes:
        print(f'  {"":25s} Note: {notes}')
    return len(mismatches) > 0


def main():
    print('=' * 100)
    print('HYPERPARAMETER CONSISTENCY AUDIT')
    print('=' * 100)

    cfg = load_yaml_config()
    yaml_training = cfg.get('training', {})
    yaml_seed = cfg.get('seed')
    _ = cfg.get('augmentation', {})

    # --- Extract from all scripts ---
    scripts = {
        '4.2_train_resnet50_pv.py': 'Baseline ResNet-50 training',
        '6.3_train_coral.py': 'CORAL domain adaptation training',
        '5.5.2_finetune_target.py': 'Fine-tuning on target domain',
        '4.2b_train_efficientnet_pv.py': 'EfficientNet baseline',
        '4.2c_train_mobilenet_baseline_pv.py': 'MobileNet baseline',
        '9.0_train_mobilenetv2_pv.py': 'MobileNetV2 training',
        '9.1_train_coral_mobilenetv2.py': 'MobileNetV2 CORAL',
    }

    script_info = {}
    for script_name, _desc in scripts.items():
        script_info[script_name] = extract_from_script(SCRIPTS_DIR / script_name)

    # --- YAML config summary ---
    print(f'\n--- YAML Config ({CONFIG}) ---')
    print(f'  seed:           {yaml_seed}')
    print(f'  optimizer:      {yaml_training.get("optimizer")}')
    print(f'  lr:             {yaml_training.get("lr")}')
    print(f'  batch_size:     {yaml_training.get("batch_size")}')
    print(f'  epochs:         {yaml_training.get("epochs")}')
    print(f'  scheduler:      {yaml_training.get("scheduler")}')
    print(f'  weight_decay:   {yaml_training.get("weight_decay")}')
    print(f'  early_stopping: patience={yaml_training.get("early_stopping_patience")}')

    # --- Paper claims ---
    print('\n--- Paper Claims (Section 3.2) ---')
    print(f'  optimizer:  {PAPER_CLAIMS["optimizer"]}')
    print(f'  momentum:   {PAPER_CLAIMS["momentum"]}')
    print(f'  lr:         {PAPER_CLAIMS["lr"]}')
    print(f'  batch_size: {PAPER_CLAIMS["batch_size"]}')
    print(f'  epochs:     {PAPER_CLAIMS["epochs"]}')
    print(f'  scheduler:  {PAPER_CLAIMS["scheduler"]}')
    print(f'  Source:     {PAPER_CLAIMS["source"]}')

    # --- Detailed comparison per script ---
    total_mismatches = 0

    for script_name, desc in scripts.items():
        info = script_info[script_name]
        if 'error' in info:
            print(f'\n--- {script_name} ({desc}): {info["error"]} ---')
            continue

        print(f'\n--- {script_name} ({desc}) ---')

        yaml_opt = yaml_training.get('optimizer', '').upper()
        code_opt = info.get('optimizer')
        paper_opt = PAPER_CLAIMS['optimizer']

        total_mismatches += compare(
            'Optimizer',
            yaml_opt,
            code_opt,
            paper_opt,
            notes=(
                'YAML=AdamW, Paper=SGD. This is a critical '
                'discrepancy. Which was actually used?'
                if yaml_opt != paper_opt
                else ''
            ),
        )

        yaml_lr = yaml_training.get('lr')
        if 'lr_multiplier' in info:
            code_lr = (
                f'{yaml_lr}*{info["lr_multiplier"]}'
                f'={yaml_lr * info["lr_multiplier"]:.5f}'
            )
            _ = yaml_lr * info['lr_multiplier']
        elif 'lr_value' in info:
            code_lr = str(info['lr_value'])
            _ = info['lr_value']
        else:
            code_lr = f'from yaml ({yaml_lr})'
            _ = yaml_lr
        paper_lr = PAPER_CLAIMS['lr']
        total_mismatches += compare(
            'Learning rate',
            yaml_lr,
            code_lr,
            paper_lr,
        )

        yaml_bs = yaml_training.get('batch_size')
        code_bs = info.get('batch_size_source', '?')
        if code_bs == "cfg['training']['batch_size']":
            code_bs = f'from yaml ({yaml_bs})'
        paper_bs = PAPER_CLAIMS['batch_size']
        total_mismatches += compare('Batch size', yaml_bs, code_bs, paper_bs)

        yaml_ep = yaml_training.get('epochs')
        code_ep_src = info.get('epochs_source', '?')
        if code_ep_src == "cfg['training']['epochs']":
            code_ep = f'from yaml ({yaml_ep})'
        elif code_ep_src == 'args.epochs':
            default = info.get('epochs_default', '?')
            code_ep = f'args (default={default})'
        else:
            code_ep = code_ep_src
        paper_ep = PAPER_CLAIMS['epochs']
        total_mismatches += compare('Epochs', yaml_ep, code_ep, paper_ep)

        yaml_sched = yaml_training.get('scheduler', 'none')
        code_sched = info.get('scheduler', 'none')
        paper_sched = PAPER_CLAIMS['scheduler']
        total_mismatches += compare(
            'Scheduler',
            yaml_sched,
            code_sched,
            paper_sched,
        )

        code_seed = 'Yes' if info.get('seed_set') else 'No'
        compare('Seed set', f'seed={yaml_seed}', code_seed, '---')

        code_det = 'Yes' if info.get('cudnn_deterministic') else 'No'
        compare('cudnn.deterministic', '---', code_det, '---')

    # --- reproduce.sh consistency ---
    reproduce_sh = Path('reproduce.sh')
    if reproduce_sh.exists():
        content = reproduce_sh.read_text()
        print('\n--- reproduce.sh CORAL defaults ---')
        best_lam = re.search(r'BEST_LAMBDA=([\d.]+)', content)
        lambdas = re.search(r'LAMBDA_VALUES=\(([^)]+)\)', content)
        if best_lam:
            print(f'  BEST_LAMBDA={best_lam.group(1)}')
        if lambdas:
            print(f'  LAMBDA_VALUES=({lambdas.group(1)})')

    # --- Summary ---
    print(f'\n{"=" * 100}')
    print('SUMMARY')
    print('=' * 100)
    print(f'  Total mismatches detected: {total_mismatches}')

    if total_mismatches > 0:
        print('\n  CRITICAL DISCREPANCIES:')
        print(
            '\n  1. OPTIMIZER: YAML says "adamw", code uses AdamW, '
            'paper says "SGD with momentum 0.9"'
        )
        print(
            '     This is the most serious mismatch. A reviewer running '
            'the code will get AdamW results,'
        )
        print(
            '     but the paper describes SGD. They are fundamentally '
            'different optimizers.'
        )

        print('\n  2. LEARNING RATE: YAML says 3e-4, paper says 0.001')
        print('     3e-4 = 0.0003, paper claims 0.001 -- a 3.3x difference.')

        print('\n  3. BATCH SIZE: YAML says 32, paper says 64')
        print('     2x difference in batch size affects effective LR and convergence.')

        print('\n  4. EPOCHS: YAML says 30 (with early stopping), paper says 50')
        print('     CORAL script defaults to 20 epochs (--epochs default=20),')
        print('     baseline uses yaml=30, paper claims 50 for all.')

        print('\n  5. CORAL script has NO scheduler (no LR scheduler object)')
        print('     Paper claims "cosine annealing over 50 epochs" for CORAL.')

    print(f'\n{"=" * 100}')
    print('REQUIRED ACTIONS')
    print('=' * 100)
    print('  1. Determine which hyperparameters were ACTUALLY used in the experiments')
    print('     that produced the submitted results.')
    print()
    print('  2. If code is correct (AdamW, lr=3e-4, bs=32, epochs=30):')
    print('     -> Update paper Section 3.2 to match the code.')
    print('     -> Change "SGD with momentum 0.9" to "AdamW with weight decay 1e-4"')
    print('     -> Change "learning rate of 0.001" to "learning rate of 3x10^-4"')
    print('     -> Change "batch size is 64" to "batch size of 32"')
    print('     -> Change "50 epochs" to "30 epochs with early stopping (patience=5)"')
    print()
    print('  3. If paper is correct (SGD, lr=0.001, bs=64, epochs=50):')
    print('     -> Update configs/train_resnet50.yaml')
    print('     -> Update code in 6.3_train_coral.py')
    print('     -> Re-run ALL experiments with correct params')
    print('     -> This invalidates all current results!')
    print()
    print('  4. CORAL-specific: paper says SGD but 6.3_train_coral.py uses AdamW.')
    print('     The CORAL script also defaults to 20 epochs, not 50 as stated.')
    print('     And it has NO LR scheduler despite paper claiming "cosine annealing".')
    print()
    print('  5. Add a complete hyperparameter table to Supplementary material:')
    print('     | Param | Baseline | CORAL | Fine-tune |')
    print('     covering optimizer, LR, batch, epochs, scheduler, augmentation.')


if __name__ == '__main__':
    main()
