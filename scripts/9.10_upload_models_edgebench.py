#!/usr/bin/env python3
"""
Upload ECCV TFLite models to Edge-Bench server.

Scans export/ directory for .tflite models and uploads them
to the Edge-Bench server's model repository.

Usage:
    python scripts/9.10_upload_models_edgebench.py [--server URL] [--dir PATH]

Examples:
    # Upload all models from export/ to local server
    python scripts/9.10_upload_models_edgebench.py

    # Upload from custom directory to remote server
    python scripts/9.10_upload_models_edgebench.py \\
        --server http://192.168.1.4:8000 \\
        --dir ./export

    # Upload only EdgeTPU models
    python scripts/9.10_upload_models_edgebench.py --pattern "*_edgetpu.tflite"

    # Upload and deploy to device
    python scripts/9.10_upload_models_edgebench.py --deploy-to raspberrypi
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import requests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Upload ECCV TFLite models to Edge-Bench server'
    )
    parser.add_argument(
        '--server',
        default='http://localhost:8000',
        help='Edge-Bench server URL (default: http://localhost:8000)',
    )
    parser.add_argument(
        '--dir',
        default='export',
        help='Directory with .tflite models (default: export/)',
    )
    parser.add_argument(
        '--pattern',
        default='*.tflite',
        help='Glob pattern for model files (default: *.tflite)',
    )
    parser.add_argument(
        '--deploy-to',
        default=None,
        help='Device name to auto-deploy models after upload',
    )
    parser.add_argument(
        '--edgetpu-dir',
        default=None,
        help='Directory with EdgeTPU-compiled models (default: {dir}/edgetpu/)',
    )
    return parser.parse_args()


def _discover_models(models_dir: Path, pattern: str) -> list[Path]:
    """Find all .tflite model files."""
    models = sorted(models_dir.glob(pattern))
    return models


def _check_server(server: str) -> bool:
    """Verify Edge-Bench server is reachable."""
    try:
        resp = requests.get(f'{server}/api/files?file_type=model', timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _get_existing_models(server: str) -> dict[str, str]:
    """Get existing models on server (name -> id)."""
    try:
        resp = requests.get(f'{server}/api/files?file_type=model', timeout=10)
        if resp.status_code == 200:
            return {f['name']: f['id'] for f in resp.json()}
    except Exception:
        pass
    return {}


def _upload_model(server: str, model_path: Path) -> dict | None:
    """Upload a single model to the server."""
    try:
        with open(model_path, 'rb') as f:
            files = {'file': (model_path.name, f, 'application/octet-stream')}
            resp = requests.post(
                f'{server}/api/files/upload',
                files=files,
                data={'file_type': 'model'},
                timeout=120,
            )

        if resp.status_code == 200:
            data = resp.json()
            if data.get('duplicate'):
                return {'status': 'duplicate', 'name': data['name'], 'id': data['id']}
            return {'status': 'uploaded', 'name': data['name'], 'id': data['id']}
        else:
            return {'status': 'error', 'name': model_path.name, 'error': resp.text}
    except Exception as e:
        return {'status': 'error', 'name': model_path.name, 'error': str(e)}


def _get_device_id(server: str, device_name: str) -> str | None:
    """Find device ID by name."""
    try:
        resp = requests.get(f'{server}/api/devices', timeout=5)
        if resp.status_code == 200:
            for dev in resp.json():
                if dev['name'] == device_name:
                    return dev['id']
    except Exception:
        pass
    return None


def _deploy_model(server: str, device_id: str, model_name: str) -> bool:
    """Deploy a model to device by name."""
    try:
        resp = requests.post(
            f'{server}/api/devices/{device_id}/deploy-by-name',
            json={'model_name': model_name},
            timeout=120,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _print_section(title: str, items: list, fmt: str = '    {item}') -> None:
    """Print a named section of results."""
    if not items:
        return
    print(f'\n  {title}: {len(items)}')
    for item in items:
        print(fmt.format(item=item))


def _print_results(
    uploaded: list[dict],
    duplicates: list[dict],
    errors: list[dict],
    deployed: list[str],
    deploy_errors: list[str],
) -> None:
    """Print summary of operations."""
    total = len(uploaded) + len(duplicates) + len(errors)
    print(f'\n{"=" * 50}')
    print(f'Результаты загрузки ({total} моделей)')
    print(f'{"=" * 50}')

    _print_section('Загружено', [m['name'] for m in uploaded], '    + {item}')
    _print_section('Уже существуют', [m['name'] for m in duplicates], '    = {item}')
    _print_section(
        'Ошибки',
        [f'{m["name"]}: {m.get("error", "?")}' for m in errors],
        '    ! {item}',
    )
    _print_section('Развёрнуто на устройство', deployed, '    -> {item}')
    _print_section('Ошибки развёртывания', deploy_errors, '    ! {item}')
    print()


def _upload_all_models(
    server: str, models: list[Path], existing: dict[str, str]
) -> tuple[list[dict], list[dict], list[dict]]:
    """Upload models to server, returning (uploaded, duplicates, errors)."""
    uploaded: list[dict] = []
    duplicates: list[dict] = []
    errors: list[dict] = []

    for model_path in models:
        if model_path.name in existing:
            duplicates.append(
                {'name': model_path.name, 'id': existing[model_path.name]}
            )
            print(f'  = {model_path.name} (already exists)')
            continue

        print(f'  Uploading {model_path.name}...', end=' ', flush=True)
        result = _upload_model(server, model_path)

        if result is None:
            errors.append({'name': model_path.name, 'error': 'No response'})
            print('ERROR')
        elif result['status'] == 'uploaded':
            uploaded.append(result)
            print('OK')
        elif result['status'] == 'duplicate':
            duplicates.append(result)
            print('duplicate')
        else:
            errors.append(result)
            print(f'ERROR: {result.get("error", "?")}')

    return uploaded, duplicates, errors


def _deploy_to_device(
    server: str, device_name: str, model_names: list[str]
) -> tuple[list[str], list[str]]:
    """Deploy models to a device, returning (deployed, deploy_errors)."""
    device_id = _get_device_id(server, device_name)
    if not device_id:
        print(f'\nWARNING: Device "{device_name}" not found, skipping deploy')
        return [], []

    deployed: list[str] = []
    deploy_errors: list[str] = []
    print(f'\nDeploying to device: {device_name} ({device_id})')

    for model_name in model_names:
        print(f'  Deploying {model_name}...', end=' ', flush=True)
        if _deploy_model(server, device_id, model_name):
            deployed.append(model_name)
            print('OK')
        else:
            deploy_errors.append(model_name)
            print('FAILED')

    return deployed, deploy_errors


def main() -> None:
    args = _parse_args()
    server = args.server.rstrip('/')
    models_dir = Path(args.dir)
    edgetpu_dir = Path(args.edgetpu_dir) if args.edgetpu_dir else models_dir / 'edgetpu'

    print('Edge-Bench Model Uploader')
    print(f'Server: {server}')
    print(f'Models: {models_dir}')
    print()

    if not _check_server(server):
        print(f'ERROR: Cannot connect to server {server}')
        sys.exit(1)
    print('Server: OK')

    # Discover models
    models = _discover_models(models_dir, args.pattern)
    if edgetpu_dir.exists() and args.pattern == '*.tflite':
        models.extend(_discover_models(edgetpu_dir, '*.tflite'))

    if not models:
        print(f'No models found matching {args.pattern} in {models_dir}')
        sys.exit(0)

    print(f'Found {len(models)} model(s):')
    for m in models:
        print(f'  {m.name} ({m.stat().st_size / 1024 / 1024:.1f} MB)')
    print()

    existing = _get_existing_models(server)
    print(f'Existing models on server: {len(existing)}')

    uploaded, duplicates, errors = _upload_all_models(server, models, existing)

    deployed, deploy_errors = [], []
    if args.deploy_to:
        all_names = [m['name'] for m in uploaded + duplicates]
        deployed, deploy_errors = _deploy_to_device(server, args.deploy_to, all_names)

    _print_results(uploaded, duplicates, errors, deployed, deploy_errors)


if __name__ == '__main__':
    main()
