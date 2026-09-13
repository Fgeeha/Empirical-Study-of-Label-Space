#!/usr/bin/env python3
"""
Edge-Bench Integration for ECCV 2026

Full cycle: upload models -> deploy to RPi -> create experiments -> wait for results.

Usage:
    # Full cycle: upload, deploy, run experiments, wait
    python scripts/9.9_run_edgebench.py --server http://192.168.1.4:8000

    # Just upload models (no experiments)
    python scripts/9.9_run_edgebench.py --upload-only

    # Check server/device status
    python scripts/9.9_run_edgebench.py --status

    # Specify device explicitly
    python scripts/9.9_run_edgebench.py --device raspberrypi

    # Wait longer for experiments (default 600s per experiment)
    python scripts/9.9_run_edgebench.py --timeout 900
"""

import argparse
from pathlib import Path
import sys
import time

try:
    import httpx
except ImportError:
    print('[ERROR] httpx not installed. Run: pip install httpx')
    sys.exit(1)


EXPORT_DIR = Path('export')
EDGETPU_DIR = EXPORT_DIR / 'edgetpu'

_STRATEGIES = ['hybrid', 'Fuzzy', 'sbert']

# EdgeTPU-compatible architectures (run both CPU and EdgeTPU backends)
_EDGETPU_ARCHS = ['mobilenetv2', 'mobilenetv1', 'efficientnet_lite0', 'resnet50']
# CPU-only architectures (Swish + SE blocks not supported by EdgeTPU)
_CPU_ONLY_ARCHS = ['efficientnet']


# ---------------------------------------------------------------------------
# Server API helpers
# ---------------------------------------------------------------------------


def check_server(base_url: str) -> bool:
    try:
        resp = httpx.get(f'{base_url}/api/devices', timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_devices(base_url: str) -> list:
    resp = httpx.get(f'{base_url}/api/devices', timeout=10)
    resp.raise_for_status()
    return resp.json()


def upload_model(base_url: str, model_path: Path) -> dict:
    with open(model_path, 'rb') as f:
        files = {'file': (model_path.name, f, 'application/octet-stream')}
        data = {'file_type': 'model'}
        resp = httpx.post(
            f'{base_url}/api/files/upload', files=files, data=data, timeout=120
        )
    resp.raise_for_status()
    return resp.json()


def get_existing_models(base_url: str) -> dict[str, str]:
    """Return {model_name: model_id} for models already on server."""
    try:
        resp = httpx.get(f'{base_url}/api/files?file_type=model', timeout=10)
        if resp.status_code == 200:
            return {f['name']: f['id'] for f in resp.json()}
    except Exception:
        pass
    return {}


def deploy_model_to_device(base_url: str, device_id: str, model_name: str) -> bool:
    """Deploy (copy) a model from server to a device."""
    try:
        resp = httpx.post(
            f'{base_url}/api/devices/{device_id}/deploy-by-name',
            json={'model_name': model_name},
            timeout=120,
        )
        return resp.status_code == 200
    except Exception:
        return False


def create_experiment(
    base_url: str,
    device_id: str,
    model_name: str,
    backend: str,
    runs: int = 100,
) -> dict:
    # Server schema: ExperimentCreate requires name, device_id, model_path
    # Agent resolves model_path via os.path.expanduser + os.path.abspath.
    # Models are deployed to ~/models/ on the RPi, so use that path.
    payload = {
        'name': f'{model_name}_{backend}',
        'device_id': device_id,
        'model_path': f'~/models/{model_name}',
        'params': {
            'backend': backend,
            'benchmark_runs': runs,
            'warmup_runs': 20,
            'num_threads': 4,
        },
    }
    resp = httpx.post(f'{base_url}/api/experiments', json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def wait_for_experiment(base_url: str, experiment_id: str, timeout: int = 600) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        resp = httpx.get(f'{base_url}/api/experiments/{experiment_id}', timeout=10)
        exp = resp.json()
        status = exp.get('status', 'unknown')
        if status in ['completed', 'failed', 'error']:
            return exp
        elapsed = int(time.time() - start)
        print(f'    status={status} ({elapsed}s)...', end='\r', flush=True)
        time.sleep(3)
    return {'status': 'timeout'}


def export_t4_csv(base_url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resp = httpx.get(f'{base_url}/api/results/export/csv', timeout=30)
    resp.raise_for_status()
    output_path.write_text(resp.text)
    print(f'[OK] T4 exported: {output_path}')


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------


def find_all_models() -> list[Path]:
    """Find all ECCV INT8 TFLite + EdgeTPU models."""
    models: list[Path] = []
    all_archs = _EDGETPU_ARCHS + _CPU_ONLY_ARCHS

    for arch in all_archs:
        for strat in _STRATEGIES:
            # INT8 model
            int8 = EXPORT_DIR / f'{arch}_int8_ptq_{strat}.tflite'
            if int8.exists():
                models.append(int8)
            # EdgeTPU-compiled model
            edgetpu = EDGETPU_DIR / f'{arch}_int8_ptq_{strat}_edgetpu.tflite'
            if edgetpu.exists():
                models.append(edgetpu)

    return models


def model_backend(model_path: Path) -> str:
    """Determine backend for a model."""
    return 'edgetpu' if '_edgetpu' in model_path.name else 'cpu'


# ---------------------------------------------------------------------------
# Main workflow steps
# ---------------------------------------------------------------------------


def step_upload(server: str, models: list[Path]) -> list[str]:
    """Upload models to server, return list of model names on server."""
    existing = get_existing_models(server)
    uploaded_names: list[str] = list(existing.keys())

    print('\n[STEP 1] Upload models to server')
    print(f'  Found locally:      {len(models)}')
    print(f'  Already on server:  {len(existing)}')

    for model_path in models:
        if model_path.name in existing:
            print(f'  = {model_path.name} (exists)')
            continue
        print(f'  + {model_path.name}...', end=' ', flush=True)
        try:
            upload_model(server, model_path)
            uploaded_names.append(model_path.name)
            print('OK')
        except Exception as e:
            print(f'FAILED ({e})')

    print(f'  Total on server: {len(uploaded_names)}')
    return uploaded_names


def step_deploy(
    server: str, device_id: str, device_name: str, model_names: list[str]
) -> int:
    """Deploy all models to RPi device. Returns count of deployed."""
    print(f'\n[STEP 2] Deploy models to device: {device_name}')
    deployed = 0
    for name in model_names:
        print(f'  -> {name}...', end=' ', flush=True)
        if deploy_model_to_device(server, device_id, name):
            deployed += 1
            print('OK')
        else:
            print('FAILED')
    print(f'  Deployed: {deployed}/{len(model_names)}')
    return deployed


def step_run_experiments(
    server: str, device_id: str, models: list[Path], runs: int, timeout: int
) -> list[dict]:
    """Create experiments and wait for each to complete."""
    print(f'\n[STEP 3] Run benchmark experiments (runs={runs}, timeout={timeout}s)')

    results = []
    total = len(models)

    for i, model_path in enumerate(models, 1):
        backend = model_backend(model_path)
        model_name = model_path.name
        print(f'\n  [{i}/{total}] {model_name} [{backend}]')

        # Create experiment
        print('    Creating experiment...', end=' ', flush=True)
        try:
            exp = create_experiment(server, device_id, model_name, backend, runs)
            exp_id = exp['id']
            print(f'OK (id={exp_id})')
        except Exception as e:
            print(f'FAILED ({e})')
            results.append(
                {
                    'model': model_name,
                    'backend': backend,
                    'status': 'create_failed',
                    'error': str(e),
                }
            )
            continue

        # Wait for completion
        print('    Waiting for results...', flush=True)
        result = wait_for_experiment(server, exp_id, timeout=timeout)
        status = result.get('status', 'unknown')
        print(f'    Result: {status.upper()}')

        results.append(
            {
                'model': model_name,
                'backend': backend,
                'status': status,
                'experiment_id': exp_id,
            }
        )

    return results


def find_device(server: str, device_name: str | None) -> tuple[str, str]:
    """Find device ID, optionally by name. Returns (device_id, device_name)."""
    devices = get_devices(server)
    if not devices:
        print('[ERROR] No devices registered on server.')
        print('        Register a device in the Edge-Bench Web UI first.')
        sys.exit(1)

    if device_name:
        for d in devices:
            if d['name'] == device_name:
                return d['id'], d['name']
        print(f'[ERROR] Device "{device_name}" not found.')
        print(f'        Available: {[d["name"] for d in devices]}')
        sys.exit(1)

    # Auto-select: prefer online, fall back to first
    online = [d for d in devices if d.get('online')]
    if online:
        d = online[0]
        print(f'[INFO] Auto-selected online device: {d["name"]}')
        return d['id'], d['name']

    # No online devices -- use first registered and warn
    d = devices[0]
    print(f'[WARN] No online devices. Using: {d["name"]} (may be offline)')
    print('       Make sure RPi agent is running: cd edge-bench && make agent')
    return d['id'], d['name']


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Edge-Bench: upload, deploy, benchmark'
    )
    parser.add_argument(
        '--server',
        '-s',
        default='http://192.168.1.4:8000',
        help='Edge-Bench server URL',
    )
    parser.add_argument(
        '--device',
        '-d',
        default=None,
        help='Device name (auto-selects online device if omitted)',
    )
    parser.add_argument(
        '--upload-only',
        action='store_true',
        help='Only upload models, do not deploy or run',
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Check server and device status',
    )
    parser.add_argument(
        '--runs',
        '-r',
        type=int,
        default=100,
        help='Benchmark runs per experiment (default: 100)',
    )
    parser.add_argument(
        '--timeout',
        '-t',
        type=int,
        default=600,
        help='Timeout per experiment in seconds (default: 600)',
    )
    parser.add_argument(
        '--export-csv',
        type=Path,
        default=None,
        help='Export results to CSV after all experiments complete',
    )
    return parser.parse_args()


def _print_status(server: str):
    devices = get_devices(server)
    print(f'\nRegistered devices: {len(devices)}')
    for dev in devices:
        status = 'online' if dev.get('online') else 'offline'
        print(
            f'  - {dev["name"]} ({dev.get("ip", "?")}:{dev.get("port", "?")}) [{status}]'
        )

    existing = get_existing_models(server)
    print(f'\nModels on server: {len(existing)}')
    for name in sorted(existing.keys()):
        print(f'  - {name}')


def _ensure_server(server: str) -> None:
    """Check server connectivity or exit."""
    print('Checking server...', end=' ', flush=True)
    if not check_server(server):
        print('FAILED')
        print(f'[ERROR] Cannot reach server: {server}')
        print('        Start server: cd edge-bench && make server')
        sys.exit(1)
    print('OK')


def _discover_or_exit() -> list[Path]:
    """Find local models or exit."""
    models = find_all_models()
    if not models:
        print('[ERROR] No INT8 TFLite models found')
        print(f'        Expected in: {EXPORT_DIR}/')
        sys.exit(1)
    print(f'\nDiscovered {len(models)} model(s):')
    for m in models:
        size_mb = m.stat().st_size / 1024 / 1024
        print(f'  {m.name} ({size_mb:.1f} MB) [{model_backend(m)}]')
    return models


def _print_summary(server: str, results: list[dict], export_csv: Path | None) -> int:
    """Print experiment summary, export CSV. Return failure count."""
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] != 'completed')
    print(f'\n{"=" * 60}')
    print('SUMMARY')
    print(f'{"=" * 60}')
    print(f'  Total experiments:  {len(results)}')
    print(f'  Completed:          {completed}')
    print(f'  Failed/timeout:     {failed}')
    if failed > 0:
        print('\n  Failed experiments:')
        for r in results:
            if r['status'] != 'completed':
                print(f'    - {r["model"]} [{r["backend"]}]: {r["status"]}')
    if export_csv:
        export_t4_csv(server, export_csv)
    print(f'\n[DONE] View results: {server}/results')
    return failed


def main():
    args = _parse_args()
    server = args.server.rstrip('/')

    print('=' * 60)
    print('Edge-Bench: Upload -> Deploy -> Benchmark')
    print('=' * 60)
    print(f'  Server:  {server}')
    print(f'  Runs:    {args.runs}')
    print(f'  Timeout: {args.timeout}s per experiment')
    print()

    _ensure_server(server)

    if args.status:
        _print_status(server)
        return

    models = _discover_or_exit()
    model_names = step_upload(server, models)

    if args.upload_only:
        print('\n[DONE] Upload complete (--upload-only)')
        return

    device_id, device_name = find_device(server, args.device)
    step_deploy(server, device_id, device_name, model_names)
    results = step_run_experiments(server, device_id, models, args.runs, args.timeout)

    if _print_summary(server, results, args.export_csv) > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
