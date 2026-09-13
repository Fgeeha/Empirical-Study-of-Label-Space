#!/usr/bin/env python3
"""Deterministic metadata / Unicode hygiene for the built manuscript files through the
local watermarks-remover HTTP service (WATERMARKS_SERVICE_URL, default
http://127.0.0.1:8765): document properties, embedded-image text chunks, invisible
characters, PDF Info/XMP. Overwrites the files in place and re-inspects them.

    python3 clean_outputs.py [files...]   # default: submission docx, preprint pdf, figures
Exits with a message and status 2 if the service is not reachable.
"""

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WM = os.environ.get('WATERMARKS_SERVICE_URL', 'http://127.0.0.1:8765')
DEFAULT = [
    HERE / 'submission/Kolesnikov_Kravets_AI.docx',
    HERE / 'preprint/Kolesnikov_Kravets_preprint.pdf',
    *sorted((HERE / 'figures').glob('*.png')),
]


def call(endpoint: str, path: Path, options: dict | None = None) -> dict:
    body = {'file': base64.b64encode(path.read_bytes()).decode(), 'name': path.name}
    if options:
        body['options'] = options
    req = urllib.request.Request(
        f'{WM}/{endpoint}', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'}
    )
    return json.load(urllib.request.urlopen(req, timeout=600))


def main() -> None:
    try:
        urllib.request.urlopen(f'{WM}/health', timeout=5)
    except (urllib.error.URLError, OSError):
        sys.exit(f'watermarks service not reachable at {WM}; start it (docker compose up -d wr-core) and rerun')
    files = [Path(a) for a in sys.argv[1:]] or DEFAULT
    for f in files:
        before = call('inspect', f)
        rep = call('clean', f, {'keep_non_ai_metadata': True})
        data = base64.b64decode(rep['cleaned'])
        changed = data != f.read_bytes()
        if changed:
            f.write_bytes(data)
        after = call('inspect', f)
        acts = [a for a in rep.get('report', {}).get('actions', []) if 'no ' not in a[:4]]
        print(
            f'{f.relative_to(HERE)}: layer-A hits {before.get("report", {}).get("suspicious_total")} -> '
            f'{after.get("report", {}).get("suspicious_total")}, verdict {before.get("suspicious", {}).get("verdict")} -> '
            f'{after.get("suspicious", {}).get("verdict")}, {"rewritten" if changed else "unchanged"}; '
            f'{len(acts)} actions'
        )
        for a in acts:
            print('   -', a)


if __name__ == '__main__':
    main()
