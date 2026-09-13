#!/usr/bin/env python3
"""Generate Figure 1 (study overview) as SVG; render to PNG with headless Chrome.

Usage: python3 make_figure1.py && google-chrome --headless=new --no-sandbox \
    --force-device-scale-factor=4 --window-size=900,760 \
    --screenshot=figure1_pipeline.png file://$PWD/figure1_pipeline.svg
All numbers must match main.md (checked by ../check_numbers.py).
"""

from pathlib import Path

W, H = 900, 760
FONT = 'Helvetica, Arial, sans-serif'
out = []


def rect(x, y, w, h, fill='#FFFFFF', stroke='#555', sw=1.5, rx=6):
    out.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" ry="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def text(x, y, s, size=13, bold=False, anchor='middle', fill='#1a1a1a'):
    fw = ' font-weight="bold"' if bold else ''
    out.append(
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}"{fw} '
        f'text-anchor="{anchor}" fill="{fill}">{s}</text>'
    )


def lines(x, y, rows, size=12, dy=16, bold_first=False, anchor='middle'):
    for i, r in enumerate(rows):
        text(x, y + i * dy, r, size, bold=(bold_first and i == 0), anchor=anchor)


def arrow(x1, y1, x2, y2):
    out.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#555" '
        f'stroke-width="1.8" marker-end="url(#ah)"/>'
    )


def box(x, y, w, h, rows, fill='#FFFFFF', size=12, dy=15):
    rect(x, y, w, h, fill)
    top = y + h / 2 - (len(rows) - 1) * dy / 2 + 4
    lines(x + w / 2, top, rows, size, dy, bold_first=True)


out.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}"><defs><marker id="ah" markerWidth="8" markerHeight="8" '
    'refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#555"/>'
    f'</marker></defs><rect width="{W}" height="{H}" fill="white"/>'
)

# --- Row 1: data sources -----------------------------------------------------
box(
    40,
    20,
    250,
    58,
    ['PlantVillage (source, labelled)', '38 classes, laboratory images'],
    '#E8E8E8',
)
box(
    325,
    20,
    250,
    58,
    ['PlantDoc (target, unlabelled)', '28 class folders, field images'],
    '#E8E8E8',
)
box(
    610,
    20,
    250,
    58,
    ['PlantWild v2 (second target, block C)', '115 classes, 11,488 web images'],
    '#E8E8E8',
)

# --- Block A: label-space alignment ------------------------------------------
rect(40, 112, 820, 118, '#D6EAF8')
text(450, 134, 'A. Label-space alignment (PlantVillage ↔ PlantDoc)', 14, bold=True)
box(60, 148, 240, 66, ['Fuzzy: token-set ratio ≥ 80', '16 shared classes'])
box(330, 148, 240, 66, ['SBERT: cosine ≥ 0.75', '15 shared classes'])
box(
    600,
    148,
    240,
    66,
    ['Hybrid: fuzzy, SBERT fallback', '18 shared classes (main setting)'],
    '#FFF6D5',
)
for cx in (165, 450):
    arrow(cx, 78, cx, 112)

# --- Block B: fair-protocol UDA benchmark ------------------------------------
arrow(450, 230, 450, 262)
rect(40, 262, 820, 150, '#D6EAF8')
text(
    450,
    284,
    'B. Unsupervised domain adaptation, fair protocol (ResNet-50, Hybrid 18 classes)',
    14,
    bold=True,
)
for i, (name, note) in enumerate(
    [
        ('AdaBN', 'BN statistics'),
        ('CORAL', '2nd-order'),
        ('DANN', 'adversarial'),
        ('CDAN+E', 'conditional adv.'),
        ('MCC', 'class confusion'),
    ]
):
    box(60 + i * 160, 300, 140, 50, [name, note])
box(
    60,
    360,
    780,
    40,
    [
        'Checkpoint chosen on PlantVillage validation macro-F1 only · no target labels at any stage · seeds 42/43/44 · paired bootstrap CI'
    ],
    '#FFFFFF',
    size=11.5,
    dy=0,
)

# --- Block C: evaluation -----------------------------------------------------
arrow(450, 412, 450, 444)
rect(40, 444, 820, 110, '#D6EAF8')
text(450, 466, 'C. Evaluation', 14, bold=True)
box(60, 482, 240, 58, ['PlantDoc test, n = 1,969', 'source-only 0.156 → DANN 0.246'])
box(
    330,
    482,
    240,
    58,
    ['PlantWild v2, n = 2,336 (17 classes)', 'DANN 0.238, CDAN+E 0.230'],
)
box(
    600,
    482,
    240,
    58,
    ['Shared 13-class subset, n = 1,509', 'fair cross-strategy comparison'],
)

# --- Block D: edge deployment -------------------------------------------------
arrow(450, 554, 450, 586)
rect(40, 586, 820, 150, '#D6EAF8')
text(
    450,
    608,
    'D. Edge deployment (INT8 post-training quantization, 200 calibration images)',
    14,
    bold=True,
)
box(
    60,
    624,
    380,
    96,
    [
        'Raspberry Pi 4 Model B (8 GB) + Coral USB Edge TPU',
        'TFLite 2.14, Edge TPU compiler 16.0',
        '20 warm-up + 200 timed runs, batch 1, 4 threads',
    ],
)
box(
    460,
    624,
    380,
    96,
    [
        'Five architectures: MobileNetV1/V2,',
        'EfficientNet-Lite0, ResNet-50, EfficientNet-B3',
        'best: MobileNetV1 4.6 ms (216 FPS), 3.36 MiB',
    ],
)
text(
    450,
    752,
    'Figure 1. Overview of the study: alignment (A) → adaptation (B) → evaluation (C) → deployment (D).',
    11,
    fill='#444',
)
out.append('</svg>')
Path(__file__).with_name('figure1_pipeline.svg').write_text('\n'.join(out))
print('svg written')
