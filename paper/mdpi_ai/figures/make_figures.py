#!/usr/bin/env python3
"""Figures 2 and 4 for the MDPI manuscript, generated from canonical result files.

Run from the repository root with an environment that has matplotlib:
    python paper/mdpi_ai/figures/make_figures.py
Figure 2: UDA benchmark on PlantDoc (mean ± std over seeds, per-seed points,
           bootstrap CI of Δ vs source-only) — results/uda_benchmark/.
Figure 4: edge latency and throughput, Hybrid models — results/final/T4_system_reeval.csv (re-measured 2026-09-13).
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DPI = 600
plt.rcParams.update(
    {
        'font.size': 9,
        'font.family': 'DejaVu Sans',
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
)

# ---------------- Figure 2 -----------------------------------------------------
summ = json.load(open(ROOT / 'results/uda_benchmark/aggregated/uda_summary.json'))
boot = json.load(
    open(ROOT / 'results/uda_benchmark/bootstrap/bootstrap_vs_source.json')
)
src = summ['source_only']
order = [
    ('adabn', 'AdaBN'),
    ('coral', 'CORAL'),
    ('dann', 'DANN'),
    ('cdan', 'CDAN+E'),
    ('mcc', 'MCC'),
]
t1 = {
    r['strategy']: r
    for r in csv.DictReader(open(ROOT / 'results/final/T1_domain_shift.csv'))
}
src_f1 = float(t1['hybrid']['target_macro_f1'])

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(7.2, 3.0), gridspec_kw={'width_ratios': [1.25, 1]}
)
xs = range(len(order))
means = [summ['methods'][m]['macro_f1_mean'] for m, _ in order]
stds = [summ['methods'][m]['macro_f1_std'] for m, _ in order]
ax1.bar(
    xs, means, yerr=stds, capsize=3, color='#9dbfe0', edgecolor='#2b4c7e', linewidth=0.8
)
for i, (m, _) in enumerate(order):
    per = [v['macro_f1'] for v in summ['methods'][m]['per_seed'].values()]
    ax1.scatter([i] * len(per), per, s=14, color='#1f3a5f', zorder=3)
ax1.axhline(src_f1, color='#b03a2e', linestyle='--', linewidth=1)
ax1.text(
    len(order) - 0.55,
    src_f1 + 0.006,
    f'source-only {round(src_f1 + 1e-9, 3):.3f}',
    color='#b03a2e',
    ha='right',
    fontsize=8,
)
ax1.set_xticks(list(xs))
ax1.set_xticklabels([n for _, n in order])
ax1.set_ylabel('Target macro-F1 (PlantDoc, 18 classes)')
ax1.set_ylim(0, 0.32)
ax1.set_title('(a) Mean ± std over seeds 42/43/44', fontsize=9)

d = [boot[m]['vs_source_only'] for m, _ in order]
ax2.errorbar(
    [x['mean_diff'] * 100 for x in d],
    xs,
    xerr=[
        [(x['mean_diff'] - x['ci_lower']) * 100 for x in d],
        [(x['ci_upper'] - x['mean_diff']) * 100 for x in d],
    ],
    fmt='o',
    color='#1f3a5f',
    ecolor='#2b4c7e',
    capsize=3,
    markersize=4,
)
ax2.axvline(0, color='#b03a2e', linestyle='--', linewidth=1)
ax2.set_yticks(list(xs))
ax2.set_yticklabels([n for _, n in order])
ax2.invert_yaxis()
ax2.set_xlabel('Δ macro-F1 vs source-only, percentage points', fontsize=8)
ax2.set_title('(b) Paired bootstrap 95% CI (seed 42)', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'figure2_uda_benchmark.png', dpi=DPI, metadata={'Software': None})
print('figure2 ok', {m: round(v, 4) for m, v in zip([o[0] for o in order], means)})

# ---------------- Figure 4 -----------------------------------------------------
rows = [
    r
    for r in csv.DictReader(open(ROOT / 'results/final/T4_system_reeval.csv'))
    if r['Strategy'] == 'hybrid'
]
archs = ['MobileNetV1', 'MobileNetV2', 'EffNet-Lite0', 'ResNet-50', 'EffNet-B3']
labels = [
    'MobileNetV1',
    'MobileNetV2',
    'EfficientNet-Lite0',
    'ResNet-50',
    'EfficientNet-B3',
]
cpu = {r['Model']: float(r['Latency (ms)']) for r in rows if r['Backend'] == 'CPU'}
tpu = {r['Model']: float(r['Latency (ms)']) for r in rows if r['Backend'] == 'EDGETPU'}
size = {r['Model']: float(r['Size (MB)']) for r in rows if r['Backend'] == 'CPU'}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
w = 0.38
ax1.bar(
    [i - w / 2 for i in range(5)],
    [cpu[a] for a in archs],
    w,
    label='Raspberry Pi 4 CPU',
    color='#c9d6e3',
    edgecolor='#2b4c7e',
    linewidth=0.8,
)
ax1.bar(
    [i + w / 2 for i in range(5) if archs[i] in tpu],
    [tpu[a] for a in archs if a in tpu],
    w,
    label='Coral USB Edge TPU',
    color='#2b4c7e',
    edgecolor='#2b4c7e',
)
for i, a in enumerate(archs):
    ax1.text(i - w / 2, cpu[a] * 1.08, f'{cpu[a]:.0f}', ha='center', fontsize=7)
    if a in tpu:
        ax1.text(i + w / 2, tpu[a] * 1.08, f'{tpu[a]:.1f}', ha='center', fontsize=7)
    else:
        ax1.text(i + w / 2, 4, 'n/a', ha='center', fontsize=7, color='#b03a2e')
ax1.set_yscale('log')
ax1.set_ylim(3, 400)
ax1.set_xticks(range(5))
ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
ax1.set_ylabel('Latency per image, ms (log scale)')
ax1.set_title('(a) INT8 latency, 200 timed runs', fontsize=9)
ax1.legend(fontsize=7, frameon=False)
fps_tpu = {a: 1000 / tpu[a] for a in tpu}
ax2.scatter(
    [size[a] for a in archs if a in tpu],
    [fps_tpu[a] for a in archs if a in tpu],
    s=28,
    color='#2b4c7e',
)
for a, lab in zip(archs, labels):
    if a in tpu:
        ax2.annotate(
            lab,
            (size[a], fps_tpu[a]),
            textcoords='offset points',
            xytext=(5, 3),
            fontsize=7,
        )
ax2.set_xscale('log')
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
ax2.xaxis.set_major_locator(FixedLocator([2, 3, 5, 10, 20, 30]))
ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.xaxis.set_minor_formatter(NullFormatter())
ax2.set_xlabel('INT8 model size, MiB (log scale)')
ax2.set_ylabel('Throughput on Edge TPU, frames/s')
ax2.set_title('(b) Size vs throughput (Edge TPU)', fontsize=9)
fig.tight_layout()
fig.savefig(OUT / 'figure4_edge.png', dpi=DPI, metadata={'Software': None})
print('figure4 ok', {a: (cpu[a], tpu.get(a)) for a in archs})
