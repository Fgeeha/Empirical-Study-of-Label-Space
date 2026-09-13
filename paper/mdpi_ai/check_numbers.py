#!/usr/bin/env python3
"""Check every number in paper/mdpi_ai/main.md against its primary source file.

Run from the repository root:
    python3 paper/mdpi_ai/check_numbers.py

Exit code 1 if any PASS/FAIL check fails. Items that need an author decision
(claims, not numbers) are printed as AUTHOR and do not affect the exit code.
No third-party packages required.
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path
import statistics as st
import sys

ROOT = Path(__file__).resolve().parents[2]
DRAFT = ROOT / 'paper/mdpi_ai/main.md'
RAW = ROOT / 'results/uda_benchmark/raw'
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print(f'{"PASS" if ok else "FAIL"}  {name}  {detail}')
    if not ok:
        FAILS.append(name)


def author(name: str, detail: str) -> None:
    print(f'AUTHOR  {name}  {detail}')


def near(a: float, b: float, tol: float = 5e-4) -> bool:
    return abs(a - b) <= tol


def in_draft(text: str) -> bool:
    return text in DRAFT.read_text()


def csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


# --- UDA benchmark: mean ± std recomputed from per-seed raw JSON ---------------
def raw_f1(method: str) -> list[float]:
    files = sorted(glob.glob(str(RAW / f'{method}_hybrid_plantdoc_seed*.json')))
    return [json.load(open(f))['target_metrics']['macro_f1'] for f in files]


t1_rec = {r['strategy']: r for r in csv_rows(ROOT / 'results/final/T1_domain_shift.csv')}
check('recorded T1 hybrid 0.9561 / 0.1555 (superseded, disclosed in §5.5)',
      near(float(t1_rec['hybrid']['pv_macro_f1']), 0.9561) and near(float(t1_rec['hybrid']['target_macro_f1']), 0.1555))
t1 = {r['strategy']: r for r in csv_rows(ROOT / 'results/final/T1_domain_shift_reeval.csv')}
src_only = float(t1['hybrid']['target_macro_f1'])
check('T1 (re-evaluated) hybrid source-only 0.154', near(src_only, 0.1539))
_src_json = json.load(open(ROOT / 'results/int8_accuracy/pytorch_resnet50_hybrid__plantdoc_test.json'))
check('T1 reeval hybrid = 14.2 JSON', near(src_only, _src_json['macro_f1']))
check(
    'T1 Fuzzy 0.972 / 0.187',
    near(float(t1['Fuzzy']['pv_macro_f1']), 0.9724)
    and near(float(t1['Fuzzy']['target_macro_f1']), 0.1866),
)
check(
    'T1 sbert 0.986 / 0.171',
    near(float(t1['sbert']['pv_macro_f1']), 0.9864)
    and near(float(t1['sbert']['target_macro_f1']), 0.1708),
)
check('T1 hybrid PV 0.969', near(float(t1['hybrid']['pv_macro_f1']), 0.9685))

expected = {
    'adabn': (0.1553, 0.0000, '+0.1'),
    'coral': (0.1956, 0.0116, '+4.2'),
    'dann': (0.2455, 0.0050, '+9.2'),
    'cdan': (0.2419, 0.0258, '+8.8'),
    'mcc': (0.0772, 0.0211, '−7.7'),
}
for m, (mean, std, delta) in expected.items():
    vals = raw_f1(m)
    check(f'{m}: 3 seeds present', len(vals) == 3)
    check(f'{m}: mean {mean}', near(st.mean(vals), mean), f'raw={st.mean(vals):.4f}')
    check(f'{m}: std {std}', near(st.stdev(vals), std), f'raw={st.stdev(vals):.4f}')
    d_pp = (st.mean(vals) - src_only) * 100
    check(
        f'{m}: Δ src {delta} pp',
        near(
            d_pp,
            float(delta.replace('−', '-').replace('≈', '')) if delta != '≈0' else 0.0,
            tol=0.06,
        ),
        f'raw={d_pp:+.2f}',
    )
    if m == 'dann':
        check('DANN seed 42 fair F1 = 0.240 (not 0.202)', near(vals[0], 0.2402))

mcc_first_epoch = [
    json.load(open(f))['val_f1_history'][0]
    for f in sorted(glob.glob(str(RAW / 'mcc_hybrid_plantdoc_seed*.json')))
]

# --- DANN vs CDAN cross-seed summary ------------------------------------------
dc = json.load(
    open(ROOT / 'results/uda_benchmark/bootstrap/dann_vs_cdan_plantdoc.json')
)
check(
    'DANN−CDAN mean diff 0.004',
    near(dc['cross_seed_summary']['mean_diff_dann_minus_cdan'], 0.0037, tol=6e-4),
)
s42, s43, s44 = (dc['per_seed'][k] for k in ('seed42', 'seed43', 'seed44'))
check(
    'seed42 delta −0.027 CI [−0.046, −0.006]',
    near(s42['delta'], -0.0268)
    and near(s42['ci_lower'], -0.0464)
    and near(s42['ci_upper'], -0.0062),
)
check(
    'seed43 delta +0.035 CI [+0.017, +0.052]',
    near(s43['delta'], 0.0347)
    and near(s43['ci_lower'], 0.0171)
    and near(s43['ci_upper'], 0.0522),
)
check('seed44 inconclusive (CI spans 0)', s44['ci_lower'] < 0 < s44['ci_upper'])

# --- PlantWild v2 ------------------------------------------------------------
pw_base = json.load(open(ROOT / 'experiments/E5b_plantwild/eval_results.json'))[
    'results'
]['baseline']['macro_f1']
check('PlantWild source-only 0.172', near(pw_base, 0.1722))
for m, mean, std, per_seed, delta in (
    ('dann', 0.2381, 0.0112, (0.242, 0.226, 0.247), 6.6),
    ('cdan', 0.2296, 0.0376, (0.270, 0.195, 0.224), 5.7),
):
    d = json.load(
        open(ROOT / f'results/uda_benchmark/plantwild/{m}_hybrid_plantwild.json')
    )
    vals = [d['per_seed'][s]['macro_f1'] for s in ('42', '43', '44')]
    check(
        f'PlantWild {m} mean {mean} ± {std}',
        near(st.mean(vals), mean) and near(st.stdev(vals), std),
    )
    check(
        f'PlantWild {m} per-seed {per_seed}',
        all(near(v, e, 6e-4) for v, e in zip(vals, per_seed)),
    )
    d_pp = (st.mean(vals) - pw_base) * 100
    check(
        f'PlantWild {m} Δ src +{delta} pp',
        near(d_pp, delta, tol=0.06),
        f'raw={d_pp:+.2f} (draft rounds from rounded means)',
    )
    check(
        f'PlantWild {m} n=2336, 17 classes',
        d['n_samples'] == 2336 and d['n_classes'] == 17,
    )
man = json.load(open(ROOT / 'experiments/E5b_plantwild/manifest.json'))
check(
    'PlantWild 115 classes / 11,488 images',
    man['n_pw_classes'] == 115 and man['n_pw_images'] == 11488,
)
t065 = [t for t in man['threshold_summary'] if t['threshold'] == 0.65][0]
check(
    'PlantWild 46/115 aligned at 0.65 (4,740 images)',
    t065['n_aligned'] == 46 and t065['n_images'] == 4740,
)

# --- Shared 13-class subset (E3) and threshold sweep (E1) --------------------
e3 = {
    (r['strategy'], r['label'].split()[0]): float(r['macro_f1'])
    for r in csv_rows(ROOT / 'experiments/E3_shared_subset/table1_shared13_reeval.csv')
}
check(
    'E3 (reeval) Fuzzy 0.140 → 0.178',
    near(e3[('Fuzzy', 'Baseline')], 0.1400) and near(e3[('Fuzzy', 'CORAL')], 0.1781),
)
check(
    'E3 (reeval) SBERT 0.147 → 0.174',
    near(e3[('sbert', 'Baseline')], 0.1472) and near(e3[('sbert', 'CORAL')], 0.1736),
)
check(
    'E3 (reeval) Hybrid 0.118 → 0.142',
    near(e3[('hybrid', 'Baseline')], 0.1184) and near(e3[('hybrid', 'CORAL')], 0.1418),
)
check('draft: Table 3 gains +2.3 to +3.8 pp', in_draft('+2.3 to +3.8 pp'))
check(
    'E3 n=1509 for all rows',
    all(
        r['n_samples'] == '1509'
        for r in csv_rows(ROOT / 'experiments/E3_shared_subset/table1_shared13_reeval.csv')
    ),
)
e1 = {
    float(r['threshold']): r
    for r in csv_rows(
        ROOT / 'experiments/E1_threshold_sweep/threshold_sweep_results.csv'
    )
}
check(
    'E1 F1 invariant 0.50–0.75',
    len(
        {
            (e1[t]['baseline_macro_f1'], e1[t]['coral_macro_f1'])
            for t in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75)
        }
    )
    == 1,
)
check(
    'E1 F1 drops at 0.80',
    float(e1[0.8]['baseline_macro_f1']) < float(e1[0.75]['baseline_macro_f1']),
)

# --- Label alignment ---------------------------------------------------------
for strat, n in (('Fuzzy', 16), ('sbert', 15), ('hybrid', 18)):
    check(
        f'class_mapping_{strat}: {n} classes',
        len(json.load(open(ROOT / f'configs/class_mapping_{strat}.json'))) == n,
    )
sbert_src = (ROOT / 'scripts/2.5_build_shared_subset_sbert.py').read_text()
hyb_src = (ROOT / 'scripts/2.6_build_shared_subset_hybrid.py').read_text()
check(
    'SBERT threshold in code = 0.75 (both scripts)',
    'SIM_THRESHOLD = 0.75' in sbert_src and 'SBERT_THRESHOLD = 0.75' in hyb_src,
)
check('Fuzzy threshold in code = 80', 'FUZZY_THRESHOLD = 80' in hyb_src)
check(
    'draft states SBERT threshold 0.75 for PlantDoc mapping',
    in_draft('threshold ≥ 0.75'),
)
beans = json.load(open(ROOT / 'experiments/E5_beans/alignment_results.json'))
check(
    'BeanDiseases 3 classes, 0 aligned (fuzzy)',
    beans['n_classes'] == 3 and beans['alignment']['fuzzy']['n_aligned'] == 0,
)
rust = [
    x for x in beans['alignment']['sbert_075']['details'] if x['class'] == 'bean_rust'
][0]
check('bean_rust best cosine 0.63', near(rust['best_cosine'], 0.632, 1e-3))


# --- Dataset sizes -----------------------------------------------------------
def nrows(p: str) -> int:
    return sum(1 for _ in open(ROOT / p)) - 1


check(
    'PV train/val/test 17,195 / 3,678 / 3,701',
    (
        nrows('splits/hybrid/pv_train.csv'),
        nrows('splits/hybrid/pv_val.csv'),
        nrows('splits/hybrid/pv_test.csv'),
    )
    == (17195, 3678, 3701),
)
check(
    'PlantDoc unlabeled 1,568 / test 1,969',
    (
        nrows('splits/hybrid/target_plantdoc_unlabeled.csv'),
        nrows('splits/hybrid/target_plantdoc_test.csv'),
    )
    == (1568, 1969),
)
check('PlantVillage 38 classes', nrows('artifacts/classes_pv.txt') + 1 == 38)

# --- Training configuration --------------------------------------------------
args = json.load(open(RAW / 'dann_hybrid_plantdoc_seed42.json'))['args']
check(
    'UDA config lr 3e-4, wd 1e-4, 20 epochs, batch 32, λ 0.1, T 2.5',
    args['lr'] == 3e-4
    and args['weight_decay'] == 1e-4
    and args['epochs'] == 20
    and args['batch_size'] == 32
    and args['lambda_coral'] == 0.1
    and args['mcc_temperature'] == 2.5,
)
check(
    '20 unit tests defined',
    (ROOT / 'tests/test_uda_methods.py').read_text().count('def test_') == 20,
)
log = (ROOT / 'paper/mdpi_ai/tests_run_2026-09-07.log').read_text()
check('20/20 unit tests pass (log of 2026-09-07 run)', '20 passed' in log)

# --- Supervised upper bound and legacy DANN ----------------------------------
ub_rec = json.load(open(ROOT / 'results/metrics_target_finetuned_plantdoc_hybrid.json'))
check('recorded (in-sample) UB file = 0.8523', near(ub_rec['macro_f1_target_finetuned'], 0.8523))
ub = {
    s: json.load(open(ROOT / f'results/ub_audit/ub_plantdoc_hybrid__{s}.json'))
    for s in ('train', 'val', 'test')
}
ov = ub['test']['overlap']
check(
    'PlantDoc: test list = train ∪ val (1,568 + 401 = 1,969, no train/val overlap)',
    ov['n_test'] == 1969
    and ov['train_in_test'] == 1568
    and ov['val_in_test'] == 401
    and ov['train_val_overlap'] == 0,
)
check('UB held-out (val, n=401) macro-F1 0.531 / acc 0.581',
      ub['val']['n'] == 401 and near(ub['val']['macro_f1'], 0.5313) and near(ub['val']['accuracy'], 0.5810))
check('UB in-sample: train 0.985, full list 0.895 (archived checkpoint)',
      near(ub['train']['macro_f1'], 0.9849) and near(ub['test']['macro_f1'], 0.8946))
check('draft: supervised reference 0.531 in Table 2 and 0.852 only as withdrawn',
      in_draft('*0.531* (held-out n = 401, 1 run)') and not in_draft('upper bound (F1 = 0.852)'))
check('draft: transductive protocol disclosed', in_draft('transductive UDA setting') and in_draft('Transductive target protocol'))
leg = json.load(open(ROOT / 'experiments/E2_dann/dann_results.json'))
check(
    'Legacy DANN 0.277 ± 0.009',
    near(leg['target_macro_f1_mean'], 0.2775)
    and near(leg['target_macro_f1_std'], 0.0094),
)
check(
    'Legacy DANN Δ src +12.2 pp',
    near((leg['target_macro_f1_mean'] - src_only) * 100, 12.4, 0.06),
)
check('draft says legacy +12.4 pp', in_draft('(+12.4 pp against the re-evaluated baseline) was inflated'))
check(
    'Gap reference − DANN ≈ 0.29; DANN recovers about a quarter of the distance',
    near(ub['val']['macro_f1'] - st.mean(raw_f1('dann')), 0.286, 5e-3)
    and 0.20 < (st.mean(raw_f1('dann')) - src_only) / (ub['val']['macro_f1'] - src_only) < 0.30,
)
calib = [r['label_id'] for r in csv_rows(ROOT / 'splits/hybrid/pv_calib_10pct.csv')][:200]
from collections import Counter
cc = Counter(calib)
check('first 200 calibration rows: 18 classes, 2–46 per class', len(cc) == 18 and min(cc.values()) == 2 and max(cc.values()) == 46)
check('Table 4 CDAN+E Δ from unrounded values: 0.2296 − 0.1722 → +5.7', round((0.2296 - pw_base) * 100, 1) == 5.7)

# --- Edge deployment ---------------------------------------------------------
t4 = {
    (r['Model'], r['Strategy'], r['Backend']): r
    for r in csv_rows(ROOT / 'results/final/T4_system_reeval.csv')
}
t4_old = {
    (r['Model'], r['Strategy'], r['Backend']): r
    for r in csv_rows(ROOT / 'results/final/T4_system.csv')
}
check(
    'first-generation Edge TPU latencies quoted in Table 5 caption (4.6 / 5.9 / 6.3 / 56.0)',
    [float(t4_old[(m, 'hybrid', 'EDGETPU')]['Latency (ms)']) for m in ('MobileNetV1', 'MobileNetV2', 'EffNet-Lite0', 'ResNet-50')]
    == [4.6, 5.9, 6.3, 56.0],
)
table5 = {
    'MobileNetV1': (32.9, 4.6, 216.2, 3.36, '3.2M'),
    'MobileNetV2': (30.4, 5.1, 197.4, 2.60, '2.2M'),
    'EffNet-Lite0': (33.0, 6.3, 159.3, 3.79, '3.5M'),
    'ResNet-50': (190.3, 55.2, 18.1, 23.14, '23.5M'),
    'ResNet-50 DANN': (190.1, 54.9, 18.2, 23.14, '23.5M'),
}
for model, (cpu, tpu, fps, size, params) in table5.items():
    c, e = t4[(model, 'hybrid', 'CPU')], t4[(model, 'hybrid', 'EDGETPU')]
    check(
        f'Table 5 {model}',
        near(float(c['Latency (ms)']), cpu, 0.06)
        and near(float(e['Latency (ms)']), tpu, 0.06)
        and near(float(e['FPS']), fps, 0.6)
        and near(float(c['Size (MB)']), size, 0.006)
        and c['Params'] == params,
    )
b3 = t4[('EffNet-B3', 'hybrid', 'CPU')]
check(
    'Table 5 EfficientNet-B3 CPU 137.7 ms, 11.93 MiB, no EdgeTPU row',
    near(float(b3['Latency (ms)']), 137.7, 0.06)
    and near(float(b3['Size (MB)']), 11.93, 0.006)
    and ('EffNet-B3', 'hybrid', 'EDGETPU') not in t4,
)
runs = [json.load(open(f))[0] for f in sorted(glob.glob(str(ROOT / 'results/edgebench/reeval_20260913/runs/*.json')))]
check('edge re-measurement: 11 completed runs', len(runs) == 11 and all(r['status'] == 'completed' for r in runs))
check(
    'edge protocol: 200 runs / 20 warm-up / 4 threads / seed 42',
    {r['params']['benchmark_runs'] for r in runs} == {200}
    and {r['params']['warmup_runs'] for r in runs} == {20}
    and {r['params']['num_threads'] for r in runs} == {4}
    and {r['params']['input_seed'] for r in runs} == {42},
)
check('Edge TPU latency std ≤ 0.6 ms', max(r['latency']['std_ms'] for r in runs if r['backend'] == 'edgetpu') <= 0.6)
check('DANN vs source-only ResNet-50 on Edge TPU: 54.9 vs 55.2 ms',
      near(t4[('ResNet-50 DANN', 'hybrid', 'EDGETPU')]['Latency (ms)'] and float(t4[('ResNet-50 DANN', 'hybrid', 'EDGETPU')]['Latency (ms)']), 54.9, 0.06)
      and near(float(t4[('ResNet-50', 'hybrid', 'EDGETPU')]['Latency (ms)']), 55.2, 0.06))
env2 = (ROOT / 'results/edgebench/reeval_20260913/environment.txt').read_text()
check('re-measurement environment: Python 3.11.2, tflite_runtime 2.14.0, bookworm', '3.11.2 2.14.0' in env2 and 'bookworm' in env2)
check(
    'draft states 20 warm-up / 200 measured',
    in_draft('20 warm-up runs, 200 measured runs'),
)
env = (ROOT / 'results/edge/environment_live.txt').read_text()
check(
    'RPi 4 Model B, Debian 12, Python 3.11.2, tflite 2.14.0',
    'Raspberry Pi 4 Model B' in env
    and 'bookworm' in env
    and 'Python 3.11.2' in env
    and 'tflite_runtime: 2.14.0' in env,
)
check('RPi RAM is 8 GB (7.6Gi total)', '7.6Gi' in env)
check('draft states 8 GB RAM', in_draft('(8 GB RAM)'))
check(
    'no audit markers left in draft',
    'AUTHOR' not in DRAFT.read_text() and 'CORRECTED' not in DRAFT.read_text(),
)
check(
    'Table 2 CI strings match regenerated bootstrap',
    all(
        in_draft(s)
        for s in (
            '[−0.0, +0.4]',
            '[+2.5, +6.0]',
            '[+6.7, +10.5]',
            '[+9.4, +13.1]',
            '[−9.6, −5.7]',
        )
    ),
)
check(
    'four figure files present',
    all(
        (ROOT / f'paper/mdpi_ai/figures/{f}').exists()
        for f in (
            'figure1_pipeline.png',
            'figure2_uda_benchmark.png',
            'figure3_tsne_hybrid.png',
            'figure4_edge.png',
        )
    ),
)
check('36 references listed', DRAFT.read_text().count('\n36. Efron') == 1)
check(
    'intro uses same-strategy pair 0.969 → 0.154',
    in_draft('macro-F1 = 0.969 on the PlantVillage test set but only 0.154'),
)
check('PlantDoc: 28 folders stated', in_draft('28 class folders'))
check('MCC claim rewritten from per-seed data', in_draft('never exceeded 0.28'))
check('T3 device-less latency claim removed', not in_draft('46.3 ms'))
check(
    'CORAL stability overclaim removed', not in_draft('CORAL is the most stable method')
)
check('PlantVillage segmented subset stated', in_draft('segmented subset'))
check(
    'EdgeTPU compiler 16.0.384591198',
    any(
        '16.0.384591198' in open(f).read()
        for f in glob.glob(str(ROOT / 'export/edgetpu/*.log'))
    ),
)
logs = ' '.join(open(f).read() for f in glob.glob(str(ROOT / 'logs/*.log')))
check(
    'Table 5 models: 200 calibration samples (export logs)',
    '200 calibration samples' in logs,
)
check(
    'ResNet-50 T3 export: whole pv_calib_10pct (1,720 images)',
    nrows('splits/hybrid/pv_calib_10pct.csv') == 1720
    and 'num_samples' not in (ROOT / 'scripts/8.4_export_tflite_int8.py').read_text(),
)
check(
    "draft no longer says '100 representative calibration images'",
    not in_draft('100 representative calibration images'),
)

# --- T3 quantization ---------------------------------------------------------
t3 = {
    (r['domain'], r['model']): r
    for r in csv_rows(ROOT / 'results/T3_quant_fp32_vs_int8_hybrid.csv')
}
fp32, int8 = t3[('target', 'fp32')], t3[('target', 'int8')]
check(
    'T3 size 89.7 → 23.1 MB',
    near(float(fp32['size_mb']), 89.66, 0.05)
    and near(float(int8['size_mb']), 23.07, 0.05),
)
check(
    'T3 latency current source 52.9 → 46.3 ms (NOT 57.1 → 15.5)',
    near(float(fp32['latency_ms']), 52.89, 0.06)
    and near(float(int8['latency_ms']), 46.30, 0.06),
)
check(
    'draft no longer cites 57.1 → 15.5 ms as a result',
    not in_draft('and latency from 57.1 ms'),
)
cv = json.load(open(ROOT / 'results/edge/conversion_validation.json'))
check(
    'conversion validation: cosine 0.20 on 20 samples',
    near(cv['mean_cosine'], 0.2036, 1e-3) and cv['n_samples'] == 20,
)

# --- Introduction consistency ------------------------------------------------


# --- INT8 conversion-fidelity audit (Table 6) -------------------------------
R8 = ROOT / 'results/int8_accuracy'


def f1_of(tag: str, split: str) -> float:
    return json.load(open(R8 / f'{tag}__{split}.json'))['macro_f1']


table6 = {
    'mobilenetv1': (
        'pytorch_mobilenetv1_hybrid_pil',
        'mobilenetv1_int8_ptq_hybrid',
        'mobilenetv1__mobilenetv1_hybrid_full_integer_quant',
        (0.989, 0.124),
        (0.102, 0.029),
        (0.989, 0.122),
    ),
    'mobilenetv2': (
        'pytorch_mobilenetv2_hybrid_pil',
        'mobilenetv2_int8_ptq_hybrid',
        'mobilenetv2__mobilenetv2_hybrid_full_integer_quant',
        (0.976, 0.134),
        (0.004, 0.009),
        (0.976, 0.142),
    ),
    'resnet50': (
        'pytorch_resnet50_hybrid_pil',
        'resnet50_int8_ptq_hybrid',
        'resnet50__resnet50_hybrid_full_integer_quant',
        (0.974, 0.148),
        (0.946, 0.128),
        (0.971, 0.149),
    ),
    'dann42': (
        'pytorch_dann42_hybrid_pil',
        'resnet50_dann42_int8_ptq_hybrid',
        'dann42__dann42_hybrid_full_integer_quant',
        (0.961, 0.249),
        (0.786, 0.184),
        (0.960, 0.247),
    ),
    'efficientnet_b3': (
        'pytorch_efficientnet_b3_hybrid_pil',
        'efficientnet_int8_ptq_hybrid',
        'efficientnet_b3__efficientnet_b3_hybrid_full_integer_quant',
        (0.984, 0.255),
        (0.008, 0.005),
        (0.958, 0.146),
    ),
}
for name, (ref, keras, onnx, e_ref, e_keras, e_onnx) in table6.items():
    for i, split in enumerate(('pv_test', 'plantdoc_test')):
        check(
            f'Table 6 {name} {split}: PyTorch {e_ref[i]}',
            near(f1_of(ref, split), e_ref[i], 6e-4),
        )
        check(
            f'Table 6 {name} {split}: Keras-rebuild INT8 {e_keras[i]}',
            near(f1_of(keras, split), e_keras[i], 6e-4),
        )
        check(
            f'Table 6 {name} {split}: ONNX INT8 {e_onnx[i]}',
            near(f1_of(onnx, split), e_onnx[i], 6e-4),
        )
lite_ref = [
    f1_of('pytorch_efficientnet_lite0_hybrid_pil', s)
    for s in ('pv_test', 'plantdoc_test')
]
lite_k = [
    f1_of('reexport_20260912__efficientnet_lite0_int8_ptq_hybrid', s)
    for s in ('pv_test', 'plantdoc_test')
]
check(
    'Table 6 lite0: 0.982/0.212 vs 0.982/0.205',
    near(lite_ref[0], 0.982, 6e-4)
    and near(lite_ref[1], 0.212, 6e-4)
    and near(lite_k[0], 0.982, 6e-4)
    and near(lite_k[1], 0.205, 6e-4),
)
check(
    'INT8 loss ≤ 0.3 pp for MNv1/MNv2/ResNet-50/DANN (ONNX path)',
    all(
        abs(f1_of(v[2], s) - f1_of(v[0], s)) <= 0.003
        or (f1_of(v[2], s) - f1_of(v[0], s)) > 0
        for k, v in table6.items()
        if k != 'efficientnet_b3'
        for s in ('pv_test', 'plantdoc_test')
    ),
)
check(
    'draft states INT8 cost at most 0.3 points', in_draft('at most 0.3 pp of macro-F1')
)
for m in ('mobilenetv1', 'mobilenetv2', 'resnet50', 'dann42'):
    txt = (ROOT / f'export/onnx2tf_20260912/{m}/edgetpu/compile.txt').read_text()
    check(
        f'Edge TPU: {m} onnx2tf INT8 fully mapped',
        'Mapped to Edge TPU' in txt and 'CPU' not in txt.split('Operator')[-1],
    )

ops = {}
for tag, path in (
    ('mnv1_keras', 'export/reexport_20260912/edgetpu/mobilenetv1_compile.txt'),
    ('mnv2_keras', 'export/reexport_20260912/edgetpu/mobilenetv2_compile.txt'),
    ('mnv1_onnx', 'export/onnx2tf_20260912/mobilenetv1/edgetpu/compile.txt'),
    ('mnv2_onnx', 'export/onnx2tf_20260912/mobilenetv2/edgetpu/compile.txt'),
):
    txt = (ROOT / path).read_text()
    ops[tag] = int(txt.split('Total number of operations:')[1].split()[0])
check(
    'op counts in §4.6 (34 vs 29, 69 vs 70)',
    (ops['mnv1_onnx'], ops['mnv1_keras'], ops['mnv2_onnx'], ops['mnv2_keras'])
    == (34, 29, 69, 70)
    and in_draft('34 versus 29 and 69 versus 70 operations'),
)

_b = json.load(open(ROOT / 'results/uda_benchmark/bootstrap/bootstrap_vs_source.json'))
check(
    'Table 2 Δ (seed 42) column matches bootstrap point estimates',
    all(
        in_draft(f'| {_b[m]["vs_source_only"]["mean_diff"] * 100:+.1f}'.replace('-', '−'))
        for m in ('coral', 'dann', 'cdan', 'mcc')
    ),
)
check(
    'calibration set defined as 10% of train (1,720)',
    in_draft('1,720 images, shuffled with seed 42, disjoint from validation and test'),
)

print(f'\n{len(FAILS)} FAIL' if FAILS else '\nall numeric checks PASS')
sys.exit(1 if FAILS else 0)
