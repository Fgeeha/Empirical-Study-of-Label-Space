#!/usr/bin/env python3
"""
Audit paper references: detect duplicates, misattributions, and orphans.

Checks performed:
    1. Duplicate references (same paper cited under different numbers)
    2. Misattributed references (cited for wrong reason)
    3. Orphan references (listed but never cited in text)
    4. Irrelevant references (paper topic doesn't match citation context)
    5. Missing critical references

Input: the paper text extracted from .docm (hardcoded reference database
       based on manual analysis of the document).

Usage:
    python scripts/13.2_audit_references.py
"""

import json
from pathlib import Path

REFERENCE_DB = [
    {
        'id': 1,
        'authors': 'Hughes & Salathe',
        'year': 2015,
        'title': 'An open access repository of images on plant health',
        'topic': 'PlantVillage dataset',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 2,
        'authors': 'Chen et al.',
        'year': 2018,
        'title': 'Learning to See in the Dark',
        'topic': 'Low-light RAW image processing',
        'status': 'ERROR',
        'notes': (
            'WRONG PAPER. Cited for "Backgrounds are complex, lighting '
            'varies with time of day" (field photography). The actual paper '
            'is about RAW sensor data processing in extremely low light. '
            'Has nothing to do with plant disease or field photography.'
        ),
        'fix': (
            'Replace with: Barbedo, J.G.A. (2018). "Factors influencing '
            'the use of deep learning for plant disease recognition." '
            'Biosystems Engineering, 172, 84-91. '
            'OR: Ferentinos, K.P. (2018). "Deep learning models for plant '
            'disease detection and diagnosis." Computers and Electronics '
            'in Agriculture, 145, 311-318.'
        ),
    },
    {
        'id': 3,
        'authors': 'Singh et al.',
        'year': 2019,
        'title': 'PlantDoc: A Dataset for Visual Plant Disease Detection',
        'topic': 'PlantDoc dataset',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 4,
        'authors': 'Ganin et al.',
        'year': 2015,
        'title': 'Domain-Adversarial Training of Neural Networks',
        'topic': 'DANN',
        'status': 'DUPLICATE',
        'notes': 'Same paper as [15] and [17]. Keep one, remove others.',
        'fix': (
            'Keep ONE entry (preferably the 2016 JMLR version: '
            'Ganin et al. "Domain-Adversarial Training of Neural Networks." '
            'JMLR 17(59):1-35, 2016). Remove [15] and [17].'
        ),
    },
    {
        'id': 5,
        'authors': 'Sun, Feng & Saenko',
        'year': 2016,
        'title': 'Correlation Alignment for Unsupervised Domain Adaptation',
        'topic': 'CORAL original',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 6,
        'authors': 'Reimers & Gurevych',
        'year': 2019,
        'title': 'Sentence-BERT',
        'topic': 'SBERT sentence embeddings',
        'status': 'DUPLICATE',
        'notes': 'Same paper as [25]. Keep one.',
        'fix': 'Remove [25], keep [6].',
    },
    {
        'id': 7,
        'authors': 'Kubler et al.',
        'year': 2026,
        'title': 'When LLMs get significantly worse',
        'topic': 'LLM degradation detection',
        'status': 'ERROR',
        'notes': (
            'WRONG PAPER. Cited for McNemar test and bootstrap CI. '
            'This paper is about detecting LLM performance degradation, '
            'not about statistical tests for classification comparison.'
        ),
        'fix': (
            'Replace with: McNemar, Q. (1947). "Note on the sampling error '
            'of the difference between correlated proportions or '
            'percentages." Psychometrika, 12(2), 153-157. '
            'AND/OR: Dietterich, T.G. (1998). "Approximate statistical '
            'tests for comparing supervised classification learning '
            'algorithms." Neural Computation, 10(7), 1895-1923.'
        ),
    },
    {
        'id': 8,
        'authors': 'Ignatov et al.',
        'year': 2018,
        'title': 'AI Benchmark: Running Deep Neural Networks on Android',
        'topic': 'Mobile DNN benchmarking',
        'status': 'WARNING',
        'notes': (
            'Cited in TWO contexts: (a) "Edge deployment" in Section 1 -- '
            'acceptable but weak; (b) Section 3.1 for label-space '
            'alignment -- WRONG context, ref [8] has nothing to do '
            'with label alignment.'
        ),
        'fix': (
            'Remove citation from Section 3.1. For edge deployment, '
            'consider replacing with: Banbury et al. (2021). '
            '"MLPerf Tiny Benchmark." NeurIPS Datasets and Benchmarks. '
            'OR: Howard et al. (2019). "Searching for MobileNetV3." ICCV.'
        ),
    },
    {
        'id': 9,
        'authors': 'Boroumand et al.',
        'year': 2021,
        'title': 'Google Neural Network Models for Edge Devices',
        'topic': 'EdgeTPU analysis',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 10,
        'authors': 'Mohanty, Hughes & Salathe',
        'year': 2016,
        'title': 'Using Deep Learning for Image-Based Plant Disease Detection',
        'topic': 'Deep learning for plant disease (PlantVillage)',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 11,
        'authors': 'Thapa et al.',
        'year': 2020,
        'title': 'The Plant Pathology Challenge 2020 data set',
        'topic': 'Apple foliar disease dataset (FGVC7)',
        'status': 'ERROR',
        'notes': (
            'WRONG PAPER. The paper says "Thapa et al. collected PlantDoc" '
            'but Thapa et al. created the Plant Pathology Challenge 2020 '
            '(apple leaves), NOT PlantDoc. PlantDoc was created by '
            'Singh et al. [3].'
        ),
        'fix': (
            'Either: (a) Remove this citation and refer to [3] for '
            'PlantDoc; or (b) keep but fix the text: "The Plant '
            'Pathology Challenge [11] and PlantDoc [3] were created '
            'because the field lacked challenging evaluation benchmarks." '
            'Best option: replace with the actual PlantDoc follow-up paper '
            'if one exists, or just cite [3] again.'
        ),
    },
    {
        'id': 12,
        'authors': 'Kowshik et al.',
        'year': 2021,
        'title': 'Plant Disease Detection Using Deep Learning',
        'topic': 'Plant disease deep learning survey',
        'status': 'OK',
        'notes': (
            'Low-impact venue (IRJASH). Consider replacing with a '
            'higher-quality survey or using a more specific augmentation '
            'reference.'
        ),
    },
    {
        'id': 13,
        'authors': 'Kang et al.',
        'year': 2019,
        'title': 'Contrastive Adaptation Network for UDA',
        'topic': 'CAN for domain adaptation',
        'status': 'DUPLICATE',
        'notes': 'Same paper as [18]. Keep one.',
        'fix': 'Remove [18], keep [13].',
    },
    {
        'id': 14,
        'authors': 'Wang et al.',
        'year': 2020,
        'title': 'Rethink Maximum Mean Discrepancy for DA',
        'topic': 'MMD for domain adaptation',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 15,
        'authors': 'Ganin et al.',
        'year': 2015,
        'title': 'Domain-Adversarial Training of Neural Networks',
        'topic': 'DANN',
        'status': 'DUPLICATE',
        'notes': 'Duplicate of [4] and [17].',
        'fix': 'Remove. Keep only [4] (renumber).',
    },
    {
        'id': 16,
        'authors': 'Sun, Feng & Saenko',
        'year': 2015,
        'title': 'Return of Frustratingly Easy Domain Adaptation',
        'topic': 'CORAL theoretical foundation',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 17,
        'authors': 'Ganin et al.',
        'year': 2015,
        'title': 'Domain-Adversarial Training of Neural Networks',
        'topic': 'DANN',
        'status': 'DUPLICATE',
        'notes': 'Duplicate of [4] and [15].',
        'fix': 'Remove. Keep only [4] (renumber).',
    },
    {
        'id': 18,
        'authors': 'Kang et al.',
        'year': 2019,
        'title': 'Contrastive Adaptation Network for UDA',
        'topic': 'CAN',
        'status': 'DUPLICATE',
        'notes': 'Duplicate of [13].',
        'fix': 'Remove. Keep only [13] (renumber).',
    },
    {
        'id': 19,
        'authors': 'Sundaresan et al.',
        'year': 2021,
        'title': 'Comparison of domain adaptation techniques for WMH segmentation',
        'topic': 'DA in medical imaging',
        'status': 'OK',
        'notes': (
            'Acceptable but domain-specific (brain MRI). Consider a more '
            'general DA survey if page space is tight.'
        ),
    },
    {
        'id': 20,
        'authors': 'Cao et al.',
        'year': 2018,
        'title': 'Partial Adversarial Domain Adaptation',
        'topic': 'Partial DA',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 21,
        'authors': 'Yan',
        'year': 2023,
        'title': (
            'Study for Performance of MobileNetV1 and '
            'MobileNetV2 Based on Breast Cancer'
        ),
        'topic': 'MobileNet for breast cancer classification',
        'status': 'WARNING',
        'notes': (
            'Cited for "MobileNetV1 and MobileNetV2 have served as '
            'standard baselines for mobile deployment." This is a weak '
            'citation -- a breast cancer study is not the standard ref '
            'for mobile architectures.'
        ),
        'fix': (
            'Replace with: Howard, A.G. et al. (2017). "MobileNets: '
            'Efficient Convolutional Neural Networks for Mobile Vision '
            'Applications." arXiv:1704.04861. '
            'This is the ORIGINAL MobileNet paper and the correct '
            'citation for architectural baselines.'
        ),
    },
    {
        'id': 22,
        'authors': 'Tan & Le',
        'year': 2019,
        'title': 'EfficientNet: Rethinking Model Scaling',
        'topic': 'EfficientNet architecture',
        'status': 'OK',
        'notes': (
            'This is the EfficientNet paper, not EfficientNet-Lite. '
            'The Lite variant is described in TF Model Garden docs, '
            'not in a separate paper. Consider adding a footnote.'
        ),
    },
    {
        'id': 23,
        'authors': 'Coral',
        'year': None,
        'title': 'Edge TPU Compiler',
        'topic': 'Coral EdgeTPU compiler documentation',
        'status': 'OK',
        'notes': 'Technical documentation, acceptable as reference.',
    },
    {
        'id': 24,
        'authors': 'Acikgoz',
        'year': 2026,
        'title': 'Fuzzy Aura Topological Spaces with Applications to Rough Set Theory',
        'topic': 'Fuzzy mathematics / topology',
        'status': 'ERROR',
        'notes': (
            'ORPHAN + IRRELEVANT. Not cited anywhere in the paper text. '
            'The paper is about mathematical fuzzy topology, which has '
            'NOTHING to do with fuzzy string matching used in label '
            'alignment.'
        ),
        'fix': (
            'Remove entirely. If a fuzzy matching reference is needed, '
            'cite: Cohen, W.W. et al. (2003). "A comparison of string '
            'distance metrics for name-matching tasks." IJCAI Workshop. '
            'Or cite the RapidFuzz library documentation.'
        ),
    },
    {
        'id': 25,
        'authors': 'Reimers & Gurevych',
        'year': 2019,
        'title': 'Sentence-BERT',
        'topic': 'SBERT',
        'status': 'DUPLICATE',
        'notes': 'Duplicate of [6].',
        'fix': 'Remove. Keep only [6].',
    },
    {
        'id': 26,
        'authors': 'Sun & Saenko',
        'year': 2016,
        'title': 'Deep CORAL',
        'topic': 'Deep CORAL method',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 27,
        'authors': 'Zhu et al.',
        'year': 2017,
        'title': 'Unpaired Image-to-Image Translation using CycleGAN',
        'topic': 'CycleGAN / image translation',
        'status': 'ERROR',
        'notes': (
            'WRONG PAPER. Cited in Section 3.2 for "target-domain images '
            'are included in each batch to estimate covariance." CycleGAN '
            'has nothing to do with batching strategy for covariance '
            'estimation. This appears to be a random/incorrect insertion.'
        ),
        'fix': (
            'Remove citation. The statement about batch composition for '
            'CORAL does not need a citation -- it is a description of '
            'your own method. If a reference is needed, cite the '
            'original CORAL paper [5] or Deep CORAL [26].'
        ),
    },
    {
        'id': 28,
        'authors': 'Nagel et al.',
        'year': 2019,
        'title': 'Data-Free Quantization Through Weight Equalization',
        'topic': 'Post-training quantization',
        'status': 'OK',
        'notes': '',
    },
    {
        'id': 29,
        'authors': 'Sandler et al.',
        'year': 2018,
        'title': 'MobileNetV2: Inverted Residuals and Linear Bottlenecks',
        'topic': 'MobileNetV2 architecture',
        'status': 'OK',
        'notes': '',
    },
]

MISSING_REFS = [
    {
        'topic': 'McNemar test (original)',
        'suggestion': (
            'McNemar, Q. (1947). "Note on the sampling error of the '
            'difference between correlated proportions or percentages." '
            'Psychometrika, 12(2), 153-157.'
        ),
        'reason': (
            'McNemar test is used throughout but never cited with the '
            'original source. Current ref [7] is an LLM degradation paper.'
        ),
    },
    {
        'topic': 'MobileNetV1 (original)',
        'suggestion': (
            'Howard, A.G. et al. (2017). "MobileNets: Efficient '
            'Convolutional Neural Networks for Mobile Vision Applications." '
            'arXiv:1704.04861.'
        ),
        'reason': 'MobileNetV1 is used but the original paper is not cited.',
    },
    {
        'topic': 'Bootstrap confidence intervals',
        'suggestion': (
            'Efron, B. & Tibshirani, R.J. (1993). "An Introduction to '
            'the Bootstrap." Chapman & Hall/CRC.'
        ),
        'reason': 'Bootstrap CI is used but never cited.',
    },
    {
        'topic': 'Fuzzy string matching',
        'suggestion': (
            'Cohen, W.W., Ravikumar, P. & Fienberg, S.E. (2003). '
            '"A Comparison of String Distance Metrics for Name-Matching '
            'Tasks." IIWeb Workshop, IJCAI.'
        ),
        'reason': (
            'Fuzzy string matching is a core method but has no proper '
            'citation. Current orphan ref [24] is about fuzzy topology.'
        ),
    },
    {
        'topic': 'ResNet-50',
        'suggestion': (
            'He, K. et al. (2016). "Deep Residual Learning for Image '
            'Recognition." CVPR.'
        ),
        'reason': (
            'ResNet-50 is the primary backbone but the original paper is not cited.'
        ),
    },
    {
        'topic': 'ImageNet pretraining',
        'suggestion': (
            'Deng, J. et al. (2009). "ImageNet: A large-scale hierarchical '
            'image database." CVPR.'
        ),
        'reason': (
            'All models use ImageNet-pretrained weights but ImageNet is not cited.'
        ),
    },
]


def main():
    print('=' * 70)
    print('REFERENCE AUDIT REPORT')
    print('=' * 70)

    errors = [r for r in REFERENCE_DB if r['status'] == 'ERROR']
    warnings = [r for r in REFERENCE_DB if r['status'] == 'WARNING']
    duplicates = [r for r in REFERENCE_DB if r['status'] == 'DUPLICATE']
    ok = [r for r in REFERENCE_DB if r['status'] == 'OK']

    # --- Errors ---
    print(f'\n{"=" * 70}')
    print(f'ERRORS ({len(errors)} found) -- must fix before submission')
    print('=' * 70)
    for r in errors:
        print(f'\n  [{r["id"]}] {r["authors"]} ({r["year"]})')
        print(f'      Title: {r["title"]}')
        print(f'      Topic: {r["topic"]}')
        print(f'      Problem: {r["notes"]}')
        if 'fix' in r:
            print(f'      Fix: {r["fix"]}')

    # --- Duplicates ---
    print(f'\n{"=" * 70}')
    print(f'DUPLICATES ({len(duplicates)} found)')
    print('=' * 70)

    dup_groups: dict[str, list[int]] = {}
    for r in duplicates:
        key = r['title'][:40]
        dup_groups.setdefault(key, []).append(r['id'])

    for r in REFERENCE_DB:
        if r['status'] != 'DUPLICATE':
            continue
        key = r['title'][:40]
        if key in dup_groups:
            existing = [
                x
                for x in REFERENCE_DB
                if x['title'][:40] == key and x['status'] != 'DUPLICATE'
            ]
            existing_ids = [x['id'] for x in existing]
            all_ids = sorted(dup_groups[key] + existing_ids)
            print(
                f'\n  [{", ".join(str(i) for i in all_ids)}] '
                f'{r["authors"]} -- "{r["title"][:50]}..."'
            )
            if 'fix' in r:
                print(f'      Fix: {r["fix"]}')
            del dup_groups[key]

    # --- Warnings ---
    print(f'\n{"=" * 70}')
    print(f'WARNINGS ({len(warnings)} found)')
    print('=' * 70)
    for r in warnings:
        print(f'\n  [{r["id"]}] {r["authors"]} ({r["year"]})')
        print(f'      Title: {r["title"]}')
        print(f'      Problem: {r["notes"]}')
        if 'fix' in r:
            print(f'      Fix: {r["fix"]}')

    # --- Missing ---
    print(f'\n{"=" * 70}')
    print(f'MISSING REFERENCES ({len(MISSING_REFS)} found)')
    print('=' * 70)
    for m in MISSING_REFS:
        print(f'\n  Topic: {m["topic"]}')
        print(f'  Why: {m["reason"]}')
        print(f'  Add: {m["suggestion"]}')

    # --- Summary ---
    print(f'\n{"=" * 70}')
    print('SUMMARY')
    print('=' * 70)
    print(f'  Total references: {len(REFERENCE_DB)}')
    print(f'  OK:               {len(ok)}')
    print(f'  Errors:           {len(errors)} (MUST FIX)')
    print(f'  Duplicates:       {len(duplicates)} (MUST DEDUPLICATE)')
    print(f'  Warnings:         {len(warnings)} (SHOULD FIX)')
    print(f'  Missing:          {len(MISSING_REFS)} (SHOULD ADD)')
    print()

    after_dedup = len(REFERENCE_DB) - len(duplicates)
    after_fix = after_dedup - len(errors) + len(MISSING_REFS)
    print(f'  After dedup + error removal + additions: ~{after_fix} references')

    # --- Proposed clean reference list ---
    print(f'\n{"=" * 70}')
    print('PROPOSED CLEAN REFERENCE LIST (renumbered)')
    print('=' * 70)

    clean = []
    skip_ids = {15, 17, 18, 24, 25}  # duplicates + orphan
    replace_map = {
        2: {
            'authors': 'Barbedo',
            'year': 2018,
            'title': (
                'Factors influencing the use of deep learning'
                ' for plant disease recognition'
            ),
        },
        7: {
            'authors': 'McNemar',
            'year': 1947,
            'title': (
                'Note on the sampling error of the difference'
                ' between correlated proportions'
            ),
        },
        11: {
            'authors': 'Singh et al.',
            'year': 2019,
            'title': 'PlantDoc (same as [3], merge or remove)',
        },
        21: {
            'authors': 'Howard et al.',
            'year': 2017,
            'title': 'MobileNets: Efficient CNNs for Mobile Vision Applications',
        },
        27: None,  # remove
    }

    new_num = 1
    for r in REFERENCE_DB:
        if r['id'] in skip_ids:
            continue
        if r['id'] in replace_map:
            rep = replace_map[r['id']]
            if rep is None:
                continue
            clean.append(
                f'  [{new_num}] (was [{r["id"]}]) '
                f'{rep["authors"]} ({rep["year"]}). '
                f'{rep["title"]}'
            )
        else:
            clean.append(
                f'  [{new_num}] (was [{r["id"]}]) '
                f'{r["authors"]} ({r["year"]}). '
                f'{r["title"]}'
            )
        new_num += 1

    for m in MISSING_REFS:
        clean.append(f'  [{new_num}] (NEW) {m["suggestion"][:80]}...')
        new_num += 1

    print('\n'.join(clean))

    # --- Save report as JSON ---
    report = {
        'errors': [
            {
                'id': r['id'],
                'title': r['title'],
                'problem': r['notes'],
                'fix': r.get('fix', ''),
            }
            for r in errors
        ],
        'duplicates': [
            {'id': r['id'], 'title': r['title'], 'fix': r.get('fix', '')}
            for r in duplicates
        ],
        'warnings': [
            {
                'id': r['id'],
                'title': r['title'],
                'problem': r['notes'],
                'fix': r.get('fix', ''),
            }
            for r in warnings
        ],
        'missing': MISSING_REFS,
    }

    out = Path('results/final/reference_audit.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'\n[SAVED] {out}')


if __name__ == '__main__':
    main()
