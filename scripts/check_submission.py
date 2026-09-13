"""Submission readiness checker for CVPPA 2026.

Exits with code 1 if any blocking issue is found.
Run: python scripts/check_submission.py [--latex]

Checks latex source by default. Pass --markdown to check the old Markdown draft.

Official CVPPA 2026 requirements (verified from cvppa2026/website_src/public/content/cfp.md):
  deadline_known = true
  full_paper_limit = 14 pages
  references_excluded_from_limit = true
  double_blind = true
  eccv_2026_template_required = true
  paper_id_required = true
  line_numbers_required = true
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).parent.parent

# Primary source to check: LaTeX sections
LATEX_DIR = ROOT / 'paper' / 'latex'
LATEX_MAIN = LATEX_DIR / 'main.tex'
SECTIONS_DIR = LATEX_DIR / 'sections'
BIB_FILE = LATEX_DIR / 'references.bib'
SUBMISSION_RESULTS = ROOT / 'results' / 'uda_benchmark' / 'submission_results.csv'
ANONYMOUS_PDF = ROOT / 'build' / 'cvppa2026_anonymous_submission.pdf'

# Files that are allowed to contain legacy numbers (audit docs)
AUDIT_ALLOWLIST = {
    'docs/revision_audit.md',
    'docs/uda_protocol_audit.md',
    'docs/uda_hyperparameter_policy.md',
    'docs/numerical_consistency_audit.md',
    'docs/revision_notes.md',
    'docs/reviewer_concern_matrix.md',
    'docs/claim_evidence_matrix.md',
    'docs/statistical_validation.md',
    'experiments/E2_dann/dann_results.json',
    'experiments/E2_dann/train_dann.py',
    'rebuttal/',
    '.git/',
    '.claude/',
    'results/uda_benchmark/bootstrap/',
}

errors = []
warnings = []


def is_allowlisted(path: Path) -> bool:
    rel = str(path.relative_to(ROOT))
    return any(rel.startswith(a) for a in AUDIT_ALLOWLIST)


def check_latex_main():
    """Check main.tex for anonymous mode, line numbers, Paper ID."""
    if not LATEX_MAIN.exists():
        errors.append(f'LATEX MAIN NOT FOUND: {LATEX_MAIN}')
        return

    text = LATEX_MAIN.read_text()

    # Anonymous mode must be active
    if r'\anonymoustrue' not in text or r'%\anonymoustrue' in text.replace(' ', ''):
        if r'\anonymoustrue' not in text:
            errors.append(
                'Anonymous mode not enabled: \\anonymoustrue not found in main.tex'
            )
    # Check it's not commented out (allow trailing comment after the command)
    anon_lines = [
        line.strip() for line in text.splitlines() if '\\anonymoustrue' in line
    ]
    commented_out = (
        all(line.startswith('%') for line in anon_lines) if anon_lines else True
    )
    if commented_out:
        errors.append('\\anonymoustrue is commented out — not in anonymous mode')

    # Line numbers must be present
    if r'\usepackage{lineno}' not in text:
        errors.append('Line numbers not enabled: \\usepackage{lineno} not in main.tex')
    if r'\linenumbers' not in text:
        errors.append('Line numbers not activated: \\linenumbers not in main.tex')

    # Paper ID macro must be defined
    if r'\def\paperID' not in text:
        errors.append('Paper ID macro missing: \\def\\paperID not in main.tex')

    # ECCV 2024 provisional template marker must not be present as active claim
    if 'PROVISIONAL TEMPLATE: Based on ECCV 2024' in text:
        errors.append(
            'Old ECCV 2024 provisional template comment still present — update to ECCV 2026 template'
        )

    # No personal absolute paths in main.tex
    for pp in ['/home/nkolesnikov', '/Users/', 'C:\\Users\\']:
        if pp in text:
            errors.append(f'Personal path in main.tex: {pp}')


def check_latex_sections():
    """Check all section files for known issues."""
    if not SECTIONS_DIR.exists():
        errors.append(f'sections/ dir not found: {SECTIONS_DIR}')
        return

    for tex_file in sorted(SECTIONS_DIR.glob('*.tex')):
        text = tex_file.read_text()
        lines = text.splitlines()

        for i, line in enumerate(lines, 1):
            # No TODO/TBD/FIXME
            if re.search(
                r'\b(TODO|TBD|FIXME|pending verification)\b', line, re.IGNORECASE
            ):
                errors.append(f'TODO/TBD at {tex_file.name}:{i}: {line.strip()[:80]}')

            # No personal paths
            for pp in ['/home/nkolesnikov', '/Users/', 'C:\\Users\\']:
                if pp in line:
                    errors.append(
                        f'Personal path at {tex_file.name}:{i}: {line.strip()[:80]}'
                    )

            # Uncontextualized legacy DANN 0.277
            if '0.277' in line:
                ok = any(
                    ctx in line.lower()
                    for ctx in [
                        'legacy',
                        'protocol violation',
                        'test-set',
                        'inflated',
                        'not a fair',
                        'fair-protocol',
                        'prior result',
                    ]
                )
                if not ok:
                    errors.append(
                        f'Uncontextualized 0.277 at {tex_file.name}:{i}: {line.strip()[:80]}'
                    )

            # Invalid INT8 accuracy claims
            bad_int8 = [
                'negligible accuracy loss',
                'accuracy-preserving',
                'INT8 accuracy preserved',
                'validated end-to-end INT8',
            ]
            for claim in bad_int8:
                if claim.lower() in line.lower():
                    errors.append(
                        f'Invalid INT8 accuracy claim at {tex_file.name}:{i}: {line.strip()[:80]}'
                    )

            # Forbidden overclaiming language
            forbidden = [
                'significantly outperforms CDAN',
                'universally best',
                'state of the art',
                'performance ceiling',
                'dominates all methods',
                'consistent superiority beyond',
            ]
            for phrase in forbidden:
                if phrase.lower() in line.lower():
                    errors.append(
                        f'Overclaiming language at {tex_file.name}:{i}: {line.strip()[:80]}'
                    )

            # Old Jeon citation key must not appear
            if 'Jeon2025Background' in line:
                errors.append(
                    f'Old Jeon2025Background citation key at {tex_file.name}:{i} — use jeon2026bridging'
                )

            # CDAN+E claimed to use Jeon's method without verification
            if (
                'jeon' in line.lower()
                and 'cdan+e' in line.lower()
                and 'apply' in line.lower()
            ):
                if 'independently' in line.lower() or 'same' in line.lower():
                    errors.append(
                        f'Unverified Jeon+CDAN+E claim at {tex_file.name}:{i}: {line.strip()[:80]}'
                    )


def check_bibliography():
    """Check references.bib for required entries and correct Jeon record."""
    if not BIB_FILE.exists():
        errors.append(f'references.bib not found: {BIB_FILE}')
        return

    text = BIB_FILE.read_text()

    # Jeon 2026 must use updated key and have DOI/volume/year
    if 'Jeon2025Background' in text:
        errors.append('Old citation key Jeon2025Background still in references.bib')
    if 'jeon2026bridging' not in text:
        errors.append('Updated jeon2026bridging entry missing from references.bib')

    # Must have DOI for Jeon — find block by scanning from entry start to next @
    jeon_start = text.find('@article{jeon2026bridging')
    if jeon_start == -1:
        errors.append('jeon2026bridging @article block not found in references.bib')
    else:
        next_entry = text.find('\n@', jeon_start + 1)
        block = text[jeon_start:next_entry] if next_entry != -1 else text[jeon_start:]
        if 'doi' not in block.lower():
            errors.append('jeon2026bridging entry missing DOI field')
        if '2026' not in block:
            errors.append('jeon2026bridging year is not 2026')
        if 'Kim, S' not in block:
            warnings.append('jeon2026bridging may be missing Kim Seong Yeop author')
        if 'Song, M' not in block:
            warnings.append('jeon2026bridging may be missing Song Mungyeong author')

    # DOI pending markers must not exist
    if 'DOI pending' in text or 'doi pending' in text.lower():
        errors.append('DOI pending markers still in references.bib')


def check_submission_results():
    """Verify the authoritative submission_results.csv exists and has expected values."""
    if not SUBMISSION_RESULTS.exists():
        errors.append(
            f'Authoritative submission_results.csv missing: {SUBMISSION_RESULTS}'
        )
        return

    import csv

    rows = list(csv.DictReader(open(SUBMISSION_RESULTS)))
    dann_plantdoc = next(
        (
            r
            for r in rows
            if r['method'] == 'dann' and r['target_dataset'] == 'PlantDoc'
        ),
        None,
    )
    cdan_plantdoc = next(
        (
            r
            for r in rows
            if r['method'] == 'cdan' and r['target_dataset'] == 'PlantDoc'
        ),
        None,
    )
    dann_plantwild = next(
        (
            r
            for r in rows
            if r['method'] == 'dann' and r['target_dataset'] == 'PlantWild_v2'
        ),
        None,
    )
    cdan_plantwild = next(
        (
            r
            for r in rows
            if r['method'] == 'cdan' and r['target_dataset'] == 'PlantWild_v2'
        ),
        None,
    )

    expected = {
        'DANN PlantDoc': (dann_plantdoc, '0.2455', '0.005'),
        'CDAN PlantDoc': (cdan_plantdoc, '0.2419', '0.0258'),
        'DANN PlantWild': (dann_plantwild, '0.2381', '0.0112'),
        'CDAN PlantWild': (cdan_plantwild, '0.2296', '0.0376'),
    }
    for label, (row, exp_mean, _exp_std) in expected.items():
        if row is None:
            errors.append(f'submission_results.csv missing row: {label}')
        elif row['macro_f1_mean'] != exp_mean:
            errors.append(
                f'submission_results.csv {label} mean mismatch: '
                f'got {row["macro_f1_mean"]}, expected {exp_mean}'
            )


def check_anonymous_pdf():
    """Basic check that the anonymous PDF exists and has expected page count."""
    if not ANONYMOUS_PDF.exists():
        errors.append(f'Anonymous PDF not built: {ANONYMOUS_PDF}')
        return

    import subprocess

    r = subprocess.run(['pdfinfo', str(ANONYMOUS_PDF)], capture_output=True, text=True)
    if r.returncode != 0:
        warnings.append(f'pdfinfo failed on anonymous PDF: {r.stderr[:80]}')
        return

    pages_lines = [line for line in r.stdout.splitlines() if line.startswith('Pages:')]
    if pages_lines:
        total_pages = int(pages_lines[0].split()[-1])
        if total_pages > 16:
            errors.append(
                f'PDF has {total_pages} pages — likely too long (14 main + refs; check carefully)'
            )
        else:
            print(
                f'  PDF pages: {total_pages} (main text approximately {total_pages - 2})'
            )

    # Check for author info in PDF text
    try:
        r2 = subprocess.run(
            ['pdftotext', str(ANONYMOUS_PDF), '-'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r2.returncode == 0:
            pdf_text = r2.stdout
            personal = ['/home/nkolesnikov', 'kolesnikov', 'VolgGTU']
            for p in personal:
                if p.lower() in pdf_text.lower():
                    errors.append(f'Personal info found in PDF text: {p}')
            if 'Anonymous' not in pdf_text:
                errors.append('Anonymous author block not found in PDF')
    except Exception:
        warnings.append('Could not extract PDF text for anonymization check')


def check_result_files():
    """Verify key result files exist."""
    required = [
        'results/uda_benchmark/aggregated/uda_summary.json',
        'results/uda_benchmark/plantwild/cdan_hybrid_plantwild.json',
        'experiments/E5b_plantwild/eval_results.json',
        'results/T4_edgebench.csv',
        'results/coral_multiseed_summary.json',
        'results/uda_benchmark/bootstrap/dann_vs_cdan_plantdoc.json',
        'results/edge/environment_live.txt',
    ]
    for r in required:
        if not (ROOT / r).exists():
            errors.append(f'Required result file missing: {r}')


def main():
    print('=== CVPPA 2026 Submission Readiness Check ===')
    print('Official deadline: 2026-07-10 (timezone: verify in OpenReview UI)')
    print('Limit: 14 pages main text, references excluded, double-blind\n')

    check_latex_main()
    check_latex_sections()
    check_bibliography()
    check_submission_results()
    check_result_files()
    check_anonymous_pdf()

    if warnings:
        print('WARNINGS:')
        for w in warnings:
            print(f'  ⚠  {w}')
        print()

    if errors:
        print('ERRORS (blocking):')
        for e in errors:
            print(f'  ✗  {e}')
        print(f'\n{len(errors)} error(s) found. Submission NOT ready.')
        sys.exit(1)
    else:
        print(f'✓ No blocking errors found ({len(warnings)} warning(s)).')
        print('\nRemaining manual actions before submission:')
        print(
            '  1. Obtain official ECCV 2026 Author Kit (.sty files) from submission system'
        )
        print('  2. Register on CVPPA OpenReview to obtain Paper ID')
        print('  3. Replace \\def\\paperID{XXXX} with actual 4-digit ID')
        print('  4. Verify final PDF page count after template replacement')
        sys.exit(0)


if __name__ == '__main__':
    main()
