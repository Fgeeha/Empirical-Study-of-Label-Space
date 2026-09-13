#!/usr/bin/env python3
"""Build an arXiv-style preprint PDF from paper/mdpi_ai/main.md.

    python3 build_preprint.py          # -> preprint/Kolesnikov_Kravets_preprint.pdf

Single-column article class, 11 pt, Linux Libertine via XeLaTeX. Front matter,
figure markers ([[FIGURE n]]), table captions (**Table n.** paragraphs) and the
figure-caption list of main.md are translated into pandoc syntax; sections keep
their manual numbers. Needs pandoc and xelatex.
"""

import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / 'main.md'
OUT_DIR = HERE / 'preprint'
OUT = OUT_DIR / 'Kolesnikov_Kravets_preprint'

HEADER = r"""
\usepackage[a4paper,margin=25mm]{geometry}
\usepackage{fontspec}
\setmainfont{Linux Libertine O}
\setsansfont{Linux Biolinum O}
\setmonofont[Scale=0.85]{DejaVu Sans Mono}
\usepackage{microtype}
\usepackage{booktabs,longtable,array,etoolbox}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{caption}
\captionsetup{font=small,labelfont=bf,skip=4pt}
\AtBeginEnvironment{longtable}{\small}
\usepackage{enumitem}
\setlist{nosep,leftmargin=*}
\usepackage{titlesec}
\titleformat*{\section}{\large\bfseries}
\titleformat*{\subsection}{\normalsize\bfseries}
\setlength{\parskip}{3pt plus 1pt}
\setlength{\parindent}{0pt}
\usepackage[hidelinks]{hyperref}
\urlstyle{same}
\renewcommand{\abstractname}{Abstract}
"""


def main() -> None:
    t = MD.read_text()
    lines = t.split('\n')
    title = next(ln[2:].strip() for ln in lines if ln.startswith('# '))
    i_abs = t.index('\n## Abstract'); i_intro = t.index('\n## 1. Introduction')
    front = t[: i_abs]
    abs_block = t[i_abs:i_intro].split('\n')
    abstract = ' '.join(ln for ln in abs_block if ln.strip() and not ln.startswith('#') and not ln.startswith('**Keywords') and ln.strip() != '---')
    keywords = next(ln for ln in abs_block if ln.startswith('**Keywords:**'))
    authors = re.search(r'^\*\*(.+?)\*\*\s*\^(.+?)\^\s*and\s*\*\*(.+?)\*\*\s*\^(.+?)\^', front, re.M)
    aff = [ln for ln in front.split('\n') if ln.startswith('^1^') or ln.startswith('\\* Correspondence')]
    body = t[i_intro:]
    body = body[: body.index('\n## Figure Captions')]
    figs = {}
    for m in re.finditer(r'\*\*Figure (\d+)\.\*\*\s*(.*)', t[t.index('\n## Figure Captions'):]):
        cap = m.group(2)
        f = re.search(r'File: `([^`]+)`', cap).group(1)
        cap = re.sub(r'\s*File: `[^`]+`.*$', '', cap).strip()
        figs[int(m.group(1))] = (f, cap)
    # figure markers -> pandoc figures
    def fig_repl(m):
        f, cap = figs[int(m.group(1))]
        return f'![{cap}]({f}){{width=100%}}'
    body = re.sub(r'^\[\[FIGURE (\d+)\]\]$', fig_repl, body, flags=re.M)
    # **Table n.** caption paragraph before a pipe table -> pandoc caption after the table
    out, i, blines = [], 0, body.split('\n')
    while i < len(blines):
        m = re.match(r'^\*\*Table (\d+)\.\*\*\s*(.*)$', blines[i])
        if m and i + 2 < len(blines) and blines[i + 2].startswith('|'):
            cap = m.group(2)
            j = i + 2
            while j < len(blines) and blines[j].startswith('|'):
                out.append(blines[j]); j += 1
            out.append(''); out.append(f': {cap}'); out.append('')
            i = j
            continue
        out.append(blines[i]); i += 1
    body = '\n'.join(out)
    body = body.replace('\n## Back Matter\n', '\n## Declarations {.unnumbered}\n')
    body = body.replace('\n## References\n', '\n## References {.unnumbered}\n')
    body = re.sub(r'\n---\n', '\n', body)
    meta = (
        '---\n'
        f'title: |\n  {title}\n'
        f'author: |\n  {authors.group(1)}^{authors.group(2)}^ and {authors.group(3)}^{authors.group(4)}^\n'
        f'date: |\n  {aff[0]}\\\n  {aff[1]}\n'
        f'abstract: |\n  {abstract}\n'
        'lang: en-GB\n'
        '---\n\n'
    )
    md = meta + keywords + '\n\n' + body
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / 'header.tex').write_text(HEADER)
    (OUT_DIR / 'preprint.md').write_text(md)
    # figures are referenced relative to main.md
    if (OUT_DIR / 'figures').exists():
        shutil.rmtree(OUT_DIR / 'figures')
    shutil.copytree(HERE / 'figures', OUT_DIR / 'figures', ignore=shutil.ignore_patterns('*.py', '*.svg'))
    cmd = ['pandoc', str(OUT_DIR / 'preprint.md'), '-f', 'markdown', '-o', f'{OUT}.pdf', '--pdf-engine=xelatex',
           '-H', str(OUT_DIR / 'header.tex'), '--resource-path', str(OUT_DIR), '--wrap=none', '-V', 'fontsize=11pt', '-V', 'documentclass=article']
    subprocess.run(cmd, check=True, cwd=OUT_DIR)
    tex_cmd = [c for c in cmd if c != '--pdf-engine=xelatex']
    tex_cmd[tex_cmd.index(f'{OUT}.pdf')] = f'{OUT}.tex'
    subprocess.run(tex_cmd + ['-s'], check=True, cwd=OUT_DIR)
    print(f'[OK] {OUT}.pdf')


if __name__ == '__main__':
    main()
