"""Assemble draft/sections/*.md, resolving <equations/...>, <tables/...>, and
<figures/...> placement lines, into draft/manuscript/latex/main.tex against
the real IEEEtran conference class.

Mirrors scripts/build_docx.py over the same source of truth, so both outputs
stay in sync as draft/sections/*.md changes -- important with multiple
writers editing the same source.

Unlike the docx path, equation source in draft/equations/*.md is already
real LaTeX (\\frac, \\sum, \\sigma, ...) and passes through nearly verbatim;
LaTeX/IEEEtran auto-numbers \\section/\\subsection, so section headings drop
the manual "3.1" prefix used in the docx heading text.

Usage: uv run python scripts/build_latex.py
Output: draft/manuscript/latex/main.tex (+ figures/ copied alongside)
"""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SECTIONS = REPO / "draft/sections"
EQUATIONS = REPO / "draft/equations"
TABLES = REPO / "draft/tables"
FIGURES = REPO / "draft/figures"
REFERENCES = REPO / "draft/references"
OUT_DIR = REPO / "draft/manuscript/latex"
OUT_FIGURES = OUT_DIR / "figures"

SECTION_FILE_RE = re.compile(r"^\d+(\.\d+)*-.+\.md$")
PLACEMENT_RE = re.compile(r"^<(equations|tables|figures)/(.+)>$")

# Per-figure override width as a fraction of \columnwidth. Default 1.0.
FIGURE_WIDTH = {
    "3.2-gradcam-qualitative.png": 1.0,
}

# Figures that span both columns (figure*, width=\textwidth) instead of \columnwidth.
FIGURE_DOUBLE_COLUMN = {"2.2-hierarchy-diagram.png"}

FRONT_MATTER_ABSTRACT = (
    "Chest X-rays serve as the first-line imaging exam for many thoracic "
    "conditions. The shortage of radiologists, not the images, limits "
    "throughput. Earlier work has used the dependence among findings, "
    "since a specific pattern can point to a broader diagnostic category, "
    "to improve multi-label classification. It is unknown whether that "
    "gain holds when several public datasets are combined under one "
    "label set, with only partial co-annotation of parent and child "
    "labels. We merge seven public chest X-ray datasets into 527,745 "
    "images labeled with 51 standard findings from an Indonesian "
    "clinical guideline. A flat multi-label model is compared to one "
    "regularized with a hierarchy-consistency penalty across 13 "
    "parent-child relations, evaluated on discrimination, calibration, "
    "selective prediction quality, hierarchy violation rate, and "
    "localization. The hierarchy-consistency penalty does not clearly "
    "outperform the flat baseline: discrimination is statistically "
    "similar, flat wins on ranking and selective prediction, and "
    "hierarchical wins only on hierarchy violation rate, traceable to "
    "the small number of co-annotated parent-child pairs. Localization "
    "is not usable at standard thresholds for either approach."
)

FRONT_MATTER_KEYWORDS = (
    "Chest radiography, multi-label classification, hierarchical "
    "classification, deep learning, medical image analysis"
)

END_MATTER_CONCLUSION = (
    "This paper tested whether a hierarchy-consistency penalty improves "
    "multi-label chest radiograph classification once training data is "
    "pooled from seven public sources rather than drawn from one. It does "
    "not, at least not clearly: hierarchical training ties flat training "
    "on overall discrimination, loses on ranking and selective-prediction "
    "quality, and gains only a modest reduction in hierarchy violations, "
    "concentrated on the single parent-child pair in this corpus with "
    "enough co-annotated data to give the penalty a gradient signal. This "
    "points to a data problem rather than a modeling one. Grad-CAM "
    "localization was also not usable at standard thresholds for either "
    "condition. Both conditions come from a single training run each, so "
    "these results bound sampling noise on one test set, not "
    "seed-to-seed variance, and repeated runs remain the most direct way "
    "to close that gap."
)

LATEX_SPECIALS = {
    "&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#",
    "$": r"\$",  # bare, not already-escaped math delimiters -- see escape_text
}


def section_files():
    files = [f for f in SECTIONS.iterdir() if SECTION_FILE_RE.match(f.name)]
    return sorted(files, key=lambda f: tuple(int(p) for p in f.stem.split("-", 1)[0].split(".")))


def heading_level(stem: str) -> int:
    return stem.split("-", 1)[0].count(".") + 1


def heading_text(stem: str) -> str:
    # No manual number: \section/\subsection auto-number under IEEEtran.
    _, _, slug = stem.partition("-")
    return slug.replace("-", " ").capitalize()


LATEX_CMD = {"section", "subsection", "subsubsection"}


def escape_text(text: str) -> str:
    # Protect inline math ($...$) from escaping, escape everything else.
    parts = re.split(r"(\$[^$]*\$)", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:  # inside $...$
            out.append(part)
            continue
        for ch, esc in LATEX_SPECIALS.items():
            part = part.replace(ch, esc)
        out.append(part)
    return "".join(out)


CITATION_RE = re.compile(r"\{cite:(\w+)\}")


def resolve_citations_latex(text: str) -> str:
    # Inline citation marker {cite:key} -> \cite{key}, IEEEtran/bibtex numbers
    # it by citation order automatically. Braces aren't in LATEX_SPECIALS so
    # escape_text() never touches this marker; safe to resolve after escaping.
    return CITATION_RE.sub(lambda m: rf"\cite{{{m.group(1)}}}", text)


def add_prose_paragraph(tex: list, text: str):
    text = text.replace("\n", " ").strip()
    tex.append(resolve_citations_latex(escape_text(text)))
    tex.append("")


def add_equation_block(tex: list, latex_line: str):
    tex.append(r"\begin{equation}")
    tex.append(latex_line.strip())
    tex.append(r"\end{equation}")
    tex.append("")


def _to_number(text: str):
    try:
        return float(re.sub(r"[^0-9.\-]", "", text))
    except ValueError:
        return None


NON_METRIC_LABELS = {"n boxes", "n", "condition", "metric", "finding", "group"}


def _direction(label: str):
    if label.strip().lower() in NON_METRIC_LABELS:
        return None
    if "(lower is better)" in label.lower():
        return label.replace(" (lower is better)", "").replace(" (Lower is better)", ""), True
    return label, False


def _arrow(lower_better: bool) -> str:
    return r"$\downarrow$" if lower_better else r"$\uparrow$"


def add_table(tex: list, csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return
    header, body = rows[0], rows[1:]
    row_wise = "Flat" in header and "Hierarchical" in header
    column_wise = bool(header) and header[0] == "Condition"
    metric_labeled = bool(header) and header[0].strip().lower() == "metric"

    display_header = list(header)
    if column_wise:
        display_header = []
        for h in header:
            d = _direction(h)
            display_header.append(h if d is None else f"{d[0]} {_arrow(d[1])}")

    best = {}  # (row_idx, col_idx) -> True if this cell should be bold
    if row_wise:
        flat_i, hier_i = header.index("Flat"), header.index("Hierarchical")
        for r, row in enumerate(body):
            lower_better = False
            if metric_labeled:
                d = _direction(row[0])
                if d is None:
                    continue
                lower_better = d[1]
            a, b = _to_number(row[flat_i]), _to_number(row[hier_i])
            if a is None or b is None or a == b:
                continue
            winner = flat_i if (a < b if lower_better else a > b) else hier_i
            best[(r, winner)] = True
    elif column_wise:
        by_condition = {row[0]: (i, row) for i, row in enumerate(body)}
        if "Flat" in by_condition and "Hierarchical" in by_condition:
            fi, frow = by_condition["Flat"]
            hi, hrow = by_condition["Hierarchical"]
            for c in range(1, len(header)):
                if _direction(header[c]) is None:
                    continue
                a, b = _to_number(frow[c]), _to_number(hrow[c])
                if a is None or b is None or a == b:
                    continue
                best[(fi if a > b else hi, c)] = True

    ncols = len(header)
    # Long free-text columns (e.g. "Significance") overflow a plain "c" column
    # regardless of table width -- always wrap those in a fixed paragraph column.
    LONG_TEXT_HEADERS = {"significance", "note", "notes"}
    wide_col = any(h.strip().lower() in LONG_TEXT_HEADERS for h in header[1:])
    colspec = "l" + "".join(
        r">{\centering\arraybackslash}p{3.2cm}" if h.strip().lower() in LONG_TEXT_HEADERS else "c"
        for h in header[1:]
    )
    # Estimate total row width from the widest content per column (chars).
    # Use the plain label length, not display_header's raw LaTeX arrow macro
    # source ("$\uparrow$" is ~10 chars of source but renders ~1 glyph wide) --
    # add a small constant instead for columns that get a direction arrow.
    # Calibrated empirically against Table II (est_width 67, fits single-column
    # fine) and Table I (est_width 81, needs table*). Tables that don't fit
    # span both columns (table*) instead of fighting it with font size.
    header_len = [
        len(header[c]) + (2 if display_header[c] != header[c] else 0)
        for c in range(ncols)
    ]
    col_widths = [
        max(header_len[c], max((len(row[c]) for row in body), default=0))
        for c in range(ncols)
    ]
    est_width = sum(col_widths) + 2 * ncols
    table_env = "table*" if (wide_col or est_width > 68) else "table"
    tex.append(rf"\begin{{{table_env}}}[htbp]")
    tex.append(r"\centering")
    tex.append(rf"\caption{{{_caption_text(csv_path)}}}")
    tex.append(rf"\label{{tab:{csv_path.stem}}}")
    tex.append(rf"\begin{{tabular}}{{{colspec}}}")
    tex.append(r"\toprule")
    tex.append(" & ".join(rf"\textbf{{{escape_text(h)}}}" for h in display_header) + r" \\")

    if row_wise:
        tex.append(r"\midrule")
        for r, row in enumerate(body):
            label = row[0]
            if metric_labeled:
                d = _direction(label)
                if d is not None:
                    label = f"{d[0]} {_arrow(d[1])}"
            cells = [escape_text(label)]
            for c in range(1, ncols):
                text = escape_text(row[c])
                cells.append(rf"\textbf{{{text}}}" if best.get((r, c)) else text)
            tex.append(" & ".join(cells) + r" \\")
    elif column_wise:
        tex.append(r"\midrule")
        for r, row in enumerate(body):
            cells = [escape_text(row[0])]
            for c in range(1, ncols):
                text = escape_text(row[c])
                cells.append(rf"\textbf{{{text}}}" if best.get((r, c)) else text)
            tex.append(" & ".join(cells) + r" \\")
    else:
        tex.append(r"\midrule")
        for row in body:
            tex.append(" & ".join(escape_text(c) for c in row) + r" \\")

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")
    tex.append(rf"\end{{{table_env}}}")
    tex.append("")


def _caption_text(asset_path: Path) -> str:
    caption_path = asset_path.parent / f"{asset_path.name}.caption.txt"
    text = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else "Caption to be added."
    return escape_text(text)


def add_figure(tex: list, png_path: Path):
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    shutil.copy(png_path, OUT_FIGURES / png_path.name)
    double_col = png_path.name in FIGURE_DOUBLE_COLUMN
    fig_env = "figure*" if double_col else "figure"
    width_unit = r"\textwidth" if double_col else r"\columnwidth"
    width = FIGURE_WIDTH.get(png_path.name, 1.0)
    tex.append(rf"\begin{{{fig_env}}}[htbp]")
    tex.append(r"\centering")
    tex.append(rf"\includegraphics[width={width}{width_unit}]{{figures/{png_path.name}}}")
    tex.append(rf"\caption{{{_caption_text(png_path)}}}")
    tex.append(rf"\label{{fig:{png_path.stem}}}")
    tex.append(rf"\end{{{fig_env}}}")
    tex.append("")


def add_content_blocks(tex: list, text: str):
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        placement = PLACEMENT_RE.match(block)
        if placement:
            kind, name = placement.groups()
            if kind == "equations":
                add_content_blocks(tex, (EQUATIONS / name).read_text(encoding="utf-8"))
            elif kind == "tables":
                add_table(tex, TABLES / name)
            elif kind == "figures":
                add_figure(tex, FIGURES / name)
            continue
        if block.startswith("$$") and block.endswith("$$"):
            add_equation_block(tex, block.strip("$"))
            continue
        add_prose_paragraph(tex, block)


def build():
    tex: list[str] = []
    tex.append(r"\documentclass[conference,a4paper]{IEEEtran}")
    tex.append(r"\IEEEoverridecommandlockouts")
    tex.append(r"\usepackage[T1]{fontenc}")  # correct </> glyphs; OT1 default mis-renders them as inverted punctuation
    tex.append(r"\usepackage{cite}")
    tex.append(r"\usepackage{amsmath,amssymb,amsfonts}")
    tex.append(r"\usepackage{graphicx}")
    tex.append(r"\usepackage{textcomp}")
    tex.append(r"\usepackage{xcolor}")
    tex.append(r"\usepackage{booktabs}")
    tex.append(r"\usepackage{array}")
    # IEEEtran's default label is "Index Terms"; match its own \@IEEEabskeysecsize
    # (\small, same size the Abstract uses) and \ignorespaces (no gap after the
    # dash, matching Abstract's spacing) -- bfseries carries through to bold the
    # whole keyword list, not just the label. @ is an internal-command char
    # outside the class file, hence makeatletter/other.
    tex.append(r"\makeatletter")
    tex.append(r"\renewenvironment{IEEEkeywords}{\normalfont\@IEEEabskeysecsize\par\vskip 3pt\noindent\bfseries\textit{Keywords}---\ignorespaces}{\par}")
    tex.append(r"\makeatother")
    tex.append("")

    files = section_files()
    title = ""
    if files and files[0].stem == "0-title":
        title = files[0].read_text(encoding="utf-8").strip()
        files = files[1:]

    tex.append(rf"\title{{{escape_text(title)}}}")
    tex.append("")
    # Double-blind submission: no real author identity anywhere in the paper --
    # bracketed placeholders keep the real IEEE author-block field structure
    # (name, degree/program, university, city/country, email) ready to fill
    # in for the camera-ready version, without stating anything as real now.
    # \\{} (not bare \\) before a [-led line: LaTeX parses \\[...] as the
    # linebreak-with-spacing syntax, silently swallowing a literal [City, ...].
    ORDINALS = {1: "st", 2: "nd", 3: "rd"}
    author_blocks = []
    for i in range(1, 4):
        author_blocks.append(
            rf"\IEEEauthorblockN{{{i}\textsuperscript{{{ORDINALS[i]}}} [Name]}}"
            "\n"
            r"\IEEEauthorblockA{\footnotesize\textit{[Degree]} \\{}"
            "\n"
            r"\textit{[University]} \\{}"
            "\n"
            r"[City, Country] \\{}"
            "\n"
            r"[email]}"
        )
    tex.append(r"\author{" + "\n\\and\n".join(author_blocks) + "}")
    tex.append("")
    tex.append(r"\begin{document}")
    tex.append(r"\maketitle")
    tex.append("")
    tex.append(r"\begin{abstract}")
    tex.append(escape_text(FRONT_MATTER_ABSTRACT))
    tex.append(r"\end{abstract}")
    tex.append("")
    tex.append(r"\begin{IEEEkeywords}")
    tex.append(escape_text(FRONT_MATTER_KEYWORDS))
    tex.append(r"\end{IEEEkeywords}")
    tex.append("")

    for f in files:
        level = min(heading_level(f.stem), 3)
        cmd = ["section", "subsection", "subsubsection"][level - 1]
        tex.append(rf"\{cmd}{{{escape_text(heading_text(f.stem))}}}")
        add_content_blocks(tex, f.read_text(encoding="utf-8"))

    tex.append(r"\section{Conclusion}")
    tex.append(escape_text(END_MATTER_CONCLUSION))
    tex.append("")

    bib_files = list(REFERENCES.glob("*.bib"))
    if bib_files:
        shutil.copy(bib_files[0], OUT_DIR / bib_files[0].name)
        tex.append(r"\bibliographystyle{IEEEtran}")
        tex.append(r"\nocite{*}")  # ponytail: no in-text \cite keys wired into prose yet
        tex.append(rf"\bibliography{{{bib_files[0].stem}}}")
        tex.append("")

    tex.append(r"\end{document}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "main.tex"
    out_path.write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build()
