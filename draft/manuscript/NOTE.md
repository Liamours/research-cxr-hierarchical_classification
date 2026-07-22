just the manuscript. naming:
{date}-{time}.{pdf or docx}

date = {YYMMDD}
time = {HHMM}

OPEN DECISION: docx vs LaTeX for final construction (both installed: Word at
C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE, and a full LaTeX
distro with pdflatex/xelatex/lualatex/latexmk on PATH, likely MiKTeX).

- docx: scripts/build_docx.py already works end to end against draft/sections,
  reusable now, Word can hand-polish before submission.
- LaTeX: template/conference-latex-template (IEEEtran.cls) + template/IEEEtranBST2
  give IEEE's own two-column typesetting, proper eq/table numbering, and a real
  .bst bibliography -- more reliable camera-ready fidelity, but needs a new
  constructor script (no .tex equivalent of build_docx.py exists yet).

Not yet decided. Revisit before final submission prep.
