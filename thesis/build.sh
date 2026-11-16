#!/usr/bin/env bash
# Build Thesis.docx from the Markdown chapters:  bash thesis/build.sh
# Needs pandoc (>= 3.1). Citations: [@key] from references.bib (IEEE style);
# cross-references: Figure @fig:x / Table @tbl:x / Section @sec:x (crossref.lua).
# Figures: python3 figures/make_figures.py (data) and figures/src/*.mmd (mermaid-cli).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3 make_bib.py
pandoc metadata.yaml chapters/0*.md \
  --from markdown+pipe_tables+table_captions+implicit_figures \
  --lua-filter crossref.lua \
  --citeproc --bibliography references.bib --csl ieee.csl \
  --reference-doc reference.docx --resource-path .:figures \
  --toc --toc-depth 2 --number-sections \
  -o Thesis.docx
python3 fix_docx.py Thesis.docx
echo "wrote $(pwd)/Thesis.docx"
