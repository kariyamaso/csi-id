Overleaf upload (paper)

This folder is a minimal Overleaf-ready subset of the project for compiling the paper.

How to use
- Upload the contents of this `overleaf/` folder (not the parent repo) to Overleaf as a new project.
- Set the main file to `paper_main.tex` in Overleaf Project Settings.
- Compiler: set to LuaLaTeX.

Included assets
- `paper_main.tex`
- `figures/ntu_fi_results/validation_accuracy_bar.png`
- `figures/ntu_fi_results/training_curves.png`
- `figures/ntu_fi_results/results.tex`

Notes
- The LaTeX engine requires LuaLaTeX + luatexja packages (available on Overleaf).
- If you build locally from this folder, run:
- `latexmk -C -outdir=out paper_main.tex`
- `latexmk -g -f -lualatex -interaction=nonstopmode -file-line-error -outdir=out paper_main.tex`
