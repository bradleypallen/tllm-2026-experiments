# talk

Presentation materials based on the 2026 experimental results.

- `tllm2026.tex` — LaTeX source for the talk
- `tllm2026.pdf` — built slide deck
- `*.pdf` — figures embedded in the talk (regenerable from `../analysis/visualize_*.py` against `../results/2026_03/`)

The build artifacts (`.aux`, `.log`, `.nav`, `.out`, `.snm`, `.toc`, `.vrb`) are kept alongside the source so the deck reproduces deterministically. They are mostly LaTeX overhead — feel free to `latexmk -c` and rebuild if needed.
