# Analysis

Visualization and analysis scripts that consume `results/` and produce figures + tables in `results/derived/`.

| Script | Purpose |
|--------|---------|
| `visualize_tv_distributions.py` | 4×6 grid of stacked bars showing VM(s₀,p) bilateral truth value distributions (per benchmark × model). |
| `visualize_modal_distributions.py` | Same grid for [[□p]] modal aggregated distributions. |
| `visualize_src_vs_modal.py` | Side-by-side: VM(s₀,p) vs [[□p]]. |
| `visualize_full_comparison.py` | F1-macro comparison across approaches; delta heatmaps. |
| `visualize_proper_pilot.py` | TruthfulQA pilot comparison (3-panel). |
| `visualize_tv_diagnostic.py` | Diagnostic plot of TV distributions, separately for positive vs negative ground-truth subsets. |
| `visualize_bilattice_heatmap.py` | NINE bilattice rendering. |

Outputs are written as both PDF and PNG into `results/derived/2026_03/`.

## November 2026 revision

Run in this order; none makes API calls.

| Script | Purpose |
|--------|---------|
| `revision_2026_11.py` | Derived tables for E1–E3, coverage, the transition matrix and rankings → `results/derived/2026_11/`. |
| `make_f3prime_figure.py` | The four-option figure (E2 / E2-0) → `results/derived/2026_11/f3prime_four_option.{pdf,png}`. |
| `emit_revision_tables.py` | LaTeX tables from the derived JSON, including the tabular-only fragments in `tables/lncs/`. |
| `revision_e4.py` | E4 / E4-0 analysis and its fragment → `results/derived/2026_11_e4/`. |
| `sync_generated_blocks.py` | Copies the fragments into the manuscript's `BEGIN GENERATED` / `END GENERATED` blocks and stamps the commit. `--check` reports drift without writing. |

The manuscript is `paper.tex` in `bradleypallen/bbl-paper-lncs-v2`. After any change to a fragment: commit here, then

```bash
python3 analysis/sync_generated_blocks.py /path/to/bbl-paper-lncs-v2/paper.tex
```
