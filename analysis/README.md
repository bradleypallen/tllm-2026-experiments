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
