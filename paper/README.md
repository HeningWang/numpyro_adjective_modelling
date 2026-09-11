# Manuscript and figures

Edit `draft.tex` and `references.bib`. Run `make draft` from the repository root.
All main-text and appendix figures are already supplied; rebuilding the manuscript
does not rerun analyses or regenerate figures.

## Figure sources

Run R scripts from this directory. Each reads the supplied summaries in `data/`.

| Manuscript figure | Source |
|---|---|
| 1: overview | `figures/overview_figure.html` |
| 2: trial, discriminability, and design | Layout in `draft.tex`; screenshot in `figures/experiment_trial_snapshot.png`; `scripts/plot_size_discriminability.R` |
| 3: slider studies | `scripts/plot_slider_interactions.R` |
| 4–5: production behaviour | `scripts/plot_revision_figures.R` |
| 6–7: model comparison and predictive checks | `scripts/plot_production_main_figures.R` |
| 8: controlled simulation | `scripts/plot_controlled_simulation.R --main-only` |
| Appendix: observed slider ratings | `scripts/plot_slider_empirical.R` |
| Appendix: model structure | `scripts/plot_model_inference_plate.R` |
| Appendix: architecture residuals | `scripts/plot_architecture_diagnostics.R` |
| Appendix: participant hierarchy | `scripts/plot_kappa_hierarchy_diagnostics.R` |
| Appendix: semantic decomposition, fitted-display predictions, and parameter sweeps | `scripts/plot_semantic_diagnostics.R` |
| Appendix: controlled architecture comparison | `scripts/plot_controlled_simulation.R` |

For example:

```sh
Rscript scripts/plot_production_main_figures.R
```

The production-behaviour script also invokes the slider and architecture scripts.
The architecture script invokes the main production-model figures. The common
visual style is defined in `scripts/csp_figure_style.R`.

The overview has an editable HTML source and supplied PDF and PNG exports.
Its optional `figures/render_overview.sh` helper uses Chrome and ImageMagick.

Some summary filenames retain version identifiers for compatibility with the
export scripts. Each retained table supports the current manuscript; these
identifiers do not designate alternative manuscript drafts.
