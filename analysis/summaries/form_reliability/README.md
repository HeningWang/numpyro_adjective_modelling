# Form-reliability sensitivity (S6)

Six shared-kappa production models refitted with informative form semantics,
nu_F in {.59, .75} (primary fits: nu_F = .50). Warmup 1000, samples 1000, 4 chains,
tree depth 6, seed 4711. All twelve fits: R-hat <= 1.010, no divergences,
max Pareto k < .5.

- `elpd/elpd_contrasts.csv`: PSIS-LOO differences against nu_F = .50, overall and per
  adjective pair, plus architecture and semantics contrasts within each nu_F
  (participant Dirichlet bootstrap, 200,000 draws). Script: `analysis/form_reliability_elpd.py`.
- `slider/nuF*/`: cross-task slider predictions from the refitted posteriors
  (1,000 draws x 20 virtual raters). Script: `analysis/slider_cross_task_prediction.py
  --form-reliability`.
- `slider/intervals_nuF*.csv`: gradients and contrasts with 95% intervals.
  Script: `analysis/slider_cross_task_intervals.py`.

CSP workdir: `/home/csp/Documents/00_HW/form_reliability_20261110`.
