Cross-task prediction of slider order ratings from the production posteriors (no refit).

Script: `analysis/slider_cross_task_prediction.py`, run on CSP on 2026-11-10 from the model code
that produced the primary posteriors (modelSpecification.py sha 9502779f…), in
`/home/csp/Documents/00_HW/slider_cross_task_20261110`.
Settings: 400 posterior draws per model, 10 virtual raters per draw (participant effects
drawn from the fitted population), seed 20261110. Slider rows: 18,985 (original and
replication, all three adjective pairs); 4,729 distinct displays.

Link: the predicted rating is P(canonical order) / (P(canonical) + P(reversed)) from the
production distribution over fifteen utterances (DC/CD, DF/FD, CF/FC).

Summary: `analysis/slider_cross_task_summary.py` → `summary.csv`, `slider_cross_task_cells.csv`.
Metrics use posterior-mean cell predictions: r (36 cells), r_within (pair means removed),
held-out RMSE of a two-parameter logit calibration fitted on one study and tested on the
other, and the context gradient (logit first-sufficient − logit second-sufficient).
No credible intervals for the metrics yet; per-draw cell predictions were not saved.

## Per-draw run with intervals (2026-11-10)
`draws/`: 1,000 posterior draws × 20 virtual raters per model; per-draw cell shares in
`*_cell_draws.npy` (columns follow the rows of `*_cells.csv`).
`analysis/slider_cross_task_intervals.py` → `intervals.csv`: context gradient per pair and
averaged, within-pair r, and model contrasts (draws paired by index; independent posteriors).
Observed gradients: participant bootstrap within study, 2,000 reps, seed 20261110.
Caution: the averaged gradient of the fully incremental model matches the observed one partly
because it overshoots colour–form (2.10 vs 1.12) and undershoots size–form (0.18 vs 1.09).
