Participant-kappa ablations (2026-11-24): reference model for the reframed manuscript.

Reference: intermediate speaker with participant-specific kappa (K-HKO, K-UPD-HKO) and primary
salience (property baseline plus display contrast). Full fits are the paper primary fits.

Fits: CSP `/home/csp/Documents/00_HW/hkappa_20261124/fits/{no_order,no_info,no_sal}`,
`models/production/run_corrected_primary.py --drop-*`, 4,000 warm-up, 1,000 draws, 4 chains,
max tree depth 6. `fits/convergence.csv`: no divergences; no_info converged (R-hat 1.01); no_order
and no_sal mix slowly (R-hat 1.12 to 1.66), because one to three participants have two modes
(for example kappa near .1 with alpha near .1, or kappa near 1 with alpha near 3.7). Chains agree
on the total log-likelihood except K-UPD-HKO no_order (spread about 20).
`refit_td8/`: the same three cells refitted with max tree depth 8. Mixing did not improve, and
one no_sal chain stopped at kappa_mean = 0. Response ELPD differs from the depth-6 fits by at most
6 for no_order; the depth-6 fits are used.

Files
- `component_factors/`: pointwise LOO factors of the six K cells (`analysis/component_decomposition.py`).
- `slider/`: cross-task slider predictions, 1,000 draws (`analysis/slider_cross_task_prediction.py`,
  run from `models/production`); `no_order_GI_upd/` adds G-UPD-HO and I-UPD-HO without the stable
  order term (third-factor posteriors).
- `inputs/`: the layout read by `analysis/reframing_contrasts.py` (labels primary, no_order,
  no_info, no_sal). G and I factors are copied from CSP `third_factor_20261104` (label `order`
  renamed `primary`); G and I slider predictions from `slider_cross_task/draws` and
  `salience_split/slider`.
- `sim/`: convention x salience simulation (`simulation/run_convention_salience.py`, beta =
  exp(.962), lambda = 2.0747 relative to alpha = 1, 24 scene files x 1,000 scenes).

Contrast table: `paper/data/reframing_contrasts.csv`, figures by `paper/scripts/plot_reframing.R`.
