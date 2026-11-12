Salience split (2026-11-10): property baseline versus display contrast.

Fits: CSP `/home/csp/Documents/00_HW/salience_split_20261110/fits/{split,no_base,no_contrast}`,
models G-HO, K-HO, I-HO (context-fixed), `models/production/run_tuned.py`, standard sampler
contract (1,000 warm-up, 1,000 draws, 4 chains). No divergences; maximum R-hat 1.005 to 1.014
(the two highest, 1.014, are the global and plan-guided fits without contrast).

`split`: separate coefficients for the centred property baseline (0, 1, .25) and the centred
display contrast. `no_base` / `no_contrast`: the respective coefficient fixed at zero.

Files
- `component_factors/`: pointwise LOO factors of the nine fits (`analysis/component_decomposition.py --tuned ...`).
- `reference_factors/`: factors of the primary and no-salience fits (copied from `third_factor_20261104`).
- `salience_split_gains.csv`: participant-bootstrap gains (20,000 draws) by response, length,
  set given length, order given set.
- `slider/`: cross-task slider predictions (1,000 draws, 20 virtual raters) for the nine fits and
  for the existing no-order, no-informativeness and no-salience posteriors.
- `slider_switch_gradients_replication.csv`: context gradients and switch contrasts, replication
  study only (the original study's preprocessed file carries template display values for the
  third property).
- `fits/`: run manifests.

Baseline-only model (2026-11-10): salience restricted to the property baseline (`split_salience`,
`drop_salience_contrast`). Fits on CSP under `fits/{base_upd,base_no_info,base_no_order,base_no_suff}`;
the context-fixed cells are the `no_contrast` fits. Thirteen of fifteen fits converged (no
divergences, R-hat at most 1.02). The plan-guided fits without the stable order term are bimodal
(kappa near .89 or near .08) also with 4,000 warm-up steps (`base_no_order_w4000`); the kappa .89
mode has the higher log density by about 770. `base_no_order_modeA` keeps the chains in that mode
(three for K-HO, two for K-UPD-HO) and is the version used in the gains table.
- `base_model_gains.csv`: participant-bootstrap gains for the baseline-only model's contrasts.
- `base_model_slider_gradients_replication.csv`: slider context gradients per switch (replication
  study); its plan-guided no-order rows predate the mode selection and use the unconverged fits.
