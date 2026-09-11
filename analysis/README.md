# Analysis sources

The manuscript uses the frozen tables in `../paper/data/` and `summaries/`.
No inference is needed to compile the paper or draw its figures.

| Entry point | Purpose |
|---|---|
| `behavioral_bayes.py` | Hierarchical models for the two slider studies and the two production outcomes |
| `../models/production/run_corrected_primary.py` | The eight production fits reported in the manuscript |
| `../models/production/diagnose_corrected_primary.py` | Sampling diagnostics and response-wise PSIS-LOO |
| `semantic_diagnostics.py` | Decomposition of the semantic predictive contrast |
| `reconcile_primary_exports.py` | Export selected production predictions and participant summaries from a completed run |
| `exports/` | Helpers called by the production exporter |
| `../simulation/run_reproducible_sweep.py` | Original two-order random-scene simulation |
| `../simulation/run_controlled_random_scenes.py` | Controlled comparisons on the same saved scenes |
| `../simulation/run_fitted_bridge.py` | Supplementary predictions on experimental displays |

The behavioural source is restored from the commit recorded by the fitted
analyses (`711af39`). Its input files are included, along with the four reported
behavioural summary tables. The deterministic production input is
`../data/production_model_input.csv` (9,100 responses from 113 participants).

Production and simulation runners require a CUDA environment with JAX x64.
Consult each script's command-line arguments before starting a run. Store new
outputs in a separate run directory; existing posterior files should be retained
unchanged. The selected fits use 1,000 warm-up iterations and four chains. Global
models use 4,000 retained draws per chain and depth 8; the remaining models use
1,000 retained draws and depth 6.

The eight production identifiers are `G-HO`, `I-HO`, `K-HO`, their `UPD`
counterparts, and `K-HKO` / `K-UPD-HKO`. `G`, `I`, and `K` denote global, fully
incremental, and plan-guided production. `UPD` denotes sequential context updating;
`HKO` adds participant-specific successive-choice weights. The manuscript explains
the model components and all statistical contrasts.

Large posterior arrays and saved random scenes are stored separately from
this manuscript checkout. Export and diagnostic scripts that consume them require
the completed run directory; the supplied tables can be inspected directly.
