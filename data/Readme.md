# Experimental data

| File | Contents |
|---|---|
| `00-stimuli-table-raw.csv` | Recorded object properties and experimental design |
| `00-slider-data-raw.csv` / `00-slider-subj-info.csv` | Study 1a trials and participant records |
| `00-slider-data-replication-raw.csv` / `00-slider-data-replication-raw-subj-info.csv` | Study 1b trials and participant records |
| `00-production-data-raw.csv` / `00-production-subj-info.csv` | Study 2 trials and participant records |
| `01-slider-data-preprocessed.csv` | Cleaned Study 1a data |
| `01-slider-data-replication-preprocessed.csv` | Cleaned Study 1b data used in the Bayesian analysis |
| `01-production-data-preprocessed.csv` | Cleaned and annotated Study 2 data |
| `production_model_input.csv` | Deterministic reconstruction of the recorded displays for all 9,100 modelled production responses |
| `01-preprocessing_n_manipulation.ipynb` | Original preprocessing code |

The current production fits use `production_model_input.csv`. Its SHA-256 is
`44a41b44b83dfef7a2d1788c6ac399a637cb9cb48b63eb3751339ed849bf4ac1`.
The input preserves recorded size, colour, and form values. The model loader
recodes colour and form as target-match relations.

The original preprocessing notebook documents earlier data preparation, including
an earlier feature encoding. Use the supplied deterministic model input for the
reported production fits. The manuscript appendix describes exclusions, coding,
and the behavioural subsets. In production annotations, `D` denotes size, `C`
colour, and `F` form; their sequence records adjective order.
