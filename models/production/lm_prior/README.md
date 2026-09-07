# Fixed language-model order feature

The production model uses scores from `dbmdz/german-gpt2`. The original generator and its archived score table are retained here. They were stored under `05-modelling-production-data/generate_LM_prior/` before the publication-directory restructure.

`generate_sequences.py` uses the size adjective *großer*, six colour adjectives, six form adjectives, and the head noun *Aufkleber*. For each of the 36 colour–form pairs it generates the fifteen nonempty adjective orders without repetition, appending the noun with spaces and no surrounding sentence. The resulting table contains 540 rows, including repeated shorter strings. The generator obtains the model's mean causal-language-model loss and converts nats to bits per token. It requires PyTorch, Transformers, and access to the named pretrained model. Its historical model revision and dependency versions were not pinned.

The original aggregation averages bits-per-token surprisal within each order category, converts each mean to a weight using `2 ** (-mean_surprisal)`, and normalises across fifteen categories. These are normalised weights derived from average token surprisal. The production constants retain the original notebook's rounded float32 printout. The model then centres the log weights within each unordered adjective set, so every singleton residual is zero.

To reproduce the fixed weights from the archived table without running GPT-2:

```sh
python models/production/lm_prior/aggregate_prior.py --output /tmp/production_order_reconstruction.csv
```

The script checks all fifteen reconstructed values against the existing model constants with an absolute tolerance of `1e-8`. No model constants are updated.

## Source provenance

- Original generator Git blob: `173e876175d75b5da01dba75768024fd5cd4dfa2` (`generate_seqeunces.py`). The restored generator differs only in trailing comment whitespace.
- Archived score-table Git blob: `7bc524fc6f7aa2056c79ccaf1d2ddc091537ee4a`. The restored table is byte-identical.
- Original aggregation notebook Git blob: `0fd2aa922cd863c23349859d1ec4d903854f0e79` (`generate_LM_prior.ipynb`). `aggregate_prior.py` reproduces its deterministic calculation and checks the fixed model constants.
