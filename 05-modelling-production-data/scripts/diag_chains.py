"""Per-chain diagnostic for Colab NC: divergences, chain means for worst params."""

from pathlib import Path

import arviz as az
import numpy as np

NC = Path(__file__).resolve().parent.parent / "inference_data" / \
    "mcmc_results_contextual_speaker_hier_dc_warmup2000_samples1000_chains4_colab.nc"

idata = az.from_netcdf(str(NC))

# Divergences
if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
    div = idata.sample_stats["diverging"].values
    print(f"Divergences per chain: {div.sum(axis=1).tolist()}  (total {int(div.sum())})")
else:
    print("No diverging field in sample_stats.")

posterior = idata.posterior
chains = posterior.dims["chain"]
print(f"\nChains={chains}, draws={posterior.dims['draw']}")

# Per-chain means for the worst-mixing params
def per_chain_summary(name: str, idx=None):
    arr = posterior[name].values  # (chain, draw, ...)
    if idx is not None:
        arr = arr[..., idx]
    means = arr.reshape(arr.shape[0], -1).mean(axis=1)
    stds = arr.reshape(arr.shape[0], -1).std(axis=1)
    label = f"{name}" + (f"[{idx}]" if idx is not None else "")
    print(f"\n{label}:")
    for c, (m, s) in enumerate(zip(means, stds)):
        print(f"  chain {c}: mean={m:+.4f}  std={s:.4f}")
    print(f"  spread of chain-means: {means.max() - means.min():+.4f}  "
          f"(within-chain std median: {np.median(stds):.4f})")

per_chain_summary("alpha_D")
per_chain_summary("alpha_C")
per_chain_summary("alpha_F")
per_chain_summary("log_beta_lm")
per_chain_summary("tau")
per_chain_summary("delta", idx=30)
per_chain_summary("delta", idx=5)
per_chain_summary("delta", idx=47)
