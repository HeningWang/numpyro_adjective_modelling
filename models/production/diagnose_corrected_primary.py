"""Validate one corrected primary artifact and export its prespecified LOO scores."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import arviz as az
import numpy as np
import pandas as pd


def diagnose(artifact: Path, output: Path, encoded: Path):
    output.mkdir(parents=True, exist_ok=True)
    name = artifact.stem
    manifest = json.loads(artifact.with_name(f"{name}_manifest.json").read_text())
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if not manifest["complete"] or digest != manifest["artifact_sha256"]:
        raise ValueError("Incomplete artifact or hash mismatch")
    idata = az.from_netcdf(artifact)
    role = idata.attrs.get("artifact_role")
    recovery = role == "recovery" and name in {"G-HO", "G-UPD-HO"}
    expected_draws, expected_depth = (4000, 8) if recovery else (1000, 6)
    if role not in {"primary", "recovery"} or (role == "recovery" and not recovery):
        raise ValueError("Timing pilots cannot enter primary summaries")
    if (manifest["warmup"], manifest["samples"], manifest["max_tree_depth"]) != (1000, expected_draws, expected_depth):
        raise ValueError("Artifact does not match its approved sampler contract")
    with np.load(encoded, allow_pickle=False) as data:
        expected = data["empirical_seq_flat"]
    np.testing.assert_array_equal(idata.observed_data.obs.values, expected)
    if idata.posterior.sizes["chain"] != 4 or idata.posterior.sizes["draw"] != expected_draws:
        raise ValueError("Unexpected posterior dimensions")
    ll = idata.log_likelihood.obs.values
    if ll.shape != (4, expected_draws, 9100) or not np.isfinite(ll).all():
        raise ValueError("Invalid likelihood dimensions or values")
    pp = idata.posterior_predictive.obs.values
    if pp.shape != ll.shape or not ((pp >= 0) & (pp < 15)).all():
        raise ValueError("Invalid posterior-predictive response codes")
    deterministic = {"alpha_by_participant", "delta", "epsilon", "nu_F",
                     "kappa_by_participant", "log_beta_order_by_participant",
                     "beta_order_by_participant"}
    if name.startswith(("G-", "I-")):
        deterministic.add("kappa")
    sampled = idata.posterior[[n for n in idata.posterior if n not in deterministic]]
    summary = az.summary(sampled, kind="diagnostics", round_to="none")
    summary.to_csv(output/f"{name}_parameter_diagnostics.csv", index_label="parameter")
    # Save interpretable parameters too, with constants explicitly excluded.
    variable = [n for n in idata.posterior
                if np.any(np.std(idata.posterior[n].values, axis=(0, 1)) > 0)]
    az.summary(idata, var_names=variable, round_to="none").to_csv(
        output/f"{name}_posterior_summary.csv", index_label="parameter")
    loo = az.loo(idata, pointwise=True)
    pareto = np.asarray(loo.pareto_k).reshape(-1)
    pointwise = np.asarray(loo.loo_i).reshape(-1)
    pd.DataFrame({"canonical_row_position": np.arange(9100), name: pointwise,
                  "pareto_k": pareto}).to_csv(output/f"{name}_pointwise_loo.csv", index=False)
    bfmi = np.asarray(az.bfmi(idata))
    stats = idata.sample_stats
    step_cap = 2**expected_depth - 1
    row = dict(model=name, artifact_role=role, retained_draws_per_chain=expected_draws,
        max_tree_depth=expected_depth, elpd_loo=float(loo.elpd_loo), p_loo=float(loo.p_loo),
        se_elpd_loo=float(loo.se), divergences=int(stats.diverging.values.sum()),
        max_rhat=float(summary.r_hat.max()), min_bulk_ess=float(summary.ess_bulk.min()),
        min_tail_ess=float(summary.ess_tail.min()), min_bfmi=float(bfmi.min()),
        max_pareto_k=float(pareto.max()), pareto_k_gt_0_7=int((pareto > .7).sum()),
        tree_depth_saturation_fraction=float((stats.n_steps.values >= step_cap).mean()),
        mean_acceptance=float(stats.accept_prob.values.mean()),
        finite_diagnostics=bool(np.isfinite(summary[["r_hat", "ess_bulk", "ess_tail"]]).all().all()),
        pointwise_reconstruction_error=float(abs(pointwise.sum()-loo.elpd_loo)),
        artifact_sha256=digest)
    row["sampler_gate_pass"] = bool(row["finite_diagnostics"] and row["divergences"] == 0
        and row["max_rhat"] < 1.01 and row["min_bulk_ess"] > 400
        and row["min_tail_ess"] > 400 and row["min_bfmi"] > .3
        and row["max_pareto_k"] < .7 and row["pointwise_reconstruction_error"] < 1e-8)
    pd.DataFrame([row]).to_csv(output/f"{name}_diagnostics.csv", index=False)
    pd.DataFrame({"chain": np.arange(4), "bfmi": bfmi,
        "divergences": stats.diverging.values.sum(axis=1),
        "tree_depth_saturation_fraction": (stats.n_steps.values >= step_cap).mean(axis=1),
        "mean_acceptance": stats.accept_prob.values.mean(axis=1)}).to_csv(
            output/f"{name}_chain_diagnostics.csv", index=False)
    print(json.dumps(row), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("artifact", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--encoded", type=Path, required=True)
    args = p.parse_args()
    diagnose(args.artifact, args.output, args.encoded)
