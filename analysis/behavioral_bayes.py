"""Bayesian mixed-effects complements for the behavioral hypothesis tests.

The four fitted models reproduce the fixed and random structures used in the
reported frequentist analyses.  Posterior draws are converted into
population-marginal predictive cell means and directional hypothesis
contrasts on the outcome scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


CONTEXTS = ("size_sufficient", "both_necessary", "colour_sufficient")
DISCRIMINABILITY = ("low", "high")
CELL_LABELS = tuple(
    f"{context}:{discriminability}"
    for context in CONTEXTS
    for discriminability in DISCRIMINABILITY
)
FIT_IDS = (
    "study1a_slider",
    "study1b_slider",
    "production_size_initial",
    "production_overinformative",
)
FIT_SEEDS = {
    "study1a_slider": 1401,
    "study1b_slider": 1402,
    "production_size_initial": 2401,
    "production_overinformative": 2402,
}


@dataclass(frozen=True)
class PreparedData:
    fit_id: str
    likelihood: str
    outcome_name: str
    outcome: np.ndarray
    fixed_design: np.ndarray
    random_design: np.ndarray
    participant_index: np.ndarray
    participant_labels: tuple[str, ...]
    item_index: np.ndarray | None
    item_labels: tuple[str, ...]
    context: np.ndarray
    discriminability: np.ndarray

    @property
    def n_observations(self) -> int:
        return int(self.outcome.shape[0])

    @property
    def n_participants(self) -> int:
        return len(self.participant_labels)

    @property
    def n_items(self) -> int:
        return len(self.item_labels)


def _factorize(values: pd.Series) -> tuple[np.ndarray, tuple[str, ...]]:
    labels = tuple(sorted(values.astype(str).unique()))
    lookup = {label: index for index, label in enumerate(labels)}
    indices = values.astype(str).map(lookup).to_numpy(dtype=np.int32)
    return indices, labels


def _design_matrices(
    context: Sequence[str], discriminability: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    context_array = np.asarray(context, dtype=str)
    discriminability_array = np.asarray(discriminability, dtype=str)
    both = (context_array == "both_necessary").astype(float)
    colour = (context_array == "colour_sufficient").astype(float)
    high = (discriminability_array == "high").astype(float)
    fixed = np.column_stack(
        [np.ones(len(context_array)), both, colour, high, both * high, colour * high]
    )
    random = np.column_stack([np.ones(len(context_array)), both, colour])
    return fixed.astype(np.float64), random.astype(np.float64)


def _cell_design() -> tuple[np.ndarray, np.ndarray]:
    contexts = [context for context in CONTEXTS for _ in DISCRIMINABILITY]
    discriminability = list(DISCRIMINABILITY) * len(CONTEXTS)
    return _design_matrices(contexts, discriminability)


def _prepare_slider(
    repo_root: Path, fit_id: str
) -> tuple[pd.DataFrame, np.ndarray]:
    if fit_id == "study1a_slider":
        frame = pd.read_csv(repo_root / "paper/data/slider_empirical.csv")
        outcome = frame["human_slider"].to_numpy(dtype=float) - 0.5
    else:
        frame = pd.read_csv(
            repo_root / "data/01-slider-data-replication-preprocessed.csv"
        )
        frame = frame.loc[frame["combination"].eq("dimension_color")].copy()
        outcome = frame["rating_centered"].to_numpy(dtype=float)
    frame["context"] = frame["relevant_property"].replace(
        {
            "first": "size_sufficient",
            "both": "both_necessary",
            "second": "colour_sufficient",
        }
    )
    frame["discriminability"] = frame["sharpness"].replace(
        {"blurred": "low", "sharp": "high"}
    )
    return frame, outcome


def _prepare_production(
    repo_root: Path, fit_id: str
) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.read_csv(repo_root / "data/01-production-data-preprocessed.csv")
    frame = frame.loc[
        frame["conditions"].isin(("zrdc", "erdc", "brdc"))
        & frame["annotation"].notna()
        & frame["annotation"].ne("")
    ].copy()
    keys = ["id", "item", "conditions"]
    conflicting = (
        frame.groupby(keys, sort=False)["annotation"].nunique().gt(1).sum()
    )
    if int(conflicting) != 0:
        raise ValueError("canonical production trials contain conflicting outcomes")
    frame = frame.drop_duplicates(keys, keep="first").copy()
    frame["context"] = frame["relevant_property"].replace(
        {
            "first": "size_sufficient",
            "both": "both_necessary",
            "second": "colour_sufficient",
        }
    )
    frame["discriminability"] = frame["sharpness"].replace(
        {"blurred": "low", "sharp": "high"}
    )
    adjective_count = frame["annotation"].str.len()
    if fit_id == "production_size_initial":
        outcome = frame["annotation"].str.startswith("D").to_numpy(dtype=float)
    else:
        outcome = (
            (frame["context"].eq("both_necessary") & adjective_count.eq(3))
            | (~frame["context"].eq("both_necessary") & adjective_count.gt(1))
        ).to_numpy(dtype=float)
    return frame, outcome


def load_analysis_dataset(repo_root: Path | str, fit_id: str) -> PreparedData:
    root = Path(repo_root).resolve()
    if fit_id not in FIT_IDS:
        raise ValueError(f"unknown behavioral fit: {fit_id}")
    if fit_id.endswith("slider"):
        frame, outcome = _prepare_slider(root, fit_id)
        likelihood = "gaussian"
        outcome_name = "centered_rating"
        item_index = None
        item_labels: tuple[str, ...] = ()
    else:
        frame, outcome = _prepare_production(root, fit_id)
        likelihood = "bernoulli"
        outcome_name = (
            "size_initial"
            if fit_id == "production_size_initial"
            else "overinformative"
        )
        item_index, item_labels = _factorize(frame["item"])
    invalid_contexts = set(frame["context"].unique()).difference(CONTEXTS)
    invalid_discriminability = set(
        frame["discriminability"].unique()
    ).difference(DISCRIMINABILITY)
    if invalid_contexts or invalid_discriminability:
        raise ValueError("analysis data contain unknown experimental levels")
    fixed, random = _design_matrices(
        frame["context"], frame["discriminability"]
    )
    participant_index, participant_labels = _factorize(frame["id"])
    return PreparedData(
        fit_id=fit_id,
        likelihood=likelihood,
        outcome_name=outcome_name,
        outcome=np.asarray(outcome, dtype=np.float64),
        fixed_design=fixed,
        random_design=random,
        participant_index=participant_index,
        participant_labels=participant_labels,
        item_index=item_index,
        item_labels=item_labels,
        context=frame["context"].to_numpy(dtype=str),
        discriminability=frame["discriminability"].to_numpy(dtype=str),
    )


def _mixed_model(prepared: PreparedData):
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    beta_scale = (
        jnp.asarray([0.30] * 6)
        if prepared.likelihood == "gaussian"
        else jnp.asarray([2.0, 1.5, 1.5, 1.5, 1.5, 1.5])
    )
    participant_prior_scale = 0.20 if prepared.likelihood == "gaussian" else 1.0

    def model(
        fixed_design,
        random_design,
        participant_index,
        outcome=None,
        item_index=None,
    ):
        beta = numpyro.sample(
            "beta", dist.Normal(jnp.zeros(6), beta_scale).to_event(1)
        )
        participant_scale = numpyro.sample(
            "participant_scale",
            dist.HalfNormal(participant_prior_scale).expand([3]).to_event(1),
        )
        participant_corr_cholesky = numpyro.sample(
            "participant_corr_cholesky", dist.LKJCholesky(3, concentration=2.0)
        )
        with numpyro.plate("participant", prepared.n_participants):
            participant_z = numpyro.sample(
                "participant_z", dist.Normal(0, 1).expand([3]).to_event(1)
            )
        participant_cholesky = (
            participant_scale[..., :, None] * participant_corr_cholesky
        )
        participant_effect = participant_z @ participant_cholesky.T
        linear_predictor = fixed_design @ beta + jnp.sum(
            random_design * participant_effect[participant_index], axis=-1
        )
        if item_index is not None:
            item_scale = numpyro.sample("item_scale", dist.HalfNormal(1.0))
            with numpyro.plate("item", prepared.n_items):
                item_z = numpyro.sample("item_z", dist.Normal(0, 1))
            linear_predictor = linear_predictor + item_scale * item_z[item_index]
        if prepared.likelihood == "gaussian":
            residual_scale = numpyro.sample(
                "residual_scale", dist.HalfNormal(0.25)
            )
            numpyro.sample(
                "obs", dist.Normal(linear_predictor, residual_scale), obs=outcome
            )
        else:
            numpyro.sample(
                "obs", dist.Bernoulli(logits=linear_predictor), obs=outcome
            )

    return model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_commit(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()


def _minimum_bfmi(energy: np.ndarray) -> float:
    values = np.asarray(energy, dtype=float)
    variance = np.var(values, axis=1, ddof=1)
    transitions = np.mean(np.diff(values, axis=1) ** 2, axis=1)
    if np.any(variance <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("BFMI is undefined for the supplied energy draws")
    return float(np.min(transitions / variance))


def _sampler_diagnostics(
    samples: Mapping[str, np.ndarray],
    extras: Mapping[str, np.ndarray],
    max_tree_depth: int,
) -> dict[str, Any]:
    import arviz as az

    inference_data = az.from_dict(posterior=dict(samples))
    summary = az.summary(inference_data, kind="diagnostics", round_to=None)
    tree_limit = 2**max_tree_depth - 1
    return {
        "n_chains": int(next(iter(samples.values())).shape[0]),
        "draws_per_chain": int(next(iter(samples.values())).shape[1]),
        "divergences": int(np.count_nonzero(extras["diverging"])),
        "tree_depth_hits": int(
            np.count_nonzero(np.asarray(extras["num_steps"]) >= tree_limit)
        ),
        "max_rhat": float(summary["r_hat"].dropna().max()),
        "min_ess_bulk": float(summary["ess_bulk"].dropna().min()),
        "min_ess_tail": float(summary["ess_tail"].dropna().min()),
        "min_bfmi": _minimum_bfmi(np.asarray(extras["energy"])),
        "maximum_num_steps": int(np.max(extras["num_steps"])),
    }


def fit_dataset(
    prepared: PreparedData,
    output_dir: Path,
    *,
    warmup: int,
    samples: int,
    chains: int,
    chain_method: str,
    target_accept: float,
    max_tree_depth: int,
    seed: int,
) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC, NUTS

    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"FIT_START fit={prepared.fit_id} n={prepared.n_observations} "
        f"participants={prepared.n_participants}",
        flush=True,
    )
    kernel = NUTS(
        _mixed_model(prepared),
        target_accept_prob=target_accept,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=samples,
        num_chains=chains,
        chain_method=chain_method,
        progress_bar=True,
    )
    mcmc.run(
        jax.random.PRNGKey(seed),
        fixed_design=jnp.asarray(prepared.fixed_design),
        random_design=jnp.asarray(prepared.random_design),
        participant_index=jnp.asarray(prepared.participant_index),
        outcome=jnp.asarray(prepared.outcome),
        item_index=(
            None if prepared.item_index is None else jnp.asarray(prepared.item_index)
        ),
        extra_fields=("diverging", "num_steps", "potential_energy", "energy"),
    )
    posterior = {
        name: np.asarray(value)
        for name, value in mcmc.get_samples(group_by_chain=True).items()
    }
    extras = {
        name: np.asarray(value)
        for name, value in mcmc.get_extra_fields(group_by_chain=True).items()
    }
    diagnostics = _sampler_diagnostics(posterior, extras, max_tree_depth)
    np.savez_compressed(output_dir / "posterior_samples.npz", **posterior)
    metadata = {
        "fit_id": prepared.fit_id,
        "likelihood": prepared.likelihood,
        "outcome": prepared.outcome_name,
        "n_observations": prepared.n_observations,
        "n_participants": prepared.n_participants,
        "n_items": prepared.n_items,
        "seed": seed,
        "warmup": warmup,
        "samples": samples,
        "chains": chains,
        "chain_method": chain_method,
        "target_accept": target_accept,
        "max_tree_depth": max_tree_depth,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"FIT_COMPLETE fit={prepared.fit_id} divergences={diagnostics['divergences']} "
        f"max_rhat={diagnostics['max_rhat']:.4f} "
        f"min_ess_bulk={diagnostics['min_ess_bulk']:.1f}",
        flush=True,
    )
    return {**metadata, **diagnostics}


def posterior_cell_draws(
    samples: Mapping[str, np.ndarray], *, likelihood: str
) -> np.ndarray:
    fixed, random = _cell_design()
    beta = np.asarray(samples["beta"], dtype=float).reshape(-1, 6)
    linear = beta @ fixed.T
    if likelihood == "gaussian":
        return linear
    if likelihood != "bernoulli":
        raise ValueError(f"unknown likelihood: {likelihood}")
    participant_scale = np.asarray(
        samples["participant_scale"], dtype=float
    ).reshape(-1, 3)
    corr_cholesky = np.asarray(
        samples["participant_corr_cholesky"], dtype=float
    ).reshape(-1, 3, 3)
    scale_cholesky = participant_scale[:, :, None] * corr_cholesky
    covariance = scale_cholesky @ np.swapaxes(scale_cholesky, 1, 2)
    variance = np.einsum("ci,dij,cj->dc", random, covariance, random)
    if "item_scale" in samples:
        item_scale = np.asarray(samples["item_scale"], dtype=float).reshape(-1)
        variance = variance + item_scale[:, None] ** 2
    nodes, weights = np.polynomial.hermite.hermgauss(30)
    logits = linear[:, :, None] + np.sqrt(2.0 * variance[:, :, None]) * nodes
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -35.0, 35.0)))
    return np.sum(probabilities * weights, axis=2) / math.sqrt(math.pi)


def _contrast_row(
    fit_id: str,
    outcome: str,
    hypothesis: str,
    contrast: np.ndarray,
    definition: str,
) -> dict[str, Any]:
    values = np.asarray(contrast, dtype=float)
    lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
    probability = float(np.mean(values > 0))
    return {
        "study": fit_id,
        "outcome": outcome,
        "hypothesis": hypothesis,
        "definition": definition,
        "mean": float(np.mean(values)),
        "median": float(median),
        "lower": float(lower),
        "upper": float(upper),
        "posterior_probability": probability,
        "credible_direction": bool(lower > 0),
    }


def hypothesis_rows(cell_draws: np.ndarray, fit_id: str) -> pd.DataFrame:
    cells = np.asarray(cell_draws, dtype=float)
    if cells.ndim != 2 or cells.shape[1] != 6:
        raise ValueError("cell draws must have shape (draw, 6)")
    outcome = {
        "study1a_slider": "centered_rating",
        "study1b_slider": "centered_rating",
        "production_size_initial": "size_initial_probability",
        "production_overinformative": "overinformative_probability",
    }[fit_id]
    size = cells[:, 0:2].mean(axis=1)
    both = cells[:, 2:4].mean(axis=1)
    colour = cells[:, 4:6].mean(axis=1)
    rows: list[dict[str, Any]] = []
    if fit_id != "production_overinformative":
        size_both = size - both
        both_colour = both - colour
        rows.extend(
            [
                _contrast_row(
                    fit_id,
                    outcome,
                    "size_sufficient_above_both_necessary",
                    size_both,
                    "size-sufficient minus both-necessary",
                ),
                _contrast_row(
                    fit_id,
                    outcome,
                    "both_necessary_above_colour_sufficient",
                    both_colour,
                    "both-necessary minus colour-sufficient",
                ),
                _contrast_row(
                    fit_id,
                    outcome,
                    "graded_context_order",
                    np.minimum(size_both, both_colour),
                    "minimum of the two adjacent context contrasts",
                ),
            ]
        )
    if fit_id.endswith("slider"):
        rows.append(
            _contrast_row(
                fit_id,
                outcome,
                "colour_sufficient_above_neutral",
                colour,
                "colour-sufficient centered rating minus zero",
            )
        )
        for context_index, context in enumerate(CONTEXTS):
            low = cells[:, 2 * context_index]
            high = cells[:, 2 * context_index + 1]
            rows.append(
                _contrast_row(
                    fit_id,
                    outcome,
                    f"{context}_low_minus_high_discriminability",
                    low - high,
                    f"{context}: low minus high discriminability",
                )
            )
    elif fit_id == "production_size_initial":
        high_minus_low = np.column_stack(
            [cells[:, 1] - cells[:, 0], cells[:, 3] - cells[:, 2], cells[:, 5] - cells[:, 4]]
        )
        for context_index, context in enumerate(CONTEXTS):
            rows.append(
                _contrast_row(
                    fit_id,
                    outcome,
                    f"{context}_high_minus_low_discriminability",
                    high_minus_low[:, context_index],
                    f"{context}: high minus low discriminability",
                )
            )
        interaction = high_minus_low[:, 1] - high_minus_low[:, [0, 2]].mean(axis=1)
        rows.append(
            _contrast_row(
                fit_id,
                outcome,
                "both_necessary_high_discriminability_advantage",
                interaction,
                "both-necessary high-minus-low contrast minus the mean "
                "contrast in sufficient contexts",
            )
        )
    else:
        low_minus_high = np.column_stack(
            [cells[:, 0] - cells[:, 1], cells[:, 2] - cells[:, 3], cells[:, 4] - cells[:, 5]]
        )
        for context_index, context in enumerate(CONTEXTS):
            rows.append(
                _contrast_row(
                    fit_id,
                    outcome,
                    f"{context}_low_minus_high_discriminability",
                    low_minus_high[:, context_index],
                    f"{context}: low minus high discriminability",
                )
            )
        interaction = low_minus_high[:, 0] - low_minus_high[:, [1, 2]].mean(axis=1)
        rows.append(
            _contrast_row(
                fit_id,
                outcome,
                "size_sufficient_low_discriminability_advantage",
                interaction,
                "size-sufficient low-minus-high contrast minus the mean "
                "contrast in the other contexts",
            )
        )
    return pd.DataFrame(rows)


def _prediction_rows(
    prepared: PreparedData, cell_draws: np.ndarray
) -> pd.DataFrame:
    records = []
    for cell_index, label in enumerate(CELL_LABELS):
        context, discriminability = label.split(":")
        values = cell_draws[:, cell_index]
        observed_mask = (prepared.context == context) & (
            prepared.discriminability == discriminability
        )
        lower, median, upper = np.quantile(values, [0.025, 0.5, 0.975])
        records.append(
            {
                "study": prepared.fit_id,
                "outcome": prepared.outcome_name,
                "context": context,
                "discriminability": discriminability,
                "observed_mean": float(prepared.outcome[observed_mask].mean()),
                "posterior_mean": float(values.mean()),
                "median": float(median),
                "lower": float(lower),
                "upper": float(upper),
                "n_observations": int(np.count_nonzero(observed_mask)),
            }
        )
    return pd.DataFrame(records)


def summarize_run(
    repo_root: Path,
    run_dir: Path,
    paper_data_dir: Path | None = None,
) -> dict[str, Any]:
    prediction_frames = []
    hypothesis_frames = []
    diagnostic_records = []
    for fit_id in FIT_IDS:
        fit_dir = run_dir / fit_id
        metadata = json.loads((fit_dir / "metadata.json").read_text())
        diagnostics = json.loads((fit_dir / "diagnostics.json").read_text())
        with np.load(fit_dir / "posterior_samples.npz") as archive:
            posterior = {name: archive[name] for name in archive.files}
        prepared = load_analysis_dataset(repo_root, fit_id)
        cells = posterior_cell_draws(posterior, likelihood=metadata["likelihood"])
        prediction_frames.append(_prediction_rows(prepared, cells))
        hypothesis_frames.append(hypothesis_rows(cells, fit_id))
        diagnostic_records.append({"study": fit_id, **diagnostics})
    summary_dir = run_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    hypotheses = pd.concat(hypothesis_frames, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_records)
    outputs = {
        "behavioral_bayesian_predictions.csv": predictions,
        "behavioral_bayesian_hypotheses.csv": hypotheses,
        "behavioral_bayesian_diagnostics.csv": diagnostics,
    }
    for filename, frame in outputs.items():
        frame.to_csv(summary_dir / filename, index=False)
        if paper_data_dir is not None:
            paper_data_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(paper_data_dir / filename, index=False)
    result = {
        "status": "summary_complete",
        "run_dir": str(run_dir),
        "prediction_rows": int(len(predictions)),
        "hypothesis_rows": int(len(hypotheses)),
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def fit_all(
    repo_root: Path,
    run_dir: Path,
    *,
    warmup: int,
    samples: int,
    chains: int,
    chain_method: str,
    target_accept: float,
    max_tree_depth: int,
) -> dict[str, Any]:
    import importlib.metadata
    import jax

    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": _source_commit(repo_root),
        "python": sys.version,
        "platform": platform.platform(),
        "device_backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "settings": {
            "warmup": warmup,
            "samples": samples,
            "chains": chains,
            "chain_method": chain_method,
            "target_accept": target_accept,
            "max_tree_depth": max_tree_depth,
        },
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("jax", "jaxlib", "numpyro", "arviz", "numpy", "pandas")
        },
        "data_hashes": {
            str(path.relative_to(repo_root)): _sha256(path)
            for path in (
                repo_root / "paper/data/slider_empirical.csv",
                repo_root / "data/01-slider-data-replication-preprocessed.csv",
                repo_root / "data/01-production-data-preprocessed.csv",
            )
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    fit_records = []
    for fit_id in FIT_IDS:
        prepared = load_analysis_dataset(repo_root, fit_id)
        fit_records.append(
            fit_dataset(
                prepared,
                run_dir / fit_id,
                warmup=warmup,
                samples=samples,
                chains=chains,
                chain_method=chain_method,
                target_accept=target_accept,
                max_tree_depth=max_tree_depth,
                seed=FIT_SEEDS[fit_id],
            )
        )
    result = {"status": "fit_complete", "run_dir": str(run_dir), "fits": fit_records}
    (run_dir / "fit_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "fit_complete", "run_dir": str(run_dir)}), flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit-all")
    fit.add_argument("--repo-root", default=".")
    fit.add_argument("--run-dir", required=True)
    fit.add_argument("--warmup", type=int, default=1000)
    fit.add_argument("--samples", type=int, default=1000)
    fit.add_argument("--chains", type=int, default=4)
    fit.add_argument(
        "--chain-method",
        choices=("parallel", "vectorized", "sequential"),
        default="vectorized",
    )
    fit.add_argument("--target-accept", type=float, default=0.90)
    fit.add_argument("--max-tree-depth", type=int, default=9)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--repo-root", default=".")
    summarize.add_argument("--run-dir", required=True)
    summarize.add_argument("--paper-data-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repo_root = Path(arguments.repo_root).resolve()
    if arguments.command == "fit-all":
        fit_all(
            repo_root,
            Path(arguments.run_dir).resolve(),
            warmup=arguments.warmup,
            samples=arguments.samples,
            chains=arguments.chains,
            chain_method=arguments.chain_method,
            target_accept=arguments.target_accept,
            max_tree_depth=arguments.max_tree_depth,
        )
    else:
        summarize_run(
            repo_root,
            Path(arguments.run_dir).resolve(),
            None if arguments.paper_data_dir is None else Path(arguments.paper_data_dir).resolve(),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
