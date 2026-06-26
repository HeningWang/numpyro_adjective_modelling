"""CLI entry point for running MCMC inference on the slider dataset.

Usage:
    python run_inference.py --speaker_type incremental --hierarchical \
        --num_warmup 500 --num_samples 500 --num_chains 4
"""
import os
# Default local runs to CPU with 4 host devices, but honour env overrides.
# Server pilots should set `JAX_PLATFORMS=cuda` before launching this script.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
if os.environ.get("JAX_PLATFORMS", "").lower() == "cpu":
    os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import argparse
import string
import subprocess
import numpy as np
import arviz as az
import jax
from jax import random
from jax.random import PRNGKey
import numpyro
from numpyro.infer import MCMC, NUTS, Predictive

from modelSpecification import (
    import_dataset,
    import_dataset_hier,
    precompute_listeners,
    precompute_listeners_frozen,
    precompute_listeners_at_csv,
    precompute_listeners_frozen_at_csv,
    precompute_listeners_production_anchor,
    slider_is_sharp_vector,
    canonicalize_speaker_type,
    likelihood_gb_speaker,
    likelihood_inc_speaker,
    likelihood_planned_usefulness_order_speaker,
    likelihood_planned_signed_usefulness_order_speaker,
    likelihood_planned_usefulness_mixture_speaker,
    likelihood_planned_usefulness_mixture_anchored_speaker,
    likelihood_planned_usefulness_order_speaker_static,
    likelihood_planned_signed_usefulness_order_speaker_static,
    likelihood_planned_usefulness_mixture_speaker_static,
    likelihood_planned_usefulness_mixture_anchored_speaker_static,
    likelihood_production_anchor_inc_speaker,
    likelihood_production_anchor_global_speaker,
    likelihood_gb_speaker_static,
    likelihood_inc_speaker_frozen,
    likelihood_gb_speaker_hier,
    likelihood_inc_speaker_hier,
    likelihood_planned_usefulness_order_speaker_hier,
    likelihood_planned_signed_usefulness_order_speaker_hier,
    likelihood_planned_usefulness_mixture_speaker_hier,
    likelihood_planned_usefulness_mixture_anchored_speaker_hier,
    likelihood_planned_usefulness_order_speaker_static_hier,
    likelihood_planned_signed_usefulness_order_speaker_static_hier,
    likelihood_planned_usefulness_mixture_speaker_static_hier,
    likelihood_planned_usefulness_mixture_anchored_speaker_static_hier,
    likelihood_production_anchor_inc_speaker_hier,
    likelihood_production_anchor_global_speaker_hier,
    likelihood_gb_speaker_static_hier,
    likelihood_inc_speaker_frozen_hier,
    likelihood_inc_speaker_hier_free_csv,
    likelihood_inc_speaker_frozen_hier_free_csv,
    ZOIB,
)

RECURSIVE_LISTENER_SPEAKERS = {
    "global",
    "incremental",
    "planned_usefulness_order",
    "planned_usefulness_signed_order",
    "planned_usefulness_mixture",
    "planned_usefulness_mixture_anchored",
    "production_anchor_sizesharp_2x2_inc_rec",
    "production_anchor_sizesharp_2x2_glob_rec",
    "production_anchor_reliabilitybackup_2x2_inc_rec",
    "production_anchor_reliabilitybackup_2x2_glob_rec",
}

PRODUCTION_ANCHOR_SPEAKERS = {
    "production_anchor_sizesharp_2x2_inc_rec",
    "production_anchor_sizesharp_2x2_inc_static",
    "production_anchor_sizesharp_2x2_glob_rec",
    "production_anchor_sizesharp_2x2_glob_static",
    "production_anchor_reliabilitybackup_2x2_inc_rec",
    "production_anchor_reliabilitybackup_2x2_inc_static",
    "production_anchor_reliabilitybackup_2x2_glob_rec",
    "production_anchor_reliabilitybackup_2x2_glob_static",
}

SPEAKER_CHOICES = [
    "global",
    "incremental",
    "global_static",
    "incremental_static",
    "incremental_frozen",
    "planned_usefulness_order",
    "planned_usefulness_order_static",
    "planned_usefulness_signed_order",
    "planned_usefulness_signed_order_static",
    "planned_usefulness_mixture",
    "planned_usefulness_mixture_static",
    "planned_usefulness_mixture_anchored",
    "planned_usefulness_mixture_anchored_static",
    "production_anchor_sizesharp_2x2_inc_rec",
    "production_anchor_sizesharp_2x2_inc_static",
    "production_anchor_sizesharp_2x2_glob_rec",
    "production_anchor_sizesharp_2x2_glob_static",
    "production_anchor_reliabilitybackup_2x2_inc_rec",
    "production_anchor_reliabilitybackup_2x2_inc_static",
    "production_anchor_reliabilitybackup_2x2_glob_rec",
    "production_anchor_reliabilitybackup_2x2_glob_static",
]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def artifact_tag_suffix(artifact_tag: str = "") -> str:
    """Return a safe filename suffix for optional artifact-provenance tags."""
    if not artifact_tag:
        return ""
    allowed = set(string.ascii_letters + string.digits + "_.-")
    if any(ch not in allowed for ch in artifact_tag):
        raise ValueError("--artifact_tag may only contain letters, digits, '_', '.', or '-'.")
    return f"_{artifact_tag}"


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def git_dirty_state() -> str:
    try:
        unstaged = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        return "true" if unstaged or staged else "false"
    except Exception:
        return "unknown"


def run_metadata(**fields) -> dict[str, str]:
    metadata = {
        "project_git_commit": git_value(["rev-parse", "--short", "HEAD"]),
        "project_git_dirty": git_dirty_state(),
        **fields,
    }
    return {key: "" if value is None else str(value) for key, value in metadata.items()}


def attach_run_metadata(idata, metadata: dict[str, str]) -> None:
    for group in ("posterior", "sample_stats", "prior", "posterior_predictive", "observed_data"):
        dataset = getattr(idata, group, None)
        if dataset is not None:
            dataset.attrs.update(metadata)


def balanced_fold_ids(
    df,
    num_folds: int,
    seed: int = 13,
    group_cols=("relevant_property", "sharpness"),
):
    """Deterministic condition-balanced observation folds."""
    if num_folds < 2:
        raise ValueError("--num_folds must be at least 2.")
    rng = np.random.default_rng(seed)
    fold_ids = np.empty(len(df), dtype=np.int32)
    grouped = df.reset_index(drop=True).groupby(list(group_cols), sort=True)
    for idx in grouped.indices.values():
        idx = np.asarray(idx, dtype=np.int32)
        rng.shuffle(idx)
        fold_ids[idx] = np.arange(len(idx), dtype=np.int32) % num_folds
    return fold_ids


def select_listener_precompute(canonical_speaker_type: str, states, color_semvalue=None):
    if canonical_speaker_type in PRODUCTION_ANCHOR_SPEAKERS:
        return precompute_listeners_production_anchor(
            states,
            recursive=canonical_speaker_type in RECURSIVE_LISTENER_SPEAKERS,
        )
    if color_semvalue is not None:
        if canonical_speaker_type in RECURSIVE_LISTENER_SPEAKERS:
            return precompute_listeners_at_csv(states, color_semvalue)
        return precompute_listeners_frozen_at_csv(states, color_semvalue)
    if canonical_speaker_type in RECURSIVE_LISTENER_SPEAKERS:
        return precompute_listeners(states)
    return precompute_listeners_frozen(states)


def get_hier_model(canonical_speaker_type: str, free_color_semvalue: bool = False):
    if free_color_semvalue:
        if canonical_speaker_type == "incremental":
            return likelihood_inc_speaker_hier_free_csv
        if canonical_speaker_type == "incremental_static":
            return likelihood_inc_speaker_frozen_hier_free_csv
        raise ValueError(
            "--free_color_semvalue is only implemented for the incremental cells "
            "(speaker_type=incremental or incremental_static)."
        )
    if canonical_speaker_type == "global":
        return likelihood_gb_speaker_hier
    if canonical_speaker_type == "incremental":
        return likelihood_inc_speaker_hier
    if canonical_speaker_type == "planned_usefulness_order":
        return likelihood_planned_usefulness_order_speaker_hier
    if canonical_speaker_type == "planned_usefulness_signed_order":
        return likelihood_planned_signed_usefulness_order_speaker_hier
    if canonical_speaker_type == "planned_usefulness_mixture":
        return likelihood_planned_usefulness_mixture_speaker_hier
    if canonical_speaker_type == "planned_usefulness_mixture_anchored":
        return likelihood_planned_usefulness_mixture_anchored_speaker_hier
    if canonical_speaker_type == "global_static":
        return likelihood_gb_speaker_static_hier
    if canonical_speaker_type == "incremental_static":
        return likelihood_inc_speaker_frozen_hier
    if canonical_speaker_type == "planned_usefulness_order_static":
        return likelihood_planned_usefulness_order_speaker_static_hier
    if canonical_speaker_type == "planned_usefulness_signed_order_static":
        return likelihood_planned_signed_usefulness_order_speaker_static_hier
    if canonical_speaker_type == "planned_usefulness_mixture_static":
        return likelihood_planned_usefulness_mixture_speaker_static_hier
    if canonical_speaker_type == "planned_usefulness_mixture_anchored_static":
        return likelihood_planned_usefulness_mixture_anchored_speaker_static_hier
    if canonical_speaker_type in (
        "production_anchor_sizesharp_2x2_inc_rec",
        "production_anchor_reliabilitybackup_2x2_inc_rec",
    ):
        return likelihood_production_anchor_inc_speaker_hier
    if canonical_speaker_type in (
        "production_anchor_sizesharp_2x2_inc_static",
        "production_anchor_reliabilitybackup_2x2_inc_static",
    ):
        return likelihood_production_anchor_inc_speaker_hier
    if canonical_speaker_type in (
        "production_anchor_sizesharp_2x2_glob_rec",
        "production_anchor_reliabilitybackup_2x2_glob_rec",
    ):
        return likelihood_production_anchor_global_speaker_hier
    if canonical_speaker_type in (
        "production_anchor_sizesharp_2x2_glob_static",
        "production_anchor_reliabilitybackup_2x2_glob_static",
    ):
        return likelihood_production_anchor_global_speaker_hier
    raise ValueError(
        "Invalid speaker type. Choose 'global', 'incremental', "
        "planned usefulness variants, 'global_static', "
        "'incremental_static', or planned usefulness static variants "
        "(legacy alias: 'incremental_frozen')."
    )


def run_inference(
    speaker_type: str = "global",
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    artifact_tag: str = "",
):
    canonical_speaker_type = canonicalize_speaker_type(speaker_type)
    states_train, empirical_train, df = import_dataset()
    is_sharp_train = slider_is_sharp_vector(df)

    empirical_train_np = np.asarray(empirical_train)
    pi0 = float(np.mean(np.isclose(empirical_train_np, 0.0)))
    pi1 = float(np.mean(np.isclose(empirical_train_np, 1.0)))
    if (pi0 + pi1) >= 0.95:
        raise ValueError(f"Boundary masses too large for ZOIB: pi0+pi1={pi0+pi1:.3f}")

    L1_all, L2_all = select_listener_precompute(canonical_speaker_type, states_train)
    print("Listeners precomputed.")

    run_tag = artifact_tag_suffix(artifact_tag)
    output_file_name = (
        f"./inference_data/mcmc_results_{canonical_speaker_type}_speaker"
        f"{run_tag}_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    print(f"Output file: {output_file_name}")
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Removed existing file: {output_file_name}")

    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    if canonical_speaker_type == "global":
        model = likelihood_gb_speaker
    elif canonical_speaker_type == "incremental":
        model = likelihood_inc_speaker
    elif canonical_speaker_type == "planned_usefulness_order":
        model = likelihood_planned_usefulness_order_speaker
    elif canonical_speaker_type == "planned_usefulness_signed_order":
        model = likelihood_planned_signed_usefulness_order_speaker
    elif canonical_speaker_type == "planned_usefulness_mixture":
        model = likelihood_planned_usefulness_mixture_speaker
    elif canonical_speaker_type == "planned_usefulness_mixture_anchored":
        model = likelihood_planned_usefulness_mixture_anchored_speaker
    elif canonical_speaker_type == "global_static":
        model = likelihood_gb_speaker_static
    elif canonical_speaker_type == "incremental_static":
        model = likelihood_inc_speaker_frozen
    elif canonical_speaker_type == "planned_usefulness_order_static":
        model = likelihood_planned_usefulness_order_speaker_static
    elif canonical_speaker_type == "planned_usefulness_signed_order_static":
        model = likelihood_planned_signed_usefulness_order_speaker_static
    elif canonical_speaker_type == "planned_usefulness_mixture_static":
        model = likelihood_planned_usefulness_mixture_speaker_static
    elif canonical_speaker_type == "planned_usefulness_mixture_anchored_static":
        model = likelihood_planned_usefulness_mixture_anchored_speaker_static
    elif canonical_speaker_type in (
        "production_anchor_sizesharp_2x2_inc_rec",
        "production_anchor_reliabilitybackup_2x2_inc_rec",
    ):
        model = likelihood_production_anchor_inc_speaker
    elif canonical_speaker_type in (
        "production_anchor_sizesharp_2x2_inc_static",
        "production_anchor_reliabilitybackup_2x2_inc_static",
    ):
        model = likelihood_production_anchor_inc_speaker
    elif canonical_speaker_type in (
        "production_anchor_sizesharp_2x2_glob_rec",
        "production_anchor_reliabilitybackup_2x2_glob_rec",
    ):
        model = likelihood_production_anchor_global_speaker
    elif canonical_speaker_type in (
        "production_anchor_sizesharp_2x2_glob_static",
        "production_anchor_reliabilitybackup_2x2_glob_static",
    ):
        model = likelihood_production_anchor_global_speaker
    else:
        raise ValueError(
            "Invalid speaker type. Choose 'global', 'incremental', "
            "planned usefulness variants, 'global_static', "
            "'incremental_static', or planned usefulness static variants "
            "(legacy alias: 'incremental_frozen')."
        )

    kernel = NUTS(model, dense_mass=True, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="vectorized")
    if canonical_speaker_type in PRODUCTION_ANCHOR_SPEAKERS:
        mcmc.run(
            rng_key_,
            states_train,
            empirical_train,
            pi0,
            pi1,
            L1_all,
            L2_all,
            is_sharp_train,
        )
    else:
        mcmc.run(rng_key_, states_train, empirical_train, pi0, pi1, L1_all, L2_all)
    mcmc.print_summary()

    posterior_samples = mcmc.get_samples()
    if canonical_speaker_type in PRODUCTION_ANCHOR_SPEAKERS:
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), states_train, None, pi0, pi1, L1_all, L2_all, is_sharp_train
        )
        prior = Predictive(model, num_samples=1000)(
            PRNGKey(2), states_train, None, pi0, pi1, L1_all, L2_all, is_sharp_train
        )
    else:
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), states_train, None, pi0, pi1, L1_all, L2_all
        )
        prior = Predictive(model, num_samples=1000)(
            PRNGKey(2), states_train, None, pi0, pi1, L1_all, L2_all
        )

    N = states_train.shape[0]
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={"states": np.arange(N)},
        dims={"obs": ["states"]},
    )
    attach_run_metadata(
        numpyro_data,
        run_metadata(
            dataset="slider",
            run_kind="flat",
            speaker_type=speaker_type,
            canonical_speaker_type=canonical_speaker_type,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            artifact_tag=artifact_tag,
            artifact_file=os.path.basename(output_file_name),
            n_observations=N,
        ),
    )
    az.to_netcdf(numpyro_data, output_file_name)
    assert os.path.exists(output_file_name), f"Save failed: {output_file_name} not found"
    size_mb = os.path.getsize(output_file_name) / 1024 / 1024
    print(f"Saved: {output_file_name}  ({size_mb:.1f} MB)")


def run_inference_hier(
    speaker_type: str = "global",
    num_samples: int = 1000,
    num_warmup: int = 1000,
    num_chains: int = 4,
    free_color_semvalue: bool = False,
    color_semvalue: float | None = None,
    heldout_fold: int | None = None,
    num_folds: int = 5,
    fold_seed: int = 13,
    artifact_tag: str = "",
):
    """Run MCMC for the hierarchical (random participant intercept) speaker model.

    When ``free_color_semvalue`` is True (only valid for incremental cells), the
    colour reliability nu is sampled with ``Uniform(0.5, 0.999)`` and the literal
    listeners are recomputed inside the model at each MCMC step. The output
    filename is suffixed with ``_free_csv`` to keep it distinct from the fixed-nu
    fits.

    When ``color_semvalue`` is given, the literal listeners are precomputed at
    that value instead of the module-level ``FIXED_COLOR_SEMVALUE``. The output
    filename is suffixed with ``_csv{NNN}`` (e.g. ``_csv063`` for 0.63). This
    is mutually exclusive with ``free_color_semvalue``.
    """
    canonical_speaker_type = canonicalize_speaker_type(speaker_type)
    states_all, empirical_all, df, participant_idx_all, n_participants = import_dataset_hier()
    print(f"Hierarchical model: {n_participants} participants, {len(states_all)} observations")

    fold_tag = ""
    if heldout_fold is not None:
        if free_color_semvalue:
            raise ValueError("--heldout_fold is not implemented with --free_color_semvalue.")
        if heldout_fold < 0 or heldout_fold >= num_folds:
            raise ValueError("--heldout_fold must be in [0, num_folds).")
        fold_ids = balanced_fold_ids(df, num_folds=num_folds, seed=fold_seed)
        train_mask = fold_ids != heldout_fold
        heldout_mask = fold_ids == heldout_fold
        states_train = states_all[train_mask]
        empirical_train = empirical_all[train_mask]
        participant_idx = participant_idx_all[train_mask]
        fold_tag = f"_fold{heldout_fold}of{num_folds}"
        print(
            f"Heldout fold {heldout_fold}/{num_folds}: "
            f"{int(train_mask.sum())} train, {int(heldout_mask.sum())} heldout"
        )
    else:
        states_train = states_all
        empirical_train = empirical_all
        participant_idx = participant_idx_all
        train_mask = None

    empirical_train_np = np.asarray(empirical_train)
    pi0 = float(np.mean(np.isclose(empirical_train_np, 0.0)))
    pi1 = float(np.mean(np.isclose(empirical_train_np, 1.0)))
    if (pi0 + pi1) >= 0.95:
        raise ValueError(f"Boundary masses too large for ZOIB: pi0+pi1={pi0+pi1:.3f}")

    if free_color_semvalue and canonical_speaker_type not in ("incremental", "incremental_static"):
        raise ValueError(
            "--free_color_semvalue is only implemented for the incremental cells "
            "(speaker_type=incremental or incremental_static)."
        )
    if free_color_semvalue and color_semvalue is not None:
        raise ValueError(
            "--free_color_semvalue and --color_semvalue are mutually exclusive."
        )

    if color_semvalue is not None:
        L1_all, L2_all = select_listener_precompute(
            canonical_speaker_type,
            states_train,
            color_semvalue=color_semvalue,
        )
        print(f"Listeners precomputed at color_semvalue = {color_semvalue}.")
    else:
        L1_all, L2_all = select_listener_precompute(canonical_speaker_type, states_train)
    print("Listeners precomputed.")
    is_sharp_all = slider_is_sharp_vector(df)
    is_sharp_train = is_sharp_all if train_mask is None else is_sharp_all[train_mask]

    if free_color_semvalue:
        suffix = "_free_csv"
    elif color_semvalue is not None:
        suffix = f"_csv{int(round(color_semvalue * 100)):03d}"
    else:
        suffix = ""
    run_tag = artifact_tag_suffix(artifact_tag)
    output_file_name = (
        f"./inference_data/mcmc_results_{canonical_speaker_type}_speaker_hier{suffix}{fold_tag}"
        f"{run_tag}_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    print(f"Output file: {output_file_name}")
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Removed existing file: {output_file_name}")

    rng_key = random.PRNGKey(11)
    rng_key, rng_key_ = random.split(rng_key)

    model = get_hier_model(canonical_speaker_type, free_color_semvalue=free_color_semvalue)

    kernel = NUTS(model, dense_mass=False, max_tree_depth=8, target_accept_prob=0.9)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )
    if canonical_speaker_type in PRODUCTION_ANCHOR_SPEAKERS:
        mcmc.run(
            rng_key_, states_train, empirical_train,
            pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
            is_sharp_train,
        )
    else:
        mcmc.run(
            rng_key_, states_train, empirical_train,
            pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
        )
    mcmc.print_summary(exclude_deterministic=False)

    posterior_samples = mcmc.get_samples()
    if canonical_speaker_type in PRODUCTION_ANCHOR_SPEAKERS:
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), states_train, None,
            pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
            is_sharp_train,
        )
        prior = Predictive(model, num_samples=1000)(
            PRNGKey(2), states_train, None,
            pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
            is_sharp_train,
        )
    else:
        posterior_predictive = Predictive(model, posterior_samples)(
            PRNGKey(1), states_train, None,
            pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
        )
        prior = Predictive(model, num_samples=1000)(
            PRNGKey(2), states_train, None,
            pi0, pi1, participant_idx, n_participants, L1_all, L2_all,
        )

    N = states_train.shape[0]
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={
            "states": np.arange(N),
            "participants": np.arange(n_participants),
        },
        dims={"obs": ["states"], "delta": ["participants"]},
    )
    attach_run_metadata(
        numpyro_data,
        run_metadata(
            dataset="slider",
            run_kind="hierarchical",
            speaker_type=speaker_type,
            canonical_speaker_type=canonical_speaker_type,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            artifact_tag=artifact_tag,
            artifact_file=os.path.basename(output_file_name),
            free_color_semvalue=free_color_semvalue,
            color_semvalue=color_semvalue,
            heldout_fold=heldout_fold,
            num_folds=num_folds,
            fold_seed=fold_seed,
            n_observations=N,
            n_participants=n_participants,
        ),
    )
    az.to_netcdf(numpyro_data, output_file_name)
    assert os.path.exists(output_file_name), f"Save failed: {output_file_name} not found"
    size_mb = os.path.getsize(output_file_name) / 1024 / 1024
    print(f"Saved: {output_file_name}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker inference with NumPyro.")
    parser.add_argument("--speaker_type", type=str, choices=SPEAKER_CHOICES,
                        default="global", help="Choose the speaker model type.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=500, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Run hierarchical model with random participant intercepts."
    )
    parser.add_argument(
        "--free_color_semvalue", action="store_true",
        help=(
            "Sample colour reliability nu ~ Uniform(0.5, 0.999) inside the model "
            "and recompute literal listeners each step. Only valid for the "
            "incremental cells; produces an *_hier_free_csv_*.nc artifact."
        ),
    )
    parser.add_argument(
        "--color_semvalue", type=float, default=None,
        help=(
            "Override the module-level FIXED_COLOR_SEMVALUE when precomputing "
            "literal listeners. Output filename is suffixed with _csv{NNN}. "
            "Mutually exclusive with --free_color_semvalue."
        ),
    )
    parser.add_argument(
        "--heldout_fold", type=int, default=None,
        help="Fit all observations except this deterministic condition-balanced fold.",
    )
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--fold_seed", type=int, default=13)
    parser.add_argument(
        "--artifact_tag", type=str, default="",
        help=(
            "Optional safe token appended to inference artifact filenames "
            "before the warmup/sample/chains tag."
        ),
    )

    args = parser.parse_args()

    jax.local_device_count()

    if args.test:
        pass
    elif args.hierarchical:
        run_inference_hier(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            free_color_semvalue=args.free_color_semvalue,
            color_semvalue=args.color_semvalue,
            heldout_fold=args.heldout_fold,
            num_folds=args.num_folds,
            fold_seed=args.fold_seed,
            artifact_tag=args.artifact_tag,
        )
    else:
        if args.heldout_fold is not None:
            raise ValueError("--heldout_fold requires --hierarchical.")
        if args.free_color_semvalue:
            raise ValueError("--free_color_semvalue requires --hierarchical.")
        if args.color_semvalue is not None:
            raise ValueError("--color_semvalue requires --hierarchical.")
        run_inference(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            artifact_tag=args.artifact_tag,
        )
