"""CLI entry point for running MCMC inference on the production dataset.

Usage:
    python run_inference.py --speaker_type incremental --hierarchical \
        --num_warmup 500 --num_samples 500 --num_chains 4
"""
import os
# Default local runs to CPU with 4 host devices, but honour env overrides.
# Server runs should set `JAX_PLATFORMS=cuda` before launching this script.
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

from helper import import_dataset, import_dataset_hier
from modelSpecification import (
    canonicalize_speaker_type,
    likelihood_function_global_speaker,
    likelihood_function_incremental_speaker,
    likelihood_function_global_speaker_static,
    likelihood_function_incremental_speaker_frozen,
    likelihood_function_global_speaker_hier,
    likelihood_function_incremental_speaker_hier,
    likelihood_function_incremental_speaker_lowcol_hier,
    likelihood_function_contextual_hier,
    likelihood_function_contextual_lambdaunc_hier,
    likelihood_function_simplified_lm_resid_hier,
    likelihood_function_simplified_lm_raw_hier,
    likelihood_function_simplified_hand_order_hier,
    likelihood_function_simplified_no_frontload_hier,
    likelihood_function_simplified_no_uncertainty_len_hier,
    likelihood_function_simplified_no_order_hier,
    likelihood_function_principled_hier,
    likelihood_function_principled_no_order_hier,
    likelihood_function_principled_no_salience_hier,
    likelihood_function_principled_no_uncertainty_len_hier,
    likelihood_function_principled_salience_stop_hier,
    likelihood_function_principled_salience_stop_regularized_hier,
    likelihood_function_principled_salience_stop_regularized_2x2_inc_rec_hier,
    likelihood_function_principled_salience_stop_regularized_2x2_inc_static_hier,
    likelihood_function_principled_salience_stop_regularized_2x2_glob_rec_hier,
    likelihood_function_principled_salience_stop_regularized_2x2_glob_static_hier,
    likelihood_function_principled_salience_stop_regularized_2x2_glob_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_2x2_glob_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_tmcc_2x2_inc_rec_hier,
    likelihood_function_principled_salience_stop_regularized_tmcc_2x2_inc_static_hier,
    likelihood_function_principled_salience_stop_regularized_plannedprefix_2x2_inc_rec_hier,
    likelihood_function_principled_salience_stop_regularized_plannedprefix_2x2_inc_static_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_static_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_static_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_rec_hier,
    likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_static_hier,
    likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_rec_fixedeps_hier,
    likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_static_fixedeps_hier,
    likelihood_function_principled_salience_stop_strong_regularized_hier,
    likelihood_function_contextual_freewf_hier,
    likelihood_function_contextual_anchored_hier,
    likelihood_function_contextual_freewf_anchored_hier,
    likelihood_function_contextual_anchored_gamma_hier,
    likelihood_function_contextual_anchored_gamma_fixedwf_hier,
    likelihood_function_contextual_anchored_gamma_fixedwf_pcalpha_hier,
    likelihood_function_contextual_pcalpha_gammasharp_hier,
    likelihood_function_contextual_pcalpha_formmod_hier,
    likelihood_function_contextual_pcalpha_canon_hier,
    likelihood_function_contextual_pcalpha_canon_betafixed_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_gammasharp_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_lambdasuff_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freecsv_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freefsv_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freek_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freewf_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freeall4_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_csv059_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_rec_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_static_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_2x2_glob_rec_hier,
    likelihood_function_contextual_pcalpha_canon_parsimony_2x2_glob_static_hier,
    likelihood_function_global_speaker_static_hier,
    likelihood_function_incremental_speaker_frozen_hier,
    likelihood_function_incremental_lm_only_hier,
    likelihood_function_incremental_rsa_only_hier,
    likelihood_function_incremental_speaker_lookahead_hier,
    likelihood_function_incremental_speaker_extended_hier,
    likelihood_function_incremental_speaker_mixture_hier,
    likelihood_function_incremental_speaker_mixture_simple_hier,
    likelihood_function_reported_hier,
    likelihood_function_reported_lowcol_hier,
    likelihood_function_v5_hier,
    likelihood_function_v5_no_lm_hier,
    likelihood_function_v5a_hier,
    likelihood_function_v5b_hier,
    likelihood_function_v5_inc_static_hier,
    likelihood_function_v5_global_hier,
    likelihood_function_v5_global_static_hier,
    likelihood_function_v5_global_full_hier,
    likelihood_function_v5_global_static_full_hier,
)


# Model dispatch tables
FLAT_MODELS = {
    "global": (likelihood_function_global_speaker, 0.9, 5),
    "incremental": (likelihood_function_incremental_speaker, 0.85, 5),
    "global_static": (likelihood_function_global_speaker_static, 0.9, 5),
    "incremental_static": (likelihood_function_incremental_speaker_frozen, 0.85, 5),
}

HIER_MODELS = {
    "global": (likelihood_function_global_speaker_hier, 0.9, 5),
    "incremental": (likelihood_function_incremental_speaker_hier, 0.85, 5),
    "global_static": (likelihood_function_global_speaker_static_hier, 0.9, 5),
    "incremental_static": (likelihood_function_incremental_speaker_frozen_hier, 0.85, 5),
    "incremental_lm_only": (likelihood_function_incremental_lm_only_hier, 0.85, 5),
    "incremental_rsa_only": (likelihood_function_incremental_rsa_only_hier, 0.85, 5),
    "incremental_lookahead": (likelihood_function_incremental_speaker_lookahead_hier, 0.85, 5),
    "incremental_extended": (likelihood_function_incremental_speaker_extended_hier, 0.85, 5),
    "incremental_mixture": (likelihood_function_incremental_speaker_mixture_hier, 0.80, 5),
    "incremental_mixture_simple": (likelihood_function_incremental_speaker_mixture_simple_hier, 0.80, 5),
    "reported": (likelihood_function_reported_hier, 0.85, 5),
    "reported_lowcol": (likelihood_function_reported_lowcol_hier, 0.85, 5),
    "incremental_lowcol": (likelihood_function_incremental_speaker_lowcol_hier, 0.85, 5),
    "contextual": (likelihood_function_contextual_hier, 0.85, 5),
    "contextual_lambdaunc": (likelihood_function_contextual_lambdaunc_hier, 0.85, 5),
    "simplified_lm_resid": (likelihood_function_simplified_lm_resid_hier, 0.85, 5),
    "simplified_lm_raw": (likelihood_function_simplified_lm_raw_hier, 0.85, 5),
    "simplified_hand_order": (likelihood_function_simplified_hand_order_hier, 0.85, 5),
    "simplified_no_frontload": (likelihood_function_simplified_no_frontload_hier, 0.85, 5),
    "simplified_no_uncertainty_len": (likelihood_function_simplified_no_uncertainty_len_hier, 0.85, 5),
    "simplified_no_order": (likelihood_function_simplified_no_order_hier, 0.85, 5),
    "principled": (likelihood_function_principled_hier, 0.85, 5),
    "principled_no_order": (likelihood_function_principled_no_order_hier, 0.85, 5),
    "principled_no_salience": (likelihood_function_principled_no_salience_hier, 0.85, 5),
    "principled_no_uncertainty_len": (likelihood_function_principled_no_uncertainty_len_hier, 0.85, 5),
    "principled_salience_stop": (likelihood_function_principled_salience_stop_hier, 0.85, 5),
    "principled_salience_stop_regularized": (likelihood_function_principled_salience_stop_regularized_hier, 0.85, 5),
    "principled_salience_stop_regularized_2x2_inc_rec": (likelihood_function_principled_salience_stop_regularized_2x2_inc_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_2x2_inc_static": (likelihood_function_principled_salience_stop_regularized_2x2_inc_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_2x2_glob_rec": (likelihood_function_principled_salience_stop_regularized_2x2_glob_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_2x2_glob_static": (likelihood_function_principled_salience_stop_regularized_2x2_glob_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_2x2_glob_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_2x2_glob_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_2x2_glob_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_2x2_glob_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_tmcc_2x2_inc_rec": (likelihood_function_principled_salience_stop_regularized_tmcc_2x2_inc_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_tmcc_2x2_inc_static": (likelihood_function_principled_salience_stop_regularized_tmcc_2x2_inc_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_plannedprefix_2x2_inc_rec": (likelihood_function_principled_salience_stop_regularized_plannedprefix_2x2_inc_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_plannedprefix_2x2_inc_static": (likelihood_function_principled_salience_stop_regularized_plannedprefix_2x2_inc_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec": (likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_2x2_inc_static": (likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_2x2_glob_rec": (likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_2x2_glob_static": (likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_2x2_inc_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_inc_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_2x2_glob_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_2x2_glob_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_tmcc_2x2_glob_rec": (likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_rec_hier, 0.85, 5),
    "principled_salience_stop_regularized_tmcc_2x2_glob_static": (likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_static_hier, 0.85, 5),
    "principled_salience_stop_regularized_tmcc_2x2_glob_rec_fixedeps": (likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_rec_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_regularized_tmcc_2x2_glob_static_fixedeps": (likelihood_function_principled_salience_stop_regularized_tmcc_2x2_glob_static_fixedeps_hier, 0.85, 5),
    "principled_salience_stop_strong_regularized": (likelihood_function_principled_salience_stop_strong_regularized_hier, 0.85, 5),
    "contextual_freewf": (likelihood_function_contextual_freewf_hier, 0.85, 5),
    "contextual_anchored": (likelihood_function_contextual_anchored_hier, 0.85, 5),
    "contextual_freewf_anchored": (likelihood_function_contextual_freewf_anchored_hier, 0.85, 5),
    "contextual_anchored_gamma": (likelihood_function_contextual_anchored_gamma_hier, 0.85, 5),
    "contextual_anchored_gamma_fixedwf": (likelihood_function_contextual_anchored_gamma_fixedwf_hier, 0.85, 5),
    "contextual_anchored_gamma_fixedwf_pcalpha": (likelihood_function_contextual_anchored_gamma_fixedwf_pcalpha_hier, 0.85, 5),
    "contextual_pcalpha_gammasharp": (likelihood_function_contextual_pcalpha_gammasharp_hier, 0.85, 5),
    "contextual_pcalpha_formmod": (likelihood_function_contextual_pcalpha_formmod_hier, 0.85, 5),
    "contextual_pcalpha_canon": (likelihood_function_contextual_pcalpha_canon_hier, 0.85, 5),
    "contextual_pcalpha_canon_betafixed": (likelihood_function_contextual_pcalpha_canon_betafixed_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony": (likelihood_function_contextual_pcalpha_canon_parsimony_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_gammasharp": (likelihood_function_contextual_pcalpha_canon_parsimony_no_gammasharp_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_lambdasuff": (likelihood_function_contextual_pcalpha_canon_parsimony_no_lambdasuff_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_alphaF": (likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_alphaF_freecsv": (likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freecsv_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_alphaF_freefsv": (likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freefsv_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_alphaF_freek": (likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freek_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_alphaF_freewf": (likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freewf_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_alphaF_freeall4": (likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_freeall4_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_no_alphaF_csv059": (likelihood_function_contextual_pcalpha_canon_parsimony_no_alphaF_csv059_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_2x2_inc_rec": (likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_rec_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_2x2_inc_static": (likelihood_function_contextual_pcalpha_canon_parsimony_2x2_inc_static_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_2x2_glob_rec": (likelihood_function_contextual_pcalpha_canon_parsimony_2x2_glob_rec_hier, 0.85, 5),
    "contextual_pcalpha_canon_parsimony_2x2_glob_static": (likelihood_function_contextual_pcalpha_canon_parsimony_2x2_glob_static_hier, 0.85, 5),
    "v5":        (likelihood_function_v5_hier,       0.85, 5),
    "v5_no_lm":  (likelihood_function_v5_no_lm_hier, 0.85, 5),
    "v5a":       (likelihood_function_v5a_hier,      0.85, 5),
    "v5b":       (likelihood_function_v5b_hier,      0.85, 5),
    "v5_inc_static":    (likelihood_function_v5_inc_static_hier,   0.85, 5),
    "v5_global":        (likelihood_function_v5_global_hier,        0.85, 5),
    "v5_global_static": (likelihood_function_v5_global_static_hier, 0.85, 5),
    "v5_global_full":        (likelihood_function_v5_global_full_hier,        0.85, 5),
    "v5_global_static_full": (likelihood_function_v5_global_static_full_hier, 0.85, 5),
}

SPEAKER_CHOICES = sorted(set(FLAT_MODELS) | set(HIER_MODELS) | {"incremental_frozen"})

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def artifact_tag_suffix(artifact_tag: str = "") -> str:
    """Return a safe filename suffix for optional artifact-provenance tags."""
    if not artifact_tag:
        return ""
    allowed = set(string.ascii_letters + string.digits + "_.-")
    if any(ch not in allowed for ch in artifact_tag):
        raise ValueError("--artifact-tag may only contain letters, digits, '_', '.', or '-'.")
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


def run_inference(
    speaker_type: str = "global",
    num_warmup: int = 100,
    num_samples: int = 250,
    num_chains: int = 4,
    artifact_tag: str = "",
):
    canonical_speaker_type = canonicalize_speaker_type(speaker_type)

    run_tag = artifact_tag_suffix(artifact_tag)
    output_file_name = (
        f"./inference_data/mcmc_results_{canonical_speaker_type}_speaker"
        f"{run_tag}_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Deleted existing file: {output_file_name}")

    data = import_dataset()
    states_train = data["states_train"]
    empirical_train_seq_flat = data["empirical_seq_flat"]

    print("States train shape:", states_train.shape)
    print("Empirical train flat shape:", empirical_train_seq_flat.shape)
    print("Output file name:", output_file_name)

    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    if canonical_speaker_type not in FLAT_MODELS:
        raise ValueError(
            f"Unknown speaker_type '{canonical_speaker_type}'. "
            f"Choose from: {list(FLAT_MODELS.keys())}"
        )
    model, target_accept_prob, max_tree_depth = FLAT_MODELS[canonical_speaker_type]

    kernel = NUTS(model, target_accept_prob=target_accept_prob, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=num_chains, chain_method="vectorized")
    mcmc.run(rng_key_, states_train, empirical_train_seq_flat)
    mcmc.print_summary()

    posterior_samples = mcmc.get_samples()
    posterior_predictive = Predictive(model, posterior_samples)(
        PRNGKey(1), states_train, None
    )
    prior = Predictive(model, num_samples=500)(
        PRNGKey(2), states_train, None
    )

    N = states_train.shape[0]
    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords={"item": np.arange(N)},
        dims={"obs": ["item"]},
    )
    attach_run_metadata(
        numpyro_data,
        run_metadata(
            dataset="production",
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
    numpyro_data.to_netcdf(output_file_name)
    print(f"Saved: {output_file_name}")


def run_inference_hier(
    speaker_type: str = "global",
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 4,
    min_proportion: float = 0.0,
    condition_subset: str = "",
    state_encoding: str = "target_match",
    artifact_tag: str = "",
):
    """Run MCMC for the hierarchical (random participant alpha) speaker model.

    condition_subset : str
        Comma-separated condition codes (e.g., "erdc,zrdc,brdc") to filter trials.
        Empty = use all 9 conditions. Adds suffix "_<tag>" to output filename.
    """
    canonical_speaker_type = canonicalize_speaker_type(speaker_type)

    subset_tag = ""
    if condition_subset:
        subset_codes = tuple(s.strip() for s in condition_subset.split(",") if s.strip())
        # Compact tag: hash-free, derive from condition stems (third+fourth chars).
        stems = sorted({c[2:4] for c in subset_codes})
        subset_tag = f"_{''.join(stems)}"
    else:
        subset_codes = None

    tag = f"_top" if min_proportion > 0 else ""
    run_tag = artifact_tag_suffix(artifact_tag)
    output_file_name = (
        f"./inference_data/mcmc_results_{canonical_speaker_type}_speaker_hier{tag}{subset_tag}"
        f"{run_tag}_warmup{num_warmup}_samples{num_samples}_chains{num_chains}.nc"
    )
    if os.path.exists(output_file_name):
        os.remove(output_file_name)
        print(f"Deleted existing file: {output_file_name}")

    data = import_dataset_hier(
        min_proportion=min_proportion,
        state_encoding=state_encoding,
    )
    print(f"  [encoding] state_encoding={state_encoding}")

    if subset_codes is not None:
        df = data["df"]
        keep_mask_np = df["conditions"].isin(subset_codes).to_numpy()
        n_before = len(df)
        n_after = int(keep_mask_np.sum())
        if n_after == 0:
            raise ValueError(f"condition_subset {subset_codes} matched zero trials.")
        keep_idx = np.where(keep_mask_np)[0]
        keep_idx_jnp = jax.numpy.asarray(keep_idx)

        # Filter every per-trial array; participant_idx must stay 0-indexed dense.
        for k in ("states_train", "empirical_seq_flat", "empirical_flat",
                  "empirical_seq", "seq_mask", "sharpness_idx",
                  "is_colour_sufficient", "sufficient_dim",
                  "has_one_word_solution"):
            if k in data and data[k] is not None:
                data[k] = data[k][keep_idx_jnp]
        data["df"] = df.iloc[keep_idx].reset_index(drop=True)

        # Re-index participants to 0..n-1 over the filtered trials.
        old_pid = np.asarray(data["participant_idx"])[keep_idx]
        unique_p = sorted(set(old_pid.tolist()))
        remap = {p: i for i, p in enumerate(unique_p)}
        new_pid = np.array([remap[p] for p in old_pid], dtype=np.int32)
        data["participant_idx"] = jax.numpy.asarray(new_pid, dtype=jax.numpy.int32)
        data["n_participants"] = len(unique_p)

        # Re-index conditions to 0..n-1 over the surviving subset conditions.
        if "condition_idx" in data and data["condition_idx"] is not None:
            old_cid = np.asarray(data["condition_idx"])[keep_idx]
            unique_c = sorted(set(old_cid.tolist()))
            cremap = {c: i for i, c in enumerate(unique_c)}
            new_cid = np.array([cremap[c] for c in old_cid], dtype=np.int32)
            data["condition_idx"] = jax.numpy.asarray(new_cid, dtype=jax.numpy.int32)
            data["n_conditions"] = len(unique_c)
            # Preserve labels in the same 0..n-1 order
            old_labels = data.get("condition_labels", [])
            if old_labels:
                data["condition_labels"] = [old_labels[c] for c in unique_c]

        print(f"  [subset] Kept {n_after}/{n_before} trials in conditions {subset_codes} "
              f"({len(unique_p)} participants)")

    states_train       = data["states_train"]
    empirical_seq_flat = data["empirical_seq_flat"]
    participant_idx    = data["participant_idx"]
    n_participants     = data["n_participants"]
    condition_idx      = data.get("condition_idx")
    n_conditions       = data.get("n_conditions")
    is_colour_sufficient = data.get("is_colour_sufficient")  # only present for v5 family
    is_sharp             = data.get("sharpness_idx")         # 1 if sharp, 0 if blurred
    sufficient_dim        = data.get("sufficient_dim")
    has_one_word_solution = data.get("has_one_word_solution")

    V5_FAMILY = {"v5", "v5_no_lm", "v5a", "v5b",
                 "v5_inc_static", "v5_global", "v5_global_static",
                 "v5_global_full", "v5_global_static_full"}
    CONTEXTUAL_FAMILY = {
        "contextual",
        "contextual_lambdaunc",
        "simplified_lm_resid",
        "simplified_lm_raw",
        "simplified_hand_order",
        "simplified_no_frontload",
        "simplified_no_uncertainty_len",
        "simplified_no_order",
        "principled",
        "principled_no_order",
        "principled_no_salience",
        "principled_no_uncertainty_len",
        "principled_salience_stop",
        "principled_salience_stop_regularized",
        "principled_salience_stop_regularized_2x2_inc_rec",
        "principled_salience_stop_regularized_2x2_inc_static",
        "principled_salience_stop_regularized_2x2_glob_rec",
        "principled_salience_stop_regularized_2x2_glob_static",
        "principled_salience_stop_regularized_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_2x2_glob_static_fixedeps",
        "principled_salience_stop_regularized_tmcc_2x2_inc_rec",
        "principled_salience_stop_regularized_tmcc_2x2_inc_static",
        "principled_salience_stop_regularized_plannedprefix_2x2_inc_rec",
        "principled_salience_stop_regularized_plannedprefix_2x2_inc_static",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_static",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_rec",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_static",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_fixedeps",
        "principled_salience_stop_regularized_tmcc_2x2_glob_rec",
        "principled_salience_stop_regularized_tmcc_2x2_glob_static",
        "principled_salience_stop_regularized_tmcc_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_tmcc_2x2_glob_static_fixedeps",
        "principled_salience_stop_strong_regularized",
        "contextual_freewf",
        "contextual_anchored",
        "contextual_freewf_anchored",
        "contextual_anchored_gamma",
        "contextual_anchored_gamma_fixedwf",
        "contextual_anchored_gamma_fixedwf_pcalpha",
        "contextual_pcalpha_gammasharp",
        "contextual_pcalpha_formmod",
        "contextual_pcalpha_canon",
        "contextual_pcalpha_canon_betafixed",
        "contextual_pcalpha_canon_parsimony",
        "contextual_pcalpha_canon_parsimony_no_gammasharp",
        "contextual_pcalpha_canon_parsimony_no_lambdasuff",
        "contextual_pcalpha_canon_parsimony_no_alphaF",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freecsv",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freefsv",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freek",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freewf",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freeall4",
        "contextual_pcalpha_canon_parsimony_no_alphaF_csv059",
        "contextual_pcalpha_canon_parsimony_2x2_inc_rec",
        "contextual_pcalpha_canon_parsimony_2x2_inc_static",
        "contextual_pcalpha_canon_parsimony_2x2_glob_rec",
        "contextual_pcalpha_canon_parsimony_2x2_glob_static",
    }
    # Models that take an additional (participant × condition) random effect
    # on alpha — need condition_idx and n_conditions passed through.
    PCALPHA_FAMILY = {
        "contextual_anchored_gamma_fixedwf_pcalpha",
        "contextual_pcalpha_gammasharp",
        "contextual_pcalpha_formmod",
        "contextual_pcalpha_canon",
        "contextual_pcalpha_canon_betafixed",
        "contextual_pcalpha_canon_parsimony",
        "contextual_pcalpha_canon_parsimony_no_gammasharp",
        "contextual_pcalpha_canon_parsimony_no_lambdasuff",
        "contextual_pcalpha_canon_parsimony_no_alphaF",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freecsv",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freefsv",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freek",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freewf",
        "contextual_pcalpha_canon_parsimony_no_alphaF_freeall4",
        "contextual_pcalpha_canon_parsimony_no_alphaF_csv059",
        "contextual_pcalpha_canon_parsimony_2x2_inc_rec",
        "contextual_pcalpha_canon_parsimony_2x2_inc_static",
        "contextual_pcalpha_canon_parsimony_2x2_glob_rec",
        "contextual_pcalpha_canon_parsimony_2x2_glob_static",
    }
    is_v5 = canonical_speaker_type in V5_FAMILY
    is_contextual = canonical_speaker_type in CONTEXTUAL_FAMILY
    RESPONSE_POLICY_FAMILY = {
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_static",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_rec",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_static",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_2x2_glob_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sharpform_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_sizesharp_2x2_glob_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_reliabilitybackup_2x2_inc_static_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_fixedeps",
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_static_fixedeps",
    }
    is_response_policy = canonical_speaker_type in RESPONSE_POLICY_FAMILY

    print(f"Hierarchical model: {n_participants} participants, {len(states_train)} observations")
    print(f"Output file: {output_file_name}")

    rng_key = random.PRNGKey(4711)
    rng_key, rng_key_ = random.split(rng_key)

    if canonical_speaker_type not in HIER_MODELS:
        raise ValueError(
            f"Unknown speaker_type '{canonical_speaker_type}'. "
            f"Choose from: {list(HIER_MODELS.keys())}"
        )
    model, target_accept_prob, max_tree_depth = HIER_MODELS[canonical_speaker_type]

    # dense_mass for the (alpha_D, alpha_C, alpha_F, log_beta_lm) ridge was
    # tried (commits f9e5afe, 75d2b12) but produced WORSE diagnostics than
    # diagonal mass: chains adapted to different basins during warmup and
    # could not mix between them. Diagonal mass is the stable default here.
    kernel = NUTS(model, target_accept_prob=target_accept_prob,
                  max_tree_depth=max_tree_depth)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized",
    )

    base_kwargs = dict(
        states=states_train,
        empirical=empirical_seq_flat,
        participant_idx=participant_idx,
        n_participants=n_participants,
    )
    if is_v5:
        if is_colour_sufficient is None:
            raise RuntimeError("v5 family requires is_colour_sufficient in data dict; rebuild dataset")
        if is_sharp is None:
            raise RuntimeError("v5 family requires sharpness_idx in data dict; rebuild dataset")
        base_kwargs["is_colour_sufficient"] = is_colour_sufficient
        base_kwargs["is_sharp"]             = is_sharp
    if is_contextual:
        if sufficient_dim is None or has_one_word_solution is None or is_sharp is None:
            raise RuntimeError("contextual model requires sufficient_dim, has_one_word_solution, and sharpness_idx")
        base_kwargs["sufficient_dim"] = sufficient_dim
        base_kwargs["has_one_word_solution"] = has_one_word_solution
        base_kwargs["is_sharp"] = is_sharp
    if is_response_policy:
        if is_colour_sufficient is None:
            raise RuntimeError(
                "response-policy model requires is_colour_sufficient in data dict"
            )
        base_kwargs["is_colour_sufficient"] = is_colour_sufficient
    if canonical_speaker_type in PCALPHA_FAMILY:
        if condition_idx is None or n_conditions is None:
            raise RuntimeError(
                "pcalpha variant requires condition_idx and n_conditions in data dict"
            )
        base_kwargs["condition_idx"] = condition_idx
        base_kwargs["n_conditions"] = n_conditions

    mcmc.run(rng_key_, **base_kwargs)
    mcmc.print_summary(exclude_deterministic=False)

    posterior_samples = mcmc.get_samples()
    pp_kwargs = dict(base_kwargs)
    pp_kwargs["empirical"] = None  # generate, not condition
    posterior_predictive = Predictive(model, posterior_samples)(PRNGKey(1), **pp_kwargs)
    prior = Predictive(model, num_samples=500)(PRNGKey(2), **pp_kwargs)

    N = states_train.shape[0]
    coords = {"item": np.arange(N)}
    dims   = {"obs": ["item"]}
    if "delta" in posterior_samples:
        delta_shape = posterior_samples["delta"].shape  # (chains*samples, ...)
        if len(delta_shape) == 2:
            coords["participants"] = np.arange(n_participants)
            dims["delta"] = ["participants"]
        elif len(delta_shape) == 3:
            coords["participants"] = np.arange(n_participants)
            coords["conditions"] = (data.get("condition_labels")
                                    or np.arange(n_conditions or delta_shape[2]).tolist())
            dims["delta"] = ["participants", "conditions"]
    # Non-centered "delta_raw" gets the same shape coords if present.
    if "delta_raw" in posterior_samples:
        draw_shape = posterior_samples["delta_raw"].shape
        if len(draw_shape) == 2:
            coords["participants"] = coords.get("participants", np.arange(n_participants))
            dims["delta_raw"] = ["participants"]
        elif len(draw_shape) == 3:
            coords["participants"] = coords.get("participants", np.arange(n_participants))
            coords["conditions"] = coords.get(
                "conditions",
                data.get("condition_labels") or np.arange(n_conditions or draw_shape[2]).tolist(),
            )
            dims["delta_raw"] = ["participants", "conditions"]

    numpyro_data = az.from_numpyro(
        mcmc,
        prior=prior,
        posterior_predictive=posterior_predictive,
        coords=coords,
        dims=dims,
    )
    attach_run_metadata(
        numpyro_data,
        run_metadata(
            dataset="production",
            run_kind="hierarchical",
            speaker_type=speaker_type,
            canonical_speaker_type=canonical_speaker_type,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            artifact_tag=artifact_tag,
            artifact_file=os.path.basename(output_file_name),
            state_encoding=state_encoding,
            condition_subset=condition_subset,
            subset_tag=subset_tag,
            min_proportion=min_proportion,
            n_observations=N,
            n_participants=n_participants,
            n_conditions=n_conditions,
        ),
    )
    numpyro_data.to_netcdf(output_file_name)
    assert os.path.exists(output_file_name), f"Save failed: {output_file_name} not found"
    print(f"Saved: {output_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run speaker inference with NumPyro.")
    parser.add_argument("--speaker_type", type=str,
                        choices=SPEAKER_CHOICES,
                        default="incremental",
                        help="Choose the speaker model type.")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of posterior samples.")
    parser.add_argument("--num_warmup", type=int, default=500, help="Number of warm-up iterations.")
    parser.add_argument("--num_chains", type=int, default=4, help="Number of MCMC chains.")
    parser.add_argument("--test", action="store_true", help="Run test function and exit.")
    parser.add_argument(
        "--hierarchical", action="store_true",
        help="Run hierarchical model with random per-participant alpha intercepts.",
    )
    parser.add_argument(
        "--min-proportion", type=float, default=0.0,
        help="Filter training data to utterance types with max proportion >= this in any condition.",
    )
    parser.add_argument(
        "--condition-subset", type=str, default="",
        help="Comma-separated condition codes (e.g. 'erdc,zrdc,brdc') to filter trials. Empty = all 9.",
    )
    parser.add_argument(
        "--state-encoding", type=str, default="target_match",
        choices=["target_match", "canonical"],
        help="State encoding for colour/form features. Default target_match.",
    )
    parser.add_argument(
        "--artifact-tag", type=str, default="",
        help=(
            "Optional safe token appended to inference artifact filenames "
            "before the warmup/sample/chains tag."
        ),
    )

    args = parser.parse_args()

    if args.test:
        from modelSpecification import test
        test()
    elif args.hierarchical:
        run_inference_hier(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            min_proportion=args.min_proportion,
            condition_subset=args.condition_subset,
            state_encoding=args.state_encoding,
            artifact_tag=args.artifact_tag,
        )
    else:
        run_inference(
            speaker_type=args.speaker_type,
            num_samples=args.num_samples,
            num_warmup=args.num_warmup,
            num_chains=args.num_chains,
            artifact_tag=args.artifact_tag,
        )
