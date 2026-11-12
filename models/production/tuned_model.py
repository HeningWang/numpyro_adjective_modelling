"""Theory-mapped variants of the six shared-kappa production models.

Every option of :class:`TuningSpec` changes one component of the primary
model so that the component is either tied to an account of adjective order
or removed.  The default spec reproduces the primary model exactly, so each
option can be evaluated as a matched contrast against it.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

import modelSpecification as ms
from discovery import (
    noncentered_participant_alpha,
    normalization_path_distribution,
)

SIZE_BLUR = 0.6856
LAPSE = 0.003
FORM_TERM_PRIOR_SCALE = 1.5
ORDER_SCALE_PRIOR = 0.5
SUPPORTED_MODELS = ("G-HO", "K-HO", "I-HO", "G-UPD-HO", "K-UPD-HO", "I-UPD-HO")
FIXED_KAPPA = {"G": 0.0, "I": 1.0}


@dataclass(frozen=True)
class TuningSpec:
    """Component choices; all defaults give the primary model."""

    # Remove the continuation cost and the two discriminability-gated terms.
    drop_dead_terms: bool = False
    # Size contrast in salience no longer depends on the discriminability group.
    unscaled_size_contrast: bool = False
    # Low discriminability widens the size blur by this factor (None: one blur).
    size_blur_multiplier: float | None = None
    # The form term applies in every context.
    unconditional_form_term: bool = False
    # Sufficiency is read off the display instead of the condition label.
    display_sufficiency: bool = False
    # Separate coefficients for property baseline and display contrast.
    split_salience: bool = False
    # Property baseline estimated (colour and form relative to size) instead of fixed.
    free_salience_base: bool = False
    # Theory switches.
    drop_order_term: bool = False
    drop_informativeness: bool = False
    drop_salience_base: bool = False
    drop_salience_contrast: bool = False
    drop_sufficiency: bool = False
    terminal_utility: bool = False

    def __post_init__(self) -> None:
        if self.display_sufficiency and not self.drop_dead_terms:
            raise ValueError("Display sufficiency requires drop_dead_terms: the "
                             "discriminability-gated terms use the condition label")
        if (self.drop_salience_base or self.drop_salience_contrast) and not self.split_salience:
            raise ValueError("Salience switches require split_salience")
        if self.free_salience_base and (not self.split_salience or self.drop_salience_base):
            raise ValueError("A free property baseline requires split_salience and the baseline")
        if self.size_blur_multiplier is not None and not (
                np.isfinite(self.size_blur_multiplier) and self.size_blur_multiplier >= 1.0):
            raise ValueError("The size-blur multiplier must be finite and at least 1")
        if self.terminal_utility and self.drop_informativeness:
            raise ValueError("Terminal utility is undefined without informativeness")


def display_sufficiency_mask(states: np.ndarray) -> np.ndarray:
    """Mark the properties that identify the target on their own.

    ``states`` has shape (trials, objects, 3) with the target in row 0 and
    columns size, colour match, form match.  Colour or form is sufficient
    when no other object shares the target's value.  Size is sufficient when
    the target is strictly larger than every other object.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 3 or states.shape[2] != 3:
        raise ValueError("Expected states of shape (trials, objects, 3)")
    size = states[:, 0, 0] > states[:, 1:, 0].max(axis=1)
    colour = ~(states[:, 1:, 1] == states[:, :1, 1]).any(axis=1)
    form = ~(states[:, 1:, 2] == states[:, :1, 2]).any(axis=1)
    return np.stack([size, colour, form], axis=1).astype(np.float64)


def precompute_tuned_features(
    states, is_sharp, *, recursive: bool, spec: TuningSpec,
    color_semval: float = 0.59, form_semval: float = 0.50,
) -> dict[str, jnp.ndarray]:
    """Fixed semantic, salience and sufficiency features for each trial."""
    scenes = np.asarray(states, dtype=np.float64)
    sharp = np.asarray(is_sharp, dtype=np.float64)
    if scenes.ndim != 3 or len(scenes) != len(sharp):
        raise ValueError("States and discriminability labels must align by trial")
    multiplier = 1.0 if spec.size_blur_multiplier is None else spec.size_blur_multiplier
    blur = SIZE_BLUR * (1.0 + (1.0 - sharp) * (multiplier - 1.0))
    keyed = np.concatenate([scenes.reshape(len(scenes), -1), blur[:, None]], axis=1)
    unique, inverse = np.unique(keyed, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    unique_scenes = jnp.asarray(unique[:, :-1].reshape(-1, *scenes.shape[1:]))
    log_target = jax.jit(jax.vmap(lambda scene, wf: ms.principled_prefix_log_listeners(
        scene, color_semval=color_semval, form_semval=form_semval, wf=wf,
        recursive=recursive)))(unique_scenes, jnp.asarray(unique[:, -1]))[inverse]
    contrast_sharp = np.ones_like(sharp) if spec.unscaled_size_contrast else sharp
    salience = jax.jit(jax.vmap(ms._visual_salience_scores))(
        jnp.asarray(scenes), jnp.asarray(contrast_sharp, dtype=jnp.float64))
    base = ms.BASE_VISUAL_SALIENCE - jnp.mean(ms.BASE_VISUAL_SALIENCE)
    salience_base = jnp.broadcast_to(base, salience.shape)
    return {
        "log_target": log_target,
        "salience": salience,
        "salience_base": salience_base,
        "salience_contrast": salience - salience_base,
        "stop_load": jax.vmap(ms._salience_continuation_load)(salience),
        "sufficiency_mask": jnp.asarray(display_sufficiency_mask(scenes)),
    }


def _display_policy(sufficiency_mask, form_gate, lambda_sufficient_single,
                    lambda_form, lambda_three_word_penalty):
    """Length and selection terms without condition labels for sufficiency."""
    single = (ms.N_WORDS == 1.0).astype(jnp.float64)
    named_property = jnp.argmax(ms.FULL_PRESENT_15, axis=1)
    sufficient_single = single * sufficiency_mask[named_property]
    return (lambda_sufficient_single * sufficient_single
            + lambda_form * form_gate * ms.F_PRESENT_15
            - lambda_three_word_penalty * (ms.N_WORDS == 3.0))


@partial(jax.jit, static_argnames=("terminal_utility", "display_sufficiency"))
def tuned_probabilities(
    features, salience_term, sufficient_dim, has_one_word_solution, is_sharp,
    form_context_gate, alpha, beta_order, kappa, rho_salience_stop,
    lambda_sufficient_single, lambda_form, lambda_three_word_penalty,
    lambda_size_single, lambda_size_form_pair, epsilon,
    terminal_utility=False, display_sufficiency=False,
):
    """Response distribution over the fifteen utterances for each trial."""
    path_components = (ms.principled_terminal_path_components_from_features
                       if terminal_utility else ms.principled_path_components_from_features)
    raw, normalizers, floor = jax.vmap(path_components, in_axes=(0, 0, 0, None))(
        features["log_target"], salience_term, alpha, 1.0)
    if display_sufficiency:
        policy = jax.vmap(_display_policy, in_axes=(0, 0, None, None, None))(
            features["sufficiency_mask"], 1.0 - form_context_gate,
            lambda_sufficient_single, lambda_form, lambda_three_word_penalty)
    else:
        policy = jax.vmap(ms.principled_reliability_response_logits,
                          in_axes=(0, 0, 0, 0, None, None, None, None, None))(
            sufficient_dim, has_one_word_solution, is_sharp, form_context_gate,
            lambda_sufficient_single, lambda_form, lambda_three_word_penalty,
            lambda_size_single, lambda_size_form_pair)
    terminal = (beta_order[:, None] * ms.LOG_LM_ORDER_ONLY_15
                - rho_salience_stop * features["stop_load"] + policy)
    return normalization_path_distribution(
        raw, normalizers - floor, terminal, kappa[:, None],
        jnp.asarray(epsilon, dtype=jnp.float64))


def _half_normal(name: str, scale: float, *, off: bool = False):
    return jnp.asarray(0.0) if off else numpyro.sample(name, dist.HalfNormal(scale))


def _free_salience_base(scale: float):
    """Centred baseline with estimated colour and form values; size is the reference."""
    colour = numpyro.sample("salience_base_colour", dist.Normal(0.0, scale))
    form = numpyro.sample("salience_base_form", dist.Normal(0.0, scale))
    base = jnp.stack([jnp.zeros_like(colour), colour, form])
    return base - jnp.mean(base)


def make_tuned_model(model_name: str, spec: TuningSpec):
    """Build one of the six shared-kappa cells under ``spec``."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model_name!r}; expected one of {SUPPORTED_MODELS}")
    architecture = model_name.split("-")[0]
    priors = ms.PRINCIPLED_PRIOR_PROFILES["regularized"]

    def model(states=None, empirical=None, participant_idx=None, n_participants=None,
              sufficient_dim=None, has_one_word_solution=None, is_sharp=None,
              is_colour_sufficient=None, precomputed_features=None):
        if precomputed_features is None:
            raise ValueError("Tuned models require precomputed features")
        n_trials = len(states)
        alpha = _half_normal("alpha", priors["alpha_scale"], off=spec.drop_informativeness)
        if not spec.drop_order_term:
            log_beta_order = numpyro.sample(
                "log_beta_order", dist.Normal(0.0, priors["log_beta_order_sd"]))
            tau_order = numpyro.sample("tau_order", dist.HalfNormal(ORDER_SCALE_PRIOR))
        salience_scale = priors["lambda_salience_scale"]
        if spec.split_salience:
            if spec.free_salience_base:
                base_term = jnp.broadcast_to(
                    _free_salience_base(salience_scale),
                    precomputed_features["salience_base"].shape)
            else:
                base_term = (_half_normal("lambda_salience_base", salience_scale,
                                          off=spec.drop_salience_base)
                             * precomputed_features["salience_base"])
            salience_term = (
                base_term
                + _half_normal("lambda_salience_contrast", salience_scale,
                               off=spec.drop_salience_contrast)
                * precomputed_features["salience_contrast"])
        else:
            salience_term = (_half_normal("lambda_salience", salience_scale)
                             * precomputed_features["salience"])
        rho = _half_normal("rho_salience_stop", priors["rho_salience_stop_scale"],
                           off=spec.drop_dead_terms)
        lambda_form = numpyro.sample("lambda_reliability_form",
                                     dist.HalfNormal(FORM_TERM_PRIOR_SCALE))
        lambda_sufficient = _half_normal(
            "lambda_sufficient_single", priors["lambda_sufficient_single_scale"],
            off=spec.drop_sufficiency)
        lambda_three = numpyro.sample(
            "lambda_three_word_penalty",
            dist.HalfNormal(priors["lambda_three_word_penalty_scale"]))
        lambda_size_single = _half_normal(
            "lambda_size_reliability_single_bonus",
            priors["lambda_size_reliability_single_bonus_scale"], off=spec.drop_dead_terms)
        lambda_size_form_pair = _half_normal(
            "lambda_size_reliability_form_pair_tradeoff",
            priors["lambda_size_reliability_form_pair_tradeoff_scale"],
            off=spec.drop_dead_terms)
        epsilon = numpyro.deterministic("epsilon", jnp.asarray(LAPSE))
        if architecture in FIXED_KAPPA:
            kappa = numpyro.deterministic("kappa", jnp.asarray(FIXED_KAPPA[architecture]))
        else:
            kappa = numpyro.sample("kappa", dist.Uniform(0.0, 1.0))
        if not spec.drop_informativeness:
            tau_log_alpha = numpyro.sample("tau_log_alpha", dist.HalfNormal(priors["tau_scale"]))
        with numpyro.plate("participants", n_participants):
            if not spec.drop_informativeness:
                z_alpha = numpyro.sample("z_alpha", dist.Normal(0.0, 1.0))
            if not spec.drop_order_term:
                z_order = numpyro.sample("z_order", dist.Normal(0.0, 1.0))
        if spec.drop_informativeness:
            alpha_per_trial = jnp.zeros(n_trials)
        else:
            alpha_by_participant = numpyro.deterministic(
                "alpha_by_participant",
                noncentered_participant_alpha(alpha, tau_log_alpha, z_alpha))
            alpha_per_trial = alpha_by_participant[participant_idx]
        if spec.drop_order_term:
            beta_order = jnp.zeros(n_trials)
        else:
            beta_by_participant = numpyro.deterministic(
                "beta_order_by_participant", jnp.exp(log_beta_order + tau_order * z_order))
            beta_order = beta_by_participant[participant_idx]
        form_context_gate = (jnp.zeros(n_trials) if spec.unconditional_form_term
                             else jnp.asarray(is_colour_sufficient, dtype=jnp.float64))
        with numpyro.plate("data", n_trials):
            probs = tuned_probabilities(
                precomputed_features, salience_term, sufficient_dim, has_one_word_solution,
                is_sharp, form_context_gate, alpha_per_trial, beta_order,
                jnp.broadcast_to(kappa, (n_trials,)), rho, lambda_sufficient, lambda_form,
                lambda_three, lambda_size_single, lambda_size_form_pair, epsilon,
                terminal_utility=spec.terminal_utility,
                display_sufficiency=spec.display_sufficiency)
            numpyro.sample("obs", dist.Categorical(probs=probs), obs=empirical)

    return model
