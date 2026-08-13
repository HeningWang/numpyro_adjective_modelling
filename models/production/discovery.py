"""Pure JAX kernels for theory-isolated production-model candidates."""

import jax
import jax.numpy as jnp


DISCOVERY_CANDIDATE_IDS = (
    "discovery_normpath_kappa1_inc_static_fixedeps",
    "discovery_normpath_kappa0_matched_global_static_fixedeps",
    "discovery_normpath_kappa_free_static_fixedeps",
    "discovery_form_reliability_static_fixedeps",
    "discovery_participant_kappa_static_fixedeps",
)


def final_uniform_lapse(
    probabilities: jnp.ndarray,
    epsilon: float,
) -> jnp.ndarray:
    """Mix a normalized response distribution with a uniform response lapse."""
    probabilities = jnp.asarray(probabilities)
    support_size = probabilities.shape[-1]
    return (1.0 - epsilon) * probabilities + epsilon / support_size


def normalization_path_distribution(
    raw_path_score: jnp.ndarray,
    local_log_normalizer: jnp.ndarray,
    terminal_policy: jnp.ndarray,
    kappa: float,
    epsilon: float,
) -> jnp.ndarray:
    """Normalize matched path utilities and apply one final uniform lapse.

    ``kappa=1`` retains the accumulated local-normalization correction of an
    incremental speaker. ``kappa=0`` globally normalizes the same raw path
    utilities and terminal policy.
    """
    logits = (
        jnp.asarray(raw_path_score)
        - kappa * jnp.asarray(local_log_normalizer)
        + jnp.asarray(terminal_policy)
    )
    return final_uniform_lapse(jax.nn.softmax(logits, axis=-1), epsilon)


def legacy_vs_final_lapse_max_probability_shift(
    raw_path_score: jnp.ndarray,
    local_log_normalizer: jnp.ndarray,
    base_terminal_policy: jnp.ndarray,
    response_terminal_policy: jnp.ndarray,
    kappa: float,
    epsilon: float,
) -> jnp.ndarray:
    """Return the largest response shift caused solely by lapse placement."""
    legacy_base = normalization_path_distribution(
        raw_path_score,
        local_log_normalizer,
        base_terminal_policy,
        kappa,
        epsilon,
    )
    probability_floor = jnp.finfo(legacy_base.dtype).tiny
    legacy_final = jax.nn.softmax(
        jnp.log(jnp.clip(legacy_base, probability_floor))
        + jnp.asarray(response_terminal_policy),
        axis=-1,
    )
    corrected_final = normalization_path_distribution(
        raw_path_score,
        local_log_normalizer,
        (
            jnp.asarray(base_terminal_policy)
            + jnp.asarray(response_terminal_policy)
        ),
        kappa,
        epsilon,
    )
    return jnp.max(jnp.abs(legacy_final - corrected_final), axis=-1)


def noncentered_participant_kappa(
    mean_kappa: float,
    tau_kappa: float,
    z_kappa: jnp.ndarray,
) -> jnp.ndarray:
    """Map noncentered participant deviations to the unit interval."""
    z_kappa = jnp.asarray(z_kappa)
    mean_kappa = jnp.asarray(mean_kappa, dtype=z_kappa.dtype)
    tau_kappa = jnp.asarray(tau_kappa, dtype=z_kappa.dtype)
    finfo = jnp.finfo(z_kappa.dtype)
    clipped_mean = jnp.clip(mean_kappa, finfo.eps, 1.0 - finfo.eps)
    mean_logit = jnp.log(clipped_mean) - jnp.log1p(-clipped_mean)
    varied = jax.nn.sigmoid(mean_logit + tau_kappa * z_kappa)
    recovered = jnp.broadcast_to(mean_kappa, z_kappa.shape)
    return jnp.where(tau_kappa == 0.0, recovered, varied)


def noncentered_participant_alpha(
    population_alpha: float,
    tau_log_alpha: float,
    z_alpha: jnp.ndarray,
) -> jnp.ndarray:
    """Map Gaussian participant effects to a smooth positive alpha scale.

    ``population_alpha`` is the participant-level median.  The zero-scale
    branch recovers it exactly, which keeps deterministic parent-nesting
    checks independent of floating-point exponential roundoff.
    """

    z_alpha = jnp.asarray(z_alpha)
    population_alpha = jnp.asarray(
        population_alpha,
        dtype=z_alpha.dtype,
    )
    tau_log_alpha = jnp.asarray(tau_log_alpha, dtype=z_alpha.dtype)
    varied = population_alpha * jnp.exp(tau_log_alpha * z_alpha)
    recovered = jnp.broadcast_to(population_alpha, z_alpha.shape)
    return jnp.where(tau_log_alpha == 0.0, recovered, varied)
