"""Pure NumPy feature helpers for principled production-model variants."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np


UTTERANCE_LABELS = [
    "D", "DC", "DCF", "DF", "DFC",
    "C", "CD", "CDF", "CF", "CFD",
    "F", "FD", "FDC", "FC", "FCD",
]

LM_PRIOR_15 = np.array([
    0.1669514, 0.16733019, 0.12160929, 0.11005973, 0.09253279,
    0.07532827, 0.02494562, 0.03780574, 0.05690099, 0.02470998,
    0.02651604, 0.01232579, 0.03122547, 0.0363892, 0.01536951,
], dtype=np.float64)

DEFAULT_BASE_SALIENCE = np.array([0.0, 1.0, 0.25], dtype=np.float32)


def order_only_lm_residual(
    labels: Sequence[str] = UTTERANCE_LABELS,
    lm_prior: np.ndarray = LM_PRIOR_15,
) -> np.ndarray:
    """Return LM log-probability residuals within unordered content sets.

    This preserves the LM's order preference among utterances with identical
    adjective content, while removing content and length preferences.
    """
    labels = list(labels)
    log_prior = np.log(np.clip(np.asarray(lm_prior, dtype=np.float64), 1e-12, None))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, label in enumerate(labels):
        groups["".join(sorted(label))].append(i)

    scores = np.zeros(len(labels), dtype=np.float64)
    for idxs in groups.values():
        scores[idxs] = log_prior[idxs] - float(np.mean(log_prior[idxs]))
    return scores


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _size_membership(states: np.ndarray, wf: float = 0.6856, k: float = 0.5) -> np.ndarray:
    sizes = np.asarray(states, dtype=np.float32)[:, 0]
    prior = np.ones(len(sizes), dtype=np.float32) / len(sizes)
    order = np.argsort(sizes)
    sizes_sorted = sizes[order]
    cdf = np.cumsum(prior[order])
    x_min_mid = sizes_sorted[np.argmax(cdf >= 0.2)]
    x_max_mid = sizes_sorted[np.argmax(cdf >= 0.8)]
    theta = x_max_mid - k * (x_max_mid - x_min_mid)
    denom = wf * np.sqrt(sizes ** 2 + theta ** 2 + 1e-8)
    return _normal_cdf((sizes - theta) / denom).astype(np.float32)


def _entropy(probs: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=np.float32)
    probs = probs / np.clip(probs.sum(), 1e-12, None)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def size_uncertainty_excess(
    states: np.ndarray,
    is_sharp: float,
    wf: float = 0.6856,
    k: float = 0.5,
    blur_wf_multiplier: float = 2.0,
) -> float:
    """Entropy increase in size interpretation caused by perceptual degradation."""
    sharp_membership = _size_membership(states, wf=wf, k=k)
    wf_eff = wf * (1.0 + (1.0 - float(is_sharp)) * (blur_wf_multiplier - 1.0))
    effective_membership = _size_membership(states, wf=wf_eff, k=k)

    sharp_post = sharp_membership / np.clip(sharp_membership.sum(), 1e-12, None)
    effective_post = effective_membership / np.clip(effective_membership.sum(), 1e-12, None)
    return max(_entropy(effective_post) - _entropy(sharp_post), 0.0)


def visual_salience_features(
    states: np.ndarray,
    is_sharp: float,
    base_salience: Iterable[float] = DEFAULT_BASE_SALIENCE,
) -> np.ndarray:
    """Soft visual salience for D/C/F, centered because only differences matter."""
    states = np.asarray(states, dtype=np.float32)
    sizes = states[:, 0]
    colors = states[:, 1]
    forms = states[:, 2]

    size_scale = float(np.std(sizes) + 1e-8)
    size_margin = (float(sizes[0]) - float(np.max(sizes[1:]))) / size_scale
    size_contrast = 1.0 / (1.0 + math.exp(-size_margin))
    size_contrast *= 0.5 + 0.5 * float(is_sharp)

    color_contrast = 1.0 - float(np.mean(colors[1:] == colors[0]))
    form_contrast = 1.0 - float(np.mean(forms[1:] == forms[0]))

    raw = np.asarray(base_salience, dtype=np.float32) + np.array(
        [size_contrast, color_contrast, form_contrast],
        dtype=np.float32,
    )
    return raw - float(raw.mean())


ORDER_ONLY_LM_RESID_15 = order_only_lm_residual()
