"""Utility helpers for the production modelling workflow."""

from pathlib import Path
from typing import Dict, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns

def transformation_data(slider_value, link = None):
    if link == "identity":
      slider_value = jnp.clip(slider_value, 0, 100)
      transformed_prob = slider_value / 100
    elif link == "logit":
        transformed_prob = 1 / (1 + math.exp(-slider_value))
    return transformed_prob

def link_function(x, param = 1):
    return 1 / (1 + jnp.exp(param * -(x - 0.5)))

def compute_alpha_beta_concentration(mu, v):
    alpha = mu * v
    beta = (1 - mu) * v
    return alpha, beta


def normalize(arr: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """Normalize an array along the provided axis."""
    return arr / jnp.sum(arr, axis=axis, keepdims=True)

def plot_dist(d, ax=None):
    support = d.enumerate_support()
    data = [d.log_prob(s).exp().item() for s in d.enumerate_support()]
    names = list(map(str, support))

    if ax is None:
        ax = plt.subplot(111)

    width = 0.3
    bins = [x-width/2 for x in range(1, len(data) + 1)]
    ax.bar(bins,data,width=width)
    ax.set_xticks(list(range(1, len(data) + 1)))
    ax.set_xticklabels(names, rotation=45, rotation_mode="anchor", ha="right")

def get_results(posterior):
    results = {}
    support = posterior.enumerate_support()
    data = [posterior.log_prob(s).exp().item() for s in posterior.enumerate_support()]
    results["support"] = support
    results["probs"] = data
    return results