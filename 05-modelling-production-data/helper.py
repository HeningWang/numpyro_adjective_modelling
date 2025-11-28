"""Utility helpers for the production modelling workflow."""

from pathlib import Path
from typing import Dict, Optional, Union

import jax.numpy as jnp
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = REPO_ROOT / "01-dataset" / "01-production-data-preprocessed.csv"
CONDITIONS_OF_INTEREST = ("erdc", "zrdc", "brdc")
SYMBOL_TO_INDEX: Dict[str, int] = {"D": 0, "C": 1, "F": 2}
MAX_UTTERANCE_LEN = 3
FLAT_TO_CATEGORIES: Dict[str, str] = {
    "0": "D",
    "1": "DC",
    "2": "DCF",
    "3": "DF",
    "4": "DFC",
    "5": "C",
    "6": "CD",
    "7": "CDF",
    "8": "CF",
    "9": "CFD",
    "10": "F",
    "11": "FD",
    "12": "FDC",
    "13": "FC",
    "14": "FCD",
}


def normalize(arr: jnp.ndarray, axis: int = 1) -> jnp.ndarray:
    """Normalize an array along the provided axis."""
    return arr / jnp.sum(arr, axis=axis, keepdims=True)


def import_dataset(
    file_path: Optional[Union[str, Path]] = None
):
    """
    Load and preprocess the production dataset, returning encodings used by the
    modelling pipeline.
    """
    dataset_path = Path(file_path) if file_path is not None else DEFAULT_DATASET_PATH

    df = pd.read_csv(dataset_path).dropna(subset=["annotation"]).copy()
    df = df[df["conditions"].isin(CONDITIONS_OF_INTEREST)].copy()

    # Encode states via vectorised slicing.
    sizes = df.iloc[:, 6:12].to_numpy(dtype=float)  # (N, 6)
    colors = (df.iloc[:, 12:18] == "blue").to_numpy(dtype=int)
    forms = (df.iloc[:, 18:24] == "circle").to_numpy(dtype=int)
    states_np = np.stack([sizes, colors, forms], axis=2)  # (N, 6, 3)
    states_train = jnp.array(states_np, dtype=jnp.float32)

    # Encode utterances.
    df["annotation"] = df["annotation"].astype("category")
    empirical_flat = jnp.array(
        df["annotation"].cat.codes.to_numpy(), dtype=jnp.int32
    )
    utt_strings = df["annotation"].astype(str).tolist()

    # Sequences padded with -1 (True sequence positions replaced below).
    sequences = np.full((len(utt_strings), MAX_UTTERANCE_LEN), -1, dtype=np.int32)
    for row_idx, utt in enumerate(utt_strings):
        for col_idx, symbol in enumerate(utt):
            sequences[row_idx, col_idx] = SYMBOL_TO_INDEX[symbol]
    unique_sequences = np.unique(sequences, axis=0)
    empirical_seq = jnp.array(sequences, dtype=jnp.int32)
    unique_utterances = jnp.array(unique_sequences, dtype=jnp.int32)

    seq_lengths = np.array([len(utt) for utt in utt_strings], dtype=np.int32)
    seq_mask = jnp.array(
        (np.arange(MAX_UTTERANCE_LEN)[None, :] < seq_lengths[:, None]), dtype=bool
    )

    utt_seq_to_idx = {
        tuple(seq): idx for idx, seq in enumerate(unique_utterances.tolist())
    }
    empirical_seq_flat = jnp.array(
        [utt_seq_to_idx[tuple(seq)] for seq in empirical_seq.tolist()],
        dtype=jnp.int32,
    )

    df["statesArray"] = states_train.tolist()
    df["annotation_string_flat"] = empirical_flat.tolist()
    df["annotation_seq"] = empirical_seq.tolist()
    df["annotation_seq_mask"] = seq_mask.tolist()
    df["annotation_seq_flat"] = empirical_seq_flat.tolist()

    empirical_dist_by_condition = df.groupby(["item", "list"])["annotation_seq_flat"].value_counts(normalize=True).unstack(fill_value=0)
    empirical_dist_by_condition = jnp.array(empirical_dist_by_condition.values, dtype=jnp.float32)

    return {
        "states_train": states_train,
        "empirical_flat": empirical_flat,
        "empirical_seq": empirical_seq,
        "seq_mask": seq_mask,
        "df": df,
        "unique_utterances": unique_utterances,
        "empirical_seq_flat": empirical_seq_flat,
        "empirical_dist_by_condition": empirical_dist_by_condition
    }

def build_utterance_prior_jax(
    utterance_list: jnp.ndarray,  # shape (U, 3)
    costParam_length: float = 1.0,
    costParam_bias: float = 1.0,
    costParam_subjectivity: float = 1.0
) -> jnp.ndarray:
    """
    Build a prior over utterances using JAX-friendly vectorised arithmetic.
    Assumes utterance_list is a (U, 3) array where -1 is padding.
    """

    penalized = jnp.array(
        [
            [1, 0, -1],  # "CD"
            [2, 1, -1],  # "FD"
            [1, 2, 0],   # "CFD"
            [2, 1, 0],   # "FCD"
            [2, 0, 1],   # "FDC"
        ],
        dtype=jnp.int32,
    )

    utterances = jnp.asarray(utterance_list, dtype=jnp.int32)
    valid_mask = utterances >= 0
    lengths = jnp.sum(valid_mask, axis=1)

    base_util = jnp.where(
        lengths == 1,
        3.0,
        jnp.where(lengths == 2, 2.0, 1.0),
    )

    # Match penalized sequences (broadcast compare across utterances).
    matches = jnp.all(
        utterances[:, None, :] == penalized[None, :, :],
        axis=2,
    )
    penalty = jnp.where(jnp.any(matches, axis=1), -3.0, 0.0)

    boost = jnp.where(utterances[:, 0] == 0, 2.0, 1.0)

    utils = (
        costParam_length * base_util
        + costParam_bias * penalty
        + costParam_subjectivity * boost
    )

    utils = utils - jnp.max(utils)
    probs = jnp.exp(utils)
    probs = probs / jnp.sum(probs)

    return probs.astype(jnp.float32)

# ========================
# Global Variables (Setup)
# ========================
utterance_list = import_dataset()["unique_utterances"]  # shape (U,3)
utterance_prior = build_utterance_prior_jax(utterance_list)

# ========================
def plot_utterance_distribution(distribution_array, titel = "Example Empirical Distribution"):
    df_plot_example_empirical_dist = pd.DataFrame({
        "annotation": list(FLAT_TO_CATEGORIES.values()),
        "probability": distribution_array
    })
    sns.catplot(
        data=df_plot_example_empirical_dist,
        kind="bar",
        x="annotation",
        y="probability",
        height=4,
        aspect=1.5,
        sharey=True,
        hue="annotation",
        palette="Set2"
    )
    plt.title(titel)
    plt.xlabel("Annotation Category")
    plt.ylabel("Probability")
    plt.show()

# ========================
def theme_aida(
    title_size=16,
    text_size=14,
    legend_position="top",
    show_axis=False,      # False, "x", or "y"
    show_grid=True,
    plot_margin=(0.2, 0.1, 0.2, 0.1)  # top, right, bottom, left in inches
):
    """
    Seaborn/Matplotlib version of the AIDA ggplot2 theme.
    """

    # --- Base style: equivalent to theme_classic() ---
    sns.set_theme(style="white")   # white background, classic layout
    sns.set_context("notebook", font_scale=text_size/14)

    # Font sizes
    plt.rcParams.update({
        "axes.titlesize": title_size,
        "axes.titleweight": "bold",
        "axes.labelsize": text_size,
        "axes.linewidth": 0.8,
        "xtick.labelsize": text_size - 2,
        "ytick.labelsize": text_size - 2,
        "legend.fontsize": text_size - 2,
        "legend.title_fontsize": text_size,
        "figure.titlesize": title_size,
        "figure.titleweight": "bold",
    })

    # --- Grid lines (dotted) ---
    if show_grid:
        plt.rcParams["axes.grid"] = True
        plt.rcParams["grid.linestyle"] = ":"
        plt.rcParams["grid.linewidth"] = 0.6
        plt.rcParams["grid.color"] = "#333333"
    else:
        plt.rcParams["axes.grid"] = False

    # --- Axis visibility ---
    if show_axis is False:
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["axes.spines.left"] = False
        plt.rcParams["axes.spines.bottom"] = False
        plt.rcParams["xtick.bottom"] = False
        plt.rcParams["ytick.left"] = False

    elif show_axis == "x":
        plt.rcParams["axes.spines.left"] = False
        plt.rcParams["axes.spines.right"] = False
        plt.rcParams["ytick.left"] = False

    elif show_axis == "y":
        plt.rcParams["axes.spines.top"] = False
        plt.rcParams["axes.spines.bottom"] = False
        plt.rcParams["xtick.bottom"] = False

    # --- Legend positions ---
    if legend_position == "top":
        loc = "upper center"
        bbox = (0.5, 1.15)
    elif legend_position == "bottom":
        loc = "lower center"
        bbox = (0.5, -0.15)
    elif legend_position == "left":
        loc = "center left"
        bbox = (1.02, 0.5)
    elif legend_position == "right":
        loc = "center right"
        bbox = (1.02, 0.5)
    else:  # "none"
        loc = None
        bbox = None

    # Save legend config so users can apply per-plot
    theme_config = {
        "legend_loc": loc,
        "legend_bbox": bbox,
    }

    # --- Margins ---
    t, r, b, l = plot_margin
    plt.gcf().subplots_adjust(
        top=1 - t/2,
        bottom=b/2,
        left=l/2,
        right=1 - r/2,
    )

    return theme_config
