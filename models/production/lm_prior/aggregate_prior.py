"""Reconstruct the fixed order weights from archived token-surprisal scores.

This deterministic step requires no language-model download or inference.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from principled_features import LM_PRIOR_15, UTTERANCE_LABELS, order_only_lm_residual


def reconstruct(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores: dict[str, list[float]] = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            scores[row["order_key"]].append(float(row["surprisal_bits_per_token"]))
    if set(scores) != set(UTTERANCE_LABELS):
        raise ValueError("Expected the fifteen production response categories")
    means = np.array([np.mean(scores[label]) for label in UTTERANCE_LABELS])
    weights = np.exp2(-means)
    return means, weights / weights.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name(
        "LM_adjective_sequences_with_surprisal.csv"))
    parser.add_argument("--output", type=Path, help="Optional reconstruction table")
    args = parser.parse_args()
    means, weights = reconstruct(args.csv)
    # The fitted models retain the historical notebook's rounded float32 printout.
    np.testing.assert_allclose(weights, LM_PRIOR_15, rtol=0, atol=1e-8)
    residuals = order_only_lm_residual(lm_prior=LM_PRIOR_15)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["response", "mean_surprisal_bits_per_token",
                             "reconstructed_weight", "fitted_weight", "order_residual"])
            for label, mean, weight, fitted, residual in zip(
                    UTTERANCE_LABELS, means, weights, LM_PRIOR_15, residuals):
                writer.writerow([label.replace("D", "S"), mean, weight, fitted, residual])
    print(f"All 15 fixed weights reproduced within {np.max(np.abs(weights - LM_PRIOR_15)):.3g}.")


if __name__ == "__main__":
    main()
