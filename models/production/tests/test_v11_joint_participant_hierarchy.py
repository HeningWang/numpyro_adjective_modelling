"""Contract tests for the joint alpha/kappa/stable-order hierarchy."""

from pathlib import Path
import sys

import numpy as np
import numpyro.handlers as handlers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper import import_dataset_hier
import modelSpecification as ms


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "analysis_revision/12_deterministic_encoding/model_input_raw_observed_9100.csv"


def _arguments(n_rows=16):
    data = import_dataset_hier(file_path=DATA, state_encoding="target_match")
    return {
        "states": data["states_train"][:n_rows],
        "empirical": None,
        "participant_idx": data["participant_idx"][:n_rows],
        "n_participants": data["n_participants"],
        "sufficient_dim": data["sufficient_dim"][:n_rows],
        "has_one_word_solution": data["has_one_word_solution"][:n_rows],
        "is_sharp": data["sharpness_idx"][:n_rows],
        "is_colour_sufficient": data["is_colour_sufficient"][:n_rows],
    }


def test_joint_model_exposes_all_three_participant_hierarchies():
    trace = handlers.trace(
        handlers.seed(ms.V11_JOINT_PARTICIPANT_MODELS["K-HKO"], 17)
    ).get_trace(**_arguments())

    expected = {
        "alpha_by_participant",
        "kappa_by_participant",
        "beta_order_by_participant",
        "tau_log_alpha",
        "tau_kappa",
        "tau_order",
    }
    assert expected.issubset(trace)
    assert "kappa" not in trace
    probs = np.asarray(trace["obs"]["fn"].probs)
    assert probs.shape == (16, 15)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


if __name__ == "__main__":
    test_joint_model_exposes_all_three_participant_hierarchies()
    print("PASS test_joint_model_exposes_all_three_participant_hierarchies")
