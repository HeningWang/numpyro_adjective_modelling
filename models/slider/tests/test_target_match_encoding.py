import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

import modelSpecification as ms  # noqa: E402


def test_slider_states_default_to_target_match_encoding():
    row = pd.Series(
        {
            "size_A": 8,
            "size_B": 7,
            "size_C": 6,
            "size_D": 5,
            "size_E": 4,
            "size_F": 3,
            "color_A": "red",
            "color_B": "blue",
            "color_C": "red",
            "color_D": "blue",
            "color_E": "red",
            "color_F": "blue",
            "form_A": "square",
            "form_B": "circle",
            "form_C": "square",
            "form_D": "circle",
            "form_E": "square",
            "form_F": "circle",
        }
    )

    target_match = np.asarray(ms.encode_states(row))
    canonical = np.asarray(ms.encode_states(row, state_encoding="canonical"))

    assert target_match[0, 1] == 1.0
    assert target_match[0, 2] == 1.0
    assert canonical[0, 1] == 0.0
    assert canonical[0, 2] == 0.0
    assert np.array_equal(target_match[:, 1], np.array([1, 0, 1, 0, 1, 0]))
    assert np.array_equal(target_match[:, 2], np.array([1, 0, 1, 0, 1, 0]))


if __name__ == "__main__":
    test_slider_states_default_to_target_match_encoding()
    print("PASS slider target-match encoding tests")
