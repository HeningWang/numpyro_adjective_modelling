import os
import sys
from pathlib import Path

import numpy as np


os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "production"))

import helper  # noqa: E402


def test_production_states_default_to_target_match_encoding():
    target_match = helper.import_dataset()
    canonical = helper.import_dataset(state_encoding="canonical")

    df = target_match["df"].reset_index(drop=True)
    red_target_idx = int(df.index[df["color_A"].eq("red")][0])

    target_match_states = np.asarray(target_match["states_train"])[red_target_idx]
    canonical_states = np.asarray(canonical["states_train"])[red_target_idx]

    assert target_match_states[0, 1] == 1.0
    assert canonical_states[0, 1] == 0.0
    assert np.array_equal(
        target_match_states[:, 1],
        (df.loc[red_target_idx, [f"color_{label}" for label in "ABCDEF"]].to_numpy() == "red"),
    )


if __name__ == "__main__":
    test_production_states_default_to_target_match_encoding()
    print("PASS production target-match encoding tests")
