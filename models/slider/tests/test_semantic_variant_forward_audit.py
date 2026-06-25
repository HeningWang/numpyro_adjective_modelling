import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

import semantic_variant_forward_audit as audit  # noqa: E402


def make_row(color_a="red"):
    data = {
        "id": 1,
        "item": 1,
        "conditions": "erdc",
        "list": 1,
        "trials": 1,
        "sharpness": "sharp",
        "prefer_first_1st": 80,
        "combination": "dimension_color",
        "relevant_property": "first",
        "color_A": color_a,
        "color_B": "blue",
        "color_C": "red",
        "color_D": "blue",
        "color_E": "red",
        "color_F": "blue",
        "form_A": "circle",
        "form_B": "circle",
        "form_C": "square",
        "form_D": "square",
        "form_E": "circle",
        "form_F": "square",
    }
    for label, size in zip("ABCDEF", [8, 6, 5, 20, 18, 2]):
        data[f"size_{label}"] = size
    return pd.Series(data)


def test_target_match_encoding_flips_red_target_colour_feature():
    row = make_row(color_a="red")
    canonical = audit.build_states(row, "canonical")
    target_match = audit.build_states(row, "target_match")

    assert canonical[0, 1] == 0.0
    assert target_match[0, 1] == 1.0
    assert np.array_equal(target_match[:, 1], np.array([1, 0, 1, 0, 1, 0], dtype=float))


def test_comparison_class_changes_dc_but_not_cd_size_threshold():
    states = np.asarray(
        [
            [8.0, 1.0, 1.0],
            [6.0, 1.0, 1.0],
            [20.0, 0.0, 1.0],
            [18.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    static = audit.listener_values(states, "static", color_sem=0.8, k=0.5, wf=1.0)
    comparison_class = audit.listener_values(
        states, "comparison_class", color_sem=0.8, k=0.5, wf=1.0
    )

    assert comparison_class["DC"]["target_listener_prob"] > static["DC"]["target_listener_prob"]
    assert np.isclose(
        comparison_class["CD"]["target_listener_prob"],
        static["CD"]["target_listener_prob"],
    )


def test_incremental_and_global_predictions_are_probabilities():
    states = audit.build_states(make_row(color_a="red"), "target_match")
    listener = audit.listener_values(states, "comparison_class", color_sem=0.8, k=0.5, wf=1.0)
    inc = audit.incremental_prediction(listener, alpha=1.0, bias=2.0)
    glob = audit.global_prediction(listener, alpha=1.0, bias=2.0)

    assert 0.0 <= inc["pred_slider"] <= 1.0
    assert 0.0 <= glob["pred_slider"] <= 1.0
    assert 0.0 <= inc["first_step_size_prob"] <= 1.0
    assert 0.0 <= inc["continue_after_size"] <= 1.0


if __name__ == "__main__":
    test_target_match_encoding_flips_red_target_colour_feature()
    test_comparison_class_changes_dc_but_not_cd_size_threshold()
    test_incremental_and_global_predictions_are_probabilities()
    print("PASS slider semantic variant audit tests")
