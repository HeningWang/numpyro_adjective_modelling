import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

import speaker_variant_forward_audit as audit  # noqa: E402


def make_listener(size=0.2, colour=0.7, dc=0.45, cd=0.55):
    return {
        "D": {"target_listener_prob": size},
        "C": {"target_listener_prob": colour},
        "DC": {"target_listener_prob": dc},
        "CD": {"target_listener_prob": cd},
    }


def test_planned_stop_prediction_returns_probability_components():
    out = audit.planned_stop_prediction(make_listener(), alpha=1.0, order_bias=1.0)

    assert 0.0 <= out["pred_slider"] <= 1.0
    assert 0.0 <= out["first_step_size_prob"] <= 1.0
    assert 0.0 <= out["continue_after_size"] <= 1.0
    assert 0.0 <= out["continue_after_colour"] <= 1.0
    assert out["chain_dc"] >= 0.0
    assert out["chain_cd"] >= 0.0


def test_usefulness_moderated_order_bias_drops_when_colour_is_more_useful():
    listener = make_listener(size=0.2, colour=0.8)
    moderated = audit.usefulness_moderated_order_bias(
        listener,
        base_order_bias=2.0,
        usefulness_scale=2.0,
    )

    assert moderated < 2.0
    assert moderated >= 0.0


def test_pareto_frontier_marks_dominated_rows():
    import pandas as pd

    summary = pd.DataFrame(
        [
            {
                "variant": "a",
                "condition_rmse": 0.2,
                "condition_abs_zrdc_residual": 0.2,
                "mechanism_count": 2,
            },
            {
                "variant": "b",
                "condition_rmse": 0.1,
                "condition_abs_zrdc_residual": 0.1,
                "mechanism_count": 2,
            },
            {
                "variant": "c",
                "condition_rmse": 0.09,
                "condition_abs_zrdc_residual": 0.09,
                "mechanism_count": 4,
            },
        ]
    )

    out = audit.pareto_frontier(summary)
    flags = dict(zip(out["variant"], out["forward_pareto_frontier"]))

    assert flags["a"] == False
    assert flags["b"] == True
    assert flags["c"] == True


if __name__ == "__main__":
    test_planned_stop_prediction_returns_probability_components()
    test_usefulness_moderated_order_bias_drops_when_colour_is_more_useful()
    test_pareto_frontier_marks_dominated_rows()
    print("PASS slider speaker variant audit tests")
