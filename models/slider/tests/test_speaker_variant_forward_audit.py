import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

import pandas as pd  # noqa: E402
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


def test_signed_usefulness_bias_can_be_negative_when_colour_is_more_useful():
    listener = make_listener(size=0.2, colour=0.8)

    moderated = audit.signed_usefulness_moderated_order_bias(
        listener,
        base_order_bias=2.0,
        signed_scale=4.0,
    )

    assert moderated < 0.0


def test_mixture_prediction_interpolates_slider_prediction():
    listener = make_listener(size=0.2, colour=0.8, dc=0.45, cd=0.55)
    greedy = audit.incremental_prediction(listener, alpha=1.0, bias=2.0)
    planned = audit.predict_variant(
        "planned_usefulness_order",
        listener,
        type("Args", (), {"alpha": 1.0, "order_bias": 2.0, "usefulness_order_scale": 4.0})(),
    )

    out = audit.mixture_prediction(
        listener,
        alpha=1.0,
        order_bias=2.0,
        usefulness_scale=4.0,
        planned_weight=0.25,
    )

    expected = 0.25 * planned["pred_slider"] + 0.75 * greedy["pred_slider"]
    assert abs(out["pred_slider"] - expected) < 1e-12
    assert out["planned_mixture_weight"] == 0.25


def make_grid_row(condition, relevant_property, sharpness, human_slider):
    row = {
        "id": 1,
        "item": 1,
        "conditions": condition,
        "list": 1,
        "trials": 1,
        "relevant_property": relevant_property,
        "sharpness": sharpness,
        "human_slider": human_slider,
    }
    sizes = {"A": 10, "B": 8, "C": 6, "D": 4, "E": 3, "F": 2}
    colors = {"A": "red", "B": "red", "C": "blue", "D": "blue", "E": "red", "F": "blue"}
    forms = {"A": "circle", "B": "square", "C": "circle", "D": "square", "E": "circle", "F": "square"}
    for label in "ABCDEF":
        row[f"size_{label}"] = sizes[label]
        row[f"color_{label}"] = colors[label]
        row[f"form_{label}"] = forms[label]
    return row


def test_parameter_grid_gate_exports_compact_decision():
    df = pd.DataFrame(
        [
            make_grid_row("zrdc", "second", "sharp", 0.55),
            make_grid_row("erdc", "first", "sharp", 0.80),
            make_grid_row("brdc", "both", "blurred", 0.70),
        ]
    )
    args = SimpleNamespace(
        size_context_mode="static",
        color_sem=0.8,
        k=0.5,
        wf=1.0,
        grid_alpha="1.0",
        grid_order_bias="1.0",
        grid_usefulness_order_scale="0.0,4.0",
        grid_signed_order_scale="0.0,4.0",
        grid_planned_mixture_weight="0.0,0.5,1.0",
        rmse_gate=0.01,
        zrdc_gate=0.02,
    )

    summary = audit.parameter_grid_summary(df, args)
    gate = audit.parameter_grid_gate_decision(summary, args)

    assert {"planned_usefulness_mixture", "planned_usefulness_signed_order"}.issubset(
        set(summary["variant"])
    )
    assert len(gate) == 1
    assert "run_new_ablation_pilot" in gate.columns


def test_pareto_frontier_marks_dominated_rows():
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
    test_signed_usefulness_bias_can_be_negative_when_colour_is_more_useful()
    test_mixture_prediction_interpolates_slider_prediction()
    test_parameter_grid_gate_exports_compact_decision()
    test_pareto_frontier_marks_dominated_rows()
    print("PASS slider speaker variant audit tests")
