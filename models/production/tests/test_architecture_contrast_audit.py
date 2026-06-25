import numpy as np
import pandas as pd
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from models.production.architecture_contrast_audit import (
    aggregate_category_proportions,
    build_global_misses,
    classify_utterance,
    parse_2x2_model_name,
)


def test_classify_utterance_derives_requested_categories():
    one = classify_utterance("D")
    assert one["length_class"] == "one_adjective"
    assert one["selection_class"] == "size_only"
    assert one["order_class"] == "single_adjective"

    dc = classify_utterance("DC")
    assert dc["length_class"] == "two_adjectives"
    assert dc["selection_class"] == "size_colour"
    assert dc["order_class"] == "size_initial_multi"

    cf = classify_utterance("CF")
    assert cf["selection_class"] == "redundant_form"
    assert cf["order_class"] == "colour_initial_multi"

    cdf = classify_utterance("CDF")
    assert cdf["length_class"] == "three_adjectives"
    assert cdf["selection_class"] == "redundant_form"
    assert cdf["order_class"] == "colour_initial_multi"


def test_parse_2x2_model_name_extracts_architecture_and_semantics():
    parsed = parse_2x2_model_name(
        "principled_salience_stop_regularized_2x2_inc_static"
    )
    assert parsed["architecture"] == "incremental"
    assert parsed["semantics"] == "context_fixed"
    assert parsed["architecture_short"] == "inc"
    assert parsed["semantics_short"] == "static"

    parsed = parse_2x2_model_name(
        "principled_salience_stop_regularized_2x2_glob_rec"
    )
    assert parsed["architecture"] == "global"
    assert parsed["semantics"] == "context_updating"

    parsed = parse_2x2_model_name(
        "principled_salience_stop_regularized_responsepolicy_boundedform_2x2_glob_rec_fixedeps"
    )
    assert parsed["architecture"] == "global"
    assert parsed["semantics"] == "context_updating"


def test_aggregate_category_proportions_preserves_condition_totals():
    rows = pd.DataFrame({
        "source": ["empirical"] * 3,
        "model": ["empirical"] * 3,
        "architecture": ["empirical"] * 3,
        "semantics": ["empirical"] * 3,
        "relevant_property": ["first"] * 3,
        "sharpness": ["sharp"] * 3,
        "utterance_label": ["D", "DC", "CF"],
        "proportion": [0.25, 0.50, 0.25],
        "n": [100, 100, 100],
    })

    out = aggregate_category_proportions(rows, "selection_class")

    assert set(out["category"]) == {"size_only", "size_colour", "redundant_form"}
    assert np.isclose(out["proportion"].sum(), 1.0)


def test_build_global_misses_ranks_where_global_is_worse_than_incremental():
    residuals = pd.DataFrame({
        "model": [
            "principled_salience_stop_regularized_2x2_inc_static",
            "principled_salience_stop_regularized_2x2_glob_static",
            "principled_salience_stop_regularized_2x2_inc_static",
            "principled_salience_stop_regularized_2x2_glob_static",
        ],
        "architecture": ["incremental", "global", "incremental", "global"],
        "semantics": ["context_fixed"] * 4,
        "summary_type": ["length"] * 4,
        "relevant_property": ["first", "first", "second", "second"],
        "sharpness": ["sharp", "sharp", "blurred", "blurred"],
        "category": ["one_adjective", "one_adjective", "two_adjectives", "two_adjectives"],
        "empirical_proportion": [0.8, 0.8, 0.2, 0.2],
        "model_proportion": [0.7, 0.3, 0.1, 0.4],
        "signed_residual": [-0.1, -0.5, -0.1, 0.2],
        "abs_residual": [0.1, 0.5, 0.1, 0.2],
    })

    misses = build_global_misses(residuals)

    assert misses.iloc[0]["relevant_property"] == "first"
    assert misses.iloc[0]["global_minus_incremental_abs_residual"] == 0.4


if __name__ == "__main__":
    test_classify_utterance_derives_requested_categories()
    test_parse_2x2_model_name_extracts_architecture_and_semantics()
    test_aggregate_category_proportions_preserves_condition_totals()
    test_build_global_misses_ranks_where_global_is_worse_than_incremental()
    print("PASS architecture contrast audit tests")
