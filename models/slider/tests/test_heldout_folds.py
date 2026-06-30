import os
import sys
from pathlib import Path

import pandas as pd


os.environ.setdefault("JAX_PLATFORMS", "cpu")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

from run_inference import balanced_fold_ids  # noqa: E402
import evaluate_heldout_elpd as heldout  # noqa: E402


def test_balanced_fold_ids_are_deterministic_and_condition_balanced():
    rows = []
    for relevant_property in ["first", "second", "both"]:
        for sharpness in ["sharp", "blurred"]:
            for _ in range(10):
                rows.append(
                    {
                        "relevant_property": relevant_property,
                        "sharpness": sharpness,
                    }
                )
    df = pd.DataFrame(rows)

    folds_a = balanced_fold_ids(df, num_folds=5, seed=13)
    folds_b = balanced_fold_ids(df, num_folds=5, seed=13)

    assert folds_a.tolist() == folds_b.tolist()
    assert sorted(set(folds_a.tolist())) == [0, 1, 2, 3, 4]

    df = df.assign(fold=folds_a)
    counts = (
        df.groupby(["relevant_property", "sharpness", "fold"])
        .size()
        .reset_index(name="n")
    )
    assert counts["n"].min() == 2
    assert counts["n"].max() == 2


def test_balanced_fold_ids_do_not_depend_on_dataframe_index():
    df = pd.DataFrame(
        {
            "relevant_property": ["first"] * 6 + ["second"] * 6,
            "sharpness": ["sharp"] * 12,
        },
        index=range(100, 112),
    )

    folds = balanced_fold_ids(df, num_folds=3, seed=17)

    assert len(folds) == len(df)
    assert sorted(set(folds.tolist())) == [0, 1, 2]


def test_heldout_speaker_ablation_specs_cover_signed_order_variants():
    assert heldout.MODEL_TO_SPEAKER["planned_usefulness_signed_order"] == (
        "planned_usefulness_signed_order"
    )
    assert heldout.MODEL_TO_SPEAKER["planned_usefulness_signed_order_static"] == (
        "planned_usefulness_signed_order_static"
    )

    pairs = {(candidate, baseline) for candidate, baseline, _ in heldout.PAIR_SPECS}
    assert ("planned_usefulness_signed_order", "planned_usefulness_order") in pairs
    assert (
        "planned_usefulness_signed_order_static",
        "planned_usefulness_order_static",
    ) in pairs
    assert ("planned_usefulness_signed_order", "incremental_recursive") in pairs
    assert ("planned_usefulness_signed_order_static", "incremental_static") in pairs
    assert ("planned_usefulness_mixture_anchored", "incremental_recursive") in pairs
    assert (
        "planned_usefulness_mixture_anchored_static",
        "incremental_static",
    ) in pairs
    assert heldout.MODEL_TO_SPEAKER["production_anchor_sizesharp_2x2_inc_rec"] == (
        "production_anchor_sizesharp_2x2_inc_rec"
    )
    assert heldout.MODEL_TO_SPEAKER["production_anchor_sizesharp_2x2_glob_static"] == (
        "production_anchor_sizesharp_2x2_glob_static"
    )
    assert heldout.MODEL_TO_SPEAKER["production_anchor_reliabilitybackup_2x2_inc_rec"] == (
        "production_anchor_reliabilitybackup_2x2_inc_rec"
    )
    assert heldout.MODEL_TO_SPEAKER["production_anchor_reliabilitybackup_2x2_glob_static"] == (
        "production_anchor_reliabilitybackup_2x2_glob_static"
    )
    assert heldout.MODEL_TO_SPEAKER[
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec"
    ] == "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec"
    assert heldout.MODEL_TO_SPEAKER[
        "production_anchor_reliabilitybackup_logalpha_2x2_glob_static"
    ] == "production_anchor_reliabilitybackup_logalpha_2x2_glob_static"
    assert heldout.MODEL_TO_SPEAKER[
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec"
    ] == "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec"
    assert heldout.MODEL_TO_SPEAKER[
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static"
    ] == "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static"
    assert (
        "production_anchor_sizesharp_2x2_inc_rec",
        "production_anchor_sizesharp_2x2_glob_rec",
    ) in pairs
    assert (
        "production_anchor_sizesharp_2x2_inc_static",
        "production_anchor_sizesharp_2x2_inc_rec",
    ) in pairs
    assert (
        "production_anchor_reliabilitybackup_2x2_inc_rec",
        "production_anchor_reliabilitybackup_2x2_glob_rec",
    ) in pairs
    assert (
        "production_anchor_reliabilitybackup_2x2_inc_static",
        "production_anchor_reliabilitybackup_2x2_inc_rec",
    ) in pairs
    assert (
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_logalpha_2x2_glob_rec",
    ) in pairs
    assert (
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec",
    ) in pairs
    assert (
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_rec",
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_rec",
    ) in pairs
    assert (
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static",
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_static",
    ) in pairs
    assert heldout.is_production_anchor_speaker(
        "production_anchor_reliabilitybackup_2x2_inc_static"
    )
    assert heldout.is_production_anchor_speaker(
        "production_anchor_reliabilitybackup_logalpha_2x2_inc_static"
    )
    assert heldout.is_production_anchor_speaker(
        "production_anchor_reliabilitybackup_orderplan_logalpha_2x2_inc_static"
    )


def test_full_run_recommendation_requires_ppc_success():
    assert not heldout.recommend_full_run(
        candidate_diagnostics_ok=True,
        baseline_diagnostics_ok=True,
        heldout_success=True,
        ppc_success=False,
        candidate_on_frontier=True,
    )
    assert heldout.recommend_full_run(
        candidate_diagnostics_ok=True,
        baseline_diagnostics_ok=True,
        heldout_success=True,
        ppc_success=True,
        candidate_on_frontier=False,
    )
    assert heldout.recommend_full_run(
        candidate_diagnostics_ok=True,
        baseline_diagnostics_ok=True,
        heldout_success=False,
        ppc_success=True,
        candidate_on_frontier=True,
    )


if __name__ == "__main__":
    test_balanced_fold_ids_are_deterministic_and_condition_balanced()
    test_balanced_fold_ids_do_not_depend_on_dataframe_index()
    test_heldout_speaker_ablation_specs_cover_signed_order_variants()
    test_full_run_recommendation_requires_ppc_success()
    print("PASS slider heldout fold tests")
