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


if __name__ == "__main__":
    test_balanced_fold_ids_are_deterministic_and_condition_balanced()
    test_balanced_fold_ids_do_not_depend_on_dataframe_index()
    test_heldout_speaker_ablation_specs_cover_signed_order_variants()
    print("PASS slider heldout fold tests")
