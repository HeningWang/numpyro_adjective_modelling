import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

import evaluate_speaker_pilot as pilot  # noqa: E402


def args():
    return SimpleNamespace(
        elpd_gate=0.0,
        ppc_rmse_gate=0.0,
        second_residual_gate=0.05,
        max_new_residual_harm=0.02,
    )


def condition_rows(model: str, residuals: dict[str, float]) -> list[dict]:
    rows = []
    for relevant_property, signed_residual in residuals.items():
        rows.append(
            {
                "model": model,
                "relevant_property": relevant_property,
                "sharpness": "sharp",
                "human_mean": 0.5,
                "model_mean": 0.5 + signed_residual,
                "signed_residual": signed_residual,
                "abs_residual": abs(signed_residual),
            }
        )
    return rows


def test_free_color_pair_is_registered():
    pairs = {(candidate, baseline) for candidate, baseline, _ in pilot.PAIR_SPECS}

    assert ("incremental_free_color", "incremental_recursive") in pairs
    assert ("incremental_static_free_color", "incremental_static") in pairs


def test_recommendation_requires_ppc_success_even_when_elpd_reliable():
    model_summary = pd.DataFrame(
        [
            {
                "model": "incremental_free_color",
                "diagnostics_ok": True,
                "psis_loo_reliable": True,
                "elpd_loo": -9.0,
                "ppc_rmse": 0.09,
                "ppc_r": 0.95,
            },
            {
                "model": "incremental_recursive",
                "diagnostics_ok": True,
                "psis_loo_reliable": True,
                "elpd_loo": -10.0,
                "ppc_rmse": 0.10,
                "ppc_r": 0.94,
            },
        ]
    )
    ppc_by_condition = pd.DataFrame(
        condition_rows(
            "incremental_recursive",
            {"second": 0.08, "first": -0.08, "both": -0.04},
        )
        + condition_rows(
            "incremental_free_color",
            {"second": 0.07, "first": -0.08, "both": -0.04},
        )
    )

    pairwise, _ = pilot.pairwise_summary(model_summary, ppc_by_condition, args())
    row = pairwise[pairwise["pair"].eq("free_color_vs_fixed_recursive")].iloc[0]

    assert bool(row["loo_success"])
    assert not bool(row["ppc_success"])
    assert not bool(row["recommended_for_full_run"])


if __name__ == "__main__":
    test_free_color_pair_is_registered()
    test_recommendation_requires_ppc_success_even_when_elpd_reliable()
    print("PASS slider speaker pilot evaluator tests")
