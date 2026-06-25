import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import model_pareto_frontier as frontier  # noqa: E402


def test_normalize_loo_uses_arviz_index_column():
    loo = pd.DataFrame(
        {
            "Unnamed: 0": ["a", "b"],
            "elpd_loo": [-10.0, -12.0],
            "p_loo": [2.0, 3.0],
            "se": [1.0, 1.5],
            "warning": [False, True],
        }
    )

    out = frontier.normalize_loo(loo)

    assert list(out["model"]) == ["a", "b"]
    assert "elpd_loo_se" in out.columns
    assert out.loc[out["model"].eq("a"), "delta_elpd_from_best"].iloc[0] == 0.0
    assert out.loc[out["model"].eq("b"), "delta_elpd_from_best"].iloc[0] == -2.0


def test_pareto_frontier_marks_dominated_models():
    scores = pd.DataFrame(
        [
            {"model": "a", "elpd_loo": -100.0, "ppc_rmse": 0.10, "complexity": 2},
            {"model": "b", "elpd_loo": -90.0, "ppc_rmse": 0.20, "complexity": 2},
            {"model": "c", "elpd_loo": -110.0, "ppc_rmse": 0.20, "complexity": 3},
        ]
    )
    scores["diagnostics_ok"] = True

    out = frontier.mark_pareto_frontier(scores)
    flags = dict(zip(out["model"], out["posterior_pareto_frontier"]))

    assert flags["a"] is True
    assert flags["b"] is True
    assert flags["c"] is False


def test_build_frontier_from_csvs_writes_expected_scores():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        loo_path = tmp_path / "loo.csv"
        ppc_path = tmp_path / "ppc.csv"
        diag_path = tmp_path / "diag.csv"
        complexity_path = tmp_path / "complexity.csv"

        pd.DataFrame(
            {
                "Unnamed: 0": ["a", "b", "c"],
                "elpd_loo": [-100.0, -90.0, -110.0],
                "p_loo": [10.0, 20.0, 30.0],
                "se": [1.0, 1.0, 1.0],
                "warning": [False, False, False],
            }
        ).to_csv(loo_path, index=False)
        pd.DataFrame(
            {
                "model": ["a", "b", "c"],
                "scope": ["all_cells", "all_cells", "all_cells"],
                "rmse": [0.10, 0.20, 0.20],
                "mae": [0.08, 0.18, 0.18],
                "r": [0.9, 0.8, 0.7],
            }
        ).to_csv(ppc_path, index=False)
        pd.DataFrame(
            {
                "model": ["a", "b", "c"],
                "diagnostic_status": ["pass", "pass", "pass"],
                "max_r_hat": [1.0, 1.0, 1.0],
                "n_divergent": [0, 0, 0],
            }
        ).to_csv(diag_path, index=False)
        pd.DataFrame(
            {
                "model": ["a", "b", "c"],
                "mechanism_count": [2, 2, 3],
            }
        ).to_csv(complexity_path, index=False)

        out = frontier.build_frontier_from_csvs(
            loo_path,
            ppc_path,
            diagnostics_csv=diag_path,
            complexity_csv=complexity_path,
        )

    flags = dict(zip(out["model"], out["posterior_pareto_frontier"]))
    assert flags["a"] is True
    assert flags["b"] is True
    assert flags["c"] is False
    assert out.loc[out["model"].eq("a"), "complexity_source"].iloc[0] == "mechanism_count"


if __name__ == "__main__":
    test_normalize_loo_uses_arviz_index_column()
    test_pareto_frontier_marks_dominated_models()
    test_build_frontier_from_csvs_writes_expected_scores()
    print("PASS model pareto frontier tests")
