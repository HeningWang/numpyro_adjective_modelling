import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "analysis"))

import model_selection_decision_summary as decisions  # noqa: E402


def args_for(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        slider_posterior_stats_dir=tmp_path / "slider_posterior",
        slider_posterior_prefix="slider_eval",
        slider_heldout_stats_dir=tmp_path / "slider_heldout",
        slider_heldout_prefix="slider_heldout_eval",
        production_architecture_dir=tmp_path / "production_architecture",
        production_prefix="production_2x2",
        out_dir=tmp_path / "out",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_decision_summary_reports_pending_when_csvs_are_absent():
    with tempfile.TemporaryDirectory() as tmp:
        evidence, stage_decisions, candidate_scores = decisions.build_decision_summary(
            args_for(Path(tmp))
        )

    statuses = dict(zip(stage_decisions["decision_stage"], stage_decisions["status"]))
    assert statuses["slider_posterior_ablation"] == "pending"
    assert statuses["slider_heldout_ablation"] == "pending"
    assert statuses["production_2x2"] == "pending"
    assert statuses["final_interpretable_2x2"] == "pending"
    assert not evidence.empty
    assert candidate_scores.empty


def test_decision_summary_passes_when_all_gates_pass():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        args = args_for(tmp_path)

        write_csv(
            args.slider_posterior_stats_dir / "slider_eval_pairwise_decisions.csv",
            [
                {
                    "pair": "mixture_vs_planned_recursive",
                    "candidate": "planned_usefulness_mixture",
                    "baseline": "planned_usefulness_order",
                    "recommended_for_full_run": True,
                    "delta_elpd_candidate_minus_baseline": 5.0,
                    "ppc_rmse_gain": 0.02,
                    "ppc_success": True,
                    "second_property_abs_residual_reduction": 0.06,
                }
            ],
        )
        write_csv(
            args.slider_posterior_stats_dir / "slider_eval_pareto_frontier.csv",
            [{"model": "planned_usefulness_mixture", "rank": 0, "elpd_loo": -10.0}],
        )
        write_csv(
            args.slider_heldout_stats_dir / "slider_heldout_eval_pairwise_decisions.csv",
            [
                {
                    "pair": "mixture_vs_planned_recursive",
                    "candidate": "planned_usefulness_mixture",
                    "baseline": "planned_usefulness_order",
                    "recommended_for_full_run": True,
                    "delta_heldout_elpd_candidate_minus_baseline": 4.0,
                    "ppc_rmse_gain": 0.01,
                    "ppc_success": True,
                    "second_property_abs_residual_reduction": 0.05,
                }
            ],
        )
        write_csv(
            args.slider_heldout_stats_dir / "slider_heldout_eval_pareto_frontier.csv",
            [{"model": "planned_usefulness_mixture", "rank": 0, "total_heldout_elpd": -12.0}],
        )
        write_csv(
            args.production_architecture_dir / "production_2x2_pareto_scores.csv",
            [
                {
                    "model": "contextual_pcalpha_canon_parsimony_2x2_inc_static",
                    "rank": 0,
                    "elpd_loo": -100.0,
                    "ppc_rmse": 0.08,
                    "diagnostic_status": "pass",
                    "diagnostics_ok": True,
                },
                {
                    "model": "contextual_pcalpha_canon_parsimony_2x2_glob_static",
                    "rank": 1,
                    "elpd_loo": -200.0,
                    "ppc_rmse": 0.12,
                    "diagnostic_status": "fail",
                    "diagnostics_ok": False,
                },
            ],
        )
        write_csv(
            args.production_architecture_dir / "production_2x2_pareto_frontier.csv",
            [
                {
                    "model": "contextual_pcalpha_canon_parsimony_2x2_inc_static",
                    "rank": 0,
                    "elpd_loo": -100.0,
                    "ppc_rmse": 0.08,
                    "diagnostic_status": "pass",
                    "diagnostics_ok": True,
                }
            ],
        )
        write_csv(
            args.production_architecture_dir / "architecture_contrast_loo_diagnostics.csv",
            [
                {
                    "semantics": "context_fixed",
                    "delta_elpd_incremental_minus_global": 100.0,
                    "incremental_diagnostic_status": "pass",
                    "global_diagnostic_status": "fail",
                },
                {
                    "semantics": "context_updating",
                    "delta_elpd_incremental_minus_global": 90.0,
                    "incremental_diagnostic_status": "pass",
                    "global_diagnostic_status": "fail",
                },
            ],
        )

        _, stage_decisions, candidate_scores = decisions.build_decision_summary(args)

    statuses = dict(zip(stage_decisions["decision_stage"], stage_decisions["status"]))
    assert statuses["slider_posterior_ablation"] == "pass"
    assert statuses["slider_heldout_ablation"] == "pass"
    assert statuses["production_2x2"] == "pass"
    assert statuses["final_interpretable_2x2"] == "pass"
    final = stage_decisions.query("decision_stage == 'final_interpretable_2x2'").iloc[0]
    assert final["selected_architecture"] == "incremental"
    assert final["selected_semantics"] == "context_fixed"
    assert set(candidate_scores["evidence_source"]) == {
        "slider_posterior_ablation",
        "slider_heldout_ablation",
        "production_2x2",
    }


if __name__ == "__main__":
    test_decision_summary_reports_pending_when_csvs_are_absent()
    test_decision_summary_passes_when_all_gates_pass()
    print("PASS model selection decision summary tests")
