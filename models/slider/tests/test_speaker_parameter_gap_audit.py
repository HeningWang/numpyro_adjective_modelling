import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "models" / "slider"))

import speaker_parameter_gap_audit as audit  # noqa: E402


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_gap_audit_blocks_more_existing_variant_mcmc_when_ppc_gate_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        forward_dir = root / "forward"
        posterior_dir = root / "posterior"
        heldout_dir = root / "heldout"
        out_dir = root / "out"

        write_csv(
            forward_dir / "slider_speaker_variant_grid_best_by_variant.csv",
            [
                {
                    "variant": "planned_usefulness_mixture",
                    "selection_criterion": "condition_rmse",
                    "grid_id": 1,
                    "alpha": 1.0,
                    "order_bias": 1.0,
                    "usefulness_order_scale": 6.0,
                    "signed_order_scale": pd.NA,
                    "planned_mixture_weight": 0.5,
                    "condition_rmse": 0.04,
                    "condition_abs_zrdc_residual": 0.01,
                }
            ],
        )
        write_csv(
            posterior_dir / "slider_mcmc_diagnostics.csv",
            [
                {"model": "planned_usefulness_mixture", "parameter": "alpha", "mean": 1.0},
                {"model": "planned_usefulness_mixture", "parameter": "bias", "mean": 0.7},
                {
                    "model": "planned_usefulness_mixture",
                    "parameter": "usefulness_order_scale",
                    "mean": 2.0,
                },
                {
                    "model": "planned_usefulness_mixture",
                    "parameter": "planned_mixture_weight",
                    "mean": 0.6,
                },
            ],
        )
        write_csv(
            posterior_dir / "slider_speaker_ablation_eval_pairwise_decisions.csv",
            [
                {
                    "pair": "mixture_vs_greedy_recursive",
                    "second_property_abs_residual_reduction": 0.01,
                    "ppc_success": False,
                    "recommended_for_full_run": False,
                }
            ],
        )
        write_csv(
            heldout_dir / "slider_heldout_eval_pairwise_decisions.csv",
            [
                {
                    "pair": "mixture_vs_greedy_recursive",
                    "delta_heldout_elpd_candidate_minus_baseline": 0.2,
                    "second_property_abs_residual_reduction": 0.01,
                    "ppc_success": False,
                    "recommended_for_full_run": False,
                }
            ],
        )

        args = SimpleNamespace(
            forward_dir=forward_dir,
            posterior_dir=posterior_dir,
            heldout_dir=heldout_dir,
            out_dir=out_dir,
            second_residual_gate=0.05,
        )
        gap, decision = audit.build_gap_summary(args)

    assert gap.iloc[0]["usefulness_order_scale_gap_grid_minus_posterior"] == 4.0
    assert not bool(decision.iloc[0]["existing_variant_passed_gate"])
    assert not bool(decision.iloc[0]["run_more_existing_variant_mcmc"])


if __name__ == "__main__":
    test_gap_audit_blocks_more_existing_variant_mcmc_when_ppc_gate_fails()
    print("PASS slider speaker parameter gap audit tests")
