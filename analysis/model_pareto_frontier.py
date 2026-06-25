"""Build a post-inference Pareto frontier from LOO and PPC summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_COL_CANDIDATES = ("model", "variant", "Unnamed: 0")


def ensure_model_column(df: pd.DataFrame, model_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    if model_col is not None:
        if model_col not in out.columns:
            raise ValueError(f"Requested model column '{model_col}' is absent.")
        return out.rename(columns={model_col: "model"})

    for candidate in MODEL_COL_CANDIDATES:
        if candidate in out.columns:
            return out.rename(columns={candidate: "model"})

    unnamed = [col for col in out.columns if str(col).startswith("Unnamed:")]
    if unnamed:
        return out.rename(columns={unnamed[0]: "model"})

    raise ValueError(
        "Could not infer model column. Pass --model-col or use a model/variant column."
    )


def normalize_loo(df: pd.DataFrame, model_col: str | None = None) -> pd.DataFrame:
    out = ensure_model_column(df, model_col)
    if "elpd_loo" not in out.columns:
        raise ValueError("LOO CSV must contain an elpd_loo column.")

    rename = {"se": "elpd_loo_se", "warning": "loo_warning"}
    out = out.rename(columns=rename)
    keep = [
        "model", "rank", "elpd_loo", "p_loo", "elpd_diff",
        "elpd_loo_se", "dse", "loo_warning", "scale",
    ]
    keep = [col for col in keep if col in out.columns]
    out = out[keep].copy()
    out["elpd_loo"] = pd.to_numeric(out["elpd_loo"], errors="coerce")
    if "p_loo" in out.columns:
        out["p_loo"] = pd.to_numeric(out["p_loo"], errors="coerce")
    out["delta_elpd_from_best"] = out["elpd_loo"] - out["elpd_loo"].max()
    return out


def normalize_ppc(
    df: pd.DataFrame,
    scope: str = "all_cells",
    model_col: str | None = None,
) -> pd.DataFrame:
    out = ensure_model_column(df, model_col)
    if "scope" in out.columns and scope:
        out = out[out["scope"].astype(str).eq(scope)].copy()
        if out.empty:
            raise ValueError(f"PPC CSV has no rows for scope='{scope}'.")

    metric_map = {
        "rmse": "ppc_rmse",
        "condition_rmse": "ppc_rmse",
        "mae": "ppc_mae",
        "condition_mae": "ppc_mae",
        "r": "ppc_r",
        "correlation": "ppc_r",
        "observation_correlation": "ppc_r",
        "r2": "ppc_r2",
        "observation_r2": "ppc_r2",
    }
    for source, dest in metric_map.items():
        if source in out.columns and dest not in out.columns:
            out[dest] = pd.to_numeric(out[source], errors="coerce")

    if "ppc_rmse" not in out.columns:
        raise ValueError(
            "PPC CSV must contain rmse, condition_rmse, or ppc_rmse."
        )

    if "scope" not in out.columns:
        out["scope"] = scope or "unspecified"

    numeric_cols = [col for col in ["ppc_rmse", "ppc_mae", "ppc_r", "ppc_r2"] if col in out.columns]
    grouped = out.groupby("model", as_index=False)
    metrics = grouped[numeric_cols].mean()
    scopes = grouped["scope"].agg(lambda x: ",".join(sorted(set(map(str, x)))))
    return metrics.merge(scopes, on="model", how="left").rename(columns={"scope": "ppc_scope"})


def normalize_diagnostics(
    df: pd.DataFrame | None,
    model_col: str | None = None,
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["model"])
    out = ensure_model_column(df, model_col)
    out = out.rename(columns={"max_rhat": "max_r_hat"})

    keep = [
        "model", "diagnostic_status", "max_r_hat", "n_r_hat_gt_1_01",
        "n_r_hat_gt_1_05", "min_ess_bulk", "min_ess_tail",
        "n_divergent", "divergence_rate", "n_parameters",
    ]
    keep = [col for col in keep if col in out.columns]
    out = out[keep].copy()
    for col in [c for c in keep if c != "model" and c != "diagnostic_status"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    aggregations = {}
    for col in out.columns:
        if col == "model":
            continue
        if col == "diagnostic_status":
            aggregations[col] = lambda x: "fail" if "fail" in set(x) else (
                "warn" if "warn" in set(x) else "pass"
            )
        elif col in {"min_ess_bulk", "min_ess_tail"}:
            aggregations[col] = "min"
        else:
            aggregations[col] = "max"
    return out.groupby("model", as_index=False).agg(aggregations)


def normalize_complexity(
    df: pd.DataFrame | None,
    model_col: str | None = None,
    complexity_col: str | None = None,
) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame(columns=["model"])
    out = ensure_model_column(df, model_col)
    for col in ["mechanism_count", "free_parameter_count", "n_parameters", "complexity"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if complexity_col is not None:
        if complexity_col not in out.columns:
            raise ValueError(f"Requested complexity column '{complexity_col}' is absent.")
        selected = complexity_col
    elif "complexity" in out.columns:
        selected = "complexity"
    elif "mechanism_count" in out.columns:
        selected = "mechanism_count"
    elif "free_parameter_count" in out.columns:
        selected = "free_parameter_count"
    elif "n_parameters" in out.columns:
        selected = "n_parameters"
    else:
        raise ValueError("Complexity CSV lacks a usable complexity column.")

    keep = [
        "model", "mechanism_count", "free_parameter_count", "n_parameters",
    ]
    keep = [col for col in keep if col in out.columns]
    out = out[keep].copy()
    out["complexity"] = pd.to_numeric(df[selected], errors="coerce")
    out["complexity_source"] = selected
    return out.groupby("model", as_index=False).first()


def diagnostics_ok(row: pd.Series) -> bool:
    status = str(row.get("diagnostic_status", "")).lower()
    if status == "fail":
        return False
    n_divergent = row.get("n_divergent", np.nan)
    if pd.notna(n_divergent) and float(n_divergent) > 0:
        return False
    max_r_hat = row.get("max_r_hat", np.nan)
    if pd.notna(max_r_hat) and float(max_r_hat) > 1.05:
        return False
    return True


def merge_model_scores(
    loo: pd.DataFrame,
    ppc: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
    complexity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = loo.merge(ppc, on="model", how="outer")
    if diagnostics is not None and not diagnostics.empty:
        out = out.merge(diagnostics, on="model", how="left")
    if complexity is not None and not complexity.empty:
        out = out.merge(complexity, on="model", how="left")

    if "complexity" not in out.columns:
        out["complexity"] = np.nan
        out["complexity_source"] = pd.NA
    fallback = out["complexity"].isna() & out.get("p_loo", pd.Series(np.nan, index=out.index)).notna()
    out.loc[fallback, "complexity"] = out.loc[fallback, "p_loo"]
    out.loc[fallback, "complexity_source"] = "p_loo"

    out["diagnostics_ok"] = out.apply(diagnostics_ok, axis=1)
    return out


def mark_pareto_frontier(
    scores: pd.DataFrame,
    elpd_tolerance: float = 0.0,
    ppc_tolerance: float = 0.0,
    complexity_tolerance: float = 0.0,
    exclude_diagnostic_fail: bool = False,
) -> pd.DataFrame:
    rows = []
    required = ["elpd_loo", "ppc_rmse", "complexity"]
    for _, candidate in scores.iterrows():
        complete = all(pd.notna(candidate.get(col, np.nan)) for col in required)
        eligible = complete and (
            bool(candidate.get("diagnostics_ok", True)) or not exclude_diagnostic_fail
        )
        dominated = False
        if eligible:
            for _, other in scores.iterrows():
                if other["model"] == candidate["model"]:
                    continue
                other_complete = all(pd.notna(other.get(col, np.nan)) for col in required)
                if not other_complete:
                    continue
                if exclude_diagnostic_fail and not bool(other.get("diagnostics_ok", True)):
                    continue
                no_worse = (
                    other["elpd_loo"] >= candidate["elpd_loo"] - elpd_tolerance
                    and other["ppc_rmse"] <= candidate["ppc_rmse"] + ppc_tolerance
                    and other["complexity"] <= candidate["complexity"] + complexity_tolerance
                )
                strictly_better = (
                    other["elpd_loo"] > candidate["elpd_loo"] + elpd_tolerance
                    or other["ppc_rmse"] < candidate["ppc_rmse"] - ppc_tolerance
                    or other["complexity"] < candidate["complexity"] - complexity_tolerance
                )
                if no_worse and strictly_better:
                    dominated = True
                    break

        row = candidate.to_dict()
        row["complete_objectives"] = complete
        row["eligible_for_frontier"] = eligible
        row["posterior_pareto_frontier"] = bool(eligible and not dominated)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["posterior_pareto_frontier", "elpd_loo", "ppc_rmse", "complexity"],
        ascending=[False, False, True, True],
    )


def build_frontier_from_csvs(
    loo_csv: Path,
    ppc_csv: Path,
    diagnostics_csv: Path | None = None,
    complexity_csv: Path | None = None,
    ppc_scope: str = "all_cells",
    model_col: str | None = None,
    complexity_col: str | None = None,
    exclude_diagnostic_fail: bool = False,
) -> pd.DataFrame:
    loo = normalize_loo(pd.read_csv(loo_csv), model_col=model_col)
    ppc = normalize_ppc(pd.read_csv(ppc_csv), scope=ppc_scope, model_col=model_col)
    diagnostics = normalize_diagnostics(
        pd.read_csv(diagnostics_csv) if diagnostics_csv else None,
        model_col=model_col,
    )
    complexity = normalize_complexity(
        pd.read_csv(complexity_csv) if complexity_csv else None,
        model_col=model_col,
        complexity_col=complexity_col,
    )
    scores = merge_model_scores(loo, ppc, diagnostics=diagnostics, complexity=complexity)
    return mark_pareto_frontier(
        scores,
        exclude_diagnostic_fail=exclude_diagnostic_fail,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loo-csv", type=Path, required=True)
    parser.add_argument("--ppc-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, default=None)
    parser.add_argument("--complexity-csv", type=Path, default=None)
    parser.add_argument("--complexity-col", type=str, default=None)
    parser.add_argument("--model-col", type=str, default=None)
    parser.add_argument("--ppc-scope", type=str, default="all_cells")
    parser.add_argument("--prefix", type=str, default="model")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--exclude-diagnostic-fail", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frontier = build_frontier_from_csvs(
        loo_csv=args.loo_csv,
        ppc_csv=args.ppc_csv,
        diagnostics_csv=args.diagnostics_csv,
        complexity_csv=args.complexity_csv,
        ppc_scope=args.ppc_scope,
        model_col=args.model_col,
        complexity_col=args.complexity_col,
        exclude_diagnostic_fail=args.exclude_diagnostic_fail,
    )

    scores_path = args.out_dir / f"{args.prefix}_pareto_scores.csv"
    frontier_path = args.out_dir / f"{args.prefix}_pareto_frontier.csv"
    frontier.to_csv(scores_path, index=False)
    frontier[frontier["posterior_pareto_frontier"]].to_csv(frontier_path, index=False)
    print(f"Wrote {scores_path}")
    print(f"Wrote {frontier_path}")


if __name__ == "__main__":
    main()
