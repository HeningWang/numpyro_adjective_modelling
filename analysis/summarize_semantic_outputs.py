"""Save manuscript-facing arithmetic summaries from completed analysis exports."""
from pathlib import Path
import argparse
import pandas as pd


def main(data):
    rows=[]
    for name, scope in [("simulation_cell_summary.csv", "Individual grid cells"),
                        ("simulation_size_first_advantage_summary.csv", "Parameter-averaged context groups")]:
        cells=pd.read_csv(data/name)
        for (speaker,semantics), group in cells.groupby(["speaker","semantics"]):
            v=group.mean_advantage
            rows.append(dict(scope=scope,speaker=speaker,semantics=semantics,groups=len(v),
                minimum_mean=v.min(),maximum_mean=v.max(),negative_means_below_tolerance=int((v < -1e-10).sum()),
                positive_means_above_tolerance=int((v > 1e-10).sum())))
    pd.DataFrame(rows).to_csv(data/"simulation_direction_summary.csv",index=False)
    architecture=pd.read_csv(data/"production_architecture_elpd_localization.csv")
    initial=architecture.loc[architecture.grouping_type.eq("initial_adjective")].copy()
    initial["percentage_of_total"]=100*initial.total_plan_vs_average_endpoints/initial.total_plan_vs_average_endpoints.sum()
    initial.to_csv(data/"production_architecture_initial_gain_summary.csv",index=False)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True)
    main(p.parse_args().data)
