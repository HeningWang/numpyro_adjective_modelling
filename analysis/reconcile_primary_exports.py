"""Export the corrected fixed best-model diagnostics for existing figure scripts."""
from pathlib import Path
import argparse
import importlib.util
import json
import hashlib
import subprocess
import sys
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    p = ROOT / relative
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main(campaign, out):
    out.mkdir(parents=True, exist_ok=True)
    src = campaign / "summaries"
    participants = load("analysis_revision/18_revised_factorial/export_v18_participant_figure_data.py")
    participants.ARTIFACT = campaign / "primary/K-HKO.nc"
    for attr in ["PARAMETER_OUTPUT", "PREDICTION_OUTPUT", "GAP_OUTPUT", "POPULATION_OUTPUT", "DEPENDENCE_OUTPUT"]:
        setattr(participants, attr, out / getattr(participants, attr).name)
    participants.main()
    hierarchy = load("analysis_revision/18_revised_factorial/export_kappa_hierarchy_diagnostics.py")
    hierarchy.SHARED = hierarchy.JOINT = src / "corrected_inputs/pointwise.csv"
    hierarchy.PARAMETERS = participants.PARAMETER_OUTPUT
    hierarchy.SHARED_MODEL, hierarchy.JOINT_MODEL = "K-HO", "K-HKO"
    with xr.open_dataset(campaign / "primary/K-HO.nc", group="posterior", engine="h5netcdf") as p:
        v = p.kappa.values.ravel()
        hierarchy.SHARED_KAPPA = float(v.mean())
        pd.DataFrame([dict(model="K-HO", kappa_mean=v.mean(), kappa_q025=np.quantile(v,.025),
                           kappa_q975=np.quantile(v,.975))]).to_csv(out/"production_corrected_kappa.csv", index=False)
    for attr in ["PARTICIPANT_OUTPUT", "OUTCOME_OUTPUT", "CORRELATION_OUTPUT"]:
        setattr(hierarchy, attr, out / getattr(hierarchy, attr).name)
    hierarchy.main()
    ppc = pd.read_csv(src / "primary_ppc_summary.csv")
    ppc.loc[ppc.model.eq("K-HKO")].assign(model="Plan-guided").to_csv(out/"production_v18_best_ppc_summary.csv", index=False)
    pd.read_csv(src/"shared_architecture_ppc_summary.csv").to_csv(out/"production_v10_ppc_summary.csv", index=False)
    # Preserve the frozen comparison reference explicitly; it need not be best.
    for name in ["shared_factorial_statistics.csv", "production_architecture_condition_tv_diagnostics.csv",
                 "production_architecture_cell_spread_diagnostics.csv", "production_architecture_elpd_localization.csv"]:
        target = "production_architecture_factorial_statistics_v16.csv" if name == "shared_factorial_statistics.csv" else name
        pd.read_csv(src/name).to_csv(out/target, index=False)
    subprocess.run([sys.executable, str(ROOT/"analysis_revision/14_final_architecture/export_v10_theory_ppc.py"),
        "--data", str(ROOT/"analysis_revision/12_deterministic_encoding/model_input_raw_observed_9100.csv"),
        "--inference", str(campaign/"primary/K-HKO.nc"), "--output", str(out/"production_v18_ppc_theory_outcomes.csv")], check=True)
    receipt = dict(best_model="K-HKO", shared_model="K-HO", complete=True,
        sources={str(Path(m.__file__).relative_to(ROOT)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                 for m in [participants, hierarchy]},
        outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.csv")})
    (out/"export_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--campaign",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    main(a.campaign,a.output)
