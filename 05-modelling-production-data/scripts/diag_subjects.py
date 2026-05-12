"""Inspect dc-only data for the worst-mixing subjects."""

import sys
from pathlib import Path
from collections import Counter

import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "01-dataset" / "01-production-data-preprocessed.csv"

SUBJECTS_OF_INTEREST = [30, 15, 5, 47, 10, 53, 54, 69, 87, 88, 50]
SUBSET = {"erdc", "zrdc", "brdc"}


def main() -> int:
    df = pd.read_csv(DATA)
    print(f"Full dataset: {len(df)} rows, columns: {list(df.columns)[:8]}...")

    cond_col = next(c for c in df.columns if "condition" in c.lower())
    print(f"Condition column: {cond_col!r}")

    df_dc = df[df[cond_col].isin(SUBSET)].copy()
    print(f"DC subset: {len(df_dc)} rows")

    pid_col = next(c for c in df.columns if c.lower() in ("workerid", "participant", "subject", "id"))
    print(f"Participant column: {pid_col!r}")

    pids = sorted(df_dc[pid_col].unique())
    print(f"DC subset participants: {len(pids)}")

    pid_to_idx = {p: i for i, p in enumerate(pids)}

    resp_col = next(c for c in df.columns if c.lower() in ("response", "responses", "annotation", "answer"))
    print(f"Response column: {resp_col!r}\n")

    print(f"{'idx':>4}  {'pid':>12}  {'n_trials':>8}  {'condition_breakdown':<35}  {'top_responses'}")
    for s in SUBJECTS_OF_INTEREST:
        if s >= len(pids):
            print(f"  {s} out of range (only {len(pids)} subjects)")
            continue
        pid = pids[s]
        rows = df_dc[df_dc[pid_col] == pid]
        n = len(rows)
        conds = Counter(rows[cond_col])
        resps = Counter(rows[resp_col])
        cond_str = ", ".join(f"{k}={v}" for k, v in conds.most_common())
        resp_str = ", ".join(f"{k}={v}" for k, v in resps.most_common(5))
        print(f"  {s:>4}  {pid!s:>12}  {n:>8}  {cond_str:<35}  {resp_str}")

    print(f"\n=== Overall n_trials distribution across all {len(pids)} subjects ===")
    n_trials = df_dc.groupby(pid_col).size()
    print(f"  min={n_trials.min()}  max={n_trials.max()}  median={int(n_trials.median())}  mean={n_trials.mean():.1f}")
    print(f"  subjects with <20 trials: {(n_trials < 20).sum()}")
    print(f"  subjects with <30 trials: {(n_trials < 30).sum()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
