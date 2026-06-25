#!/usr/bin/env python3
"""Check expected Vast inference artifacts for the current model-comparison queue."""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


SLIDER_FULL_SPEAKERS = (
    "incremental",
    "incremental_static",
    "planned_usefulness_order",
    "planned_usefulness_order_static",
    "planned_usefulness_signed_order",
    "planned_usefulness_signed_order_static",
    "planned_usefulness_mixture",
    "planned_usefulness_mixture_static",
)
SLIDER_ABLATION_SPEAKERS = (
    "planned_usefulness_signed_order",
    "planned_usefulness_signed_order_static",
    "planned_usefulness_mixture",
    "planned_usefulness_mixture_static",
)
PRODUCTION_2X2_SPEAKERS = (
    "contextual_pcalpha_canon_parsimony_2x2_inc_rec",
    "contextual_pcalpha_canon_parsimony_2x2_inc_static",
    "contextual_pcalpha_canon_parsimony_2x2_glob_rec",
    "contextual_pcalpha_canon_parsimony_2x2_glob_static",
)


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


@dataclass(frozen=True)
class ExpectedArtifact:
    group: str
    model: str
    relative_path: Path
    fold: str = ""
    expected_attrs: tuple[tuple[str, str], ...] = ()


def parse_items(value: str) -> list[str]:
    return [item for item in value.replace(",", " ").split() if item]


def subset_tag(condition_subset: str) -> str:
    stems = []
    for code in parse_items(condition_subset):
        if len(code) >= 4:
            stems.append(code[2:4])
    if not stems:
        return ""
    return "_" + "".join(sorted(set(stems)))


def top_tag(min_proportion: str) -> str:
    return "" if min_proportion in {"", "0", "0.0", "0.00"} else "_top"


def artifact_tag(value: str) -> str:
    return f"_{value}" if value else ""


def attr_pairs(**kwargs) -> tuple[tuple[str, str], ...]:
    return tuple((key, "" if value is None else str(value)) for key, value in kwargs.items())


def expand_tasks(tasks: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for task in tasks:
        if task == "all":
            expanded.extend(["slider_full", "slider_heldout", "production_2x2"])
        elif task == "queue":
            expanded.extend(["slider_full", "slider_heldout", "production_2x2"])
        else:
            expanded.append(task)
    return expanded


def slider_full_artifacts(args: argparse.Namespace) -> list[ExpectedArtifact]:
    artifacts = []
    suffix = (
        f"{artifact_tag(args.artifact_tag)}"
        f"_warmup{args.slider_warmup}_samples{args.slider_samples}_chains{args.slider_chains}.nc"
    )
    for speaker in parse_items(args.slider_full_speakers):
        artifacts.append(
            ExpectedArtifact(
                group="slider_full",
                model=speaker,
                relative_path=Path("models/slider/inference_data")
                / f"mcmc_results_{speaker}_speaker_hier{suffix}",
                expected_attrs=attr_pairs(
                    dataset="slider",
                    run_kind="hierarchical",
                    canonical_speaker_type=speaker,
                    artifact_tag=args.artifact_tag,
                    num_warmup=args.slider_warmup,
                    num_samples=args.slider_samples,
                    num_chains=args.slider_chains,
                ),
            )
        )
    return artifacts


def slider_ablation_artifacts(args: argparse.Namespace) -> list[ExpectedArtifact]:
    artifacts = []
    suffix = (
        f"{artifact_tag(args.artifact_tag)}"
        f"_warmup{args.slider_warmup}_samples{args.slider_samples}_chains{args.slider_chains}.nc"
    )
    for speaker in parse_items(args.slider_ablation_speakers):
        artifacts.append(
            ExpectedArtifact(
                group="slider_ablation",
                model=speaker,
                relative_path=Path("models/slider/inference_data")
                / f"mcmc_results_{speaker}_speaker_hier{suffix}",
                expected_attrs=attr_pairs(
                    dataset="slider",
                    run_kind="hierarchical",
                    canonical_speaker_type=speaker,
                    artifact_tag=args.artifact_tag,
                    num_warmup=args.slider_warmup,
                    num_samples=args.slider_samples,
                    num_chains=args.slider_chains,
                ),
            )
        )
    return artifacts


def slider_heldout_artifacts(args: argparse.Namespace) -> list[ExpectedArtifact]:
    artifacts = []
    suffix = (
        f"{artifact_tag(args.artifact_tag)}"
        f"_warmup{args.slider_warmup}_samples{args.slider_samples}_chains{args.slider_chains}.nc"
    )
    for speaker in parse_items(args.slider_heldout_speakers):
        for fold in range(args.slider_num_folds):
            artifacts.append(
                ExpectedArtifact(
                    group="slider_heldout",
                    model=speaker,
                    fold=str(fold),
                    relative_path=Path("models/slider/inference_data")
                    / f"mcmc_results_{speaker}_speaker_hier_fold{fold}of{args.slider_num_folds}{suffix}",
                    expected_attrs=attr_pairs(
                        dataset="slider",
                        run_kind="hierarchical",
                        canonical_speaker_type=speaker,
                        artifact_tag=args.artifact_tag,
                        num_warmup=args.slider_warmup,
                        num_samples=args.slider_samples,
                        num_chains=args.slider_chains,
                        heldout_fold=fold,
                        num_folds=args.slider_num_folds,
                        fold_seed=args.slider_fold_seed,
                    ),
                )
            )
    return artifacts


def production_2x2_artifacts(args: argparse.Namespace) -> list[ExpectedArtifact]:
    artifacts = []
    tag = f"{top_tag(args.production_min_proportion)}{subset_tag(args.production_condition_subset)}"
    suffix = (
        f"{artifact_tag(args.artifact_tag)}"
        f"_warmup{args.production_warmup}"
        f"_samples{args.production_samples}"
        f"_chains{args.production_chains}.nc"
    )
    for speaker in parse_items(args.production_speakers):
        artifacts.append(
            ExpectedArtifact(
                group="production_2x2",
                model=speaker,
                relative_path=Path("models/production/inference_data")
                / f"mcmc_results_{speaker}_speaker_hier{tag}{suffix}",
                expected_attrs=attr_pairs(
                    dataset="production",
                    run_kind="hierarchical",
                    canonical_speaker_type=speaker,
                    artifact_tag=args.artifact_tag,
                    num_warmup=args.production_warmup,
                    num_samples=args.production_samples,
                    num_chains=args.production_chains,
                    state_encoding=args.production_state_encoding,
                    condition_subset=args.production_condition_subset,
                    min_proportion=str(float(args.production_min_proportion)),
                ),
            )
        )
    return artifacts


def collect_expected(args: argparse.Namespace) -> list[ExpectedArtifact]:
    builders = {
        "slider_full": slider_full_artifacts,
        "slider_ablation": slider_ablation_artifacts,
        "slider_heldout": slider_heldout_artifacts,
        "production_2x2": production_2x2_artifacts,
    }
    artifacts: list[ExpectedArtifact] = []
    for task in expand_tasks(parse_items(args.tasks)):
        try:
            artifacts.extend(builders[task](args))
        except KeyError as exc:
            known = ", ".join(["all", "queue", *builders])
            raise SystemExit(f"Unknown task '{task}'. Known tasks: {known}") from exc
    seen = set()
    unique = []
    for artifact in artifacts:
        key = (artifact.group, artifact.model, artifact.fold, artifact.relative_path)
        if key not in seen:
            unique.append(artifact)
            seen.add(key)
    return unique


def metadata_mismatches(path: Path, artifact: ExpectedArtifact) -> list[str]:
    if not artifact.expected_attrs:
        return []
    try:
        import xarray as xr

        dataset = xr.open_dataset(path, group="posterior")
        attrs = {key: str(value) for key, value in dataset.attrs.items()}
        dataset.close()
    except Exception as exc:
        return [f"metadata_read_error={type(exc).__name__}: {exc}"]

    mismatches = []
    for key, expected in artifact.expected_attrs:
        observed = attrs.get(key)
        if observed != expected:
            mismatches.append(f"{key}: expected {expected!r}, observed {observed!r}")
    return mismatches


def status_row(
    repo_root: Path,
    artifact: ExpectedArtifact,
    min_size_bytes: int,
    check_metadata: bool,
) -> dict[str, str | int]:
    path = repo_root / artifact.relative_path
    mismatches: list[str] = []
    if not path.exists():
        status = "missing"
        size_bytes: int | str = ""
        modified = ""
    else:
        stat = path.stat()
        size_bytes = stat.st_size
        modified = datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")
        status = "present" if stat.st_size >= min_size_bytes else "too_small"
        if status == "present" and check_metadata:
            mismatches = metadata_mismatches(path, artifact)
            if mismatches:
                status = "metadata_mismatch"
    return {
        "group": artifact.group,
        "model": artifact.model,
        "fold": artifact.fold,
        "status": status,
        "size_bytes": size_bytes,
        "modified": modified,
        "metadata_mismatches": " | ".join(mismatches),
        "path": str(artifact.relative_path),
    }


def write_csv(path: Path, rows: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group",
                "model",
                "fold",
                "status",
                "size_bytes",
                "modified",
                "metadata_mismatches",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str | int]], max_missing: int) -> None:
    total = len(rows)
    by_status = Counter(str(row["status"]) for row in rows)
    print("Vast artifact status")
    print(f"  expected : {total}")
    print(f"  present  : {by_status.get('present', 0)}")
    print(f"  missing  : {by_status.get('missing', 0)}")
    print(f"  too small: {by_status.get('too_small', 0)}")
    print(f"  metadata : {by_status.get('metadata_mismatch', 0)} mismatch")
    print("")
    print("By group")
    for group in sorted({str(row["group"]) for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        counts = Counter(str(row["status"]) for row in group_rows)
        print(
            f"  {group}: present={counts.get('present', 0)} "
            f"missing={counts.get('missing', 0)} "
            f"too_small={counts.get('too_small', 0)} "
            f"metadata_mismatch={counts.get('metadata_mismatch', 0)}"
        )

    incomplete = [row for row in rows if row["status"] != "present"]
    if incomplete:
        print("")
        print("Incomplete artifacts")
        for row in incomplete[:max_missing]:
            fold = f" fold={row['fold']}" if row["fold"] != "" else ""
            print(f"  [{row['status']}] {row['group']} {row['model']}{fold}: {row['path']}")
            if row.get("metadata_mismatches"):
                print(f"    {row['metadata_mismatches']}")
        if len(incomplete) > max_missing:
            print(f"  ... {len(incomplete) - max_missing} more")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--tasks",
        default=env_default("ARTIFACT_TASKS", env_default("TASKS", "all")),
        help="Tasks to check: all, queue, slider_full, slider_ablation, slider_heldout, production_2x2.",
    )
    parser.add_argument(
        "--artifact-tag",
        default=env_default("ARTIFACT_TAG", "tm"),
        help="Optional artifact tag inserted before warmup/sample/chains in filenames.",
    )
    slider_default_speakers = env_default("SLIDER_FULL_SPEAKERS", " ".join(SLIDER_FULL_SPEAKERS))
    parser.add_argument("--slider-full-speakers", default=slider_default_speakers)
    parser.add_argument(
        "--slider-ablation-speakers",
        default=env_default("SLIDER_ABLATION_SPEAKERS", " ".join(SLIDER_ABLATION_SPEAKERS)),
    )
    parser.add_argument(
        "--slider-heldout-speakers",
        default=env_default("SLIDER_HELDOUT_SPEAKERS", slider_default_speakers),
    )
    parser.add_argument("--slider-warmup", type=int, default=env_int("SLIDER_WARMUP", env_int("NUM_WARMUP", 500)))
    parser.add_argument("--slider-samples", type=int, default=env_int("SLIDER_SAMPLES", env_int("NUM_SAMPLES", 500)))
    parser.add_argument("--slider-chains", type=int, default=env_int("SLIDER_CHAINS", env_int("NUM_CHAINS", 4)))
    parser.add_argument("--slider-num-folds", type=int, default=env_int("SLIDER_NUM_FOLDS", env_int("NUM_FOLDS", 5)))
    parser.add_argument("--slider-fold-seed", type=int, default=env_int("SLIDER_FOLD_SEED", env_int("FOLD_SEED", 13)))
    parser.add_argument(
        "--production-speakers",
        default=env_default("PRODUCTION_2X2_SPEAKERS", " ".join(PRODUCTION_2X2_SPEAKERS)),
    )
    parser.add_argument("--production-warmup", type=int, default=env_int("PRODUCTION_WARMUP", env_int("NUM_WARMUP", 4000)))
    parser.add_argument("--production-samples", type=int, default=env_int("PRODUCTION_SAMPLES", env_int("NUM_SAMPLES", 2000)))
    parser.add_argument("--production-chains", type=int, default=env_int("PRODUCTION_CHAINS", env_int("NUM_CHAINS", 4)))
    parser.add_argument(
        "--production-condition-subset",
        default=env_default("PRODUCTION_CONDITION_SUBSET", env_default("CONDITION_SUBSET", "erdc,zrdc,brdc")),
    )
    parser.add_argument(
        "--production-state-encoding",
        default=env_default("PRODUCTION_STATE_ENCODING", env_default("STATE_ENCODING", "target_match")),
    )
    parser.add_argument(
        "--production-min-proportion",
        default=env_default("PRODUCTION_MIN_PROPORTION", env_default("MIN_PROPORTION", "0.0")),
    )
    parser.add_argument("--min-size-bytes", type=int, default=env_int("ARTIFACT_MIN_SIZE_BYTES", 1024))
    parser.add_argument("--csv", type=Path, default=None, help="Optional path for a compact status CSV.")
    parser.add_argument("--max-missing", type=int, default=40)
    parser.add_argument(
        "--no-check-metadata",
        action="store_true",
        help="Only check file presence/size; skip NetCDF metadata validation.",
    )
    parser.add_argument("--fail-incomplete", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    rows = [
        status_row(
            repo_root,
            artifact,
            args.min_size_bytes,
            check_metadata=not args.no_check_metadata,
        )
        for artifact in collect_expected(args)
    ]
    print_summary(rows, args.max_missing)
    if args.csv is not None:
        write_csv(args.csv, rows)
        print("")
        print(f"Wrote {args.csv}")
    if args.fail_incomplete and any(row["status"] != "present" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
