#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
ARTIFACT_TAG="${ARTIFACT_TAG:-tm}"
CHECK_ARTIFACTS="${CHECK_ARTIFACTS:-1}"
REQUIRE_CLEAN_GIT="${REQUIRE_CLEAN_GIT:-1}"
REQUIRE_GPU="${REQUIRE_GPU:-auto}"
MIN_FREE_GB="${MIN_FREE_GB:-10}"
TASKS="${TASKS:-slider_full slider_heldout production_2x2}"
EXPECTED_BRANCH="${EXPECTED_BRANCH:-}"
CHECK_NETCDF_WRITE="${CHECK_NETCDF_WRITE:-auto}"
PREFLIGHT_PYTHON_HOME="${PREFLIGHT_PYTHON_HOME:-${TMPDIR:-/tmp}/vast-preflight-home}"
SLIDER_FULL_SPEAKERS="${SLIDER_FULL_SPEAKERS:-incremental incremental_static planned_usefulness_order planned_usefulness_order_static planned_usefulness_signed_order planned_usefulness_signed_order_static planned_usefulness_mixture planned_usefulness_mixture_static}"
SLIDER_ABLATION_SPEAKERS="${SLIDER_ABLATION_SPEAKERS:-planned_usefulness_signed_order planned_usefulness_signed_order_static planned_usefulness_mixture planned_usefulness_mixture_static}"
SLIDER_HELDOUT_SPEAKERS="${SLIDER_HELDOUT_SPEAKERS:-${SLIDER_FULL_SPEAKERS}}"
PRODUCTION_2X2_SPEAKERS="${PRODUCTION_2X2_SPEAKERS:-contextual_pcalpha_canon_parsimony_2x2_inc_rec contextual_pcalpha_canon_parsimony_2x2_inc_static contextual_pcalpha_canon_parsimony_2x2_glob_rec contextual_pcalpha_canon_parsimony_2x2_glob_static}"
SLIDER_WARMUP="${SLIDER_WARMUP:-${NUM_WARMUP:-500}}"
SLIDER_SAMPLES="${SLIDER_SAMPLES:-${NUM_SAMPLES:-500}}"
SLIDER_CHAINS="${SLIDER_CHAINS:-${NUM_CHAINS:-4}}"
SLIDER_NUM_FOLDS="${SLIDER_NUM_FOLDS:-${NUM_FOLDS:-5}}"
SLIDER_FOLD_SEED="${SLIDER_FOLD_SEED:-${FOLD_SEED:-13}}"
PRODUCTION_WARMUP="${PRODUCTION_WARMUP:-${NUM_WARMUP:-4000}}"
PRODUCTION_SAMPLES="${PRODUCTION_SAMPLES:-${NUM_SAMPLES:-2000}}"
PRODUCTION_CHAINS="${PRODUCTION_CHAINS:-${NUM_CHAINS:-4}}"
PRODUCTION_CONDITION_SUBSET="${PRODUCTION_CONDITION_SUBSET:-${CONDITION_SUBSET:-erdc,zrdc,brdc}}"
PRODUCTION_MIN_PROPORTION="${PRODUCTION_MIN_PROPORTION:-${MIN_PROPORTION:-0.0}}"
PRODUCTION_STATE_ENCODING="${PRODUCTION_STATE_ENCODING:-${STATE_ENCODING:-target_match}}"

: "${JAX_PLATFORMS:=cuda}"
: "${XLA_FLAGS:=}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
: "${XDG_CACHE_HOME:=${TMPDIR:-/tmp}/vast-preflight-cache}"
: "${MPLCONFIGDIR:=${TMPDIR:-/tmp}/vast-mpl-cache}"

export JAX_PLATFORMS
export XLA_FLAGS
export XLA_PYTHON_CLIENT_PREALLOCATE
export XDG_CACHE_HOME
export MPLCONFIGDIR
export TASKS
export ARTIFACT_TAG
export SLIDER_FULL_SPEAKERS
export SLIDER_ABLATION_SPEAKERS
export SLIDER_HELDOUT_SPEAKERS
export PRODUCTION_2X2_SPEAKERS
export SLIDER_WARMUP
export SLIDER_SAMPLES
export SLIDER_CHAINS
export SLIDER_NUM_FOLDS
export SLIDER_FOLD_SEED
export PRODUCTION_WARMUP
export PRODUCTION_SAMPLES
export PRODUCTION_CHAINS
export PRODUCTION_CONDITION_SUBSET
export PRODUCTION_MIN_PROPORTION
export PRODUCTION_STATE_ENCODING

cd "$(dirname "$0")"
mkdir -p "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}" "${PREFLIGHT_PYTHON_HOME}"

echo "Vast preflight"
echo "  repo             : $(pwd)"
echo "  python           : ${PYTHON_BIN}"
echo "  dry run          : ${DRY_RUN}"
echo "  tasks            : ${TASKS}"
echo "  artifact tag     : ${ARTIFACT_TAG}"
echo "  slider full      : ${SLIDER_FULL_SPEAKERS}"
echo "  slider heldout   : ${SLIDER_HELDOUT_SPEAKERS}"
echo "  production       : ${PRODUCTION_2X2_SPEAKERS}"
echo "  slider draws     : ${SLIDER_WARMUP}/${SLIDER_SAMPLES}/${SLIDER_CHAINS}"
echo "  slider folds     : ${SLIDER_NUM_FOLDS} seed ${SLIDER_FOLD_SEED}"
echo "  prod draws       : ${PRODUCTION_WARMUP}/${PRODUCTION_SAMPLES}/${PRODUCTION_CHAINS}"
echo "  prod subset      : ${PRODUCTION_CONDITION_SUBSET}"
echo "  prod encoding    : ${PRODUCTION_STATE_ENCODING}"
echo "  require clean git: ${REQUIRE_CLEAN_GIT}"
echo "  require gpu      : ${REQUIRE_GPU}"
echo "  check netcdf     : ${CHECK_NETCDF_WRITE}"
echo "  min free GB      : ${MIN_FREE_GB}"
echo "  JAX_PLATFORMS    : ${JAX_PLATFORMS}"
echo "  XLA_FLAGS        : ${XLA_FLAGS}"
echo "  preallocate      : ${XLA_PYTHON_CLIENT_PREALLOCATE}"
echo "  XDG_CACHE_HOME   : ${XDG_CACHE_HOME}"
echo "  MPLCONFIGDIR     : ${MPLCONFIGDIR}"
echo "  python HOME      : ${PREFLIGHT_PYTHON_HOME}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is not available." >&2
  exit 2
fi

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "This directory is not a git checkout." >&2
  exit 2
fi

commit="$(git rev-parse --short HEAD)"
branch="$(git branch --show-current || true)"
echo "  git commit       : ${commit}"
echo "  git branch       : ${branch:-detached}"

if [[ -n "${EXPECTED_BRANCH}" && "${branch}" != "${EXPECTED_BRANCH}" ]]; then
  echo "Expected branch '${EXPECTED_BRANCH}', found '${branch:-detached}'." >&2
  exit 2
fi

if [[ "${DRY_RUN}" != "1" && "${REQUIRE_CLEAN_GIT}" == "1" ]]; then
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked worktree changes are present; commit/pull clean code before running Vast inference." >&2
    git status --short --untracked-files=no >&2
    exit 2
  fi
fi

free_kb="$(df -Pk . | awk 'NR==2 {print $4}')"
free_gb=$((free_kb / 1024 / 1024))
echo "  free disk GB     : ${free_gb}"
if [[ "${free_gb}" -lt "${MIN_FREE_GB}" ]]; then
  echo "Free disk ${free_gb} GB is below MIN_FREE_GB=${MIN_FREE_GB}." >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo ""
  echo "nvidia-smi"
  nvidia-smi
else
  echo "  nvidia-smi       : not found"
fi

require_gpu_now=0
if [[ "${REQUIRE_GPU}" == "1" || "${REQUIRE_GPU}" == "true" ]]; then
  require_gpu_now=1
elif [[ "${REQUIRE_GPU}" == "auto" && "${DRY_RUN}" != "1" ]]; then
  require_gpu_now=1
fi

if [[ "${require_gpu_now}" == "1" ]]; then
  HOME="${PREFLIGHT_PYTHON_HOME}" "${PYTHON_BIN}" - <<'PY'
import importlib
import jax

print("")
print("Python/JAX environment")
print(f"  jax version      : {jax.__version__}")
for package in ("jaxlib", "numpyro", "arviz", "xarray", "h5py", "netCDF4"):
    try:
        module = importlib.import_module(package)
        print(f"  {package:<15}: {getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"  {package:<15}: unavailable ({type(exc).__name__}: {exc})")

devices = jax.devices()
print(f"  jax devices      : {devices}")
if not any(getattr(device, "platform", "").lower() in {"gpu", "cuda"} for device in devices):
    raise SystemExit("CUDA/GPU device is not visible to JAX.")
PY
fi

check_netcdf_now=0
if [[ "${CHECK_NETCDF_WRITE}" == "1" || "${CHECK_NETCDF_WRITE}" == "true" ]]; then
  check_netcdf_now=1
elif [[ "${CHECK_NETCDF_WRITE}" == "auto" && "${DRY_RUN}" != "1" ]]; then
  check_netcdf_now=1
fi

if [[ "${check_netcdf_now}" == "1" ]]; then
  HOME="${PREFLIGHT_PYTHON_HOME}" "${PYTHON_BIN}" - <<'PY'
import os
import tempfile

import arviz as az
import numpy as np

print("")
print("NetCDF write/read smoke test")
with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as handle:
    path = handle.name
try:
    idata = az.from_dict(posterior={"theta": np.ones((1, 2))})
    idata.to_netcdf(path)
    az.from_netcdf(path)
    print(f"  netcdf smoke     : ok ({path})")
finally:
    try:
        os.remove(path)
    except OSError:
        pass
PY
fi

if [[ "${CHECK_ARTIFACTS}" == "1" ]]; then
  echo ""
  "${PYTHON_BIN}" check_vast_artifacts.py \
    --tasks "${TASKS}" \
    --artifact-tag "${ARTIFACT_TAG}" \
    --max-missing 60
fi

echo ""
echo "Vast preflight complete."
