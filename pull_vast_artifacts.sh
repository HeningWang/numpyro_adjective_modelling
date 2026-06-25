#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/numpyro_adjective_modelling}"
AREAS="${AREAS:-slider production}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"
PRINT_ONLY="${PRINT_ONLY:-0}"
CHECK_ARTIFACTS="${CHECK_ARTIFACTS:-1}"
ARTIFACT_TASKS="${ARTIFACT_TASKS:-all}"
CHECK_FAIL_INCOMPLETE="${CHECK_FAIL_INCOMPLETE:-auto}"
ARTIFACT_STATUS_CSV="${ARTIFACT_STATUS_CSV:-analysis/results_model_selection/stats/vast_artifact_status.csv}"
ARTIFACT_TAG="${ARTIFACT_TAG:-tm}"
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
RSYNC="${RSYNC:-rsync}"
SSH_PORT="${SSH_PORT:-}"
SSH_IDENTITY_FILE="${SSH_IDENTITY_FILE:-}"
SSH_EXTRA="${SSH_EXTRA:-}"

export ARTIFACT_TASKS
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

if [[ -z "${REMOTE}" ]]; then
  echo "REMOTE is required, e.g. REMOTE=root@ssh5.vast.ai SSH_PORT=12345." >&2
  exit 2
fi

ssh_cmd=(ssh)
if [[ -n "${SSH_PORT}" ]]; then
  ssh_cmd+=(-p "${SSH_PORT}")
fi
if [[ -n "${SSH_IDENTITY_FILE}" ]]; then
  ssh_cmd+=(-i "${SSH_IDENTITY_FILE}")
fi
if [[ -n "${SSH_EXTRA}" ]]; then
  # shellcheck disable=SC2206
  extra_parts=(${SSH_EXTRA})
  ssh_cmd+=("${extra_parts[@]}")
fi

run_rsync() {
  local area="$1"
  local remote_dir="${REMOTE_REPO}/models/${area}/inference_data/"
  local local_dir="models/${area}/inference_data/"
  mkdir -p "${local_dir}"

  local cmd=(
    "${RSYNC}"
    -avz
    --progress
    --ignore-existing
    --include="*.nc"
    --exclude="*"
    -e "${ssh_cmd[*]}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    cmd+=(--dry-run)
  fi
  cmd+=("${REMOTE}:${remote_dir}" "${local_dir}")

  echo ""
  echo "Pulling ${area} artifacts"
  echo "  remote: ${REMOTE}:${remote_dir}"
  echo "  local : ${local_dir}"
  printf "Command:"
  printf " %q" "${cmd[@]}"
  printf "\n"
  if [[ "${PRINT_ONLY}" == "1" ]]; then
    return
  fi
  "${cmd[@]}"
}

echo "Vast artifact pull"
echo "  remote     : ${REMOTE}"
echo "  remote repo: ${REMOTE_REPO}"
echo "  areas      : ${AREAS}"
echo "  python     : ${PYTHON_BIN}"
echo "  dry run    : ${DRY_RUN}"
echo "  print only : ${PRINT_ONLY}"
echo "  check files: ${CHECK_ARTIFACTS}"
echo "  fail incomplete: ${CHECK_FAIL_INCOMPLETE}"
echo "  status csv : ${ARTIFACT_STATUS_CSV}"
echo "  artifact tag: ${ARTIFACT_TAG}"
echo "  artifact tasks: ${ARTIFACT_TASKS}"
echo "  slider full: ${SLIDER_FULL_SPEAKERS}"
echo "  slider heldout: ${SLIDER_HELDOUT_SPEAKERS}"
echo "  production: ${PRODUCTION_2X2_SPEAKERS}"
echo "  slider draws: ${SLIDER_WARMUP}/${SLIDER_SAMPLES}/${SLIDER_CHAINS}"
echo "  slider folds: ${SLIDER_NUM_FOLDS} seed ${SLIDER_FOLD_SEED}"
echo "  prod draws: ${PRODUCTION_WARMUP}/${PRODUCTION_SAMPLES}/${PRODUCTION_CHAINS}"
echo "  prod subset: ${PRODUCTION_CONDITION_SUBSET}"
echo "  prod encoding: ${PRODUCTION_STATE_ENCODING}"
if [[ -n "${SSH_PORT}" ]]; then
  echo "  ssh port   : ${SSH_PORT}"
fi
if [[ -n "${SSH_IDENTITY_FILE}" ]]; then
  echo "  ssh key    : ${SSH_IDENTITY_FILE}"
fi

read -r -a area_array <<< "${AREAS}"
for area in "${area_array[@]}"; do
  case "${area}" in
    slider|production)
      run_rsync "${area}"
      ;;
    *)
      echo "Unknown area '${area}'. Known areas: slider production" >&2
      exit 2
      ;;
  esac
done

echo ""
echo "Artifact pull complete."

if [[ "${CHECK_ARTIFACTS}" == "1" ]]; then
  checker_cmd=("${PYTHON_BIN}" check_vast_artifacts.py --tasks "${ARTIFACT_TASKS}" --artifact-tag "${ARTIFACT_TAG}")
  if [[ -n "${ARTIFACT_STATUS_CSV}" ]]; then
    checker_cmd+=(--csv "${ARTIFACT_STATUS_CSV}")
  fi
  case "${CHECK_FAIL_INCOMPLETE}" in
    1|true)
      checker_cmd+=(--fail-incomplete)
      ;;
    0|false)
      ;;
    auto)
      if [[ "${DRY_RUN}" != "1" && "${PRINT_ONLY}" != "1" ]]; then
        checker_cmd+=(--fail-incomplete)
      fi
      ;;
    *)
      echo "Unknown CHECK_FAIL_INCOMPLETE='${CHECK_FAIL_INCOMPLETE}'. Use auto, 1, or 0." >&2
      exit 2
      ;;
  esac
  printf "Artifact check:"
  printf " %q" "${checker_cmd[@]}"
  printf "\n"
  if [[ "${PRINT_ONLY}" != "1" ]]; then
    "${checker_cmd[@]}"
  fi
fi
