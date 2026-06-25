#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS="${TASKS:-slider_full slider_heldout production_2x2}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CHECK_ARTIFACTS="${CHECK_ARTIFACTS:-1}"
ARTIFACT_TAG="${ARTIFACT_TAG:-tm}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
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

export PYTHON_BIN
export JAX_PLATFORMS
export XLA_FLAGS
export XLA_PYTHON_CLIENT_PREALLOCATE
export DRY_RUN
export OVERWRITE
export SKIP_EXISTING
export ARTIFACT_TAG
export CHECK_ARTIFACTS
export TASKS
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

echo "Vast inference queue"
echo "  repo         : $(pwd)"
echo "  python       : ${PYTHON_BIN}"
echo "  tasks        : ${TASKS}"
echo "  dry run      : ${DRY_RUN}"
echo "  overwrite    : ${OVERWRITE}"
echo "  skip existing: ${SKIP_EXISTING}"
echo "  check files  : ${CHECK_ARTIFACTS}"
echo "  preflight    : ${RUN_PREFLIGHT}"
echo "  artifact tag : ${ARTIFACT_TAG}"
echo "  slider full  : ${SLIDER_FULL_SPEAKERS}"
echo "  slider heldout: ${SLIDER_HELDOUT_SPEAKERS}"
echo "  production   : ${PRODUCTION_2X2_SPEAKERS}"
echo "  slider draws : ${SLIDER_WARMUP}/${SLIDER_SAMPLES}/${SLIDER_CHAINS}"
echo "  slider folds : ${SLIDER_NUM_FOLDS} seed ${SLIDER_FOLD_SEED}"
echo "  prod draws   : ${PRODUCTION_WARMUP}/${PRODUCTION_SAMPLES}/${PRODUCTION_CHAINS}"
echo "  prod subset  : ${PRODUCTION_CONDITION_SUBSET}"
echo "  prod encoding: ${PRODUCTION_STATE_ENCODING}"
echo "  JAX_PLATFORMS: ${JAX_PLATFORMS}"
echo "  XLA_FLAGS    : ${XLA_FLAGS}"
echo "  preallocate  : ${XLA_PYTHON_CLIENT_PREALLOCATE}"
if git rev-parse --short HEAD >/dev/null 2>&1; then
  echo "  git commit   : $(git rev-parse --short HEAD)"
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  ./run_vast_preflight.sh
elif [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import jax

devices = jax.devices()
print(f"  jax devices  : {devices}")
if not any(getattr(device, "platform", "").lower() in {"gpu", "cuda"} for device in devices):
    raise SystemExit("CUDA/GPU device is not visible; aborting inference queue before MCMC.")
PY
fi

run_task() {
  local task="$1"
  local start_time
  local end_time

  echo ""
  echo "############################################################"
  echo "### TASK ${task} START $(date)"
  echo "############################################################"
  start_time=$(date +%s)

  case "${task}" in
    slider_full)
      (
        cd models/slider
        NUM_WARMUP="${SLIDER_WARMUP}" \
        NUM_SAMPLES="${SLIDER_SAMPLES}" \
        NUM_CHAINS="${SLIDER_CHAINS}" \
        SPEAKERS="${SLIDER_FULL_SPEAKERS}" \
          ./run_speaker_ablation_pilot.sh
      )
      ;;
    slider_ablation)
      (
        cd models/slider
        NUM_WARMUP="${SLIDER_WARMUP}" \
        NUM_SAMPLES="${SLIDER_SAMPLES}" \
        NUM_CHAINS="${SLIDER_CHAINS}" \
        SPEAKERS="${SLIDER_ABLATION_SPEAKERS}" \
          ./run_speaker_ablation_pilot.sh
      )
      ;;
    slider_heldout)
      (
        cd models/slider
        NUM_WARMUP="${SLIDER_WARMUP}" \
        NUM_SAMPLES="${SLIDER_SAMPLES}" \
        NUM_CHAINS="${SLIDER_CHAINS}" \
        NUM_FOLDS="${SLIDER_NUM_FOLDS}" \
        FOLD_SEED="${SLIDER_FOLD_SEED}" \
        SPEAKERS="${SLIDER_HELDOUT_SPEAKERS}" \
          ./run_heldout_pilot.sh
      )
      ;;
    production_2x2)
      (
        cd models/production
        NUM_WARMUP="${PRODUCTION_WARMUP}" \
        NUM_SAMPLES="${PRODUCTION_SAMPLES}" \
        NUM_CHAINS="${PRODUCTION_CHAINS}" \
        CONDITION_SUBSET="${PRODUCTION_CONDITION_SUBSET}" \
        MIN_PROPORTION="${PRODUCTION_MIN_PROPORTION}" \
        STATE_ENCODING="${PRODUCTION_STATE_ENCODING}" \
        SPEAKERS="${PRODUCTION_2X2_SPEAKERS}" \
          ./run_final_2x2_gpu.sh
      )
      ;;
    *)
      echo "Unknown task '${task}'." >&2
      echo "Known tasks: slider_full slider_ablation slider_heldout production_2x2" >&2
      exit 2
      ;;
  esac

  end_time=$(date +%s)
  echo "############################################################"
  echo "### TASK ${task} DONE in $((end_time - start_time))s $(date)"
  echo "############################################################"
}

queue_start=$(date +%s)
read -r -a task_array <<< "${TASKS}"
for task in "${task_array[@]}"; do
  run_task "${task}"
done
queue_end=$(date +%s)

echo ""
echo "Vast inference queue complete in $((queue_end - queue_start))s."

if [[ "${CHECK_ARTIFACTS}" == "1" ]]; then
  checker_args=(check_vast_artifacts.py --tasks "${TASKS}" --artifact-tag "${ARTIFACT_TAG}")
  echo ""
  "${PYTHON_BIN}" "${checker_args[@]}"
fi
