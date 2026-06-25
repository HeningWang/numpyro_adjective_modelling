#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
TASKS="${TASKS:-slider_posterior slider_decision slider_heldout production_posterior production_pareto_frontier production_architecture_audit final_decision}"
DRY_RUN="${DRY_RUN:-0}"
CHECK_ARTIFACTS="${CHECK_ARTIFACTS:-1}"
ARTIFACT_TAG="${ARTIFACT_TAG:-tm}"

SLIDER_MODELS="${SLIDER_MODELS:-incremental_recursive,incremental_static,planned_usefulness_order,planned_usefulness_order_static,planned_usefulness_signed_order,planned_usefulness_signed_order_static,planned_usefulness_mixture,planned_usefulness_mixture_static}"
SLIDER_ARTIFACT_SPEAKERS="${SLIDER_ARTIFACT_SPEAKERS:-incremental incremental_static planned_usefulness_order planned_usefulness_order_static planned_usefulness_signed_order planned_usefulness_signed_order_static planned_usefulness_mixture planned_usefulness_mixture_static}"
SLIDER_WARMUP="${SLIDER_WARMUP:-500}"
SLIDER_SAMPLES="${SLIDER_SAMPLES:-500}"
SLIDER_CHAINS="${SLIDER_CHAINS:-4}"
SLIDER_NUM_FOLDS="${SLIDER_NUM_FOLDS:-5}"
SLIDER_FOLD_SEED="${SLIDER_FOLD_SEED:-13}"
SLIDER_POSTERIOR_OUT_DIR="${SLIDER_POSTERIOR_OUT_DIR:-results_speaker_ablation_pilot}"
SLIDER_HELDOUT_STATS_DIR="${SLIDER_HELDOUT_STATS_DIR:-results_heldout_pilot/stats}"

PRODUCTION_MODELS="${PRODUCTION_MODELS:-contextual_pcalpha_canon_parsimony_2x2_inc_rec,contextual_pcalpha_canon_parsimony_2x2_inc_static,contextual_pcalpha_canon_parsimony_2x2_glob_rec,contextual_pcalpha_canon_parsimony_2x2_glob_static}"
PRODUCTION_WARMUP="${PRODUCTION_WARMUP:-4000}"
PRODUCTION_SAMPLES="${PRODUCTION_SAMPLES:-2000}"
PRODUCTION_CHAINS="${PRODUCTION_CHAINS:-4}"
PRODUCTION_CONDITION_SUBSET="${PRODUCTION_CONDITION_SUBSET:-erdc,zrdc,brdc}"
PRODUCTION_STATE_ENCODING="${PRODUCTION_STATE_ENCODING:-target_match}"
PRODUCTION_SUBSET_TAG="${PRODUCTION_SUBSET_TAG:-dc}"
PRODUCTION_OUT_DIR="${PRODUCTION_OUT_DIR:-results_final_2x2}"

MODEL_SELECTION_OUT_DIR="${MODEL_SELECTION_OUT_DIR:-analysis/results_model_selection/stats}"

: "${JAX_PLATFORMS:=cpu}"
: "${XLA_FLAGS:=--xla_force_host_platform_device_count=4}"

export JAX_PLATFORMS
export XLA_FLAGS

cd "$(dirname "$0")"

echo "Post-Vast local analysis"
echo "  repo                  : $(pwd)"
echo "  python                : ${PYTHON_BIN}"
echo "  tasks                 : ${TASKS}"
echo "  dry run               : ${DRY_RUN}"
echo "  check files           : ${CHECK_ARTIFACTS}"
echo "  artifact tag          : ${ARTIFACT_TAG}"
echo "  JAX_PLATFORMS         : ${JAX_PLATFORMS}"
echo "  XLA_FLAGS             : ${XLA_FLAGS}"
echo "  slider models         : ${SLIDER_MODELS}"
echo "  slider artifacts      : ${SLIDER_ARTIFACT_SPEAKERS}"
echo "  slider posterior out  : ${SLIDER_POSTERIOR_OUT_DIR}"
echo "  slider heldout stats  : ${SLIDER_HELDOUT_STATS_DIR}"
echo "  production models     : ${PRODUCTION_MODELS}"
echo "  production subset     : ${PRODUCTION_CONDITION_SUBSET}"
echo "  production encoding   : ${PRODUCTION_STATE_ENCODING}"
echo "  production subset tag : ${PRODUCTION_SUBSET_TAG}"
echo "  production out        : ${PRODUCTION_OUT_DIR}"
echo "  decision out          : ${MODEL_SELECTION_OUT_DIR}"
if git rev-parse --short HEAD >/dev/null 2>&1; then
  echo "  git commit            : $(git rev-parse --short HEAD)"
fi

if [[ "${CHECK_ARTIFACTS}" == "1" ]]; then
  checker_cmd=(
    "${PYTHON_BIN}" check_vast_artifacts.py
    --tasks all
    --slider-full-speakers "${SLIDER_ARTIFACT_SPEAKERS}"
    --slider-heldout-speakers "${SLIDER_ARTIFACT_SPEAKERS}"
    --artifact-tag "${ARTIFACT_TAG}"
    --slider-warmup "${SLIDER_WARMUP}"
    --slider-samples "${SLIDER_SAMPLES}"
    --slider-chains "${SLIDER_CHAINS}"
    --slider-num-folds "${SLIDER_NUM_FOLDS}"
    --slider-fold-seed "${SLIDER_FOLD_SEED}"
    --production-speakers "${PRODUCTION_MODELS}"
    --production-warmup "${PRODUCTION_WARMUP}"
    --production-samples "${PRODUCTION_SAMPLES}"
    --production-chains "${PRODUCTION_CHAINS}"
    --production-condition-subset "${PRODUCTION_CONDITION_SUBSET}"
    --production-state-encoding "${PRODUCTION_STATE_ENCODING}"
    --fail-incomplete
  )
  printf "Artifact check:"
  printf " %q" "${checker_cmd[@]}"
  printf "\n"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${checker_cmd[@]}"
  fi
fi

run_cmd() {
  printf "Command:"
  printf " %q" "$@"
  printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  "$@"
}

run_task() {
  local task="$1"
  echo ""
  echo "############################################################"
  echo "### ANALYSIS TASK ${task} START $(date)"
  echo "############################################################"

  case "${task}" in
    slider_posterior)
      (
        cd models/slider
        run_cmd "${PYTHON_BIN}" posterior_analysis.py \
          --models "${SLIDER_MODELS}" \
          --warmup "${SLIDER_WARMUP}" \
          --samples "${SLIDER_SAMPLES}" \
          --chains "${SLIDER_CHAINS}" \
          --artifact-tag "${ARTIFACT_TAG}" \
          --out-dir "${SLIDER_POSTERIOR_OUT_DIR}" \
          --format png
      )
      ;;
    slider_decision)
      (
        cd models/slider
        run_cmd "${PYTHON_BIN}" evaluate_speaker_pilot.py \
          --stats-dir "${SLIDER_POSTERIOR_OUT_DIR}/stats" \
          --prefix slider_speaker_ablation_eval
      )
      ;;
    slider_heldout)
      (
        cd models/slider
        run_cmd "${PYTHON_BIN}" evaluate_heldout_elpd.py \
          --models "${SLIDER_MODELS}" \
          --num-folds "${SLIDER_NUM_FOLDS}" \
          --fold-seed "${SLIDER_FOLD_SEED}" \
          --warmup "${SLIDER_WARMUP}" \
          --samples "${SLIDER_SAMPLES}" \
          --chains "${SLIDER_CHAINS}" \
          --artifact-tag "${ARTIFACT_TAG}" \
          --posterior-stats-dir "${SLIDER_POSTERIOR_OUT_DIR}/stats" \
          --out-dir "${SLIDER_HELDOUT_STATS_DIR}" \
          --decision-prefix slider_heldout_eval
      )
      ;;
    production_posterior)
      (
        cd models/production
        run_cmd "${PYTHON_BIN}" posterior_analysis.py \
          --models "${PRODUCTION_MODELS}" \
          --model-set simplified \
          --warmup "${PRODUCTION_WARMUP}" \
          --samples "${PRODUCTION_SAMPLES}" \
          --chains "${PRODUCTION_CHAINS}" \
          --artifact-tag "${ARTIFACT_TAG}" \
          --subset-tag "${PRODUCTION_SUBSET_TAG}" \
          --out-dir "${PRODUCTION_OUT_DIR}" \
          --format png
      )
      ;;
    production_pareto_frontier)
      run_cmd "${PYTHON_BIN}" analysis/model_pareto_frontier.py \
        --loo-csv "models/production/${PRODUCTION_OUT_DIR}/stats/production_loo_comparison.csv" \
        --ppc-csv "models/production/${PRODUCTION_OUT_DIR}/stats/production_simplified_ppc_correlation.csv" \
        --diagnostics-csv "models/production/${PRODUCTION_OUT_DIR}/stats/production_simplified_mcmc_model_summary.csv" \
        --ppc-scope all_cells \
        --prefix production_2x2 \
        --out-dir "models/production/${PRODUCTION_OUT_DIR}/stats/architecture_contrast" \
        --exclude-diagnostic-fail
      ;;
    production_architecture_audit)
      (
        cd models/production
        run_cmd "${PYTHON_BIN}" architecture_contrast_audit.py \
          --results-dir "${PRODUCTION_OUT_DIR}"
      )
      ;;
    final_decision)
      run_cmd "${PYTHON_BIN}" analysis/model_selection_decision_summary.py \
        --slider-posterior-stats-dir "models/slider/${SLIDER_POSTERIOR_OUT_DIR}/stats" \
        --slider-posterior-prefix "slider_speaker_ablation_eval" \
        --slider-heldout-stats-dir "models/slider/${SLIDER_HELDOUT_STATS_DIR}" \
        --slider-heldout-prefix "slider_heldout_eval" \
        --production-architecture-dir "models/production/${PRODUCTION_OUT_DIR}/stats/architecture_contrast" \
        --production-prefix "production_2x2" \
        --out-dir "${MODEL_SELECTION_OUT_DIR}"
      ;;
    *)
      echo "Unknown task '${task}'." >&2
      echo "Known tasks: slider_posterior slider_decision slider_heldout production_posterior production_pareto_frontier production_architecture_audit final_decision" >&2
      exit 2
      ;;
  esac

  echo "############################################################"
  echo "### ANALYSIS TASK ${task} DONE $(date)"
  echo "############################################################"
}

read -r -a task_array <<< "${TASKS}"
for task in "${task_array[@]}"; do
  run_task "${task}"
done

echo ""
echo "Post-Vast local analysis complete."
