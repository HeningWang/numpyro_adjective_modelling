#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/numpyro_adjective_modelling}"
REMOTE_BRANCH="${REMOTE_BRANCH:-$(git branch --show-current 2>/dev/null || printf analysis/model-contrast-audit)}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
ACTION="${ACTION:-start}"
SESSION="${SESSION:-numpyro_vast_queue}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-logs}"
REMOTE_PYTHON_BIN="${REMOTE_PYTHON_BIN:-/workspace/numpyro-venv/bin/python}"
DRY_RUN="${DRY_RUN:-0}"
PRINT_SCRIPT="${PRINT_SCRIPT:-0}"
TAIL_LINES="${TAIL_LINES:-80}"
SSH_PORT="${SSH_PORT:-}"
SSH_IDENTITY_FILE="${SSH_IDENTITY_FILE:-}"
SSH_EXTRA="${SSH_EXTRA:-}"

TASKS="${TASKS:-slider_full slider_heldout production_2x2}"
ARTIFACT_TAG="${ARTIFACT_TAG:-tm}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
OVERWRITE="${OVERWRITE:-0}"
CHECK_ARTIFACTS="${CHECK_ARTIFACTS:-1}"
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

if [[ -z "${REMOTE}" ]]; then
  echo "REMOTE is required, e.g. REMOTE=root@ssh5.vast.ai SSH_PORT=12345." >&2
  exit 2
fi

quote() {
  printf "%q" "$1"
}

assignment() {
  local key="$1"
  local value="$2"
  printf "export %s=%s\n" "${key}" "$(quote "${value}")"
}

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

queue_env_body="$(
  {
    printf "#!/usr/bin/env bash\n"
    printf "set -euo pipefail\n"
    printf "cd %s\n" "$(quote "${REMOTE_REPO}")"
    assignment PYTHON_BIN "${REMOTE_PYTHON_BIN}"
    assignment EXPECTED_BRANCH "${REMOTE_BRANCH}"
    assignment JAX_PLATFORMS cuda
    assignment XLA_FLAGS ""
    assignment XLA_PYTHON_CLIENT_PREALLOCATE false
    assignment TASKS "${TASKS}"
    assignment ARTIFACT_TAG "${ARTIFACT_TAG}"
    assignment SKIP_EXISTING "${SKIP_EXISTING}"
    assignment OVERWRITE "${OVERWRITE}"
    assignment CHECK_ARTIFACTS "${CHECK_ARTIFACTS}"
    assignment RUN_PREFLIGHT "${RUN_PREFLIGHT}"
    assignment SLIDER_FULL_SPEAKERS "${SLIDER_FULL_SPEAKERS}"
    assignment SLIDER_ABLATION_SPEAKERS "${SLIDER_ABLATION_SPEAKERS}"
    assignment SLIDER_HELDOUT_SPEAKERS "${SLIDER_HELDOUT_SPEAKERS}"
    assignment PRODUCTION_2X2_SPEAKERS "${PRODUCTION_2X2_SPEAKERS}"
    assignment SLIDER_WARMUP "${SLIDER_WARMUP}"
    assignment SLIDER_SAMPLES "${SLIDER_SAMPLES}"
    assignment SLIDER_CHAINS "${SLIDER_CHAINS}"
    assignment SLIDER_NUM_FOLDS "${SLIDER_NUM_FOLDS}"
    assignment SLIDER_FOLD_SEED "${SLIDER_FOLD_SEED}"
    assignment PRODUCTION_WARMUP "${PRODUCTION_WARMUP}"
    assignment PRODUCTION_SAMPLES "${PRODUCTION_SAMPLES}"
    assignment PRODUCTION_CHAINS "${PRODUCTION_CHAINS}"
    assignment PRODUCTION_CONDITION_SUBSET "${PRODUCTION_CONDITION_SUBSET}"
    assignment PRODUCTION_MIN_PROPORTION "${PRODUCTION_MIN_PROPORTION}"
    assignment PRODUCTION_STATE_ENCODING "${PRODUCTION_STATE_ENCODING}"
  }
)"
launcher_body="$(
  {
    printf "%s\n" "${queue_env_body}"
    printf "exec ./run_vast_inference_queue.sh\n"
  }
)"
queue_env_b64="$(printf "%s" "${queue_env_body}" | base64 | tr -d "\n")"
launcher_b64="$(printf "%s" "${launcher_body}" | base64 | tr -d "\n")"

remote_template=""
IFS= read -r -d '' remote_template <<'REMOTE_SCRIPT' || true
cd "${REMOTE_REPO}"

sync_repo() {
  git fetch "${GIT_REMOTE}"
  git checkout "${REMOTE_BRANCH}"
  git pull --ff-only "${GIT_REMOTE}" "${REMOTE_BRANCH}"
}

write_queue_env() {
  mkdir -p "${REMOTE_LOG_DIR}"
  env_script="${REMOTE_LOG_DIR}/${SESSION}_env.sh"
  printf "%s" "${QUEUE_ENV_B64}" | base64 -d > "${env_script}"
  chmod +x "${env_script}"
  # shellcheck disable=SC1090
  source "${env_script}"
}

case "${ACTION}" in
  start)
    sync_repo
    mkdir -p "${REMOTE_LOG_DIR}"
    launcher="${REMOTE_LOG_DIR}/${SESSION}_launcher.sh"
    log="${REMOTE_LOG_DIR}/${SESSION}.log"
    printf "%s" "${LAUNCHER_B64}" | base64 -d > "${launcher}"
    chmod +x "${launcher}"
    echo "Remote queue launcher: ${launcher}"
    echo "Remote queue log     : ${log}"
    if command -v tmux >/dev/null 2>&1; then
      if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "tmux session '${SESSION}' is already running." >&2
        exit 2
      fi
      tmux new-session -d -s "${SESSION}" "bash \"${launcher}\" 2>&1 | tee -a \"${log}\""
      echo "Started tmux session '${SESSION}'."
      echo "Tail with: ACTION=tail ./run_vast_remote_queue.sh ..."
    else
      nohup bash "${launcher}" > "${log}" 2>&1 &
      echo $! > "${REMOTE_LOG_DIR}/${SESSION}.pid"
      echo "tmux not found; started nohup pid file ${REMOTE_LOG_DIR}/${SESSION}.pid."
    fi
    ;;
  preflight)
    sync_repo
    write_queue_env
    echo "Remote GPU preflight"
    echo "  repo        : $(pwd)"
    echo "  python      : ${PYTHON_BIN}"
    echo "  tasks       : ${TASKS}"
    echo "  artifact tag: ${ARTIFACT_TAG}"
    ./run_vast_preflight.sh
    ;;
  status)
    echo "Repo: ${REMOTE_REPO}"
    git rev-parse --short HEAD
    git branch --show-current || true
    if command -v tmux >/dev/null 2>&1; then
      tmux ls 2>/dev/null || true
      if tmux has-session -t "${SESSION}" 2>/dev/null; then
        echo "tmux session '${SESSION}' is running."
      else
        echo "tmux session '${SESSION}' is not running."
      fi
    fi
    log="${REMOTE_LOG_DIR}/${SESSION}.log"
    if [[ -f "${log}" ]]; then
      echo "Log: ${log}"
      tail -n "${TAIL_LINES}" "${log}"
    else
      echo "No queue log found in ${REMOTE_LOG_DIR}."
    fi
    ;;
  tail)
    log="${REMOTE_LOG_DIR}/${SESSION}.log"
    if [[ ! -f "${log}" ]]; then
      echo "No queue log found in ${REMOTE_LOG_DIR}." >&2
      exit 2
    fi
    echo "Tailing ${log}"
    tail -n "${TAIL_LINES}" -f "${log}"
    ;;
  check)
    write_queue_env
    echo "Remote artifact check"
    echo "  repo        : $(pwd)"
    echo "  python      : ${PYTHON_BIN}"
    echo "  tasks       : ${TASKS}"
    echo "  artifact tag: ${ARTIFACT_TAG}"
    "${PYTHON_BIN}" check_vast_artifacts.py \
      --tasks "${TASKS}" \
      --artifact-tag "${ARTIFACT_TAG}" \
      --max-missing 52 \
      --fail-incomplete
    ;;
  stop)
    if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${SESSION}" 2>/dev/null; then
      tmux kill-session -t "${SESSION}"
      echo "Stopped tmux session '${SESSION}'."
    elif [[ -f "${REMOTE_LOG_DIR}/${SESSION}.pid" ]]; then
      pid=$(cat "${REMOTE_LOG_DIR}/${SESSION}.pid")
      kill "${pid}"
      rm -f "${REMOTE_LOG_DIR}/${SESSION}.pid"
      echo "Stopped nohup pid for '${SESSION}'."
    else
      echo "No running session or pid file found for '${SESSION}'."
    fi
    ;;
  *)
    echo "Unknown ACTION='${ACTION}'. Use start, preflight, status, tail, check, or stop." >&2
    exit 2
    ;;
esac
REMOTE_SCRIPT

remote_script="$(
  {
    printf "set -euo pipefail\n\n"
    printf "REMOTE_REPO=%s\n" "$(quote "${REMOTE_REPO}")"
    printf "REMOTE_BRANCH=%s\n" "$(quote "${REMOTE_BRANCH}")"
    printf "GIT_REMOTE=%s\n" "$(quote "${GIT_REMOTE}")"
    printf "ACTION=%s\n" "$(quote "${ACTION}")"
    printf "SESSION=%s\n" "$(quote "${SESSION}")"
    printf "REMOTE_LOG_DIR=%s\n" "$(quote "${REMOTE_LOG_DIR}")"
    printf "QUEUE_ENV_B64=%s\n" "$(quote "${queue_env_b64}")"
    printf "LAUNCHER_B64=%s\n" "$(quote "${launcher_b64}")"
    printf "TAIL_LINES=%s\n\n" "$(quote "${TAIL_LINES}")"
    printf "%s\n" "${remote_template}"
  }
)"

echo "Vast remote queue"
echo "  action      : ${ACTION}"
echo "  remote      : ${REMOTE}"
echo "  remote repo : ${REMOTE_REPO}"
echo "  branch      : ${REMOTE_BRANCH}"
echo "  session     : ${SESSION}"
echo "  python      : ${REMOTE_PYTHON_BIN}"
echo "  tasks       : ${TASKS}"
echo "  dry run     : ${DRY_RUN}"
if [[ -n "${SSH_PORT}" ]]; then
  echo "  ssh port    : ${SSH_PORT}"
fi
if [[ -n "${SSH_IDENTITY_FILE}" ]]; then
  echo "  ssh key     : ${SSH_IDENTITY_FILE}"
fi

printf "SSH command:"
printf " %q" "${ssh_cmd[@]}" "${REMOTE}" bash -s
printf "\n"

if [[ "${DRY_RUN}" == "1" || "${PRINT_SCRIPT}" == "1" ]]; then
  echo ""
  echo "Remote script:"
  printf "%s\n" "${remote_script}"
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

printf "%s\n" "${remote_script}" | "${ssh_cmd[@]}" "${REMOTE}" bash -s
