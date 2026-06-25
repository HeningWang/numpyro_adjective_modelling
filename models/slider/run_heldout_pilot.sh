#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-run_inference.py}"
NUM_WARMUP="${NUM_WARMUP:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
NUM_CHAINS="${NUM_CHAINS:-4}"
NUM_FOLDS="${NUM_FOLDS:-5}"
FOLD_SEED="${FOLD_SEED:-13}"
ARTIFACT_TAG="${ARTIFACT_TAG:-tm}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
SPEAKERS="${SPEAKERS:-incremental incremental_static planned_usefulness_order planned_usefulness_order_static planned_usefulness_mixture planned_usefulness_mixture_static}"

: "${JAX_PLATFORMS:=cuda}"
: "${XLA_FLAGS:=}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"

export JAX_PLATFORMS
export XLA_FLAGS
export XLA_PYTHON_CLIENT_PREALLOCATE

cd "$(dirname "$0")"
mkdir -p inference_data logs

echo "Slider heldout pilot"
echo "  python       : ${PYTHON_BIN}"
echo "  warmup       : ${NUM_WARMUP}"
echo "  samples      : ${NUM_SAMPLES}"
echo "  chains       : ${NUM_CHAINS}"
echo "  folds        : ${NUM_FOLDS}"
echo "  fold seed    : ${FOLD_SEED}"
echo "  artifact tag : ${ARTIFACT_TAG}"
echo "  speakers     : ${SPEAKERS}"
echo "  skip existing: ${SKIP_EXISTING}"
echo "  JAX_PLATFORMS: ${JAX_PLATFORMS}"
echo "  XLA_FLAGS    : ${XLA_FLAGS}"
echo "  preallocate  : ${XLA_PYTHON_CLIENT_PREALLOCATE}"
if git rev-parse --short HEAD >/dev/null 2>&1; then
  echo "  git commit   : $(git rev-parse --short HEAD)"
fi

if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import jax

devices = jax.devices()
print(f"  jax devices  : {devices}")
if not any(getattr(device, "platform", "").lower() in {"gpu", "cuda"} for device in devices):
    raise SystemExit("CUDA/GPU device is not visible; aborting heldout pilot before MCMC.")
PY
fi

run_cell() {
  local speaker="$1"
  local fold="$2"
  local tag_part=""
  if [[ -n "${ARTIFACT_TAG}" ]]; then
    tag_part="_${ARTIFACT_TAG}"
  fi
  local output_file="./inference_data/mcmc_results_${speaker}_speaker_hier_fold${fold}of${NUM_FOLDS}${tag_part}_warmup${NUM_WARMUP}_samples${NUM_SAMPLES}_chains${NUM_CHAINS}.nc"

  if [[ -e "${output_file}" && "${SKIP_EXISTING}" == "1" && "${OVERWRITE}" != "1" ]]; then
    echo "Skipping existing artifact: ${output_file}"
    return
  fi

  if [[ -e "${output_file}" && "${OVERWRITE}" != "1" ]]; then
    echo "Refusing to overwrite existing artifact: ${output_file}" >&2
    echo "Set OVERWRITE=1 to rerun, or SKIP_EXISTING=1 to resume past completed cells." >&2
    exit 2
  fi

  echo ""
  echo "============================="
  echo "  speaker : ${speaker}"
  echo "  fold    : ${fold}/${NUM_FOLDS}"
  echo "  output  : ${output_file}"
  echo "============================="

  local cmd=(
    "${PYTHON_BIN}" "${SCRIPT}"
    --speaker_type "${speaker}"
    --hierarchical
    --heldout_fold "${fold}"
    --num_folds "${NUM_FOLDS}"
    --fold_seed "${FOLD_SEED}"
    --num_warmup "${NUM_WARMUP}"
    --num_samples "${NUM_SAMPLES}"
    --num_chains "${NUM_CHAINS}"
    --artifact_tag "${ARTIFACT_TAG}"
  )

  printf "Command:"
  printf " %q" "${cmd[@]}"
  printf "\n"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi

  "${cmd[@]}"
}

start_time=$(date +%s)
read -r -a speaker_array <<< "${SPEAKERS}"
for speaker in "${speaker_array[@]}"; do
  for ((fold = 0; fold < NUM_FOLDS; fold++)); do
    run_cell "${speaker}" "${fold}"
  done
done
end_time=$(date +%s)

echo ""
echo "Slider heldout pilot complete in $((end_time - start_time))s."
