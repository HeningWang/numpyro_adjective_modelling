#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-run_inference.py}"
NUM_WARMUP="${NUM_WARMUP:-500}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
NUM_CHAINS="${NUM_CHAINS:-4}"
OVERWRITE="${OVERWRITE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SPEAKERS="${SPEAKERS:-planned_usefulness_mixture planned_usefulness_mixture_static}"

: "${JAX_PLATFORMS:=cuda}"
: "${XLA_FLAGS:=}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"

export JAX_PLATFORMS
export XLA_FLAGS
export XLA_PYTHON_CLIENT_PREALLOCATE

cd "$(dirname "$0")"
mkdir -p inference_data logs

echo "Slider speaker-ablation pilot"
echo "  python       : ${PYTHON_BIN}"
echo "  warmup       : ${NUM_WARMUP}"
echo "  samples      : ${NUM_SAMPLES}"
echo "  chains       : ${NUM_CHAINS}"
echo "  speakers     : ${SPEAKERS}"
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
    raise SystemExit("CUDA/GPU device is not visible; aborting pilot before MCMC.")
PY
fi

run_cell() {
  local speaker="$1"
  local output_file="./inference_data/mcmc_results_${speaker}_speaker_hier_warmup${NUM_WARMUP}_samples${NUM_SAMPLES}_chains${NUM_CHAINS}.nc"

  if [[ -e "${output_file}" && "${OVERWRITE}" != "1" ]]; then
    echo "Refusing to overwrite existing artifact: ${output_file}" >&2
    echo "Set OVERWRITE=1 only after confirming this rerun is intended." >&2
    exit 2
  fi

  echo ""
  echo "============================="
  echo "  speaker : ${speaker}"
  echo "  output  : ${output_file}"
  echo "============================="

  local cmd=(
    "${PYTHON_BIN}" "${SCRIPT}"
    --speaker_type "${speaker}"
    --hierarchical
    --num_warmup "${NUM_WARMUP}"
    --num_samples "${NUM_SAMPLES}"
    --num_chains "${NUM_CHAINS}"
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
  run_cell "${speaker}"
done
end_time=$(date +%s)

echo ""
echo "Speaker-ablation pilot complete in $((end_time - start_time))s."
