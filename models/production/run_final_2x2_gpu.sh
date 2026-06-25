#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT="${SCRIPT:-run_inference.py}"
NUM_WARMUP="${NUM_WARMUP:-4000}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
NUM_CHAINS="${NUM_CHAINS:-4}"
CONDITION_SUBSET="${CONDITION_SUBSET:-erdc,zrdc,brdc}"
MIN_PROPORTION="${MIN_PROPORTION:-0.0}"
STATE_ENCODING="${STATE_ENCODING:-target_match}"
ARTIFACT_TAG="${ARTIFACT_TAG:-tm}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DRY_RUN="${DRY_RUN:-0}"
SPEAKERS="${SPEAKERS:-contextual_pcalpha_canon_parsimony_2x2_inc_rec contextual_pcalpha_canon_parsimony_2x2_inc_static contextual_pcalpha_canon_parsimony_2x2_glob_rec contextual_pcalpha_canon_parsimony_2x2_glob_static}"

: "${JAX_PLATFORMS:=cuda}"
: "${XLA_FLAGS:=}"
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"

export JAX_PLATFORMS
export XLA_FLAGS
export XLA_PYTHON_CLIENT_PREALLOCATE

cd "$(dirname "$0")"
mkdir -p inference_data logs

subset_tag() {
  local subset="$1"
  if [[ -z "${subset}" ]]; then
    return
  fi
  local old_ifs="${IFS}"
  local code
  local stem
  local -a codes=()
  local -a stems=()
  IFS=","
  read -r -a codes <<< "${subset}"
  IFS="${old_ifs}"
  for code in "${codes[@]}"; do
    code="${code//[[:space:]]/}"
    if [[ "${#code}" -ge 4 ]]; then
      stem="${code:2:2}"
      stems+=("${stem}")
    fi
  done
  if [[ "${#stems[@]}" -eq 0 ]]; then
    return
  fi
  local joined
  joined="$(printf "%s\n" "${stems[@]}" | sort -u | tr -d "\n")"
  printf "_%s" "${joined}"
}

top_tag() {
  case "${MIN_PROPORTION}" in
    ""|"0"|"0.0"|"0.00") ;;
    *) printf "_top" ;;
  esac
}

SUBSET_TAG="$(subset_tag "${CONDITION_SUBSET}")"
TOP_TAG="$(top_tag)"

echo "Production final 2x2 GPU run"
echo "  python       : ${PYTHON_BIN}"
echo "  warmup       : ${NUM_WARMUP}"
echo "  samples      : ${NUM_SAMPLES}"
echo "  chains       : ${NUM_CHAINS}"
echo "  subset       : ${CONDITION_SUBSET}"
echo "  subset tag   : ${SUBSET_TAG}"
echo "  min prop     : ${MIN_PROPORTION}"
echo "  encoding     : ${STATE_ENCODING}"
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
    raise SystemExit("CUDA/GPU device is not visible; aborting production 2x2 before MCMC.")
PY
fi

run_cell() {
  local speaker="$1"
  local tag_part=""
  if [[ -n "${ARTIFACT_TAG}" ]]; then
    tag_part="_${ARTIFACT_TAG}"
  fi
  local output_file="./inference_data/mcmc_results_${speaker}_speaker_hier${TOP_TAG}${SUBSET_TAG}${tag_part}_warmup${NUM_WARMUP}_samples${NUM_SAMPLES}_chains${NUM_CHAINS}.nc"

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
  echo "  output  : ${output_file}"
  echo "============================="

  local cmd=(
    "${PYTHON_BIN}" "${SCRIPT}"
    --speaker_type "${speaker}"
    --hierarchical
    --condition-subset "${CONDITION_SUBSET}"
    --state-encoding "${STATE_ENCODING}"
    --artifact-tag "${ARTIFACT_TAG}"
    --min-proportion "${MIN_PROPORTION}"
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
echo "Production final 2x2 GPU run complete in $((end_time - start_time))s."
