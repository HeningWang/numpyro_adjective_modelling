#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
output="$(mktemp)"
trap 'rm -f "${output}"' EXIT

cd "${repo_root}"

PRINT_ONLY=1 \
REMOTE=root@example.invalid \
SSH_PORT=2222 \
SSH_IDENTITY_FILE=/tmp/fake_key \
REMOTE_REPO=/workspace/numpyro_adjective_modelling \
ARTIFACT_TASKS="slider_full production_2x2" \
ARTIFACT_TAG=tm \
  ./pull_vast_artifacts.sh > "${output}"

grep -q "fail incomplete: auto" "${output}"
grep -q "status csv : analysis/results_model_selection/stats/vast_artifact_status.csv" "${output}"
grep -q "Pulling slider artifacts" "${output}"
grep -q "Pulling production artifacts" "${output}"
grep -q "root@example.invalid:/workspace/numpyro_adjective_modelling/models/slider/inference_data/" "${output}"
grep -q "Artifact check:" "${output}"
grep -q -- "--csv analysis/results_model_selection/stats/vast_artifact_status.csv" "${output}"
if grep -q -- "--fail-incomplete" "${output}"; then
  echo "auto mode should not add --fail-incomplete during PRINT_ONLY dry planning." >&2
  exit 1
fi

PRINT_ONLY=1 \
CHECK_FAIL_INCOMPLETE=1 \
REMOTE=root@example.invalid \
REMOTE_REPO=/workspace/numpyro_adjective_modelling \
ARTIFACT_TASKS="slider_full production_2x2" \
ARTIFACT_TAG=tm \
  ./pull_vast_artifacts.sh > "${output}"

grep -q "fail incomplete: 1" "${output}"
grep -q -- "--fail-incomplete" "${output}"

echo "PASS vast artifact pull dry-run tests"
