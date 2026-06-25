#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
output="$(mktemp)"
launcher="$(mktemp)"
queue_env="$(mktemp)"
trap 'rm -f "${output}" "${launcher}" "${queue_env}"' EXIT

cd "${repo_root}"

DRY_RUN=1 \
REMOTE=root@example.invalid \
SSH_PORT=2222 \
SSH_IDENTITY_FILE=/tmp/fake_key \
REMOTE_REPO=/workspace/numpyro_adjective_modelling \
REMOTE_BRANCH=analysis/model-contrast-audit \
REMOTE_PYTHON_BIN=/workspace/numpyro-venv/bin/python \
TASKS="slider_full production_2x2" \
SLIDER_WARMUP=11 \
SLIDER_SAMPLES=12 \
SLIDER_CHAINS=2 \
PRODUCTION_WARMUP=21 \
PRODUCTION_SAMPLES=22 \
PRODUCTION_CHAINS=2 \
  ./run_vast_remote_queue.sh > "${output}"

grep -q "ACTION=start" "${output}"
grep -q "REMOTE_REPO=/workspace/numpyro_adjective_modelling" "${output}"
grep -q "SSH command: ssh -p 2222 -i /tmp/fake_key root@example.invalid bash -s" "${output}"

launcher_b64="$(sed -n 's/^LAUNCHER_B64=//p' "${output}" | head -n 1)"
if [[ -z "${launcher_b64}" ]]; then
  echo "Missing LAUNCHER_B64 in dry-run output." >&2
  exit 1
fi
printf "%s" "${launcher_b64}" | base64 -d > "${launcher}"

grep -q "cd /workspace/numpyro_adjective_modelling" "${launcher}"
grep -q "export PYTHON_BIN=/workspace/numpyro-venv/bin/python" "${launcher}"
grep -q "export EXPECTED_BRANCH=analysis/model-contrast-audit" "${launcher}"
grep -q "export JAX_PLATFORMS=cuda" "${launcher}"
grep -q "export XLA_FLAGS=''" "${launcher}"
grep -q "export TASKS=slider_full\\\\ production_2x2" "${launcher}"
grep -q "export SLIDER_WARMUP=11" "${launcher}"
grep -q "export SLIDER_SAMPLES=12" "${launcher}"
grep -q "export SLIDER_CHAINS=2" "${launcher}"
grep -q "export PRODUCTION_WARMUP=21" "${launcher}"
grep -q "export PRODUCTION_SAMPLES=22" "${launcher}"
grep -q "export PRODUCTION_CHAINS=2" "${launcher}"
grep -q "exec ./run_vast_inference_queue.sh" "${launcher}"

ACTION=status \
DRY_RUN=1 \
REMOTE=root@example.invalid \
REMOTE_REPO=/workspace/numpyro_adjective_modelling \
  ./run_vast_remote_queue.sh > "${output}"

grep -q "ACTION=status" "${output}"
grep -q 'log="${REMOTE_LOG_DIR}/${SESSION}.log"' "${output}"

ACTION=preflight \
DRY_RUN=1 \
REMOTE=root@example.invalid \
REMOTE_REPO=/workspace/numpyro_adjective_modelling \
REMOTE_PYTHON_BIN=/workspace/numpyro-venv/bin/python \
  ./run_vast_remote_queue.sh > "${output}"

grep -q "ACTION=preflight" "${output}"
grep -q "Remote GPU preflight" "${output}"
grep -q "./run_vast_preflight.sh" "${output}"

ACTION=check \
DRY_RUN=1 \
REMOTE=root@example.invalid \
REMOTE_REPO=/workspace/numpyro_adjective_modelling \
REMOTE_PYTHON_BIN=/workspace/numpyro-venv/bin/python \
TASKS="slider_full production_2x2" \
SLIDER_WARMUP=11 \
PRODUCTION_WARMUP=21 \
  ./run_vast_remote_queue.sh > "${output}"

grep -q "ACTION=check" "${output}"
grep -q "Remote artifact check" "${output}"
grep -q 'check_vast_artifacts.py \\' "${output}"
grep -q -- "--fail-incomplete" "${output}"

queue_env_b64="$(sed -n 's/^QUEUE_ENV_B64=//p' "${output}" | head -n 1)"
if [[ -z "${queue_env_b64}" ]]; then
  echo "Missing QUEUE_ENV_B64 in dry-run output." >&2
  exit 1
fi
printf "%s" "${queue_env_b64}" | base64 -d > "${queue_env}"

grep -q "export PYTHON_BIN=/workspace/numpyro-venv/bin/python" "${queue_env}"
grep -q "export TASKS=slider_full\\\\ production_2x2" "${queue_env}"
grep -q "export SLIDER_WARMUP=11" "${queue_env}"
grep -q "export PRODUCTION_WARMUP=21" "${queue_env}"

echo "PASS vast remote queue dry-run tests"
