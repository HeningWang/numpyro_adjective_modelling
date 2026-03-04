#!/bin/bash
set -e  # stop immediately if a command fails

# =====================
# User settings
# =====================

PYTHON=python
SCRIPT=modelSpecification.py
NUM_CHAINS=4

# Flat models (incremental_gamma, global_gamma)
NUM_SAMPLES_FLAT=1000
NUM_WARMUP_FLAT=1000

# Hierarchical models (incremental_hier_gamma, global_hier_gamma)
NUM_SAMPLES_HIER=1000
NUM_WARMUP_HIER=1000

# =====================
# Run
# =====================

START=$(date +%s)

run_model() {
  local speaker="$1"
  local hier="$2"
  local warmup="$3"
  local samples="$4"
  local hier_flag=""
  [ "${hier}" = "true" ] && hier_flag="--hierarchical"
  echo ""
  echo "============================="
  echo "  speaker : ${speaker}"
  echo "  hier    : ${hier}"
  echo "  warmup  : ${warmup}  samples : ${samples}  chains : ${NUM_CHAINS}"
  echo "============================="
  ${PYTHON} ${SCRIPT} \
    --speaker_type "${speaker}" \
    --num_warmup   "${warmup}"  \
    --num_samples  "${samples}" \
    --num_chains   "${NUM_CHAINS}" \
    --infer_gamma  \
    ${hier_flag}
}

run_model incremental false "${NUM_WARMUP_FLAT}"  "${NUM_SAMPLES_FLAT}"
run_model global      false "${NUM_WARMUP_FLAT}"  "${NUM_SAMPLES_FLAT}"
run_model incremental true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"
run_model global      true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"

END=$(date +%s)
echo ""
echo "All four gamma models done in $(( END - START ))s."
