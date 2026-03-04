#!/bin/bash
set -e  # stop immediately if a command fails

# =====================
# User settings
# =====================

PYTHON=/Users/heningwang/Documents/GitHub/numpyro_adjective_modelling/jax-metal/bin/python
SCRIPT=modelSpecification.py
NUM_CHAINS=4

# Flat models (incremental, global)
NUM_SAMPLES_FLAT=1000
NUM_WARMUP_FLAT=1000

# Hierarchical models (incremental_hier, global_hier)
NUM_SAMPLES_HIER=1000
NUM_WARMUP_HIER=1000

# Set to "true" to infer gamma_blurred/gamma_sharp, "false" to fix them
INFER_GAMMA=true

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
  local gamma_flag="--infer_gamma"
  [ "${INFER_GAMMA}" = "false" ] && gamma_flag="--no-infer_gamma"
  echo ""
  echo "============================="
  echo "  speaker    : ${speaker}"
  echo "  hier       : ${hier}"
  echo "  infer_gamma: ${INFER_GAMMA}"
  echo "  warmup     : ${warmup}  samples : ${samples}  chains : ${NUM_CHAINS}"
  echo "============================="
  ${PYTHON} ${SCRIPT} \
    --speaker_type "${speaker}" \
    --num_warmup   "${warmup}"  \
    --num_samples  "${samples}" \
    --num_chains   "${NUM_CHAINS}" \
    ${hier_flag} \
    ${gamma_flag}
}

run_model incremental false "${NUM_WARMUP_FLAT}"  "${NUM_SAMPLES_FLAT}"
run_model global      false "${NUM_WARMUP_FLAT}"  "${NUM_SAMPLES_FLAT}"
run_model incremental true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"
run_model global      true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"

END=$(date +%s)
echo ""
echo "All four models done in $(( END - START ))s."
