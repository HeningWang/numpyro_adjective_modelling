#!/bin/bash
set -e  # stop immediately if a command fails

# =====================
# User settings
# =====================

PYTHON=/Users/heningwang/Documents/GitHub/numpyro_adjective_modelling/jax-metal/bin/python
SCRIPT=run_inference.py
NUM_CHAINS=4

# Flat models (incremental, global)
NUM_SAMPLES_FLAT=1000
NUM_WARMUP_FLAT=1000

# Hierarchical models (incremental_hier, global_hier)
NUM_SAMPLES_HIER=500
NUM_WARMUP_HIER=500

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
  echo "  speaker    : ${speaker}"
  echo "  hier       : ${hier}"
  echo "  warmup     : ${warmup}  samples : ${samples}  chains : ${NUM_CHAINS}"
  echo "============================="
  ${PYTHON} ${SCRIPT} \
    --speaker_type "${speaker}" \
    --num_warmup   "${warmup}"  \
    --num_samples  "${samples}" \
    --num_chains   "${NUM_CHAINS}" \
    ${hier_flag}
}

run_model global              true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"
run_model incremental         true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"
run_model global_static       true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"
run_model incremental_frozen  true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"

END=$(date +%s)
echo ""
echo "All four hierarchical models done in $(( END - START ))s."
