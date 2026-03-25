#!/bin/bash
set -e  # stop immediately if a command fails

# =====================
# User settings
# =====================

PYTHON=/Users/heningwang/Documents/GitHub/numpyro_adjective_modelling/jax-metal/bin/python
SCRIPT=run_inference.py
NUM_CHAINS=4

# Hierarchical models
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
run_model incremental_static  true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"

# LM prior ablation (Priority 3)
run_model incremental_lm_only   true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"
run_model incremental_rsa_only  true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"

# Lookahead speaker (Priority 5)
run_model incremental_lookahead true  "${NUM_WARMUP_HIER}"  "${NUM_SAMPLES_HIER}"

END=$(date +%s)
echo ""
echo "All seven hierarchical models done in $(( END - START ))s."
