#!/bin/bash
set -e  # stop immediately if a command fails

# =====================
# User settings
# =====================

PYTHON=python
SCRIPT=modelSpecification.py

SPEAKER_TYPE=global        # global | incremental
NUM_SAMPLES=250
NUM_WARMUP=750
NUM_CHAINS=4


# =====================
# Run
# =====================

echo "Running NumPyro inference"
echo "Speaker type : ${SPEAKER_TYPE}"
echo "Samples      : ${NUM_SAMPLES}"
echo "Warmup       : ${NUM_WARMUP}"
echo "Chains       : ${NUM_CHAINS}"
echo "-----------------------------"

${PYTHON} ${SCRIPT} \
  --speaker_type ${SPEAKER_TYPE} \
  --num_samples ${NUM_SAMPLES} \
  --num_warmup ${NUM_WARMUP} \
  --num_chains ${NUM_CHAINS}

echo "Done."