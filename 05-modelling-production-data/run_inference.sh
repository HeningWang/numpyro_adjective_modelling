#!/bin/bash
set -e  # stop immediately if a command fails

# =====================
# User settings
# =====================

PYTHON=python
SCRIPT=modelSpecification.py

SPEAKER_TYPE=global        # global | incremental
NUM_SAMPLES=100
NUM_WARMUP=100
NUM_CHAINS=1

# =====================
# Environment setup
# =====================
# export JAX_PLATFORM_NAME=cpu
# export XLA_FLAGS="--xla_force_host_platform_device_count=4"
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