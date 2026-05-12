#!/bin/bash
#SBATCH --job-name=ctx_dc
#SBATCH --partition=single
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/ctx_dc_%j.out
#SBATCH --error=logs/ctx_dc_%j.err
#
# Helix (BWHPC) SLURM job for the contextual_dc inference run.
#
# Tested partition: `single` (CPU, single-node). If you want A100s, switch to
# `gpu-single` or whichever GPU partition you have access to and add
# `#SBATCH --gres=gpu:1` — but the JAX-CUDA install in the bootstrap block
# below will need to be replaced with `jax[cuda12]`.
#
# First run: takes ~5–10 min extra to create the conda env. Subsequent runs
# reuse the env and start immediately.

set -euo pipefail

mkdir -p logs inference_data

# --- module + conda ---
module purge 2>/dev/null || true
module load devel/miniconda/3 2>/dev/null \
  || module load devel/miniconda 2>/dev/null \
  || module load anaconda3 2>/dev/null \
  || true

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found after 'module load'." >&2
  echo "Check 'module avail' for the correct miniconda/anaconda module name on Helix." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_DIR="$HOME/conda-envs/ctx_dc"
if [ ! -d "$ENV_DIR" ]; then
  echo "Creating conda env at $ENV_DIR ..."
  conda create -y -p "$ENV_DIR" python=3.11
  conda activate "$ENV_DIR"
  pip install --quiet --upgrade pip
  pip install --quiet numpyro arviz pandas scipy numpy
else
  conda activate "$ENV_DIR"
fi

echo "=== env ==="
echo "Host:    $(hostname)"
echo "Date:    $(date)"
echo "Job ID:  ${SLURM_JOB_ID:-na}"
echo "Python:  $(which python)"
python -c "import jax, numpyro, arviz, pandas; print(f'jax={jax.__version__}, numpyro={numpyro.__version__}, arviz={arviz.__version__}'); print(f'devices={jax.devices()}')"
echo "==========="

cd "$SLURM_SUBMIT_DIR"

JAX_PLATFORMS='' XLA_FLAGS='' python -u run_inference.py \
    --speaker_type contextual --hierarchical \
    --condition-subset erdc,zrdc,brdc \
    --num_warmup 2000 --num_samples 1000 --num_chains 4

echo "=== done at $(date) ==="
