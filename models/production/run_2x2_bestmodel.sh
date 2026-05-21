#!/usr/bin/env bash
# 2x2 (speaker × semantics) on the best contextual model.
# 4 parameter-matched cells of contextual_pcalpha_canon_parsimony (drop=alpha_F,
# csv=0.59). inc_rec == the merged best model (R²≈0.94) — a built-in sanity
# check. dc-subset, 4000 warmup / 2000 samples / 4 chains, sequential (1 GPU).
set -e
# Run from models/production/
for C in inc_rec inc_static glob_rec glob_static; do
  echo "=== START $C $(date) ==="
  JAX_PLATFORMS="" XLA_FLAGS="" python -u \
    run_inference.py --speaker_type contextual_pcalpha_canon_parsimony_2x2_$C \
    --hierarchical --condition-subset erdc,zrdc,brdc \
    --num_warmup 4000 --num_samples 2000 --num_chains 4
  echo "=== DONE $C $(date) ==="
done
echo "=== ALL 2x2 BESTMODEL DONE $(date) ==="
