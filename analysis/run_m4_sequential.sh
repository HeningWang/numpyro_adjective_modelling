#!/usr/bin/env bash
# Use the existing CSP environment; no remote access, provisioning or parallel jobs.
set -euo pipefail
py=${M4_PYTHON:-python}
root=${M4_ROOT:-tmp/robustness/m4}
s=${M4_POSTERIOR_DRAWS:-400}
m=${M4_EFFECT_DRAWS:-100}
compare=${M4_COMPARE_M:-50}
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$root/logs"
JAX_PLATFORMS=cpu "$py" analysis/run_m4_fold.py --root "$root" --make-folds-only
for fold in {0..9}; do
    for model in G-HO I-HO K-HO K-HKO K-UPD-HO; do
        printf 'M4 fold %s model %s\n' "$fold" "$model"
        JAX_PLATFORMS=cuda "$py" analysis/run_m4_fold.py \
            --root "$root" --model "$model" --fold "$fold" --skip-completed \
            >> "$root/logs/fold_${fold}_${model}_fit.log" 2>&1
        stability=()
        if [[ "$fold" == 0 ]]; then stability=(--compare-m "$compare"); fi
        JAX_PLATFORMS=cuda "$py" analysis/score_m4_heldout.py \
            --root "$root" --model "$model" --fold "$fold" \
            --posterior-draws "$s" --effect-draws "$m" "${stability[@]}" --skip-completed \
            >> "$root/logs/fold_${fold}_${model}_score.log" 2>&1
    done
done
"$py" analysis/summarize_m4.py --root "$root" --posterior-draws "$s" \
    --effect-draws "$m" --compare-m "$compare" --skip-completed
