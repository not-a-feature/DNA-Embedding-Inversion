#!/bin/bash

# DNA Embedding Inversion - Replot Script
# Regenerates plots for all three mean evaluation folders

set -e

echo "Starting Replot..."
echo "==================="

# Get absolute path to script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Evaluation folders to replot (as absolute paths)
EVAL_DIRS=(
    "${SCRIPT_DIR}/outputs/eval_mean_dnabert2"
    "${SCRIPT_DIR}/outputs/eval_mean_evo2"
    "${SCRIPT_DIR}/outputs/eval_mean_ntv2"
)

for eval_dir in "${EVAL_DIRS[@]}"; do
    if [ -d "$eval_dir" ]; then
        echo "Replotting: $eval_dir"
        python evaluate.py \
            run_dir="$eval_dir" \
            only_plots=true \
            aggregate_only=true \
            hydra.run.dir="$eval_dir" \
            hydra.job.chdir=false &
    else
        echo "Warning: $eval_dir not found, skipping."
    fi
done

echo "Waiting for all replot jobs to complete..."
wait

echo "==================="
echo "Replot Complete."
