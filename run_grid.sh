#!/bin/bash
set -e

# Default parallelism settings
MAX_JOBS=20
SLEEP_TIME=5
NUM_GPUS=2
JOB_IDX=0

# Function to limit background jobs
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep $SLEEP_TIME
    done
}

echo "Starting Encoder Grid Search Experiments..."

# Dataset types (using same as run_experiments.sh)
DATASET_TYPES=("dnabert2" "evo2" "ntv2")

# for dtype in "${DATASET_TYPES[@]}"; do
#     echo "Processing Dataset Type: $dtype"
#     OUTPUT_DIR="outputs/grid_encoder_${dtype}"
    
#     # Loop over sequence lengths to run in parallel
#     # run_experiments.sh uses _10, _25, _50, _75, _100 for mean experiments
#     for seq_len in 10 25 50 100; do
#         DATA_CONFIG="${dtype}_${seq_len}_hg38_mean"
#         GPU_ID=$((JOB_IDX % NUM_GPUS))
        
#         echo "  [Train] Launching grid search multirun for $DATA_CONFIG on GPU $GPU_ID..."
#         wait_for_jobs

#         CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -m \
#             data=$DATA_CONFIG \
#             model=encoder \
#             model.d_model=128,256,512,1024 \
#             model.dim_feedforward=512,1024,2048,4096 \
#             model.num_layers=3,6,9,12 \
#             hydra.sweep.dir=$OUTPUT_DIR \
#             hydra.sweep.subdir="${DATA_CONFIG}_\${hydra.job.num}" \
#             hydra.job.chdir=True &
        
#         JOB_IDX=$((JOB_IDX + 1))
            
#     done
#     echo "  [Train] All grid search jobs launched for $dtype (running in background)"

# done

# Wait for all multiruns to complete
echo "Waiting for all grid search training jobs to complete..."
wait

echo "Starting Grid Search Evaluations..."

for dtype in "${DATASET_TYPES[@]}"; do
    OUTPUT_DIR="outputs/grid_encoder_${dtype}"
    EVAL_OUTPUT_DIR="outputs/eval_grid_encoder_${dtype}"
    
    if [ -d "$OUTPUT_DIR" ]; then
        if [ ! -d "$EVAL_OUTPUT_DIR" ]; then
            echo "  [Eval] Evaluating grid results for $dtype in $OUTPUT_DIR..."
            wait_for_jobs
            python evaluate.py run_dir=$OUTPUT_DIR hydra.run.dir="$EVAL_OUTPUT_DIR" &
        else
            echo "  [Eval] Output directory $EVAL_OUTPUT_DIR exists, skipping evaluation."
        fi
    else
        echo "  [Eval] Warning: $OUTPUT_DIR not found, skipping evaluation."
    fi
done

wait
echo "Encoder Grid Search Experiments Complete."
