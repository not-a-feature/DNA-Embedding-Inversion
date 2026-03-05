#!/bin/bash
set -e

# Default parallelism settings
MAX_JOBS=6
SLEEP_TIME=5
NUM_GPUS=1
JOB_IDX=0

# Function to limit background jobs
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep $SLEEP_TIME
    done
}

# # ==============================================================================
# # 1. Per-Token
# # ==============================================================================
# echo "Starting Per-Token Experiments..."

# PER_TOKEN_DATASETS=("dnabert2_100_hg38_per_token" "evo2_100_hg38_per_token" "ntv2_100_hg38_per_token")

# for dataset in "${PER_TOKEN_DATASETS[@]}"; do
#     echo "Processing Per-Token Dataset: $dataset"
    
#     # Define output directory
#     OUTPUT_DIR="outputs/train_per_token_${dataset}"
    
#     # 1.1 Training
#     if [ ! -f "$OUTPUT_DIR/model.pt" ]; then
#         echo "  [Train] Launching training for $dataset..."
#         wait_for_jobs
#         python train.py \
#             data=$dataset \
#             model=mlp_per_token \
#             hydra.run.dir=$OUTPUT_DIR \
#             hydra.job.chdir=True &
            
#         last_pid=$!
#         echo "  [Train] Job started with PID: $last_pid"
#     else
#         echo "  [Train] model.pt exists in $OUTPUT_DIR, skipping training."
#     fi
# done

# echo "Waiting for all per-token training jobs to complete..."
# wait

# echo "Starting Per-Token Evaluations..."
# for dataset in "${PER_TOKEN_DATASETS[@]}"; do
#     OUTPUT_DIR="outputs/train_per_token_${dataset}"
#     EVAL_OUTPUT_DIR="outputs/eval_per_token_${dataset}"
    
#     if [ -d "$OUTPUT_DIR" ]; then
#         if [ ! -d "$EVAL_OUTPUT_DIR" ]; then
#             echo "  [Eval] Evaluating $dataset in $OUTPUT_DIR..."
#             wait_for_jobs
#             python evaluate.py run_dir=$OUTPUT_DIR hydra.run.dir="$EVAL_OUTPUT_DIR" &
#         else
#             echo "  [Eval] Output directory $EVAL_OUTPUT_DIR exists, skipping evaluation."
#         fi
#     else
#         echo "  [Eval] Warning: $OUTPUT_DIR not found, skipping evaluation."
#     fi
    
#     # 1.3 Embedding Analysis
#     ANALYSIS_OUTPUT_DIR="outputs/analysis_per_token_${dataset}"
    
#     if [ -d "$OUTPUT_DIR" ]; then
#         if [ ! -d "$ANALYSIS_OUTPUT_DIR" ]; then
#             echo "  [Analysis] Analyzing embeddings for $dataset..."
#             wait_for_jobs
#             python embedding_analysis/embedding_analysis_per_token.py \
#                 data=$dataset \
#                 hydra.run.dir="$ANALYSIS_OUTPUT_DIR" &
#         else
#              echo "  [Analysis] Output directory $ANALYSIS_OUTPUT_DIR exists, skipping analysis."
#         fi
#     else
#         echo "  [Analysis] Warning: $OUTPUT_DIR not found, skipping analysis."
#     fi
# done
# wait
# echo "Per-Token Experiments Complete."
# echo "---------------------------------"


# ==============================================================================
# 2. Mean Experiments
# ==============================================================================
echo "Starting Mean Experiments..."

# Dataset types
MEAN_TYPES=("dnabert2" "evo2" "ntv2")

# Models to sweep (one job per model for more parallelism)
MODELS=("encoder" "decoder" "resnet" "knn")

for dtype in "${MEAN_TYPES[@]}"; do
    echo "Processing Mean Dataset Type: $dtype"
    DATA_CONFIGS="${dtype}_10_hg38_mean,${dtype}_15_hg38_mean,${dtype}_20_hg38_mean,${dtype}_25_hg38_mean,${dtype}_30_hg38_mean,${dtype}_35_hg38_mean,${dtype}_40_hg38_mean,${dtype}_45_hg38_mean,${dtype}_50_hg38_mean,${dtype}_60_hg38_mean,${dtype}_70_hg38_mean,${dtype}_80_hg38_mean,${dtype}_90_hg38_mean,${dtype}_100_hg38_mean"
    
    OUTPUT_DIR="outputs/train_mean_${dtype}"
    
    for model in "${MODELS[@]}"; do
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        echo "  [Train] Launching multirun for $dtype / $model on GPU $GPU_ID..."
        wait_for_jobs
        
        CUDA_VISIBLE_DEVICES=$GPU_ID python train.py -m \
            data=$DATA_CONFIGS \
            model=$model \
            hydra.sweep.dir=$OUTPUT_DIR \
            hydra.sweep.subdir=\${hydra.job.override_dirname} \
            hydra.job.chdir=True &
        JOB_IDX=$((JOB_IDX + 1))
            
        echo "  [Train] Multirun started for $dtype / $model on GPU $GPU_ID"
    done
done

# Wait for all multiruns to complete
echo "Waiting for all mean dataset multiruns to complete..."
wait

echo "Starting Mean Evaluations..."
# 2.2 Evaluation
for dtype in "${MEAN_TYPES[@]}"; do
    OUTPUT_DIR="outputs/train_mean_${dtype}"
    EVAL_OUTPUT_DIR="outputs/eval_mean_${dtype}"
    
    if [ -d "$OUTPUT_DIR" ]; then
        if [ ! -d "$EVAL_OUTPUT_DIR" ]; then
            GPU_ID=$((JOB_IDX % NUM_GPUS))
            echo "  [Eval] Evaluating aggregates for $dtype in $OUTPUT_DIR on GPU $GPU_ID..."
            wait_for_jobs
            CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate.py run_dir=$OUTPUT_DIR hydra.run.dir="$EVAL_OUTPUT_DIR" &
            JOB_IDX=$((JOB_IDX + 1))
        else
            echo "  [Eval] Output directory $EVAL_OUTPUT_DIR exists, skipping evaluation."
        fi
    else
        echo "  [Eval] Warning: $OUTPUT_DIR not found, skipping evaluation."
    fi

    # 2.3 Embedding Analysis (Multirun)
    ANALYSIS_OUTPUT_DIR="outputs/analysis_mean_${dtype}"
    
    if [ ! -d "$ANALYSIS_OUTPUT_DIR" ]; then
        echo "  [Analysis] Analyzing mean embeddings for $dtype (Multirun)..."
        DATA_CONFIGS="${dtype}_10_hg38_mean,${dtype}_15_hg38_mean,${dtype}_20_hg38_mean,${dtype}_25_hg38_mean,${dtype}_30_hg38_mean,${dtype}_35_hg38_mean,${dtype}_40_hg38_mean,${dtype}_45_hg38_mean,${dtype}_50_hg38_mean,${dtype}_60_hg38_mean,${dtype}_70_hg38_mean,${dtype}_80_hg38_mean,${dtype}_90_hg38_mean,${dtype}_100_hg38_mean"
        
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        wait_for_jobs
        CUDA_VISIBLE_DEVICES=$GPU_ID python embedding_analysis/embedding_analysis_mean.py -m \
            data=$DATA_CONFIGS \
            hydra.sweep.dir="$ANALYSIS_OUTPUT_DIR" \
            hydra.sweep.subdir=\${hydra.job.override_dirname} &
        JOB_IDX=$((JOB_IDX + 1))
    else
        echo "  [Analysis] Output directory $ANALYSIS_OUTPUT_DIR exists, skipping analysis."
    fi
done

wait

# ==============================================================================
# 4. Cross-Dataset Comparison Plots (hg38)
# ==============================================================================
echo "Creating cross-dataset comparison plots (hg38)..."
python plot_cross_dataset.py \
    --eval-dirs outputs/eval_mean_dnabert2 outputs/eval_mean_evo2 outputs/eval_mean_ntv2 \
    --analysis-dirs outputs/analysis_mean_dnabert2 outputs/analysis_mean_evo2 outputs/analysis_mean_ntv2 \
    --output-dir outputs/cross_dataset_comparison

# ==============================================================================
# 3. 1000g Out-of-Distribution Evaluation
# ==============================================================================
echo "Starting 1000g Evaluations..."
MEAN_TYPES=("dnabert2" "evo2" "ntv2")
for dtype in "${MEAN_TYPES[@]}"; do
    OUTPUT_DIR="outputs/train_mean_${dtype}"
    EVAL_1000G_DIR="outputs_1000g/eval_mean_${dtype}"

    if [ -d "$OUTPUT_DIR" ]; then
        if [ ! -d "$EVAL_1000G_DIR" ]; then
            GPU_ID=$((JOB_IDX % NUM_GPUS))
            echo "  [Eval-1000g] Evaluating 1000g for $dtype from $OUTPUT_DIR on GPU $GPU_ID..."
            wait_for_jobs
            CUDA_VISIBLE_DEVICES=$GPU_ID python evaluate_1000g.py run_dir=$OUTPUT_DIR hydra.run.dir="$EVAL_1000G_DIR" &
            JOB_IDX=$((JOB_IDX + 1))
        else
            echo "  [Eval-1000g] Output directory $EVAL_1000G_DIR exists, skipping."
        fi
    else
        echo "  [Eval-1000g] Warning: $OUTPUT_DIR not found, skipping."
    fi
done

echo "Waiting for all 1000g evaluations to complete..."
wait

# ==============================================================================
# 5. Cross-Dataset Comparison Plots (1000g)
# ==============================================================================
echo "Creating cross-dataset comparison plots (1000g)..."
python plot_cross_dataset.py \
    --eval-dirs outputs_1000g/eval_mean_dnabert2 outputs_1000g/eval_mean_evo2 outputs_1000g/eval_mean_ntv2 \
    --output-dir outputs_1000g/cross_dataset_comparison

# ==============================================================================
# 6. Tokenizer Analysis
# ==============================================================================
echo "Starting Tokenizer Analysis..."
TOKENIZER_OUTPUT_DIR="outputs/tokenizer_analysis"

if [ ! -d "$TOKENIZER_OUTPUT_DIR" ]; then
    python embedding_analysis/tokenizer_analysis.py \
        hydra.run.dir="$TOKENIZER_OUTPUT_DIR"
    echo "Tokenizer Analysis Complete."
else
    echo "  [Tokenizer] Output directory $TOKENIZER_OUTPUT_DIR exists, skipping."
fi

echo "All Done."
