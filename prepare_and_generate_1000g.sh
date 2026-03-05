#!/bin/bash
set -e

# Default parallelism settings
MAX_JOBS=10
SLEEP_TIME=5
NUM_GPUS=2
JOB_IDX=0

# Function to limit background jobs
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep $SLEEP_TIME
    done
}

# Helper: check whether ALL listed output files already exist
all_exist() {
    for f in "$@"; do
        [ -f "$f" ] || return 1
    done
    return 0
}

# Lengths to process
LENGTHS=(10 25 50 75 100)
SEED=42

echo "Starting 1000g data preparation and embedding generation (Parallel)..."
echo "Lengths: ${LENGTHS[@]}"
echo "Seed: $SEED"
echo "Max Jobs: $MAX_JOBS"
echo "Num GPUs: $NUM_GPUS"

# 1. Prepare 1000g Sequences (CPU-only, run sequentially)
for LEN in "${LENGTHS[@]}"; do
    OUTPUT_CSV="data/1000g_seq${LEN}.csv"
    if [ -f "$OUTPUT_CSV" ]; then
        echo "[Skip] $OUTPUT_CSV already exists."
    else
        echo "[Prepare] Generating $OUTPUT_CSV..."
        python scripts/prepare_1000g.py \
            --seq_length $LEN \
            --output $OUTPUT_CSV \
            --seed $SEED
    fi
done

# 2. Generate Mean-Pooling Embeddings (parallel, round-robin GPUs)
for LEN in "${LENGTHS[@]}"; do
    OUTPUT_CSV="data/1000g_seq${LEN}.csv"

    # DNABERT-2
    TEST="data/test_dnabert2_${LEN}_1000g.h5"
    if all_exist "$TEST"; then
        echo "[Skip] DNABERT-2 mean embeddings for len $LEN already exist."
    else
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        echo "[DNABERT-2] Generating mean embeddings for len $LEN on GPU $GPU_ID..."
        wait_for_jobs
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_dnabert2_embeddings.py \
            input_path=$OUTPUT_CSV \
            seq_length=$LEN \
            num_sequences=15000 \
            mean=true \
            +eval_only=true \
            update_config=false \
            train_output_path="data/train_dnabert2_${LEN}_1000g.h5" \
            val_output_path="data/val_dnabert2_${LEN}_1000g.h5" \
            test_output_path="$TEST" &
        JOB_IDX=$((JOB_IDX + 1))
    fi

    # Evo2
    TEST="data/test_evo2_${LEN}_1000g.h5"
    if all_exist "$TEST"; then
        echo "[Skip] Evo2 mean embeddings for len $LEN already exist."
    else
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        echo "[Evo2] Generating mean embeddings for len $LEN on GPU $GPU_ID..."
        wait_for_jobs
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_evo2_embeddings.py \
            input_path=$OUTPUT_CSV \
            seq_length=$LEN \
            num_sequences=15000 \
            mean=true \
            +eval_only=true \
            update_config=false \
            train_output_path="data/train_evo2_${LEN}_1000g.h5" \
            val_output_path="data/val_evo2_${LEN}_1000g.h5" \
            test_output_path="$TEST" &
        JOB_IDX=$((JOB_IDX + 1))
    fi

    # NTV2
    TEST="data/test_ntv2_${LEN}_1000g.h5"
    if all_exist "$TEST"; then
        echo "[Skip] NTV2 mean embeddings for len $LEN already exist."
    else
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        echo "[NTV2] Generating mean embeddings for len $LEN on GPU $GPU_ID..."
        wait_for_jobs
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_ntv2_embeddings.py \
            input_path=$OUTPUT_CSV \
            seq_length=$LEN \
            num_sequences=15000 \
            mean=true \
            +eval_only=true \
            update_config=false \
            train_output_path="data/train_ntv2_${LEN}_1000g.h5" \
            val_output_path="data/val_ntv2_${LEN}_1000g.h5" \
            test_output_path="$TEST" &
        JOB_IDX=$((JOB_IDX + 1))
    fi
done

echo "Waiting for all jobs to complete..."
wait
echo "All tasks completed successfully."
