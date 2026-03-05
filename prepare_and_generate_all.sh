#!/bin/bash
set -e

# Default parallelism settings
MAX_JOBS=20
SLEEP_TIME=5
NUM_GPUS=2
JOB_IDX=0

# limit background jobs
wait_for_jobs() {
    while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
        sleep $SLEEP_TIME
    done
}

# Check whether ALL listed output files already exist
all_exist() {
    for f in "$@"; do
        [ -f "$f" ] || return 1
    done
    return 0
}

# Lengths to process
LENGTHS=(10 15 20 25 30 35 40 45 50 60 70 80 90 100)
#LENGTHS=(100)
SEED=42

echo "Starting data preparation and embedding generation (Parallel)..."
echo "Lengths: ${LENGTHS[@]}"
echo "Seed: $SEED"
echo "Max Jobs: $MAX_JOBS"
echo "Num GPUs: $NUM_GPUS"

# 1. Prepare HG38 Sequences (CPU-only, run sequentially)
for LEN in "${LENGTHS[@]}"; do
    OUTPUT_CSV="data/hg38_seq${LEN}.csv"
    if [ -f "$OUTPUT_CSV" ]; then
        echo "[Skip] $OUTPUT_CSV already exists."
    else
        echo "[Prepare] Generating $OUTPUT_CSV..."
        python scripts/prepare_hg38.py \
            --seq_length $LEN \
            --output $OUTPUT_CSV \
            --max_sequences 100000 \
            --seed $SEED
    fi
done

# 2. Generate Mean-Pooling Embeddings (parallel, round-robin GPUs)
for LEN in "${LENGTHS[@]}"; do
    OUTPUT_CSV="data/hg38_seq${LEN}.csv"

    # DNABERT-2
    TRAIN="data/train_dnabert2_${LEN}_hg38.h5"
    VAL="data/val_dnabert2_${LEN}_hg38.h5"
    TEST="data/test_dnabert2_${LEN}_hg38.h5"
    if all_exist "$TRAIN" "$VAL" "$TEST"; then
        echo "[Skip] DNABERT-2 mean embeddings for len $LEN already exist."
    else
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        echo "[DNABERT-2] Generating mean embeddings for len $LEN on GPU $GPU_ID..."
        wait_for_jobs
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_dnabert2_embeddings.py \
            input_path=$OUTPUT_CSV \
            seq_length=$LEN \
            mean=true \
            train_output_path="$TRAIN" \
            val_output_path="$VAL" \
            test_output_path="$TEST" \
            update_config="conf/data/dnabert2_${LEN}_hg38_mean.yaml" &
        JOB_IDX=$((JOB_IDX + 1))
    fi

    # Evo2
    TRAIN="data/train_evo2_deep_${LEN}_hg38.h5"
    VAL="data/val_evo2_deep_${LEN}_hg38.h5"
    TEST="data/test_evo2_deep_${LEN}_hg38.h5"
    if all_exist "$TRAIN" "$VAL" "$TEST"; then
        echo "[Skip] Evo2 mean embeddings for len $LEN already exist."
    else
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        echo "[Evo2] Generating mean embeddings for len $LEN on GPU $GPU_ID..."
        wait_for_jobs
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_evo2_embeddings.py \
            input_path=$OUTPUT_CSV \
            seq_length=$LEN \
            mean=true \
            train_output_path="$TRAIN" \
            val_output_path="$VAL" \
            test_output_path="$TEST" \
            update_config="conf/data/evo2_${LEN}_hg38_mean.yaml" &
        JOB_IDX=$((JOB_IDX + 1))
    fi

    # NTV2
    TRAIN="data/train_ntv2_${LEN}_hg38.h5"
    VAL="data/val_ntv2_${LEN}_hg38.h5"
    TEST="data/test_ntv2_${LEN}_hg38.h5"
    if all_exist "$TRAIN" "$VAL" "$TEST"; then
        echo "[Skip] NTV2 mean embeddings for len $LEN already exist."
    else
        GPU_ID=$((JOB_IDX % NUM_GPUS))
        echo "[NTV2] Generating mean embeddings for len $LEN on GPU $GPU_ID..."
        wait_for_jobs
        CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_ntv2_embeddings.py \
            input_path=$OUTPUT_CSV \
            seq_length=$LEN \
            mean=true \
            train_output_path="$TRAIN" \
            val_output_path="$VAL" \
            test_output_path="$TEST" \
            update_config="conf/data/ntv2_${LEN}_hg38_mean.yaml" &
        JOB_IDX=$((JOB_IDX + 1))
    fi
done

# 3. Generate Per-Nucleotide Embeddings (Length 100 ONLY)
LEN=100
OUTPUT_CSV="data/hg38_seq${LEN}.csv"
echo "[Per-Token] Generating per-nucleotide embeddings for len $LEN..."

# DNABERT-2
TRAIN="data/train_dnabert2_${LEN}_hg38_per_token.h5"
VAL="data/val_dnabert2_${LEN}_hg38_per_token.h5"
TEST="data/test_dnabert2_${LEN}_hg38_per_token.h5"
if all_exist "$TRAIN" "$VAL" "$TEST"; then
    echo "[Skip] DNABERT-2 per-token embeddings for len $LEN already exist."
else
    GPU_ID=$((JOB_IDX % NUM_GPUS))
    echo "[DNABERT-2] Generating per-token embeddings for len $LEN on GPU $GPU_ID..."
    wait_for_jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_dnabert2_embeddings.py \
        input_path=$OUTPUT_CSV \
        seq_length=$LEN \
        mean=false \
        train_output_path="$TRAIN" \
        val_output_path="$VAL" \
        test_output_path="$TEST" \
        update_config="conf/data/dnabert2_${LEN}_hg38_per_token.yaml" &
    JOB_IDX=$((JOB_IDX + 1))
fi

# Evo2
TRAIN="data/train_evo2_${LEN}_hg38_per_token.h5"
VAL="data/val_evo2_${LEN}_hg38_per_token.h5"
TEST="data/test_evo2_${LEN}_hg38_per_token.h5"
if all_exist "$TRAIN" "$VAL" "$TEST"; then
    echo "[Skip] Evo2 per-token embeddings for len $LEN already exist."
else
    GPU_ID=$((JOB_IDX % NUM_GPUS))
    echo "[Evo2] Generating per-token embeddings for len $LEN on GPU $GPU_ID..."
    wait_for_jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_evo2_embeddings.py \
        input_path=$OUTPUT_CSV \
        seq_length=$LEN \
        mean=false \
        train_output_path="$TRAIN" \
        val_output_path="$VAL" \
        test_output_path="$TEST" \
        update_config="conf/data/evo2_${LEN}_hg38_per_token.yaml" &
    JOB_IDX=$((JOB_IDX + 1))
fi

# NTV2
TRAIN="data/train_ntv2_${LEN}_hg38_per_token.h5"
VAL="data/val_ntv2_${LEN}_hg38_per_token.h5"
TEST="data/test_ntv2_${LEN}_hg38_per_token.h5"
if all_exist "$TRAIN" "$VAL" "$TEST"; then
    echo "[Skip] NTV2 per-token embeddings for len $LEN already exist."
else
    GPU_ID=$((JOB_IDX % NUM_GPUS))
    echo "[NTV2] Generating per-token embeddings for len $LEN on GPU $GPU_ID..."
    wait_for_jobs
    CUDA_VISIBLE_DEVICES=$GPU_ID python generate/generate_ntv2_embeddings.py \
        input_path=$OUTPUT_CSV \
        seq_length=$LEN \
        mean=false \
        train_output_path="$TRAIN" \
        val_output_path="$VAL" \
        test_output_path="$TEST" \
        update_config="conf/data/ntv2_${LEN}_hg38_per_token.yaml" &
    JOB_IDX=$((JOB_IDX + 1))
fi

echo "Waiting for all jobs to complete..."
wait
echo "All tasks completed successfully."
