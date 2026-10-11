"""Fix HDF5 embedding files with array attributes by converting to scalars.

This script fixes the issue where emb_min, emb_max, emb_mean, emb_std are stored
as arrays instead of scalars, which causes errors during data loading.

Example usage:
    python scrips/fix_embedding_attributes.py data/train_evo2_101_random.h5
    python scrips/fix_embedding_attributes.py data/val_evo2_101_random.h5
    python scrips/fix_embedding_attributes.py data/test_evo2_101_random.h5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np


def fix_attributes(file_path: str) -> None:
    """Convert array attributes to scalar values in HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file to fix.
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Processing {file_path}")

    attr_names = ["emb_min", "emb_max", "emb_mean", "emb_std"]
    fixed = []

    with h5py.File(file_path, "r+") as f:
        if "embeddings" not in f:
            logger.error(f"  'embeddings' dataset not found in {file_path}")
            return

        embeddings = f["embeddings"]
        n_samples = len(embeddings)
        logger.info(f"  Computing statistics for {n_samples} embeddings...")

        # Initialize statistics
        # We'll compute min/max/mean/std globally across all scalar values in all embeddings
        # Use incremental algorithms or just load chunks if memory allows
        # Since we need global min/max/mean/std, we can iterate in chunks

        chunk_size = 10000
        min_val = float("inf")
        max_val = float("-inf")
        sum_val = 0.0
        sum_sq_val = 0.0
        count = 0

        for i in range(0, n_samples, chunk_size):
            # Load chunk
            chunk = embeddings[i : i + chunk_size]
            
            # Handle variable-length embeddings (object array of arrays) or fixed-length (2D array)
            if chunk.dtype == object:
                # Flatten list of arrays
                chunk_flat = np.concatenate([np.asarray(x).flatten() for x in chunk])
            else:
                chunk_flat = chunk.flatten()
            
            if len(chunk_flat) == 0:
                continue

            # Update min/max
            min_val = min(min_val, float(np.min(chunk_flat)))
            max_val = max(max_val, float(np.max(chunk_flat)))

            # Update sums for mean/std
            sum_val += float(np.sum(chunk_flat))
            sum_sq_val += float(np.sum(chunk_flat**2))
            count += len(chunk_flat)
            
            if (i // chunk_size) % 10 == 0:
                logger.info(f"  Processed {min(i + chunk_size, n_samples)}/{n_samples} samples...")

        if count == 0:
            logger.warning("  No data found in embeddings")
            return

        mean_val = sum_val / count
        # std = sqrt(E[x^2] - (E[x])^2)
        std_val = np.sqrt((sum_sq_val / count) - (mean_val**2))

        logger.info(f"  Computed stats - min: {min_val:.6f}, max: {max_val:.6f}, mean: {mean_val:.6f}, std: {std_val:.6f}")

        # Update attributes
        f.attrs["emb_min"] = min_val
        f.attrs["emb_max"] = max_val
        f.attrs["emb_mean"] = mean_val
        f.attrs["emb_std"] = std_val
        
        fixed.extend(["emb_min", "emb_max", "emb_mean", "emb_std"])

    if fixed:
        logger.info(f"Updated attributes: {', '.join(fixed)}")
    else:
        logger.info("No attributes updated")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Fix HDF5 embedding files with array attributes")
    parser.add_argument("file_paths", type=str, nargs="+", help="Path(s) to HDF5 file(s) to fix")

    args = parser.parse_args()

    for file_path in args.file_paths:
        if not Path(file_path).exists():
            logger.error(f"File not found: {file_path}")
            continue

        try:
            fix_attributes(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    logger.info("Done")


if __name__ == "__main__":
    main()
