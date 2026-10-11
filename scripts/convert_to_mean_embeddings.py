"""Convert per-nucleotide embeddings to mean-pooled embeddings.

This script loads HDF5 files containing per-nucleotide embeddings and converts
them to mean-pooled embeddings, saving space and potentially speeding up training
for mean-based models.

The input HDF5 file should contain:
- 'sequences': variable-length string dataset
- 'embeddings': variable-length float dataset (flattened per-nucleotide embeddings)

The output HDF5 file will contain:
- 'sequences': same as input
- 'embeddings': 1D arrays of length embedding_dim (mean-pooled)

Example usage:
    python scrips/convert_to_mean_embeddings.py data/train_evo2_embeddings.h5 data/train_evo2_mean_embeddings.h5 4096
    python scrips/convert_to_mean_embeddings.py data/val_evo2_embeddings.h5 data/val_evo2_mean_embeddings.h5 4096
    python scrips/convert_to_mean_embeddings.py data/test_evo2_embeddings.h5 data/test_evo2_mean_embeddings.h5 4096
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import file_sha256


def convert_to_mean_embeddings(input_path: str, output_path: str, embedding_dim: int) -> None:
    """Convert per-nucleotide embeddings to mean-pooled embeddings.

    Parameters
    ----------
    input_path : str
        Path to input HDF5 file with per-nucleotide embeddings.
    output_path : str
        Path to output HDF5 file for mean-pooled embeddings.
    embedding_dim : int
        Dimension of embeddings.
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading embeddings from {input_path}")

    sequences = []
    mean_embeddings = []
    input_attrs = {}

    with h5py.File(input_path, "r", swmr=True) as f_in:
        assert "sequences" in f_in, "Missing 'sequences' dataset in input file"
        assert "embeddings" in f_in, "Missing 'embeddings' dataset in input file"

        num_samples = len(f_in["sequences"])
        logger.info(f"Processing {num_samples} sequences")

        for i in range(num_samples):
            seq_bytes = f_in["sequences"][i]
            seq_str = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)
            sequences.append(seq_str)

            emb_flat = np.asarray(f_in["embeddings"][i], dtype=np.float32)
            seq_length = len(emb_flat) // embedding_dim
            emb = emb_flat.reshape(seq_length, embedding_dim)

            assert (
                emb.shape[1] == embedding_dim
            ), f"Expected embedding_dim={embedding_dim}, got {emb.shape[1]}"

            emb_mean = np.mean(emb, axis=0)
            mean_embeddings.append(emb_mean)

            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1}/{num_samples} sequences")

    # Compute statistics on mean-pooled embeddings
    all_mean_values = np.concatenate([emb.flatten() for emb in mean_embeddings])
    computed_min = float(np.min(all_mean_values))
    computed_max = float(np.max(all_mean_values))
    computed_mean = float(np.mean(all_mean_values))
    computed_std = float(np.std(all_mean_values))

    logger.info(f"Mean-pooled embedding statistics:")
    logger.info(f"  min:  {computed_min:.6f}")
    logger.info(f"  max:  {computed_max:.6f}")
    logger.info(f"  mean: {computed_mean:.6f}")
    logger.info(f"  std:  {computed_std:.6f}")

    logger.info(f"Writing mean-pooled embeddings to {output_path}")
    with h5py.File(output_path, "w") as f_out:
        dt_str = h5py.string_dtype(encoding="utf-8")
        seq_dataset = f_out.create_dataset("sequences", (num_samples,), dtype=dt_str)
        for i, seq in enumerate(sequences):
            seq_dataset[i] = seq

        emb_dataset = f_out.create_dataset(
            "embeddings", (num_samples, embedding_dim), dtype=np.float32
        )
        for i, emb_mean in enumerate(mean_embeddings):
            emb_dataset[i] = emb_mean

        f_out.attrs["embedding_dim"] = embedding_dim
        f_out.attrs["emb_min"] = computed_min
        f_out.attrs["emb_max"] = computed_max
        f_out.attrs["emb_mean"] = computed_mean
        f_out.attrs["emb_std"] = computed_std

    logger.info(f"Successfully saved {num_samples} mean-pooled embeddings")

    sha256_hash = file_sha256(output_path)
    logger.info(f"SHA256 hash: {sha256_hash}")
    print(f"\nSHA256: {sha256_hash}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Convert per-nucleotide embeddings to mean-pooled embeddings"
    )
    parser.add_argument(
        "input_path", type=str, help="Path to input HDF5 file with per-nucleotide embeddings"
    )
    parser.add_argument(
        "output_path", type=str, help="Path to output HDF5 file for mean-pooled embeddings"
    )
    parser.add_argument("embedding_dim", type=int, help="Dimension of embeddings (e.g., 4096)")

    args = parser.parse_args()

    logger.info("Starting conversion")
    logger.info(f"Input: {args.input_path}")
    logger.info(f"Output: {args.output_path}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")

    convert_to_mean_embeddings(args.input_path, args.output_path, args.embedding_dim)
    logger.info("Conversion complete")


if __name__ == "__main__":
    main()
