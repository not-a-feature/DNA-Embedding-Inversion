"""Script to generate synthetic DNA sequences and per-nucleotide embeddings for testing.

This script creates random DNA sequences with per-nucleotide embeddings where each position
uses a shifted 5-dimensional block (first 5 values for first nt, next 5 for second nt, etc).
The output is three HDF5 files for train, val, and test with datasets 'sequences' and 'embeddings'

where embeddings contains per-nucleotide embedding matrices
of shape [seq_length x embedding_dim] for each sequence.

Example usage:
    python generate_dummy_mean_embeddings.py --config-name generate_data num_sequences=1000 seq_length=50 embedding_dim=128
    python generate_dummy_mean_embeddings.py num_sequences=500 train_split=0.7 val_split=0.15
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import h5py
from omegaconf import DictConfig
import hydra
import hydra.utils as hy_utils

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import NUCLEOTIDES, file_sha256


def generate_random_sequence(length: int, seed: int | None = None) -> str:
    """Generate a random DNA sequence.

    Parameters
    ----------
    length : int
        Length of the sequence to generate.
    seed : int | None
        Random seed for reproducibility. If None, uses current random state.

    Returns
    -------
    str
        Random DNA sequence.
    """
    if seed is not None:
        np.random.seed(seed)

    nucleotides = np.array(NUCLEOTIDES)
    sequence = "".join(np.random.choice(nucleotides, size=length))
    return sequence


def generate_embedding(
    sequence: str, embedding_dim: int, noise_std: float = 0.1, signal_strength: float = 0.9
) -> np.ndarray:
    """Generate a per-nucleotide embedding for a DNA sequence with shifted blocks.

    Each position in the sequence contributes to ALL rows in a position-specific 4-dimensional block:
    - Position 0: dimensions 0-3 (all rows get signal here)
    - Position 1: dimensions 4-7 (all rows get signal here)
    - Position 2: dimensions 8-11 (all rows get signal here)
    - etc.

    When taking the mean across sequence positions, each 4-dim block becomes a noisy one-hot
    encoding for that position's nucleotide.

    Parameters
    ----------
    sequence : str
        DNA sequence string.
    embedding_dim : int
        Dimension of the embedding for each nucleotide.
    noise_std : float
        Standard deviation of Gaussian noise added to all dimensions.
    signal_strength : float
        Strength of the nucleotide signal in the appropriate 4-dim block (0-1).

    Returns
    -------
    np.ndarray
        Per-nucleotide embedding of shape (len(sequence), embedding_dim).
        Each row has signals in all position-specific blocks based on the nucleotide at that position.
    """
    assert embedding_dim >= len(sequence) * len(NUCLEOTIDES), (
        f"embedding_dim must be at least {len(sequence) * len(NUCLEOTIDES)} "
        f"to accommodate {len(sequence)} positions with {len(NUCLEOTIDES)} dimensions each"
    )
    assert 0.0 <= signal_strength <= 1.0, "signal_strength must be between 0 and 1"
    assert noise_std >= 0.0, "noise_std must be non-negative"

    nucleotide_to_idx = {nuc: i for i, nuc in enumerate(NUCLEOTIDES)}

    # Initialize with Gaussian noise for all dimensions
    embeddings = np.random.normal(0, noise_std, (len(sequence), embedding_dim)).astype(np.float32)

    # For each position in the sequence, add its nucleotide signal to ALL rows
    # in the position-specific 4-dimensional block
    for position, nuc in enumerate(sequence.upper()):
        assert nuc in nucleotide_to_idx, "Sequence contains invalid nucleotides; allowed: A,C,G,T"
        idx = nucleotide_to_idx[nuc]

        # Calculate the starting dimension for this position
        block_start = position * len(NUCLEOTIDES)

        # Add signal strength to ALL rows at this position's block dimension
        embeddings[:, block_start + idx] += signal_strength

    return embeddings


@hydra.main(config_path="../conf", config_name="generate_dummy_embeddings", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate test data and save to separate train/val/test CSVs using Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing data generation parameters.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating synthetic DNA sequences and embeddings with shifted blocks...")
    logger.info(
        f"Configuration: num_sequences={cfg.num_sequences}, seq_length={cfg.seq_length}, embedding_dim={cfg.embedding_dim}"
    )

    np.random.seed(cfg.seed)

    logger.info(f"Generating {cfg.num_sequences} sequences...")
    sequences = []
    embeddings = []

    # Get noise parameters from config with defaults
    noise_std = cfg.get("noise_std", 0.1)
    signal_strength = cfg.get("signal_strength", 0.8)
    logger.info(f"Embedding noise: noise_std={noise_std}, signal_strength={signal_strength}")

    for i in range(cfg.num_sequences):
        seq = generate_random_sequence(cfg.seq_length)
        emb = generate_embedding(seq, cfg.embedding_dim, noise_std, signal_strength)
        sequences.append(seq)
        embeddings.append(emb)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated {i + 1}/{cfg.num_sequences} sequences")

    # Split data into train, val, test
    train_split = cfg.get("train_split", 0.7)
    val_split = cfg.get("val_split", 0.15)
    test_split = 1.0 - train_split - val_split

    assert train_split + val_split + test_split > 0.99, "Split ratios must sum to approximately 1.0"

    n = len(sequences)
    train_n = int(n * train_split)
    val_n = int(n * val_split)

    # Create indices and split
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_idx = indices[:train_n]
    val_idx = indices[train_n : train_n + val_n]
    test_idx = indices[train_n + val_n :]

    logger.info(f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Save each split to NPZ
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    hashes = {}

    for split_name, split_indices in splits.items():
        split_sequences = [sequences[i] for i in split_indices]
        split_embeddings = [embeddings[i] for i in split_indices]

        output_path = hy_utils.to_absolute_path(
            cfg.get(f"{split_name}_output_path", f"data/{split_name}_embeddings.h5")
        )

        # Store as HDF5 with variable-length datasets for efficient lazy loading
        with h5py.File(output_path, "w") as f:
            # Create variable-length string dataset for sequences
            dt_str = h5py.string_dtype(encoding="utf-8")
            seq_dataset = f.create_dataset("sequences", (len(split_sequences),), dtype=dt_str)
            for i, seq in enumerate(split_sequences):
                seq_dataset[i] = seq

            # Create variable-length float dataset for embeddings
            dt_vlen = h5py.vlen_dtype(np.float32)
            emb_dataset = f.create_dataset("embeddings", (len(split_embeddings),), dtype=dt_vlen)
            for i, emb in enumerate(split_embeddings):
                emb_dataset[i] = emb.flatten()

            # Store shape metadata as attributes
            f.attrs["embedding_dim"] = cfg.embedding_dim
            f.attrs["seq_length"] = cfg.seq_length

        logger.info(f"Saved {len(split_sequences)} {split_name} sequences to {output_path}")
        logger.info(f"Number of embeddings: {len(split_embeddings)}")

        # Compute SHA256
        sha256_hash = file_sha256(output_path)
        hashes[split_name] = sha256_hash
        logger.info(f"SHA256 hash of {output_path}: {sha256_hash}")

    print(f"Update conf/config.yaml data section:")
    print(f"train_sha256: {hashes['train']}")
    print(f"val_sha256: {hashes['val']}")
    print(f"test_sha256: {hashes['test']}")


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
