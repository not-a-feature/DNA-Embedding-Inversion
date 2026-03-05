"""Script to compute and add statistical parameters to existing embedding datasets.

This script reads existing HDF5 or NPZ files containing embeddings and computes
min, max, mean, and std statistics for normalization purposes. The statistics are
added to the files without regenerating the embeddings.

Example usage:
    python compute_embedding_statistics.py data/train_evo2_embeddings.h5
    python compute_embedding_statistics.py data/*.h5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import h5py

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import file_sha256


def setup_logging():
    """Configure logging with colored output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def compute_and_add_statistics_h5(file_path: str, overwrite: bool = False) -> None:
    """Compute statistics for HDF5 file and add as attributes.

    Parameters
    ----------
    file_path : str
        Path to HDF5 file containing embeddings.
    overwrite : bool
        If True, overwrite existing statistics. If False, skip if already present.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing HDF5 file: {file_path}")

    # Check if file exists
    path = Path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    assert path.suffix in [".h5", ".hdf5"], f"Expected HDF5 file, got: {file_path}"

    # Open file in read-write mode
    with h5py.File(file_path, "r+") as f:
        # Check if statistics already exist
        if not overwrite and "emb_min" in f.attrs:
            logger.info(f"Statistics already exist in {file_path}, skipping...")
            logger.info(f"  Existing min:  {f.attrs['emb_min']:.6f}")
            logger.info(f"  Existing max:  {f.attrs['emb_max']:.6f}")
            logger.info(f"  Existing mean: {f.attrs['emb_mean']:.6f}")
            logger.info(f"  Existing std:  {f.attrs['emb_std']:.6f}")
            logger.info("Use --overwrite to recompute.")
            return

        assert "embeddings" in f, f"Key 'embeddings' missing from {file_path}"

        # Get embedding dimension from attributes
        embedding_dim = f.attrs.get("embedding_dim", None)
        assert embedding_dim is not None, f"embedding_dim attribute missing from {file_path}"

        logger.info(f"Loading embeddings from {file_path}...")
        logger.info(f"Number of samples: {len(f['embeddings'])}")
        logger.info(f"Embedding dimension: {embedding_dim}")

        # Load all embeddings and flatten
        embeddings_list = []
        for i in range(len(f["embeddings"])):
            emb_flat = np.asarray(f["embeddings"][i], dtype=np.float32)
            embeddings_list.append(emb_flat)

            if (i + 1) % 1000 == 0:
                logger.info(f"Loaded {i + 1}/{len(f['embeddings'])} embeddings...")

        all_emb_values = np.concatenate(embeddings_list)
        logger.info(f"Total embedding values: {len(all_emb_values)}")

        # Compute statistics
        emb_min = float(np.min(all_emb_values))
        emb_max = float(np.max(all_emb_values))
        emb_mean = float(np.mean(all_emb_values))
        emb_std = float(np.std(all_emb_values))

        logger.info(f"Computed embedding statistics:")
        logger.info(f"  min:  {emb_min:.6f}")
        logger.info(f"  max:  {emb_max:.6f}")
        logger.info(f"  mean: {emb_mean:.6f}")
        logger.info(f"  std:  {emb_std:.6f}")

        # Add statistics as attributes
        f.attrs["emb_min"] = emb_min
        f.attrs["emb_max"] = emb_max
        f.attrs["emb_mean"] = emb_mean
        f.attrs["emb_std"] = emb_std

        logger.info(f"Statistics added to {file_path}")

    # Compute new SHA256 hash
    sha256_hash = file_sha256(file_path)
    logger.info(f"New SHA256 hash: {sha256_hash}")
    logger.info("")


def compute_and_add_statistics_npz(file_path: str, overwrite: bool = False) -> None:
    """Compute statistics for NPZ file and re-save with statistics.

    Parameters
    ----------
    file_path : str
        Path to NPZ file containing embeddings.
    overwrite : bool
        If True, overwrite existing statistics. If False, skip if already present.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing NPZ file: {file_path}")

    # Check if file exists
    path = Path(file_path)
    assert path.exists(), f"File not found: {file_path}"
    assert path.suffix == ".npz", f"Expected NPZ file, got: {file_path}"

    # Load NPZ file
    data = np.load(file_path, allow_pickle=True)

    # Check if statistics already exist
    if not overwrite and "emb_min" in data:
        logger.info(f"Statistics already exist in {file_path}, skipping...")
        logger.info(f"  Existing min:  {data['emb_min']:.6f}")
        logger.info(f"  Existing max:  {data['emb_max']:.6f}")
        logger.info(f"  Existing mean: {data['emb_mean']:.6f}")
        logger.info(f"  Existing std:  {data['emb_std']:.6f}")
        logger.info("Use --overwrite to recompute.")
        return

    assert "embeddings" in data, f"Key 'embeddings' missing from {file_path}"

    embeddings = data["embeddings"]
    sequences = data["sequences"]

    logger.info(f"Number of samples: {len(embeddings)}")

    # Flatten all embeddings
    all_emb_values = np.concatenate([emb.flatten() for emb in embeddings])
    logger.info(f"Total embedding values: {len(all_emb_values)}")

    # Compute statistics
    emb_min = float(np.min(all_emb_values))
    emb_max = float(np.max(all_emb_values))
    emb_mean = float(np.mean(all_emb_values))
    emb_std = float(np.std(all_emb_values))

    logger.info(f"Computed embedding statistics:")
    logger.info(f"  min:  {emb_min:.6f}")
    logger.info(f"  max:  {emb_max:.6f}")
    logger.info(f"  mean: {emb_mean:.6f}")
    logger.info(f"  std:  {emb_std:.6f}")

    # Re-save with statistics
    np.savez_compressed(
        file_path,
        sequences=sequences,
        embeddings=embeddings,
        emb_min=emb_min,
        emb_max=emb_max,
        emb_mean=emb_mean,
        emb_std=emb_std,
    )

    logger.info(f"Statistics added to {file_path}")

    # Compute new SHA256 hash
    sha256_hash = file_sha256(file_path)
    logger.info(f"New SHA256 hash: {sha256_hash}")
    logger.info("")


def process_file(file_path: str, overwrite: bool = False) -> None:
    """Process a single file based on its extension.

    Parameters
    ----------
    file_path : str
        Path to the file to process.
    overwrite : bool
        If True, overwrite existing statistics.
    """
    path = Path(file_path)

    if path.suffix in [".h5", ".hdf5"]:
        compute_and_add_statistics_h5(str(path), overwrite)
    elif path.suffix == ".npz":
        compute_and_add_statistics_npz(str(path), overwrite)
    else:
        logging.warning(f"Unsupported file type: {file_path}, skipping...")


def main():
    """Main entry point for the script."""
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Compute and add statistical parameters to embedding datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Path(s) to HDF5 or NPZ files containing embeddings. Supports glob patterns.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing statistics if present.",
    )

    args = parser.parse_args()

    # Expand glob patterns
    files_to_process: List[str] = []
    for pattern in args.files:
        matches = list(Path().glob(pattern))
        if matches:
            files_to_process.extend([str(p) for p in matches])
        else:
            # If no glob match, treat as literal path
            files_to_process.append(pattern)

    if not files_to_process:
        logger.error("No files to process.")
        sys.exit(1)

    logger.info(f"Found {len(files_to_process)} file(s) to process.")
    logger.info("")

    for file_path in files_to_process:
        try:
            process_file(file_path, args.overwrite)
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue

    logger.info("Processing complete.")


if __name__ == "__main__":  # pragma: no cover
    main()
