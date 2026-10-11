"""Data utilities and dataset definitions for DNA sequence reconstruction."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import hydra.utils as hy_utils
import h5py
import logging

from omegaconf import DictConfig
from src.utils import file_sha256
from src.tokenizers import BaseTokenizer


class DNAEmbeddingDataset(Dataset):
    """Dataset wrapper for per-nucleotide embeddings and DNA sequences.

    This dataset uses lazy loading via HDF5 - embeddings are loaded from disk on-the-fly
    during __getitem__ to minimize memory usage for large datasets.

    Each sample contains:
    - embeddings: shape (seq_length, embedding_dim) - loaded on-the-fly
    - sequences_indices: shape (seq_length,) - computed on-the-fly (LongTensor)
    """

    def __init__(
        self,
        h5_file: h5py.File,
        tokenizer: BaseTokenizer,
        embedding_dim: int,
        seq_length: int | None = None,
        normalization_stats: Dict[str, float] | None = None,
        normalization_method: str = "standard",
        max_samples: int | None = None,
    ):
        assert tokenizer is not None, "Tokenizer must be provided"
        assert seq_length is None or seq_length > 0, "seq_length must be positive or None"
        assert normalization_method in [
            "standard",
            "minmax",
        ], f"Invalid normalization method: {normalization_method}"
        assert max_samples is None or max_samples > 0, "max_samples must be positive or None"

        self.h5_file = h5_file
        self.embeddings = h5_file["embeddings"]
        self.sequences = h5_file["sequences"]
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.normalization_stats = normalization_stats
        self.normalization_method = normalization_method
        self.max_samples = max_samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:  # noqa: D401
        total_len = len(self.embeddings)
        if self.max_samples is not None:
            return min(total_len, self.max_samples)
        return total_len

    def __getitem__(self, idx: int):  # noqa: D401
        # Load embedding from disk on-the-fly (HDF5 handles memory mapping)
        emb_flat = np.asarray(self.embeddings[idx], dtype=np.float32)

        # Reshape from flattened array back to 2D [seq_length, embedding_dim]
        seq_length = len(emb_flat) // self.embedding_dim
        emb = emb_flat.reshape(seq_length, self.embedding_dim)
        assert emb.ndim == 2 and emb.shape[1] == self.embedding_dim

        # Truncate embeddings if seq_length is specified
        if self.seq_length is not None:
            emb = emb[: self.seq_length, :]

        # Apply normalization using training set statistics (if provided)
        if self.normalization_stats:
            if self.normalization_method == "standard":
                emb = (emb - self.normalization_stats["mean"]) / self.normalization_stats["std"]
            else:  # minmax
                emb = (emb - self.normalization_stats["min"]) / (
                    self.normalization_stats["max"] - self.normalization_stats["min"]
                )

        # Decode sequence from bytes and tokenize on-the-fly
        seq_bytes = self.sequences[idx]
        seq_str = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)

        seq_indices = self.tokenizer.encode(seq_str)

        # Truncate sequence indices if seq_length is specified (to match embedding truncation)
        if self.seq_length is not None:
            seq_indices = seq_indices[: self.seq_length]

        # Convert to torch tensors
        emb_tensor = torch.from_numpy(emb).float()
        # seq_indices is already a LongTensor from tokenizer
        return emb_tensor, seq_indices


class DNAMeanEmbeddingDataset(Dataset):
    """Dataset wrapper for mean nucleotide embeddings and DNA sequences.

    This dataset uses lazy loading via HDF5. Supports both:
    - Pre-computed mean embeddings (data_is_precomputed=True)
    - On-the-fly mean computation from per-nucleotide embeddings (data_is_precomputed=False)

    Each sample contains:
    - embeddings: shape (embedding_dim) - mean-pooled embedding
    - sequences_indices: shape (seq_length,) - computed on-the-fly (LongTensor)
    """

    def __init__(
        self,
        h5_file: h5py.File,
        tokenizer: BaseTokenizer,
        embedding_dim: int,
        seq_length: int | None = None,
        normalization_stats: Dict[str, float] | None = None,
        normalization_method: str = "standard",
        data_is_precomputed: bool = False,
        max_samples: int | None = None,
    ):
        assert tokenizer is not None, "Tokenizer must be provided"
        assert seq_length is None or seq_length > 0, "seq_length must be positive or None"
        assert normalization_method in [
            "standard",
            "minmax",
        ], f"Invalid normalization method: {normalization_method}"
        assert max_samples is None or max_samples > 0, "max_samples must be positive or None"

        self.h5_file = h5_file
        self.embeddings = h5_file["embeddings"]
        self.sequences = h5_file["sequences"]
        self.embedding_dim = embedding_dim
        self.seq_length = seq_length
        self.normalization_stats = normalization_stats
        self.normalization_method = normalization_method
        self.data_is_precomputed = data_is_precomputed
        self.max_samples = max_samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:  # noqa: D401
        total_len = len(self.embeddings)
        if self.max_samples is not None:
            return min(total_len, self.max_samples)
        return total_len

    def __getitem__(self, idx: int):  # noqa: D401
        # Load embedding from disk on-the-fly (HDF5 handles memory mapping)
        emb_flat = np.asarray(self.embeddings[idx], dtype=np.float32)

        if self.data_is_precomputed:
            # Embeddings are already mean-pooled - just validate shape
            assert emb_flat.ndim == 1 and len(emb_flat) == self.embedding_dim
            emb_mean = emb_flat
        else:
            # Compute mean from per-nucleotide embeddings
            seq_length = len(emb_flat) // self.embedding_dim
            emb = emb_flat.reshape(seq_length, self.embedding_dim)
            assert emb.ndim == 2 and emb.shape[1] == self.embedding_dim

            # Truncate embeddings if seq_length is specified
            if self.seq_length is not None:
                emb = emb[: self.seq_length, :]

            # Compute mean embedding on-the-fly to save memory
            emb_mean = np.mean(emb, axis=0)

        # Apply normalization using training set statistics (if provided)
        if self.normalization_stats:
            if self.normalization_method == "standard":
                emb_mean = (emb_mean - self.normalization_stats["mean"]) / self.normalization_stats[
                    "std"
                ]
            else:  # minmax
                emb_mean = (emb_mean - self.normalization_stats["min"]) / (
                    self.normalization_stats["max"] - self.normalization_stats["min"]
                )

        # Decode sequence from bytes and tokenize on-the-fly
        seq_bytes = self.sequences[idx]
        seq_str = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)

        seq_indices = self.tokenizer.encode(seq_str)

        # Truncate sequence indices if seq_length is specified
        if self.seq_length is not None:
            seq_indices = seq_indices[: self.seq_length]

        # Convert to torch tensors
        emb_tensor = torch.from_numpy(emb_mean).float()
        # seq_indices is already a LongTensor from tokenizer
        return emb_tensor, seq_indices


def load_split_embeddings(
    cfg: DictConfig,
) -> Tuple[Dict[str, h5py.File], Dict[str, int], Dict[str, Dict[str, float]]]:
    """Load DNA sequences and embeddings from separate train/val/test HDF5 files.

    HDF5 format: Contains 'sequences' (variable-length string dataset) and
    'embeddings' (variable-length float dataset with per-nucleotide embeddings).

    This function uses HDF5 for true lazy loading with memory mapping.
    Data is loaded on-the-fly when accessed via dataset __getitem__.

    Parameters
    ----------
    cfg : DictConfig
        Data configuration containing paths for train/val/test HDF5 files and embedding dimension.

    Returns
    -------
    Tuple[Dict[str, h5py.File], Dict[str, int], Dict[str, float]]
        - data_dict: dictionary with keys 'train', 'val', 'test' containing HDF5 file handles
        - counts_dict: dictionary with keys 'train', 'val', 'test' containing sample counts
        - train_stats: dictionary containing training set normalization statistics
          (with keys 'min', 'max', 'mean', 'std') to be used for all splits
    """
    data_dict = {}
    counts_dict = {}
    train_stats = {}

    for split in ["train", "val", "test"]:
        csv_key = f"{split}_csv"
        sha_key = f"{split}_sha256"

        path = hy_utils.to_absolute_path(cfg[csv_key])
        assert path.endswith(".h5") or path.endswith(
            ".hdf5"
        ), f"Expected HDF5 file for {split}, got {path}"

        # Skip sha256 check if configured to do so (key is optional)
        if not cfg.skip_sha256_check:
            assert file_sha256(path) == cfg[sha_key], (
                f"SHA256 mismatch for {split} data file. "
                "Expected configured hash; data file may be corrupted or changed."
            )

        # Load HDF5 file in read-only mode with SWMR for concurrent access
        h5_file = h5py.File(path, "r", swmr=True)

        assert "sequences" in h5_file, f"Key 'sequences' missing from {split} HDF5 file"
        assert "embeddings" in h5_file, f"Key 'embeddings' missing from {split} HDF5 file"

        # Validate first sample to ensure correct format
        first_emb_flat = np.asarray(h5_file["embeddings"][0], dtype=np.float32)
        assert first_emb_flat.ndim == 1, f"Expected 1D embedding array in {split}"

        # Check if data is mean-pooled or per-nucleotide based on config
        if cfg.mean:
            # Mean embeddings: should be exactly embedding_dim length
            assert len(first_emb_flat) == cfg.embedding_dim, (
                f"Expected mean embedding of length {cfg.embedding_dim} in {split}, "
                f"got {len(first_emb_flat)}"
            )
        else:
            # Per-nucleotide embeddings: should be seq_length * embedding_dim
            seq_length = len(first_emb_flat) // cfg.embedding_dim
            first_emb = first_emb_flat.reshape(seq_length, cfg.embedding_dim)
            assert first_emb.shape[1] == cfg.embedding_dim, (
                f"Expected embedding_dim={cfg.embedding_dim} in {split}, "
                f"got {first_emb.shape[1]} for first sample"
            )

        data_dict[split] = h5_file
        counts_dict[split] = len(h5_file["embeddings"])

        # Load normalization statistics only from training set to avoid data leakage
        if split == "train" and "emb_min" in h5_file.attrs:
            # All statistics should be scalars representing global values across all embeddings
            def to_scalar(value) -> float:
                if isinstance(value, np.ndarray):
                    assert value.size == 1, (
                        f"Expected scalar attribute, got array of size {value.size}. "
                        "Statistics should be global (computed across all embedding values), not per-dimension."
                    )
                    return float(value.item())
                return float(value)

            train_stats["min"] = to_scalar(h5_file.attrs["emb_min"])
            train_stats["max"] = to_scalar(h5_file.attrs["emb_max"])
            train_stats["mean"] = to_scalar(h5_file.attrs["emb_mean"])
            train_stats["std"] = to_scalar(h5_file.attrs["emb_std"])

    return data_dict, counts_dict, train_stats


def create_dataset(
    data: h5py.File,
    mode: str,
    tokenizer: BaseTokenizer,
    embedding_dim: int,
    seq_length: int | None = None,
    normalization_stats: Dict[str, float] | None = None,
    normalization_method: str = "standard",
    data_is_mean: bool = False,
    subset_fraction: float | None = None,
    max_samples: int | None = None,
) -> Dataset:
    """Create dataset based on mode with lazy loading via HDF5.

    Parameters
    ----------
    data : h5py.File
        HDF5 file handle containing embeddings and sequences.
    mode : str
        Either "per_token" or "mean".
    tokenizer : BaseTokenizer
        Tokenizer instance to encode sequences.
    embedding_dim : int
        Expected embedding dimension for validation.
    seq_length : int | None
        Sequence length to use. If specified, sequences and embeddings will be truncated.
    normalization_stats : Dict[str, float] | None
        Training set normalization statistics to apply. Should contain keys 'min', 'max', 'mean', 'std'.
        All splits (train, val, test) should use the same training statistics to avoid data leakage.
    normalization_method : str
        Normalization method: 'standard' (z-score) or 'minmax' (0-1 range).
    data_is_mean : bool
        If True, embeddings are already mean-pooled. If False, they are per-nucleotide.
    subset_fraction : float | None
        Fraction of data to use (0.0 to 1.0). If None, use all data.
    max_samples : int | None
        Maximum number of samples to use. If provided, overrides subset_fraction.

    Returns
    -------
    Dataset
        Either DNAEmbeddingDataset or DNAMeanEmbeddingDataset, optionally wrapped in Subset.
        All use lazy loading - data is loaded from disk on-the-fly during __getitem__.
    """
    assert mode in ["per_token", "mean"], f"Invalid mode: {mode}"

    # Calculate max_samples if subset_fraction is provided
    if max_samples is None and subset_fraction is not None:
        logger = logging.getLogger(__name__)
        total_size = len(data["embeddings"])
        max_samples = int(total_size * subset_fraction)
        max_samples = max(1, max_samples)  # Ensure at least 1 sample

        logger.info(
            f"Subsetting data: {max_samples} samples "
            f"({subset_fraction * 100:.1f}% of {total_size})"
        )
    elif max_samples is not None:
        logger = logging.getLogger(__name__)
        total_size = len(data["embeddings"])
        logger.info(f"Using max_samples: {max_samples} (total available: {total_size})")

    if mode == "per_token":
        dataset = DNAEmbeddingDataset(
            data,
            tokenizer,
            embedding_dim,
            seq_length,
            normalization_stats,
            normalization_method,
            max_samples=max_samples,
        )
    else:
        assert mode == "mean"
        dataset = DNAMeanEmbeddingDataset(
            data,
            tokenizer,
            embedding_dim,
            seq_length,
            normalization_stats,
            normalization_method,
            data_is_precomputed=data_is_mean,
            max_samples=max_samples,
        )

    return dataset
