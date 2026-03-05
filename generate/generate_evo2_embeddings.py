"""Script to extract Evo2 embeddings for DNA sequences from a file.

The output is HDF5 files for train, val, and test with datasets 'sequences' and 'embeddings'.

If mean=false (default): embeddings contains per-nucleotide embedding matrices
of shape [seq_length x embedding_dim] for each sequence (variable-length).

If mean=true: embeddings contains mean-pooled embeddings of shape [embedding_dim]
for each sequence (fixed-size).

Layer name is configured in the YAML file.

Example usage:
    python generate_evo2_embeddings.py input_path=data.csv seq_length=50
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import h5py
from omegaconf import DictConfig
import hydra
import hydra.utils as hy_utils
from evo2 import Evo2

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import NUCLEOTIDES, file_sha256, load_sequences_from_file, update_yaml_keys


def embed_sequence(model: Evo2, sequence: str, layer_name: str, device: str) -> np.ndarray:
    """Extract embeddings for a DNA sequence from specified layer.

    Parameters
    ----------
    model : Evo
        Evo2 model instance.
    sequence : str
        DNA sequence string.
    layer_name : str
        Name of the Evo2 layer to extract embeddings from.
    device : str
        Device to place input tensors on.

    Returns
    -------
    np.ndarray
        Per-nucleotide embedding of shape (len(sequence), embedding_dim).
    """
    tokens = model.tokenizer.tokenize(sequence)
    assert tokens, "Sequence must tokenize to at least one token"

    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs, embeddings = model(input_ids, return_embeddings=True, layer_names=[layer_name])

    assert layer_name in embeddings, f"Layer '{layer_name}' not found in embeddings"

    tensor = embeddings[layer_name][0].detach().float().cpu()
    return tensor.numpy()


def generate_embeddings_with_layer(
    sequences: List[str], layer_name: str, checkpoint: str, device: str
) -> List[np.ndarray]:
    """Generate Evo2 embeddings for multiple sequences using specified layer.

    Parameters
    ----------
    sequences : List[str]
        List of DNA sequences.
    layer_name : str
        Evo2 layer name to extract embeddings from.
    checkpoint : str
        Evo2 model checkpoint.
    device : str
        Device to place input tensors on.

    Returns
    -------
    List[np.ndarray]
        List of per-nucleotide embeddings.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading Evo2 model: {checkpoint}")

    model = Evo2(checkpoint)

    logger.info(f"Evo2 model {checkpoint} loaded successfully")

    embeddings = []
    for i, seq in enumerate(sequences):
        emb = embed_sequence(model, seq, layer_name, device)
        embeddings.append(emb)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated embeddings for {i + 1}/{len(sequences)} sequences")

    return embeddings


@hydra.main(config_path="../conf", config_name="generate/evo2", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate DNA sequences and Evo2 embeddings using Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing data generation parameters.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating DNA sequences and Evo2 embeddings...")

    np.random.seed(cfg.seed)

    # Load sequences from file or generate random ones
    if cfg.input_path is None:
        raise ValueError("input_path must be provided in the configuration.")

    input_path = hy_utils.to_absolute_path(cfg.input_path)
    logger.info(f"Loading sequences from file: {input_path}")
    logger.info(f"Max sequences (num_sequences): {cfg.num_sequences}")
    logger.info(f"Max sequence length (seq_length): {cfg.seq_length}")
    sequences = load_sequences_from_file(
        input_path, max_length=cfg.seq_length, max_sequences=cfg.num_sequences
    )

    # Remove duplicate sequences
    original_count = len(sequences)
    sequences = list(dict.fromkeys(sequences))
    duplicates_removed = original_count - len(sequences)

    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate sequences")

    logger.info(
        f"Loaded {len(sequences)} unique sequences from file (limited to {cfg.num_sequences}, truncated to {cfg.seq_length} if needed)"
    )

    # Extract embeddings from configured layer
    layer_name = cfg.layer_name
    checkpoint = cfg.checkpoint
    device = cfg.device

    logger.info(f"\n")
    logger.info(f"Extracting embeddings from layer: {layer_name}")
    logger.info(f"Using checkpoint: {checkpoint}")
    logger.info(f"Using device: {device}")
    logger.info(f"{'=' * 80}")

    embeddings = generate_embeddings_with_layer(sequences, layer_name, checkpoint, device)

    # Split data into train, val, test
    eval_only = cfg.get("eval_only", False)
    if eval_only:
        train_split = 0.0
        val_split = 0.0
        test_split = 1.0
    else:
        train_split = cfg.train_split
        val_split = cfg.val_split
        test_split = 1.0 - train_split - val_split

        assert (
            train_split + val_split + test_split > 0.99
        ), "Split ratios must sum to approximately 1.0"

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
    if eval_only:
        splits = {"test": test_idx}
    else:
        splits = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

    # Compute global statistics from TRAINING SET only
    # This ensures consistent normalization across all splits and avoids data leakage
    use_mean = cfg["mean"]
    if eval_only:
        global_min, global_max, global_mean, global_std = 0.0, 0.0, 0.0, 0.0
        embedding_dim = embeddings[0].shape[1] if embeddings else 0
        logger.info("Skipping global statistics computation because eval_only mode is on.")
    else:
        train_sequences = [sequences[i] for i in splits["train"]]
        train_embeddings = [embeddings[i] for i in splits["train"]]

        if use_mean:
            logger.info("Computing global statistics from training set (mean-pooled embeddings)...")
            # Compute mean pooling for training embeddings
            train_mean_embeddings = [np.mean(emb, axis=0) for emb in train_embeddings]
            # Compute global statistics across ALL values in the training set
            all_train_values = np.concatenate([emb.flatten() for emb in train_mean_embeddings])
            embedding_dim = train_embeddings[0].shape[1]
        else:
            logger.info(
                "Computing global statistics from training set (per-nucleotide embeddings)..."
            )
            # Compute global statistics across ALL values in the training set
            all_train_values = np.concatenate([emb.flatten() for emb in train_embeddings])
            embedding_dim = train_embeddings[0].shape[1]

        # Compute global scalar statistics (not per-dimension)
        global_min = float(np.min(all_train_values))
        global_max = float(np.max(all_train_values))
        global_mean = float(np.mean(all_train_values))
        global_std = float(np.std(all_train_values))

        logger.info("Global statistics from training set:")
        logger.info(f"  min:  {global_min:.6f}")
        logger.info(f"  max:  {global_max:.6f}")
        logger.info(f"  mean: {global_mean:.6f}")
        logger.info(f"  std:  {global_std:.6f}")

    hashes = {}

    # Now save each split with both global (training) and split-specific statistics
    for split_name, split_indices in splits.items():
        split_sequences = [sequences[i] for i in split_indices]
        split_embeddings = [embeddings[i] for i in split_indices]

        output_path = hy_utils.to_absolute_path(cfg[f"{split_name}_output_path"])

        if use_mean:
            logger.info(f"Saving mean-pooled embeddings for {split_name} split")
            # Compute mean pooling for each embedding in this split
            mean_embeddings = [np.mean(emb, axis=0) for emb in split_embeddings]

            # Compute split-specific statistics for reference
            split_values = np.concatenate([emb.flatten() for emb in mean_embeddings])
            split_min = float(np.min(split_values))
            split_max = float(np.max(split_values))
            split_mean = float(np.mean(split_values))
            split_std = float(np.std(split_values))

            logger.info(f"{split_name} split statistics:")
            logger.info(f"  min:  {split_min:.6f}")
            logger.info(f"  max:  {split_max:.6f}")
            logger.info(f"  mean: {split_mean:.6f}")
            logger.info(f"  std:  {split_std:.6f}")

            # Store as HDF5 with fixed-size arrays for mean embeddings
            with h5py.File(output_path, "w") as f:
                dt_str = h5py.string_dtype(encoding="utf-8")
                seq_dataset = f.create_dataset("sequences", (len(split_sequences),), dtype=dt_str)
                for i, seq in enumerate(split_sequences):
                    seq_dataset[i] = seq

                # Create fixed-size float dataset for mean embeddings
                emb_dataset = f.create_dataset(
                    "embeddings", (len(mean_embeddings), embedding_dim), dtype=np.float32
                )
                for i, emb_mean in enumerate(mean_embeddings):
                    emb_dataset[i] = emb_mean

                # Store metadata as attributes
                f.attrs["embedding_dim"] = embedding_dim
                f.attrs["layer_name"] = layer_name

                # Store split-specific scalar statistics
                f.attrs["emb_min"] = split_min
                f.attrs["emb_max"] = split_max
                f.attrs["emb_mean"] = split_mean
                f.attrs["emb_std"] = split_std

            logger.info(
                f"Saved {len(split_sequences)} {split_name} sequences with mean embeddings to {output_path}"
            )
            logger.info(f"Mean embedding shape: ({embedding_dim},)")
        else:
            logger.info(f"Saving per-nucleotide embeddings for {split_name} split")

            # Compute split-specific statistics for reference
            split_values = np.concatenate([emb.flatten() for emb in split_embeddings])
            split_min = float(np.min(split_values))
            split_max = float(np.max(split_values))
            split_mean = float(np.mean(split_values))
            split_std = float(np.std(split_values))

            logger.info(f"{split_name} split statistics:")
            logger.info(f"  min:  {split_min:.6f}")
            logger.info(f"  max:  {split_max:.6f}")
            logger.info(f"  mean: {split_mean:.6f}")
            logger.info(f"  std:  {split_std:.6f}")

            # Store as HDF5 with variable-length datasets for efficient lazy loading
            with h5py.File(output_path, "w") as f:
                # Create variable-length string dataset for sequences
                dt_str = h5py.string_dtype(encoding="utf-8")
                seq_dataset = f.create_dataset("sequences", (len(split_sequences),), dtype=dt_str)
                for i, seq in enumerate(split_sequences):
                    seq_dataset[i] = seq

                # Create variable-length float dataset for embeddings
                dt_vlen = h5py.vlen_dtype(np.float32)
                emb_dataset = f.create_dataset(
                    "embeddings", (len(split_embeddings),), dtype=dt_vlen
                )
                for i, emb in enumerate(split_embeddings):
                    emb_dataset[i] = emb.flatten()

                # Store shape metadata as attributes
                f.attrs["embedding_dim"] = embedding_dim
                f.attrs["layer_name"] = layer_name

                # Store split-specific scalar statistics
                f.attrs["emb_min"] = split_min
                f.attrs["emb_max"] = split_max
                f.attrs["emb_mean"] = split_mean
                f.attrs["emb_std"] = split_std

            logger.info(f"Saved {len(split_sequences)} {split_name} sequences to {output_path}")
            logger.info(f"Number of embeddings: {len(split_embeddings)}")

        # Compute SHA256
        sha256_hash = file_sha256(output_path)
        hashes[split_name] = sha256_hash
        logger.info(f"SHA256 hash of {output_path}: {sha256_hash}")

    if "update_config" in cfg and cfg.update_config:
        config_path = hy_utils.to_absolute_path(cfg.update_config)
        logger.info(f"Updating config file {config_path}...")
        updates = {
            "test_sha256": hashes["test"],
            "skip_sha256_check": False,
            "test_csv": cfg.test_output_path,
            "embedding_dim": embedding_dim,
            "seq_length": cfg.seq_length,
        }
        if not eval_only:
            updates["train_sha256"] = hashes["train"]
            updates["val_sha256"] = hashes["val"]
            updates["train_csv"] = cfg.train_output_path
            updates["val_csv"] = cfg.val_output_path

        update_yaml_keys(str(config_path), updates)
        logger.info("Config file updated.")
    else:
        print(f"Update conf/config.yaml data section:")
        if not eval_only:
            print(f"train_sha256: {hashes['train']}")
            print(f"val_sha256: {hashes['val']}")
        print(f"test_sha256: {hashes['test']}")


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
