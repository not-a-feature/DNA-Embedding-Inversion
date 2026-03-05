"""Generate DNA embeddings from a DNABERT-2 model using sequences from a file.

This script extracts embeddings from DNABERT-2 model for sequences provided in a file.
The output is HDF5 files for train, val, and test with datasets 'sequences' and 'embeddings'.

If mean=false (default): embeddings contains per-nucleotide embedding matrices
of shape [seq_length x embedding_dim] for each sequence.

If mean=true: embeddings contains mean-pooled embeddings of shape [embedding_dim]
for each sequence.

Example usage:
    python generate_dnabert2_embeddings.py input_path=data.csv num_sequences=1000 seq_length=50
    python generate_dnabert2_embeddings.py input_path=data.csv num_sequences=1000 seq_length=50 mean=true
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
from transformers import AutoTokenizer, AutoModel, BertConfig

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import file_sha256, set_determinism, load_sequences_from_file, update_yaml_keys


def _patch_triton_flash_attn():
    """Patch DNABERT-2's cached flash_attn_triton.py for Triton ≥ 3.x compatibility.

    The bundled file uses deprecated ``tl.dot(..., trans_a/trans_b=True)`` kwargs
    that were removed in newer Triton releases.  This replaces them with the
    equivalent ``tl.dot(tl.trans(x), y)`` / ``tl.dot(x, tl.trans(y))`` calls.
    """
    for name, mod in list(sys.modules.items()):
        if "flash_attn_triton" not in name or not hasattr(mod, "__file__") or mod.__file__ is None:
            continue
        fpath = Path(mod.__file__)
        if not fpath.exists():
            continue
        src = fpath.read_text()
        if "trans_b=True" not in src and "trans_a=True" not in src:
            continue
        patched = src
        patched = patched.replace("tl.dot(q, k, trans_b=True)", "tl.dot(q, tl.trans(k))")
        patched = patched.replace("tl.dot(do, v, trans_b=True)", "tl.dot(do, tl.trans(v))")
        patched = patched.replace(
            "tl.dot(p.to(do.dtype), do, trans_a=True)",
            "tl.dot(tl.trans(p.to(do.dtype)), do)",
        )
        patched = patched.replace("tl.dot(ds, q, trans_a=True)", "tl.dot(tl.trans(ds), q)")
        fpath.write_text(patched)
        logging.getLogger(__name__).info("Patched %s for Triton 3.x compatibility", fpath)


def embed_sequence_dnabert(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    sequence: str,
    device: str,
) -> np.ndarray:
    """Extract embeddings for a DNA sequence from DNABERT-2 model.

    Parameters
    ----------
    tokenizer : AutoTokenizer
        DNABERT-2 tokenizer.
    model : AutoModel
        DNABERT-2 model instance.
    sequence : str
        DNA sequence string.
    device : str
        Device to place input tensors on.

    Returns
    -------
    np.ndarray
        Per-nucleotide embedding of shape (num_tokens, embedding_dim).
    """
    # Tokenize without special tokens so embeddings align 1:1 with sequence tokens
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
        device
    )

    with torch.no_grad():
        hidden_states = model(inputs)[0]  # [1, num_tokens, embedding_dim]

    token_embs = hidden_states[0].detach().float().cpu().numpy()  # [num_tokens, embedding_dim]

    return token_embs


def generate_embeddings_dnabert(
    sequences: List[str], checkpoint: str, device: str
) -> List[np.ndarray]:
    """Generate DNABERT-2 embeddings for multiple sequences.

    Parameters
    ----------
    sequences : List[str]
        List of DNA sequences.
    checkpoint : str
        DNABERT-2 model checkpoint (Hugging Face style).
    device : str
        Device to place input tensors on.

    Returns
    -------
    List[np.ndarray]
        List of per-nucleotide embeddings.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading DNABERT-2 tokenizer and model from: {checkpoint}")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    config = BertConfig.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint, config=config, trust_remote_code=True)

    # Patch & reload the Triton flash-attn module so the first forward pass
    # compiles the fixed kernels (DNABERT-2 ships trans_a/trans_b that Triton ≥ 3 removed).
    _patch_triton_flash_attn()
    import importlib
    for mod_name in list(sys.modules):
        if "flash_attn_triton" in mod_name:
            importlib.reload(sys.modules[mod_name])

    model.to(device)
    model.eval()

    logger.info(f"DNABERT-2 model {checkpoint} loaded successfully")

    embeddings = []
    for i, seq in enumerate(sequences):
        emb = embed_sequence_dnabert(tokenizer, model, seq, device)
        embeddings.append(emb)

        if (i + 1) % 100 == 0:
            logger.info(f"Generated embeddings for {i + 1}/{len(sequences)} sequences")

    return embeddings


@hydra.main(config_path="../conf", config_name="generate/dnabert2", version_base=None)
def main(cfg: DictConfig) -> None:
    """Generate DNA sequences and DNABERT-2 embeddings using Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing data generation parameters.
    """
    logger = logging.getLogger(__name__)
    logger.info("Generating DNA sequences and DNABERT-2 embeddings...")

    set_determinism(int(cfg.seed))

    # Load sequences from file or generate random ones
    input_path = cfg.get("input_path", None)

    if input_path is None:
        raise ValueError("input_path must be provided in the configuration.")

    input_path = hy_utils.to_absolute_path(input_path)
    logger.info(f"Loading sequences from file: {input_path}")
    logger.info(f"Max sequences (num_sequences): {cfg.num_sequences}")
    # Note: seq_length here acts as a filter/truncator for loaded sequences
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

    # Extract embeddings
    checkpoint = cfg.checkpoint
    device = cfg.device

    logger.info(f"\n")
    logger.info(f"Using checkpoint: {checkpoint}")
    logger.info(f"Using device: {device}")
    logger.info(f"{'=' * 80}")

    embeddings = generate_embeddings_dnabert(sequences, checkpoint, device)

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

    # Save each split to HDF5
    if eval_only:
        splits = {"test": test_idx}
    else:
        splits = {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }

    use_mean = cfg.get("mean", False)

    if eval_only:
        global_min, global_max, global_mean, global_std = 0.0, 0.0, 0.0, 0.0
        embedding_dim = embeddings[0].shape[1] if embeddings else 0
        logger.info("Skipping global statistics computation because eval_only mode is on.")
    else:
        # Compute global statistics from TRAINING SET only
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

    for split_name, split_indices in splits.items():
        split_sequences = [sequences[i] for i in split_indices]
        split_embeddings = [embeddings[i] for i in split_indices]

        output_path = hy_utils.to_absolute_path(cfg[f"{split_name}_output_path"])

        if use_mean:
            logger.info(f"Saving mean-pooled embeddings for {split_name} split")
            # Compute mean pooling for each embedding in this split
            mean_embeddings = [np.mean(emb, axis=0) for emb in split_embeddings]

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
                f.attrs["checkpoint"] = checkpoint

                # Store GLOBAL (training) statistics So loader uses correct normalization
                f.attrs["emb_min"] = global_min
                f.attrs["emb_max"] = global_max
                f.attrs["emb_mean"] = global_mean
                f.attrs["emb_std"] = global_std

            logger.info(
                f"Saved {len(split_sequences)} {split_name} sequences with mean embeddings to {output_path}"
            )
        else:
            logger.info(f"Saving per-nucleotide embeddings for {split_name} split")

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
                f.attrs["checkpoint"] = checkpoint

                # Store GLOBAL (training) statistics
                f.attrs["emb_min"] = global_min
                f.attrs["emb_max"] = global_max
                f.attrs["emb_mean"] = global_mean
                f.attrs["emb_std"] = global_std

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
