"""Inspect per-token HDF5 datasets: print seq_length vs num_tokens for the first N entries."""

import sys
from pathlib import Path

import h5py
import numpy as np
from transformers import AutoTokenizer

# Datasets to inspect: (h5_path, embedding_dim, tokenizer_name)
DATASETS = [
    (
        "data/train_dnabert2_100_hg38_per_token.h5",
        768,
        "zhihan1996/DNABERT-2-117M",
    ),
    (
        "data/train_ntv2_100_hg38_per_token.h5",
        1024,
        "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
    ),
]

N = 10  # number of entries to inspect


def inspect_dataset(h5_path: str, embedding_dim: int, tokenizer_name: str) -> None:
    path = Path(h5_path)
    assert path.exists(), f"File not found: {path}"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    with h5py.File(path, "r") as f:
        sequences = f["sequences"]
        embeddings = f["embeddings"]
        total = len(sequences)

        print(f"\n{'=' * 80}")
        print(f"Dataset : {h5_path}")
        print(f"Tokenizer: {tokenizer_name}")
        print(f"Total samples: {total}")
        print(f"{'=' * 80}")
        print(
            f"{'idx':>4}  {'seq_len':>8}  {'num_emb_tokens':>15}  "
            f"{'num_tok (no special)':>22}  {'num_tok (with special)':>23}"
        )
        print("-" * 80)

        for idx in range(min(N, total)):
            seq_bytes = sequences[idx]
            seq = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)
            seq_len = len(seq)

            emb_flat = np.asarray(embeddings[idx], dtype=np.float32)
            num_emb_tokens = len(emb_flat) // embedding_dim

            ids_no_special = tokenizer(seq, add_special_tokens=False)["input_ids"]
            ids_with_special = tokenizer(seq, add_special_tokens=True)["input_ids"]

            print(
                f"{idx:4d}  {seq_len:8d}  {num_emb_tokens:15d}  "
                f"{len(ids_no_special):22d}  {len(ids_with_special):23d}"
            )


def main() -> None:
    for h5_path, emb_dim, tok_name in DATASETS:
        inspect_dataset(h5_path, emb_dim, tok_name)


if __name__ == "__main__":
    main()
