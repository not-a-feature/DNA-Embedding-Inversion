"""Prepare human reference genome (hg38) sequences for training.

This script loads the hg38.fa FASTA file, chunks it into sequences of specified length,
filters out sequences containing 'N', removes duplicates, and saves to a CSV file
(one sequence per line, no header).

Example usage:
    python prepare_hg38.py --seq_length 100 --output data/hg38_seq100.csv
    python prepare_hg38.py --seq_length 200 --output data/hg38_seq200.csv --max_sequences 100000
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
import miniFasta as mf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def read_fasta_chunks(
    fasta_path: str, seq_length: int, max_sequences: int | None = None
) -> list[str]:
    """Read FASTA file and extract non-overlapping chunks of specified length.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file (e.g., hg38.fa).
    seq_length : int
        Length of each sequence chunk.
    max_sequences : int | None
        Maximum number of sequences to extract. If None, extract all possible sequences.

    Returns
    -------
    list[str]
        List of unique DNA sequences without 'N', uppercase, duplicates removed.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reading FASTA file: {fasta_path}")

    hg38_fasta_objects = mf.read(fasta_path, upper=True)

    # Only take regular chromosomes
    hg38_sequences = [str(f.body) for f in hg38_fasta_objects if "_" not in f.head]
    chunked_sequences = [_extract_chunks_from_sequence(seq, seq_length) for seq in hg38_sequences]

    all_chunks = set()
    for chunk_set in chunked_sequences:
        all_chunks.update(chunk_set)

    # Apply max_sequences limit after processing all chromosomes
    if max_sequences is not None and len(all_chunks) > max_sequences:
        all_chunks = set(random.sample(list(all_chunks), max_sequences))

    return list(all_chunks)


def _extract_chunks_from_sequence(sequence: str, seq_length: int) -> set[str]:
    """Extract non-overlapping chunks from a single sequence that do not contain 'N'.

    Parameters
    ----------
    sequence : str
        DNA sequence string.
    seq_length : int
        Length of each chunk.

    Returns
    -------
    set[str]
        Set of unique chunks without 'N'.
    """
    chunks = set()
    num_chunks = len(sequence) // seq_length

    for i in range(num_chunks):
        start = i * seq_length
        end = start + seq_length
        chunk = sequence[start:end]
        if "N" not in chunk:
            chunks.add(chunk)

    return chunks


def save_sequences_to_csv(sequences: list[str], output_path: str) -> None:
    """Save sequences to a CSV file (one sequence per line, no header).

    Parameters
    ----------
    sequences : list[str]
        List of DNA sequences.
    output_path : str
        Path to output CSV file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Saving {len(sequences)} sequences to: {output_path}")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for seq in sequences:
            f.write(f"{seq}\n")

    logger.info(f"Successfully saved sequences to {output_path}")


def main():
    """Main function to prepare hg38 sequences."""
    parser = argparse.ArgumentParser(
        description="Prepare hg38 reference genome sequences for training"
    )
    parser.add_argument(
        "--fasta_path",
        type=str,
        default="data/hg38.fa",
        help="Path to hg38 FASTA file",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        required=True,
        help="Length of each sequence chunk",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to extract (default: no limit)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    # Set seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Preparing hg38 sequences")
    logger.info(f"FASTA path: {args.fasta_path}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Max sequences: {args.max_sequences if args.max_sequences else 'No limit'}")
    logger.info("=" * 80)

    sequences = read_fasta_chunks(args.fasta_path, args.seq_length, args.max_sequences)

    assert len(sequences) > 0, "No valid sequences extracted from FASTA file"

    save_sequences_to_csv(sequences, args.output)

    logger.info("=" * 80)
    logger.info("Done!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
