"""Prepare 1000g sequences for evaluation.

This script loads the 1000g csv files, extracts the sequence column (last),
chunks it into sequences of specified length, filters out sequences containing 'N',
removes duplicates, and saves to a CSV file (one sequence per line, no header).

Example usage:
    python prepare_1000g.py --seq_length 100 --output data/1000g_seq100.csv
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def read_csv_chunks(
    csv_paths: list[str], seq_length: int, max_sequences: int | None = None
) -> list[str]:
    """Read CSV files, extract sequences, and chunk them.

    Parameters
    ----------
    csv_paths : list[str]
        Paths to the CSV files (e.g., eval_exon.csv, eval_intron.csv).
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

    all_sequences = []
    for csv_path in csv_paths:
        logger.info(f"Reading CSV file: {csv_path}")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) > 0:
                        seq = parts[-1].strip().upper()
                        all_sequences.append(seq)
        except Exception as e:
            logger.error(f"Error reading {csv_path}: {e}")

    chunked_sequences = [_extract_chunks_from_sequence(seq, seq_length) for seq in all_sequences]

    all_chunks = set()
    for chunk_set in chunked_sequences:
        all_chunks.update(chunk_set)

    # Apply max_sequences limit
    if max_sequences is not None and len(all_chunks) > max_sequences:
        all_chunks = set(random.sample(list(all_chunks), max_sequences))

    return list(all_chunks)


def _extract_chunks_from_sequence(sequence: str, seq_length: int, stride: int = 1) -> set[str]:
    """Extract overlapping chunks from a single sequence using a sliding window.

    Parameters
    ----------
    sequence : str
        DNA sequence string.
    seq_length : int
        Length of each chunk.
    stride : int
        Step size for the sliding window (default: 1 for maximum overlap).

    Returns
    -------
    set[str]
        Set of unique chunks without 'N'.
    """
    chunks = set()
    if len(sequence) < seq_length:
        return chunks

    for start in range(0, len(sequence) - seq_length + 1, stride):
        chunk = sequence[start : start + seq_length]
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
    """Main function to prepare 1000g sequences."""
    parser = argparse.ArgumentParser(description="Prepare 1000g sequences for evaluation")
    parser.add_argument(
        "--input_files",
        type=str,
        nargs="+",
        default=["data/eval_exon.csv", "data/eval_intron.csv"],
        help="Paths to 1000g CSV files",
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

    if args.seed is not None:
        random.seed(args.seed)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Preparing 1000g sequences")
    logger.info(f"Input files: {args.input_files}")
    logger.info(f"Sequence length: {args.seq_length}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Max sequences: {args.max_sequences if args.max_sequences else 'No limit'}")
    logger.info("=" * 80)

    sequences = read_csv_chunks(args.input_files, args.seq_length, args.max_sequences)

    assert len(sequences) > 0, "No valid sequences extracted from CSV files"

    save_sequences_to_csv(sequences, args.output)

    logger.info("=" * 80)
    logger.info("Done!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
