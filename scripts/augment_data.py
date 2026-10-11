"""Script to augment DNA sequences by introducing random SNPs (single nucleotide polymorphisms).

This script reads DNA sequences from a CSV file and creates augmented versions
by introducing 3-10 random SNP mutations per sequence. All resulting sequences
are guaranteed to be unique.

Example usage:
    python scrips/augment_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path to import src module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import NUCLEOTIDES, set_determinism


def load_sequences(file_path: str) -> list[str]:
    """Load DNA sequences from a CSV file (one sequence per line).

    Parameters
    ----------
    file_path : str
        Path to the input CSV file containing DNA sequences (one per line).

    Returns
    -------
    list[str]
        List of DNA sequences.
    """
    with open(file_path, "r") as f:
        sequences = [line.strip() for line in f if line.strip()]

    assert len(sequences) > 0, f"No sequences found in {file_path}"
    return sequences


def introduce_snps(sequence: str, num_snps: int) -> str:
    """Introduce random SNP mutations into a DNA sequence.

    Parameters
    ----------
    sequence : str
        Original DNA sequence.
    num_snps : int
        Number of SNPs to introduce (must be <= sequence length).

    Returns
    -------
    str
        Mutated DNA sequence with SNPs introduced.
    """
    assert 0 <= num_snps <= len(sequence), "num_snps must be between 0 and sequence length"

    seq_array = np.array(list(sequence.upper()))
    seq_len = len(seq_array)

    # Select random positions for mutations (without replacement)
    mutation_positions = np.random.choice(seq_len, size=num_snps, replace=False)

    # For each position, replace with a different nucleotide
    for pos in mutation_positions:
        original_nuc = seq_array[pos]
        assert original_nuc in NUCLEOTIDES, f"Invalid nucleotide '{original_nuc}' at position {pos}"

        # Choose a different nucleotide (not the same as original)
        available_nucleotides = [nuc for nuc in NUCLEOTIDES if nuc != original_nuc]
        seq_array[pos] = np.random.choice(available_nucleotides)

    return "".join(seq_array)


def augment_sequences(
    sequences: list[str],
    min_snps: int = 3,
    max_snps: int = 10,
    augmentations_per_sequence: int = 10,
    seed: int = 42,
) -> list[str]:
    """Augment DNA sequences by introducing random SNPs.

    For each original sequence, creates multiple augmented versions with a random number
    of SNPs between min_snps and max_snps. Ensures all resulting sequences are unique.

    Parameters
    ----------
    sequences : list[str]
        List of original DNA sequences.
    min_snps : int
        Minimum number of SNPs to introduce per sequence.
    max_snps : int
        Maximum number of SNPs to introduce per sequence.
    augmentations_per_sequence : int
        Number of augmented versions to create per original sequence.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    list[str]
        List of augmented sequences (guaranteed unique).
    """
    assert min_snps >= 1, "min_snps must be at least 1"
    assert max_snps >= min_snps, "max_snps must be >= min_snps"
    assert augmentations_per_sequence >= 1, "augmentations_per_sequence must be at least 1"

    set_determinism(seed)

    logger = logging.getLogger(__name__)
    logger.info(
        f"Augmenting {len(sequences)} sequences with {min_snps}-{max_snps} SNPs each, "
        f"{augmentations_per_sequence} augmentations per sequence"
    )

    # Track all sequences (original + augmented) to ensure uniqueness
    seen_sequences = set(sequences)
    augmented = []

    for i, seq in enumerate(sequences):
        for aug_idx in range(augmentations_per_sequence):
            # Randomly choose number of SNPs for this augmentation
            num_snps = np.random.randint(min_snps, max_snps + 1)

            # Keep generating until we get a unique sequence
            max_attempts = 1000
            for attempt in range(max_attempts):
                augmented_seq = introduce_snps(seq, num_snps)

                if augmented_seq not in seen_sequences:
                    seen_sequences.add(augmented_seq)
                    augmented.append(augmented_seq)
                    break
            else:
                # If we couldn't find a unique sequence after max_attempts, fail fast
                assert False, (
                    f"Failed to generate unique sequence for index {i}, augmentation {aug_idx} "
                    f"after {max_attempts} attempts. "
                    f"Consider increasing max_snps or reducing augmentations_per_sequence."
                )

        if (i + 1) % 5000 == 0:
            logger.info(f"Augmented {i + 1}/{len(sequences)} sequences")

    logger.info(f"Successfully generated {len(augmented)} unique augmented sequences")

    return augmented


def save_sequences(sequences: list[str], file_path: str) -> None:
    """Save DNA sequences to a CSV file (one sequence per line).

    Parameters
    ----------
    sequences : list[str]
        List of DNA sequences to save.
    file_path : str
        Output file path.
    """
    with open(file_path, "w") as f:
        for seq in sequences:
            f.write(f"{seq}\n")


def main() -> None:
    """Main function to augment dataset with random SNPs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # File paths
    input_file = "data/human_tfbs_1.csv"
    output_file = "data/human_tfbs_1_augmented.csv"

    # Load original sequences
    logger.info(f"Loading sequences from {input_file}")
    sequences = load_sequences(input_file)
    logger.info(f"Loaded {len(sequences)} sequences")

    # Verify all sequences are unique in the original dataset
    unique_count = len(set(sequences))
    logger.info(f"Original dataset contains {unique_count} unique sequences")
    if unique_count != len(sequences):
        logger.warning("Original dataset contains duplicate sequences")

    # Augment sequences with 3-10 SNPs each, 10 augmentations per sequence
    augmented = augment_sequences(
        sequences, min_snps=3, max_snps=10, augmentations_per_sequence=10, seed=42
    )

    # Remove duplicates by converting to set and back to list
    augmented_unique = list(set(augmented))
    logger.info(
        f"After deduplication: {len(augmented_unique)} unique sequences (removed {len(augmented) - len(augmented_unique)} duplicates)"
    )

    # Save augmented sequences
    logger.info(f"Saving {len(augmented_unique)} augmented sequences to {output_file}")
    save_sequences(augmented_unique, output_file)

    logger.info("Augmentation complete!")
    logger.info(
        f"Total sequences: {len(sequences)} original + {len(augmented_unique)} unique augmented"
    )


if __name__ == "__main__":
    main()
