"""Check for sequence leakage between train/val/test splits.

This script loads sequences from HDF5 files and detects identical sequences
across splits (and optionally within each split). It fails fast with an
AssertionError if any overlap is found.
"""

from __future__ import annotations

import argparse
import logging
from typing import Iterable, List, Set

import h5py


def _load_sequences(path: str) -> List[str]:
    with h5py.File(path, "r", swmr=True) as f:
        assert "sequences" in f, f"Missing 'sequences' dataset in {path}"
        seqs = f["sequences"]
        decoded = [s.decode("utf-8") if isinstance(s, (bytes, bytearray)) else str(s) for s in seqs]
    return decoded


def _as_set(seqs: Iterable[str]) -> Set[str]:
    seq_set = set(seqs)
    assert len(seq_set) > 0, "No sequences loaded; check input files"
    return seq_set


def _find_overlap(a: Set[str], b: Set[str]) -> Set[str]:
    return a.intersection(b)


def _sample(items: Iterable[str], max_items: int) -> List[str]:
    result = []
    for i, item in enumerate(items):
        if i >= max_items:
            break
        result.append(item)
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Check for train/val/test sequence leakage")
    parser.add_argument("train_path", type=str, help="Path to train HDF5 file")
    parser.add_argument("val_path", type=str, help="Path to val HDF5 file")
    parser.add_argument("test_path", type=str, help="Path to test HDF5 file")
    parser.add_argument("--max_examples", type=int, default=5, help="Max examples to print per overlap")
    args = parser.parse_args()

    logger.info("Loading sequences...")
    train_seqs = _load_sequences(args.train_path)
    val_seqs = _load_sequences(args.val_path)
    test_seqs = _load_sequences(args.test_path)

    train_set = _as_set(train_seqs)
    val_set = _as_set(val_seqs)
    test_set = _as_set(test_seqs)

    logger.info(f"Train sequences: {len(train_seqs)} (unique: {len(train_set)})")
    logger.info(f"Val sequences:   {len(val_seqs)} (unique: {len(val_set)})")
    logger.info(f"Test sequences:  {len(test_seqs)} (unique: {len(test_set)})")

    train_val = _find_overlap(train_set, val_set)
    train_test = _find_overlap(train_set, test_set)
    val_test = _find_overlap(val_set, test_set)

    logger.info(f"Train ∩ Val overlap:  {len(train_val)}")
    if train_val:
        logger.info(f"Examples: {_sample(train_val, args.max_examples)}")

    logger.info(f"Train ∩ Test overlap: {len(train_test)}")
    if train_test:
        logger.info(f"Examples: {_sample(train_test, args.max_examples)}")

    logger.info(f"Val ∩ Test overlap:   {len(val_test)}")
    if val_test:
        logger.info(f"Examples: {_sample(val_test, args.max_examples)}")

    assert len(train_val) == 0, "Data leakage detected: train/val overlap"
    assert len(train_test) == 0, "Data leakage detected: train/test overlap"
    assert len(val_test) == 0, "Data leakage detected: val/test overlap"

    logger.info("No cross-split leakage detected.")


if __name__ == "__main__":
    main()
