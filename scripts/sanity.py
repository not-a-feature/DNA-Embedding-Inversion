#!/usr/bin/env python3
"""Sanity check script for data configs, data files, and SHA256 hashes."""

import os
import yaml
import h5py
import hashlib
import math
from typing import List, Dict, Tuple

SEQ_LENGTHS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
MODELS = ["dnabert2", "evo2", "ntv2"]

# Hardcoded list of expected configuration files
CONFIG_FILES = [
    "conf/data/dnabert2_100_hg38_per_token.yaml",
    "conf/data/evo2_100_hg38_per_token.yaml",
    "conf/data/ntv2_100_hg38_per_token.yaml",
]

for model in MODELS:
    for length in SEQ_LENGTHS:
        CONFIG_FILES.append(f"conf/data/{model}_{length}_hg38_mean.yaml")

def file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file."""
    with open(path, "rb") as f:
        # Check if hashlib.file_digest exists (Python 3.11+)
        if hasattr(hashlib, 'file_digest'):
            h = hashlib.file_digest(f, "sha256")
        else:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(4096 * 1024), b""):
                h.update(chunk)
    return h.hexdigest().upper()

def load_generate_config() -> Dict:
    """Load the generate_common.yaml config to get expected sequence counts."""
    gen_config_path = "conf/generate_common.yaml"
    if not os.path.isfile(gen_config_path):
        return {}
    with open(gen_config_path, "r") as f:
        return yaml.safe_load(f)


def get_expected_split_counts(gen_cfg: Dict) -> Dict[str, int]:
    """Compute expected per-split sequence counts from generate config."""
    num_sequences = gen_cfg.get("num_sequences", 0)
    train_split = gen_cfg.get("train_split", 0.7)
    val_split = gen_cfg.get("val_split", 0.15)
    # test_split is the remainder
    n_train = math.floor(num_sequences * train_split)
    n_val = math.floor(num_sequences * val_split)
    n_test = num_sequences - n_train - n_val
    return {"train": n_train, "val": n_val, "test": n_test, "total": num_sequences}


def check_h5_file(filepath: str, expected_hash: str, split_name: str, config_name: str) -> List[str]:
    """Check existence, header (is_hdf5), and hash of a single h5 file."""
    errors = []
    
    if not os.path.isfile(filepath):
        errors.append(f"[{config_name}] Missing {split_name} data file: {filepath}")
        return errors

    # Check HDF5 valid header
    if not h5py.is_hdf5(filepath):
        errors.append(f"[{config_name}] Invalid HDF5 header/format for {split_name} file: {filepath}")

    # Compute hash and compare
    try:
        actual_hash = file_sha256(filepath)
        if actual_hash != expected_hash.upper():
            errors.append(
                f"[{config_name}] SHA256 mismatch for {split_name} file '{filepath}'.\n"
                f"  Expected: {expected_hash.upper()}\n"
                f"  Actual:   {actual_hash}"
            )
    except Exception as e:
        errors.append(f"[{config_name}] Error hashing {split_name} file '{filepath}': {e}")

    return errors


def check_h5_sequences(filepath: str, split_name: str, config_name: str,
                       expected_count: int) -> Tuple[List[str], int]:
    """Check that sequences in an H5 file are unique and count matches expected.
    
    Returns (errors, num_sequences).
    """
    errors = []

    if not os.path.isfile(filepath):
        return errors, 0

    try:
        with h5py.File(filepath, "r") as f:
            if "sequences" not in f:
                errors.append(f"[{config_name}] Missing 'sequences' dataset in {split_name} file: {filepath}")
                return errors, 0

            sequences = [s.decode() if isinstance(s, bytes) else s for s in f["sequences"][:]]
            num_seqs = len(sequences)
            num_unique = len(set(sequences))
            num_duplicates = num_seqs - num_unique

            print(f"  {split_name}: {num_seqs} sequences ({num_unique} unique)", end="")

            if expected_count > 0:
                print(f" | expected {expected_count}", end="")
                if num_seqs != expected_count:
                    errors.append(
                        f"[{config_name}] Sequence count mismatch for {split_name}: "
                        f"got {num_seqs}, expected {expected_count}"
                    )
                    print(" ✗", end="")
                else:
                    print(" ✓", end="")

            print()  # newline

            if num_duplicates > 0:
                errors.append(
                    f"[{config_name}] {num_duplicates} duplicate sequences found in {split_name} file: {filepath}"
                )

    except Exception as e:
        errors.append(f"[{config_name}] Error reading sequences from {split_name} file '{filepath}': {e}")
        return errors, 0

    return errors, num_seqs

def main():
    print("Starting sanity checks...")
    all_errors = []
    configs_checked = 0
    files_checked = 0

    # Load generate config for expected counts
    gen_cfg = load_generate_config()
    expected_counts = get_expected_split_counts(gen_cfg) if gen_cfg else {}
    if expected_counts:
        print(f"Generate config: num_sequences={expected_counts['total']}, "
              f"train={expected_counts['train']}, val={expected_counts['val']}, test={expected_counts['test']}")
    else:
        print("Warning: Could not load generate_common.yaml — skipping count comparison.")
    print()

    for idx, config_path in enumerate(CONFIG_FILES, 1):
        if not os.path.isfile(config_path):
            all_errors.append(f"Missing config file: {config_path}")
            continue

        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            all_errors.append(f"Error reading config '{config_path}': {e}")
            continue

        configs_checked += 1
        print(f"[{idx}/{len(CONFIG_FILES)}] {config_path}")

        splits = ["train", "val", "test"]
        total_seqs_for_config = 0
        for split in splits:
            csv_key = f"{split}_csv"
            sha_key = f"{split}_sha256"

            data_file = cfg.get(csv_key)
            expected_hash = str(cfg.get(sha_key, ""))

            if not data_file:
                all_errors.append(f"[{config_path}] Missing '{csv_key}' in config.")
                continue

            if not expected_hash:
                all_errors.append(f"[{config_path}] Missing '{sha_key}' in config.")
            
            # Perform hash/header checks
            split_errors = check_h5_file(data_file, expected_hash, split, config_path)
            all_errors.extend(split_errors)
            if not split_errors:
                files_checked += 1

            # Perform sequence uniqueness and count checks
            exp_count = expected_counts.get(split, 0)
            seq_errors, num_seqs = check_h5_sequences(data_file, split, config_path, exp_count)
            all_errors.extend(seq_errors)
            total_seqs_for_config += num_seqs

        # Check total across splits
        if expected_counts and total_seqs_for_config > 0:
            expected_total = expected_counts.get("total", 0)
            if total_seqs_for_config != expected_total:
                msg = (f"[{config_path}] Total sequence count mismatch: "
                       f"got {total_seqs_for_config}, expected {expected_total}")
                all_errors.append(msg)
                print(f"  TOTAL: {total_seqs_for_config} / {expected_total} ✗")
            else:
                print(f"  TOTAL: {total_seqs_for_config} / {expected_total} ✓")

    print()
    print("-" * 50)
    print("Sanity Check Summary:")
    print(f"Configs checked: {configs_checked} / {len(CONFIG_FILES)}")
    print(f"H5 files checked (passed all tests): {files_checked}")
    print(f"Total errors found: {len(all_errors)}")

    if all_errors:
        print("\nErrors Details:")
        for error in all_errors:
            print(f" - {error}")
        exit(1)
    else:
        print("\nAll checks passed successfully!")
        exit(0)

if __name__ == "__main__":
    # Ensure working from project root where 'conf/data/' is located
    if not os.path.isdir("conf/data"):
        print("Warning: Please run this script from the project root directory.")
    
    main()
