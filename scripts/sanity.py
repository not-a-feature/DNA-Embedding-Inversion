#!/usr/bin/env python3
"""Sanity checks for data configs and generated H5 files.

Checks performed
----------------
Per file:
  - config file loads as YAML
  - H5 file exists and has a valid HDF5 header
  - SHA256 matches the value recorded in the config
  - ``sequences`` dataset present, no duplicates within the split
  - ``embeddings`` dataset present with the right shape/dtype for the file kind
    (mean → ``(N, D)`` float32; per_token → ``(N,)`` variable-length float32)
  - ``embedding_dim`` attribute matches the actual dimensionality
  - row count of ``embeddings`` matches ``sequences``
  - ``emb_min``/``emb_max``/``emb_mean``/``emb_std`` attrs present and finite
    (and ``emb_std > 0`` unless the file was written in eval_only mode, which
    we detect by all four stats being exactly zero)
  - no NaN / Inf in a sampled subset of embeddings
  - sequence count matches the value derived from ``conf/generate_common.yaml``
  - every sequence has length == ``cfg.seq_length`` and uses only ACGT(N)

Per config (across splits):
  - ``embedding_dim`` consistent across train/val/test

Across configs:
  - per model, ``embedding_dim`` is the same across every seq_length and
    across mean/per_token variants
  - per seq_length, all three models read the same underlying sequence set
    (compared via sha256 of the sorted unique sequence list of the test split)
"""

import os
import sys
import yaml
import h5py
import hashlib
import math
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

SEQ_LENGTHS = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100]
MODELS = ["dnabert2", "evo2", "ntv2"]
VALID_NUCS = set("ACGTN")
SAMPLE_SIZE = 256  # embeddings sampled per file for NaN/Inf and finite-stats check
SEQ_CHECK_SAMPLE = 5000  # sequences sampled per file for length/charset check (full check is O(N))
# Historical per-token experiments used a smaller corpus than mean pooling.
PER_TOKEN_NUM_SEQUENCES = 100000

CONFIG_FILES: List[str] = [
    "conf/data/hg38/dnabert2_100_hg38_per_token.yaml",
    "conf/data/hg38/evo2_100_hg38_per_token.yaml",
    "conf/data/hg38/ntv2_100_hg38_per_token.yaml",
]
for _model in MODELS:
    for _length in SEQ_LENGTHS:
        CONFIG_FILES.append(f"conf/data/hg38/{_model}_{_length}_hg38_mean.yaml")


def file_sha256(path: str) -> str:
    """Compute the SHA256 hash of a file."""
    with open(path, "rb") as f:
        if hasattr(hashlib, "file_digest"):
            h = hashlib.file_digest(f, "sha256")
        else:
            h = hashlib.sha256()
            for chunk in iter(lambda: f.read(4096 * 1024), b""):
                h.update(chunk)
    return h.hexdigest().upper()


def load_generate_config() -> Dict:
    """Load conf/generate_common.yaml to derive expected sequence counts."""
    gen_config_path = "conf/generate_common.yaml"
    if not os.path.isfile(gen_config_path):
        return {}
    with open(gen_config_path, "r") as f:
        return yaml.safe_load(f)


def get_expected_split_counts(gen_cfg: Dict, num_sequences: Optional[int] = None) -> Dict[str, int]:
    """Compute expected per-split sequence counts (mirrors the generation logic).

    ``num_sequences`` overrides the corpus size from ``gen_cfg`` — needed for
    per_token configs, which use a smaller corpus than the mean variants.
    """
    if num_sequences is None:
        num_sequences = gen_cfg.get("num_sequences", 0)
    train_split = gen_cfg.get("train_split", 0.7)
    val_split = gen_cfg.get("val_split", 0.15)
    n_train = math.floor(num_sequences * train_split)
    n_val = math.floor(num_sequences * val_split)
    n_test = num_sequences - n_train - n_val
    return {"train": n_train, "val": n_val, "test": n_test, "total": num_sequences}


def _parse_config_metadata(config_path: str) -> Dict[str, Optional[str]]:
    """Pull (model, seq_length, kind) out of the filename pattern."""
    # Filenames are of the form {model}_{seq_length}_hg38_{kind}.yaml
    base = os.path.basename(config_path)
    name = base[: -len(".yaml")] if base.endswith(".yaml") else base
    parts = name.split("_")
    if len(parts) < 4:
        return {"model": None, "seq_length": None, "kind": None}
    model = parts[0]
    try:
        seq_length = int(parts[1])
    except ValueError:
        seq_length = None
    # everything after the {seq_length}_hg38_ prefix is the kind
    kind = "_".join(parts[3:]) if len(parts) >= 4 else None
    return {"model": model, "seq_length": seq_length, "kind": kind}


def inspect_h5_file(
    filepath: str,
    expected_hash: str,
    split_name: str,
    config_name: str,
    cfg: Dict,
    expected_count: int,
    is_per_token: bool,
) -> Tuple[List[str], Dict]:
    """Run every per-file check in a single pass over the HDF5 file.

    Returns (errors, info). ``info`` carries the bits needed for cross-split /
    cross-config checks: ``num_sequences``, ``embedding_dim``, and ``seq_hash``
    (sha256 of the sorted unique sequence list, used to compare sequence sets
    across models).
    """
    errors: List[str] = []
    info: Dict = {
        "num_sequences": 0,
        "embedding_dim": None,
        "seq_hash": None,
    }

    if not os.path.isfile(filepath):
        errors.append(f"[{config_name}] Missing {split_name} data file: {filepath}")
        return errors, info

    if not h5py.is_hdf5(filepath):
        errors.append(f"[{config_name}] Invalid HDF5 header for {split_name} file: {filepath}")
        return errors, info

    # SHA256 check first — if the hash is wrong the file content is suspect, but
    # we still inspect it so the user gets the full picture of what's wrong.
    if expected_hash:
        try:
            actual_hash = file_sha256(filepath)
            if actual_hash != expected_hash.upper():
                errors.append(
                    f"[{config_name}] SHA256 mismatch for {split_name} '{filepath}'.\n"
                    f"  Expected: {expected_hash.upper()}\n  Actual:   {actual_hash}"
                )
        except Exception as e:
            errors.append(f"[{config_name}] Error hashing {split_name} '{filepath}': {e}")

    try:
        with h5py.File(filepath, "r") as f:
            # --- sequences dataset ---
            if "sequences" not in f:
                errors.append(
                    f"[{config_name}] Missing 'sequences' dataset in {split_name}: {filepath}"
                )
                return errors, info
            seq_ds = f["sequences"]
            sequences_raw = seq_ds[:]
            sequences = [s.decode() if isinstance(s, bytes) else s for s in sequences_raw]
            num_seqs = len(sequences)
            info["num_sequences"] = num_seqs

            unique_seqs = set(sequences)
            num_unique = len(unique_seqs)
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
            print()

            if num_duplicates > 0:
                errors.append(
                    f"[{config_name}] {num_duplicates} duplicate sequences in {split_name}: {filepath}"
                )

            # Sequence length + charset (sampled to keep this O(SEQ_CHECK_SAMPLE))
            cfg_seq_length = cfg.get("seq_length")
            if num_seqs > 0:
                if num_seqs > SEQ_CHECK_SAMPLE:
                    rng = np.random.default_rng(0)
                    sample_idx = rng.choice(num_seqs, SEQ_CHECK_SAMPLE, replace=False)
                else:
                    sample_idx = range(num_seqs)
                bad_len = 0
                bad_chars = 0
                for i in sample_idx:
                    s = sequences[i]
                    if cfg_seq_length is not None and len(s) != cfg_seq_length:
                        bad_len += 1
                    if not set(s).issubset(VALID_NUCS):
                        bad_chars += 1
                if bad_len:
                    errors.append(
                        f"[{config_name}] {bad_len} sequences in {split_name} have length != "
                        f"cfg.seq_length={cfg_seq_length} (sampled {len(list(sample_idx))})"
                    )
                if bad_chars:
                    errors.append(
                        f"[{config_name}] {bad_chars} sequences in {split_name} contain non-ACGTN "
                        f"characters (sampled {len(list(sample_idx))})"
                    )

            # Stable hash of the sequence set — used cross-model.
            # Sorting unique strings: O(N log N), fine.
            seq_hash = hashlib.sha256(
                "\n".join(sorted(unique_seqs)).encode("utf-8")
            ).hexdigest()
            info["seq_hash"] = seq_hash

            # --- embeddings dataset ---
            if "embeddings" not in f:
                errors.append(
                    f"[{config_name}] Missing 'embeddings' dataset in {split_name}: {filepath}"
                )
                return errors, info

            emb_ds = f["embeddings"]
            emb_shape = emb_ds.shape
            emb_dtype = emb_ds.dtype

            # Row count vs sequences
            if emb_shape[0] != num_seqs:
                errors.append(
                    f"[{config_name}] Row count mismatch in {split_name}: "
                    f"sequences={num_seqs}, embeddings={emb_shape[0]}"
                )

            # Shape & dtype per kind
            if is_per_token:
                if emb_ds.ndim != 1:
                    errors.append(
                        f"[{config_name}] per_token {split_name}: expected 1-D vlen "
                        f"embeddings, got shape {emb_shape}"
                    )
                # vlen dtype check: h5py exposes vlen float32 as 'object' dtype
                if emb_dtype != np.dtype("O"):
                    errors.append(
                        f"[{config_name}] per_token {split_name}: expected vlen "
                        f"(object) dtype, got {emb_dtype}"
                    )
            else:
                if emb_ds.ndim != 2:
                    errors.append(
                        f"[{config_name}] mean {split_name}: expected 2-D embeddings, "
                        f"got shape {emb_shape}"
                    )
                if emb_dtype != np.float32:
                    errors.append(
                        f"[{config_name}] mean {split_name}: expected float32, got {emb_dtype}"
                    )

            # --- attributes ---
            attrs = dict(emb_ds.attrs) if emb_ds.attrs else {}
            file_attrs = dict(f.attrs)
            # Stats are written as top-level file attrs by the generators.
            for key in ("embedding_dim", "emb_min", "emb_max", "emb_mean", "emb_std"):
                if key not in file_attrs:
                    errors.append(
                        f"[{config_name}] {split_name}: missing required attribute '{key}'"
                    )

            embedding_dim_attr = file_attrs.get("embedding_dim", None)
            if embedding_dim_attr is not None:
                embedding_dim_attr = int(embedding_dim_attr)
                info["embedding_dim"] = embedding_dim_attr

                cfg_dim = cfg.get("embedding_dim")
                if cfg_dim is not None and int(cfg_dim) != embedding_dim_attr:
                    errors.append(
                        f"[{config_name}] {split_name}: cfg.embedding_dim={cfg_dim} != "
                        f"H5 attr embedding_dim={embedding_dim_attr}"
                    )

                # Verify shape matches the recorded embedding_dim
                if not is_per_token and emb_ds.ndim == 2:
                    if emb_shape[1] != embedding_dim_attr:
                        errors.append(
                            f"[{config_name}] {split_name}: embeddings.shape[1]={emb_shape[1]} "
                            f"!= embedding_dim attr={embedding_dim_attr}"
                        )

            # Stats sanity. eval_only files (used for identification benchmarks)
            # legitimately store all-zero stats; treat that as a single case.
            stats = {k: file_attrs.get(k) for k in ("emb_min", "emb_max", "emb_mean", "emb_std")}
            stats_present = all(v is not None for v in stats.values())
            if stats_present:
                stats_f = {k: float(v) for k, v in stats.items()}
                all_zero = all(v == 0.0 for v in stats_f.values())
                for k, v in stats_f.items():
                    if not math.isfinite(v):
                        errors.append(
                            f"[{config_name}] {split_name}: attr {k}={v} is not finite"
                        )
                if not all_zero:
                    if stats_f["emb_std"] <= 0:
                        errors.append(
                            f"[{config_name}] {split_name}: emb_std={stats_f['emb_std']} "
                            f"is not positive (degenerate embeddings?)"
                        )
                    if stats_f["emb_min"] > stats_f["emb_max"]:
                        errors.append(
                            f"[{config_name}] {split_name}: emb_min={stats_f['emb_min']} > "
                            f"emb_max={stats_f['emb_max']}"
                        )

            # --- NaN / Inf sample ---
            if num_seqs > 0:
                rng = np.random.default_rng(1)
                k = min(SAMPLE_SIZE, num_seqs)
                idx = sorted(rng.choice(num_seqs, k, replace=False).tolist())
                bad_nan = 0
                bad_inf = 0
                if is_per_token:
                    for i in idx:
                        row = np.asarray(emb_ds[i], dtype=np.float32)
                        if row.size == 0:
                            errors.append(
                                f"[{config_name}] {split_name}: embeddings[{i}] is empty"
                            )
                            continue
                        # Row should be flat num_tokens * D values
                        if embedding_dim_attr and row.size % embedding_dim_attr != 0:
                            errors.append(
                                f"[{config_name}] {split_name}: embeddings[{i}] size "
                                f"{row.size} not divisible by embedding_dim={embedding_dim_attr}"
                            )
                        bad_nan += int(np.isnan(row).any())
                        bad_inf += int(np.isinf(row).any())
                else:
                    sample = emb_ds[idx]  # advanced indexing returns a 2-D array
                    sample_arr = np.asarray(sample, dtype=np.float32)
                    bad_nan = int(np.isnan(sample_arr).any())
                    bad_inf = int(np.isinf(sample_arr).any())
                if bad_nan:
                    errors.append(
                        f"[{config_name}] {split_name}: NaN found in sampled embeddings "
                        f"(sample size {k})"
                    )
                if bad_inf:
                    errors.append(
                        f"[{config_name}] {split_name}: Inf found in sampled embeddings "
                        f"(sample size {k})"
                    )

    except Exception as e:
        errors.append(f"[{config_name}] Error inspecting {split_name} '{filepath}': {e}")

    return errors, info


def cross_split_checks(config_name: str, per_split: Dict[str, Dict]) -> List[str]:
    """Run checks that compare information across the train/val/test splits."""
    errors: List[str] = []
    dims = {s: info.get("embedding_dim") for s, info in per_split.items()
            if info.get("embedding_dim") is not None}
    distinct_dims = set(dims.values())
    if len(distinct_dims) > 1:
        errors.append(
            f"[{config_name}] embedding_dim differs across splits: {dims}"
        )
    return errors


def cross_config_checks(
    all_infos: Dict[str, Dict[str, Dict]],
    config_meta: Dict[str, Dict],
) -> List[str]:
    """Run checks that span multiple configs.

    Verifies (a) per-model embedding_dim consistency across every seq_length and
    variant, and (b) per-seq_length that all three models read the same
    underlying sequence set (via the test-split sequence-set hash).
    """
    errors: List[str] = []

    # --- (a) per-model embedding_dim consistency ---
    by_model: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for cfg_path, splits in all_infos.items():
        meta = config_meta.get(cfg_path, {})
        model = meta.get("model")
        if not model:
            continue
        for split, info in splits.items():
            d = info.get("embedding_dim")
            if d is not None:
                by_model[model][d].add(cfg_path)

    for model, dim_to_configs in by_model.items():
        if len(dim_to_configs) > 1:
            summary = "; ".join(
                f"dim={d} in {len(cfgs)} configs (e.g. {sorted(cfgs)[0]})"
                for d, cfgs in dim_to_configs.items()
            )
            errors.append(
                f"[cross-config] model '{model}' has inconsistent embedding_dim: {summary}"
            )

    # --- (b) cross-model sequence-set equality, per seq_length ---
    # Compare the test split (always present, smallest, exists in eval_only too).
    by_length: Dict[int, Dict[str, str]] = defaultdict(dict)
    for cfg_path, splits in all_infos.items():
        meta = config_meta.get(cfg_path, {})
        model = meta.get("model")
        seq_length = meta.get("seq_length")
        if not model or seq_length is None:
            continue
        # Only compare mean variants — per_token uses different input files
        # (e.g. seq=100 per_token vs seq=100 mean read the same csv but only one
        # variant exists per model+length, so this is mostly mean-vs-mean).
        if meta.get("kind") != "mean":
            continue
        test_info = splits.get("test", {})
        seq_hash = test_info.get("seq_hash")
        if seq_hash is None:
            continue
        by_length[seq_length][model] = seq_hash

    for seq_length, model_to_hash in sorted(by_length.items()):
        distinct = set(model_to_hash.values())
        if len(distinct) > 1:
            errors.append(
                f"[cross-config] seq_length={seq_length}: models disagree on test "
                f"sequence set (hashes: {model_to_hash}). Same input CSV should "
                f"produce identical sequences across models."
            )

    return errors


def main() -> int:
    print("Starting sanity checks...")
    all_errors: List[str] = []
    configs_checked = 0
    files_checked = 0

    gen_cfg = load_generate_config()
    expected_counts = get_expected_split_counts(gen_cfg) if gen_cfg else {}
    per_token_counts = (
        get_expected_split_counts(gen_cfg, num_sequences=PER_TOKEN_NUM_SEQUENCES)
        if gen_cfg
        else {}
    )
    if expected_counts:
        print(
            f"Generate config: num_sequences={expected_counts['total']}, "
            f"train={expected_counts['train']}, val={expected_counts['val']}, "
            f"test={expected_counts['test']}"
        )
        print(
            f"per_token override: num_sequences={per_token_counts['total']}, "
            f"train={per_token_counts['train']}, val={per_token_counts['val']}, "
            f"test={per_token_counts['test']}"
        )
    else:
        print("Warning: Could not load generate_common.yaml — skipping count comparison.")
    print()

    all_infos: Dict[str, Dict[str, Dict]] = {}
    config_meta: Dict[str, Dict] = {}

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

        meta = _parse_config_metadata(config_path)
        config_meta[config_path] = meta
        is_per_token = meta.get("kind") == "per_token"

        # skip_sha256_check should be False on committed configs — if it's True,
        # the file's contents aren't being validated against the recorded hash
        # at training time.
        if cfg.get("skip_sha256_check", False):
            all_errors.append(
                f"[{config_path}] skip_sha256_check=True (data integrity check is disabled)"
            )

        splits = ["train", "val", "test"]
        per_split_info: Dict[str, Dict] = {}
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

            counts_for_cfg = per_token_counts if is_per_token else expected_counts
            exp_count = counts_for_cfg.get(split, 0) if counts_for_cfg else 0
            split_errors, info = inspect_h5_file(
                filepath=data_file,
                expected_hash=expected_hash,
                split_name=split,
                config_name=config_path,
                cfg=cfg,
                expected_count=exp_count,
                is_per_token=is_per_token,
            )
            all_errors.extend(split_errors)
            per_split_info[split] = info
            total_seqs_for_config += info.get("num_sequences", 0)
            if not split_errors:
                files_checked += 1

        all_infos[config_path] = per_split_info

        # Per-config cross-split checks
        all_errors.extend(cross_split_checks(config_path, per_split_info))

        counts_for_cfg = per_token_counts if is_per_token else expected_counts
        if counts_for_cfg and total_seqs_for_config > 0:
            expected_total = counts_for_cfg.get("total", 0)
            if total_seqs_for_config != expected_total:
                all_errors.append(
                    f"[{config_path}] Total sequence count mismatch: "
                    f"got {total_seqs_for_config}, expected {expected_total}"
                )
                print(f"  TOTAL: {total_seqs_for_config} / {expected_total} ✗")
            else:
                print(f"  TOTAL: {total_seqs_for_config} / {expected_total} ✓")

    # Cross-config checks
    print()
    print("Running cross-config checks...")
    cross_errors = cross_config_checks(all_infos, config_meta)
    all_errors.extend(cross_errors)
    if cross_errors:
        print(f"  cross-config issues: {len(cross_errors)}")
    else:
        print("  cross-config: ok")

    print()
    print("-" * 50)
    print("Sanity Check Summary:")
    print(f"Configs checked: {configs_checked} / {len(CONFIG_FILES)}")
    print(f"H5 files checked (passed all per-file tests): {files_checked}")
    print(f"Total errors found: {len(all_errors)}")

    if all_errors:
        print("\nErrors Details:")
        for error in all_errors:
            print(f" - {error}")
        return 1

    print("\nAll checks passed successfully!")
    return 0


if __name__ == "__main__":
    if not os.path.isdir("conf/data"):
        print("Warning: Please run this script from the project root directory.")
    sys.exit(main())
