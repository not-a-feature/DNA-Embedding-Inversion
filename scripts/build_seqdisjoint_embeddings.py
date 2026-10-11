"""Build cleaned hg38 data and seed new embedding H5s from existing files.

This is a one-time, standalone migration step. It does two things:

1. Deduplicates the existing multi-length hg38 source in input order, then fills
   any per-length deficit with new, coordinate-disjoint hg38 windows so the
   original target counts are retained.
2. Copies embeddings for the retained prefix from the existing train/val/test
   H5 files into new partial files. The normal embedding pipeline can then be
   rerun: its existing resumable writer sees the copied prefix and embeds only
   the newly added suffix.

Existing H5 files are opened read-only and are never removed or overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.prepare_hg38_multilen import (
    IndexedFasta,
    SCHEMA,
    deterministic_split_codes,
    iter_acgt_runs,
    select_chromosomes,
)


LOGGER = logging.getLogger(__name__)
OLD_SCHEMA = "dna_inversion_hg38_multilen_v1"
SPLITS = ("train", "val", "test")
SPLIT_TO_CODE = {"train": 0, "val": 1, "test": 2}
DEFAULT_FMS = ("dnabert2", "evo2", "ntv2")
DEFAULT_DIMS = {"dnabert2": 768, "evo2": 4096, "ntv2": 1024}


def _as_bytes(value) -> bytes:
    return value if isinstance(value, bytes) else bytes(value)


def _append(dataset: h5py.Dataset, values: np.ndarray) -> None:
    start = len(dataset)
    dataset.resize((start + len(values),))
    dataset[start:] = values


def _create_source_group(
    root: h5py.Group, length: int, target_count: int, chunk_size: int
) -> h5py.Group:
    group = root.create_group(str(length))
    chunk = min(chunk_size, target_count)
    group.create_dataset(
        "sequences",
        shape=(0,),
        maxshape=(None,),
        dtype=f"S{length}",
        chunks=(chunk,),
        compression="lzf",
    )
    group.create_dataset(
        "chrom",
        shape=(0,),
        maxshape=(None,),
        dtype="S16",
        chunks=(chunk,),
        compression="lzf",
    )
    group.create_dataset(
        "start",
        shape=(0,),
        maxshape=(None,),
        dtype=np.uint32,
        chunks=(chunk,),
        compression="lzf",
        shuffle=True,
    )
    group.create_dataset(
        "split",
        shape=(target_count,),
        dtype=np.uint8,
        chunks=(chunk,),
        compression="lzf",
        shuffle=True,
    )
    group.attrs["seq_length"] = length
    group.attrs["target_count"] = target_count
    return group


def _collect_old_unique_prefixes(
    old_source: h5py.File,
    new_source: h5py.File,
    chunk_size: int,
) -> tuple[Dict[int, int], Dict[int, hashlib._Hash], Dict[str, list[np.ndarray]]]:
    """Copy first occurrences and collect all old occupied coordinates."""
    prefix_counts: Dict[int, int] = {}
    hashers: Dict[int, hashlib._Hash] = {}
    occupied: Dict[str, list[np.ndarray]] = {}
    root = new_source.create_group("lengths")

    for key in sorted(old_source["lengths"].keys(), key=int):
        length = int(key)
        old_group = old_source[f"lengths/{key}"]
        target = int(old_group.attrs.get("target_count", len(old_group["sequences"])))
        group = _create_source_group(root, length, target, chunk_size)
        seen: set[bytes] = set()
        digest = hashlib.sha256()

        for start in range(0, target, chunk_size):
            end = min(start + chunk_size, target)
            sequences = old_group["sequences"][start:end]
            chroms = old_group["chrom"][start:end]
            starts = np.asarray(old_group["start"][start:end], dtype=np.uint32)

            for chrom_value in np.unique(chroms):
                mask = chroms == chrom_value
                chrom = _as_bytes(chrom_value).decode("ascii")
                chrom_starts = starts[mask]
                intervals = np.column_stack(
                    (chrom_starts, chrom_starts.astype(np.uint64) + length)
                ).astype(np.uint32)
                occupied.setdefault(chrom, []).append(intervals)

            keep_local: list[int] = []
            for index, value in enumerate(sequences):
                sequence = _as_bytes(value)
                if sequence in seen:
                    continue
                seen.add(sequence)
                keep_local.append(index)
                digest.update(sequence)
                digest.update(b"\n")
            if keep_local:
                keep = np.asarray(keep_local, dtype=np.int64)
                _append(group["sequences"], sequences[keep])
                _append(group["chrom"], chroms[keep])
                _append(group["start"], starts[keep])

        prefix_counts[length] = len(seen)
        hashers[length] = digest
        group.attrs["reused_unique_prefix_count"] = len(seen)
        LOGGER.info(
            "Length %d: retained %s/%s existing rows (need %s new)",
            length,
            f"{len(seen):,}",
            f"{target:,}",
            f"{target - len(seen):,}",
        )

    return prefix_counts, hashers, occupied


def _free_intervals(
    fasta: IndexedFasta,
    occupied: Mapping[str, Sequence[np.ndarray]],
    chromosomes: Iterable[str],
) -> list[tuple[str, int, int]]:
    free: list[tuple[str, int, int]] = []
    for chrom in chromosomes:
        arrays = occupied.get(chrom, ())
        intervals = (
            np.concatenate(arrays, axis=0)
            if arrays
            else np.empty((0, 2), dtype=np.uint32)
        )
        if len(intervals):
            intervals = intervals[np.argsort(intervals[:, 0], kind="stable")]
        cursor = 0
        for start_value, end_value in intervals:
            start, end = int(start_value), int(end_value)
            assert cursor <= start, f"Old source has coordinate overlap on {chrom}"
            if cursor < start:
                free.append((chrom, cursor, start))
            cursor = end
        if cursor < fasta.records[chrom].length:
            free.append((chrom, cursor, fasta.records[chrom].length))
    return free


def _fill_unique_suffixes(
    fasta: IndexedFasta,
    new_source: h5py.File,
    prefix_counts: Mapping[int, int],
    hashers: Mapping[int, hashlib._Hash],
    free: Sequence[tuple[str, int, int]],
    scan_block_size: int,
) -> None:
    free_cursor = 0
    for length in sorted(prefix_counts):
        group = new_source[f"lengths/{length}"]
        target = int(group.attrs["target_count"])
        seen = {_as_bytes(value) for value in group["sequences"][:]}
        added = 0
        while len(seen) < target:
            assert free_cursor < len(free), (
                f"Free reference sequence exhausted while filling length {length}: "
                f"{len(seen):,}/{target:,} unique rows"
            )
            chrom, gap_start, gap_end = free[free_cursor]
            free_cursor += 1
            for block_start in range(gap_start, gap_end, scan_block_size):
                block_end = min(block_start + scan_block_size, gap_end)
                block = fasta.fetch(chrom, block_start, block_end)
                block_sequences: list[bytes] = []
                block_chroms: list[bytes] = []
                block_starts: list[int] = []
                for run_start, run_end in iter_acgt_runs(block):
                    cursor = run_start
                    while cursor + length <= run_end and len(seen) < target:
                        sequence = block[cursor : cursor + length]
                        coordinate = block_start + cursor
                        cursor += length
                        if sequence in seen:
                            continue
                        seen.add(sequence)
                        hashers[length].update(sequence)
                        hashers[length].update(b"\n")
                        block_sequences.append(sequence)
                        block_chroms.append(chrom.encode("ascii"))
                        block_starts.append(coordinate)
                        added += 1
                    if len(seen) == target:
                        break
                if block_sequences:
                    _append(
                        group["sequences"],
                        np.asarray(block_sequences, dtype=f"S{length}"),
                    )
                    _append(group["chrom"], np.asarray(block_chroms, dtype="S16"))
                    _append(
                        group["start"],
                        np.asarray(block_starts, dtype=np.uint32),
                    )
                if len(seen) == target:
                    break

        assert len(group["sequences"]) == target
        group["split"][:] = deterministic_split_codes(
            target,
            int(new_source.attrs["val_size_per_length"]),
            int(new_source.attrs["test_size_per_length"]),
            int(new_source.attrs["seed"]),
        )
        group.attrs["num_unique_sequences"] = target
        group.attrs["new_unique_suffix_count"] = added
        group.attrs["ordered_sequence_sha256"] = hashers[length].hexdigest()
        LOGGER.info("Length %d: added %s new unique rows", length, f"{added:,}")


def build_clean_source(
    *,
    old_source_path: str,
    output_path: str,
    fasta_path: str,
    seed: int,
    val_size: int,
    test_size: int,
    chunk_size: int,
    scan_block_size: int,
) -> Dict[int, int]:
    """Create the v2 source and return reusable-prefix counts by length."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    assert not output.exists(), f"Refusing to overwrite {output}"
    partial = Path(str(output) + ".partial")
    assert not partial.exists(), f"Partial output already exists: {partial}"

    with h5py.File(old_source_path, "r", swmr=True) as old_source:
        assert str(old_source.attrs["schema"]) == OLD_SCHEMA
        with h5py.File(partial, "w") as new_source:
            new_source.attrs["schema"] = SCHEMA
            new_source.attrs["source_corpus"] = os.path.abspath(old_source_path)
            new_source.attrs["reference"] = os.path.abspath(fasta_path)
            new_source.attrs["coordinate_system"] = "0-based half-open"
            new_source.attrs["seed"] = seed
            new_source.attrs["val_size_per_length"] = val_size
            new_source.attrs["test_size_per_length"] = test_size
            new_source.attrs["split_names"] = json.dumps(SPLITS)
            new_source.attrs["sequence_content_deduplicated"] = True
            new_source.attrs["sequence_content_unique_per_length"] = True
            new_source.attrs["sequence_content_split_disjoint"] = True
            new_source.attrs["cross_length_coordinate_overlap"] = False
            new_source.attrs["generation_complete"] = False

            prefix_counts, hashers, occupied = _collect_old_unique_prefixes(
                old_source, new_source, chunk_size
            )
            with IndexedFasta(fasta_path) as fasta:
                chromosomes = select_chromosomes(fasta.records.keys(), None)
                free = _free_intervals(fasta, occupied, chromosomes)
                del occupied
                _fill_unique_suffixes(
                    fasta,
                    new_source,
                    prefix_counts,
                    hashers,
                    free,
                    scan_block_size,
                )
            new_source.attrs["num_sequences"] = sum(
                len(group["sequences"])
                for group in new_source["lengths"].values()
            )
            new_source.attrs["generation_complete"] = True
            new_source.flush()
    os.replace(partial, output)
    LOGGER.info("Wrote cleaned source: %s", output)
    return prefix_counts


def _new_embedding_paths(data_dir: str, fm: str, length: int) -> Dict[str, str]:
    return {
        split: os.path.join(
            data_dir,
            f"{split}_{fm}_{length}_hg38_multilen_seqdisjoint.h5",
        )
        for split in SPLITS
    }


def _old_embedding_paths(data_dir: str, fm: str, length: int) -> Dict[str, str]:
    return {
        split: os.path.join(data_dir, f"{split}_{fm}_{length}_hg38_multilen.h5")
        for split in SPLITS
    }


def _create_partial_embedding_files(
    paths: Mapping[str, str],
    counts: Mapping[str, int],
    embedding_dim: int,
    input_digest: str,
    input_count: int,
    seq_length: int,
) -> Dict[str, h5py.File]:
    files: Dict[str, h5py.File] = {}
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for split, path in paths.items():
        assert not os.path.exists(path), f"Refusing to overwrite {path}"
        handle = h5py.File(path, "w")
        n = counts[split]
        handle.create_dataset(
            "sequences", shape=(n,), maxshape=(None,), dtype=string_dtype
        )
        handle.create_dataset(
            "embeddings",
            shape=(n, embedding_dim),
            maxshape=(None, embedding_dim),
            dtype=np.float32,
            chunks=(min(4096, max(1, n)), embedding_dim),
        )
        handle.attrs["embedding_dim"] = embedding_dim
        handle.attrs["input_sequence_sha256"] = input_digest
        handle.attrs["input_num_sequences"] = input_count
        handle.attrs["seq_length"] = seq_length
        handle.attrs["reused_prefix_num_sequences"] = sum(counts.values())
        handle.attrs["generation_complete"] = False
        files[split] = handle
    return files


def seed_embedding_prefix(
    *,
    clean_source_path: str,
    old_data_dir: str,
    new_data_dir: str,
    foundation_model: str,
    length: int,
    copy_chunk_size: int,
) -> None:
    """Copy a reusable input prefix into resumable per-split embedding files."""
    old_paths = _old_embedding_paths(old_data_dir, foundation_model, length)
    new_paths = _new_embedding_paths(new_data_dir, foundation_model, length)
    with h5py.File(clean_source_path, "r", swmr=True) as source:
        group = source[f"lengths/{length}"]
        target_count = int(group.attrs["target_count"])
        prefix_count = int(group.attrs["reused_unique_prefix_count"])
        input_digest = str(group.attrs["ordered_sequence_sha256"])
        labels = np.asarray(group["split"][:], dtype=np.uint8)
        sequence_to_input: Dict[bytes, int] = {}
        for start in range(0, prefix_count, copy_chunk_size):
            values = group["sequences"][start : min(start + copy_chunk_size, prefix_count)]
            for offset, value in enumerate(values):
                sequence_to_input[_as_bytes(value)] = start + offset
    assert len(sequence_to_input) == prefix_count

    positions = np.empty(target_count, dtype=np.int64)
    for code in range(3):
        indices = np.flatnonzero(labels == code)
        positions[indices] = np.arange(len(indices), dtype=np.int64)
    prefix_split_counts = {
        split: int(np.count_nonzero(labels[:prefix_count] == code))
        for split, code in SPLIT_TO_CODE.items()
    }

    available = all(os.path.exists(path) for path in old_paths.values())
    if available:
        with h5py.File(old_paths["train"], "r", swmr=True) as anchor:
            embedding_dim = int(anchor["embeddings"].shape[1])
    else:
        embedding_dim = DEFAULT_DIMS[foundation_model]
        prefix_count = 0
        prefix_split_counts = {split: 0 for split in SPLITS}
        sequence_to_input.clear()
        LOGGER.warning(
            "%s length %d: old split files incomplete; creating empty resume files",
            foundation_model,
            length,
        )

    Path(new_data_dir).mkdir(parents=True, exist_ok=True)
    output_files = _create_partial_embedding_files(
        new_paths,
        prefix_split_counts,
        embedding_dim,
        input_digest,
        target_count,
        length,
    )
    try:
        if prefix_count:
            filled = np.zeros(prefix_count, dtype=np.bool_)
            for old_split in SPLITS:
                with h5py.File(old_paths[old_split], "r", swmr=True) as old_file:
                    assert old_file["embeddings"].shape[1] == embedding_dim
                    old_sequences = old_file["sequences"]
                    old_embeddings = old_file["embeddings"]
                    for start in range(0, len(old_sequences), copy_chunk_size):
                        end = min(start + copy_chunk_size, len(old_sequences))
                        sequence_chunk = old_sequences[start:end]
                        chosen: Dict[int, int] = {}
                        for local_index, value in enumerate(sequence_chunk):
                            input_index = sequence_to_input.get(_as_bytes(value))
                            if (
                                input_index is not None
                                and not filled[input_index]
                                and input_index not in chosen
                            ):
                                chosen[input_index] = local_index
                        if not chosen:
                            continue
                        embedding_chunk = np.asarray(
                            old_embeddings[start:end], dtype=np.float32
                        )
                        for new_split, code in SPLIT_TO_CODE.items():
                            selected = [
                                (input_index, local_index)
                                for input_index, local_index in chosen.items()
                                if int(labels[input_index]) == code
                            ]
                            if not selected:
                                continue
                            selected.sort(key=lambda item: int(positions[item[0]]))
                            target_rows = np.asarray(
                                [positions[item[0]] for item in selected], dtype=np.int64
                            )
                            local_rows = np.asarray(
                                [item[1] for item in selected], dtype=np.int64
                            )
                            output_files[new_split]["sequences"][target_rows] = [
                                _as_bytes(sequence_chunk[index]).decode("ascii")
                                for index in local_rows
                            ]
                            output_files[new_split]["embeddings"][target_rows] = (
                                embedding_chunk[local_rows]
                            )
                            for input_index, _local_index in selected:
                                filled[input_index] = True
            assert bool(np.all(filled)), (
                f"Could not find {int((~filled).sum())} retained sequences in old "
                f"{foundation_model} length-{length} embedding files"
            )
        for handle in output_files.values():
            handle.flush()
    finally:
        for handle in output_files.values():
            handle.close()
    LOGGER.info(
        "%s length %d: copied %s-row prefix; regular generation will add %s rows",
        foundation_model,
        length,
        f"{prefix_count:,}",
        f"{target_count - prefix_count:,}",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy reusable embeddings into a sequence-disjoint resume dataset"
    )
    parser.add_argument(
        "--old-source", default="data/hg38_multilen/hg38_multilen.h5"
    )
    parser.add_argument(
        "--new-source",
        default="data/hg38_multilen_seqdisjoint/hg38_multilen_seqdisjoint.h5",
    )
    parser.add_argument("--old-data-dir", default="data/hg38_multilen")
    parser.add_argument("--new-data-dir", default="data/hg38_multilen_seqdisjoint")
    parser.add_argument("--fasta", default="data/hg38.fa")
    parser.add_argument(
        "--foundation-models", nargs="+", choices=DEFAULT_FMS, default=DEFAULT_FMS
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=50_000)
    parser.add_argument("--test-size", type=int, default=50_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--copy-chunk-size", type=int, default=4096)
    parser.add_argument("--scan-block-size", type=int, default=1_000_000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )
    build_clean_source(
        old_source_path=args.old_source,
        output_path=args.new_source,
        fasta_path=args.fasta,
        seed=args.seed,
        val_size=args.val_size,
        test_size=args.test_size,
        chunk_size=args.chunk_size,
        scan_block_size=args.scan_block_size,
    )
    with h5py.File(args.new_source, "r", swmr=True) as source:
        lengths = sorted(map(int, source["lengths"].keys()))
    for length in lengths:
        for foundation_model in args.foundation_models:
            seed_embedding_prefix(
                clean_source_path=args.new_source,
                old_data_dir=args.old_data_dir,
                new_data_dir=args.new_data_dir,
                foundation_model=foundation_model,
                length=length,
                copy_chunk_size=args.copy_chunk_size,
            )
    LOGGER.info(
        "Migration complete. Re-run prepare_and_generate_multilen.sh; it will "
        "resume each new H5 after the copied prefix."
    )


if __name__ == "__main__":
    main()
