"""Prepare, extract, train and evaluate the four-layer LCQD study.

Run from the repository root; see outputs/layer_sensitivity_v1/README.md.
"""

import argparse
import importlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.generation_io import SPLITS, _TrainStats, ordered_sequence_sha256
from src.utils import file_sha256, set_determinism

ROOT = Path(__file__).resolve().parents[1]
LAYERS = {"dnabert2": [3, 6, 9, 12], "ntv2": [7, 15, 22, 29], "evo2": [7, 15, 26, 31]}
ANCHORS = {"dnabert2": 12, "ntv2": 29, "evo2": 26}
LENGTHS = [25, 50, 100]
COUNTS = [100_000, 5_000, 10_000]
SCHEMA = "dna_inversion_hg38_multilen_v2"


def prepare(source_path, destination, counts=COUNTS):
    """Sample within each original split and retain original row IDs."""
    assert not destination.exists(), f"Subset already exists: {destination}"
    assert len(counts) == 3 and min(counts) > 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(destination, "x") as out:
        assert source.attrs["schema"] == SCHEMA
        for key in ("generation_complete", "sequence_content_unique_per_length", "sequence_content_split_disjoint"):
            assert source.attrs[key], key
        assert not source.attrs["cross_length_coordinate_overlap"]
        out.attrs.update(dict(source.attrs))
        out.attrs.update(generation_complete=False, source_path=str(source_path), seed=42,
                         val_size_per_length=counts[1], test_size_per_length=counts[2])
        for length in LENGTHS:
            original = source[f"lengths/{length}"]
            labels = original["split"][:]
            rng = np.random.default_rng(42 + length)
            selected = []
            for code, count in enumerate([50_000, counts[1], counts[2]]):
                candidates = np.flatnonzero(labels == code)
                assert len(candidates) >= count, (length, code, len(candidates), count)
                selected.extend(rng.choice(candidates, count, replace=False))
            # Reproduce the published two-stage selection, including row order.
            # The second draw extends training only; held-out rows stay fixed.
            old_rows = np.sort(selected)
            candidates = np.setdiff1d(np.flatnonzero(labels == 0), old_rows)
            extra_count = counts[0] - 50_000
            assert extra_count > 0 and len(candidates) >= extra_count
            extra = np.random.default_rng(42 + length).choice(candidates, extra_count, replace=False)
            rows = np.concatenate([old_rows, np.sort(extra)])
            group = out.create_group(f"lengths/{length}")
            for key in ("sequences", "split", "chrom", "start"):
                # Read contiguous HDF5 data, then select rows in memory. Large
                # HDF5 point selections are prohibitively slow on this corpus.
                group.create_dataset(key, data=original[key][:][rows])
            group.create_dataset("source_row", data=rows)
            sequences = group["sequences"].asstr()[:].tolist()
            assert len(set(sequences)) == len(sequences)
            assert all(len(s) == length for s in sequences)
            group.attrs.update(seq_length=length, target_count=len(rows),
                               ordered_sequence_sha256=ordered_sequence_sha256(sequences),
                               parent_sequence_sha256=original.attrs["ordered_sequence_sha256"])
            logging.info("Prepared length %d: %d rows", length, len(rows))
        out.attrs["generation_complete"] = True


def audit_subset(path):
    report = {"lengths": {}, "coordinate_overlaps": 0}
    coordinates = []
    with h5py.File(path, "r") as source:
        assert source.attrs["schema"] == SCHEMA
        assert source.attrs["generation_complete"]
        assert not source.attrs["cross_length_coordinate_overlap"]
        for length in LENGTHS:
            group = source[f"lengths/{length}"]
            seqs = group["sequences"].asstr()[:].tolist()
            labels = group["split"][:]
            rows = group["source_row"][:]
            assert len(set(seqs)) == len(seqs), f"Duplicate sequences at length {length}"
            assert len(np.unique(rows)) == len(rows), f"Repeated source rows at length {length}"
            assert all(len(s) == length and set(s) <= set("ACGT") for s in seqs)
            assert ordered_sequence_sha256(seqs) == group.attrs["ordered_sequence_sha256"]
            assert np.isin(labels, [0, 1, 2]).all()
            counts = np.bincount(labels, minlength=3)
            assert len(seqs) == group.attrs["target_count"]
            assert counts[1] == source.attrs["val_size_per_length"]
            assert counts[2] == source.attrs["test_size_per_length"]
            report["lengths"][str(length)] = dict(train=int(counts[0]), validation=int(counts[1]),
                                                  test=int(counts[2]), exact_duplicates=0,
                                                  cross_split_exact_duplicates=0)
            starts = group["start"][:]
            assert (starts >= 0).all()
            coordinates.append((group["chrom"][:], starts, starts + length))
    chrom = np.concatenate([c[0] for c in coordinates])
    start = np.concatenate([c[1] for c in coordinates])
    end = np.concatenate([c[2] for c in coordinates])
    order = np.lexsort((start, chrom))
    chrom, start, end = chrom[order], start[order], end[order]
    overlaps = int(np.count_nonzero((chrom[1:] == chrom[:-1]) & (start[1:] < end[:-1])))
    assert overlaps == 0, f"Found {overlaps} genomic-coordinate overlaps across lengths/splits"
    return report


def extraction_name(fm, layer):
    return f"blocks.{layer}.mlp.l3" if fm == "evo2" else f"hidden_layer_{layer}"


def write_embeddings(source_path, directory, fm, layer, dim, checkpoint, embed, chunk_size=128):
    """Resume each split at its last flushed batch, then stamp train-only statistics."""
    directory.mkdir(parents=True, exist_ok=True)
    source_hash = file_sha256(str(source_path))
    with h5py.File(source_path, "r") as source:
        assert source.attrs["generation_complete"]
        for length in LENGTHS:
            group = source[f"lengths/{length}"]
            labels = group["split"][:]
            sequences = group["sequences"].asstr()[:]
            paths = []
            stats = _TrainStats()
            for code, split in enumerate(SPLITS):
                seqs = sequences[labels == code].tolist()
                path = directory / f"{split}_{fm}_{length}_hg38_layer_sensitivity.h5"
                paths.append(path)
                identity = dict(source_sha256=source_hash, extraction_layer=extraction_name(fm, layer),
                                checkpoint=checkpoint, embedding_dim=dim, seq_length=length,
                                input_sequence_sha256=group.attrs["ordered_sequence_sha256"])
                if not path.exists():
                    with h5py.File(path, "x") as out:
                        out.attrs.update(identity)
                        out.attrs.update(completed_rows=0, generation_complete=False)
                        out.create_dataset("sequences", data=seqs, dtype=h5py.string_dtype())
                        out.create_dataset("embeddings", shape=(len(seqs), dim), dtype="f4")
                with h5py.File(path, "r+") as out:
                    for key, value in identity.items():
                        assert out.attrs[key] == value, f"{path}: {key} changed"
                    assert out["sequences"].asstr()[:].tolist() == seqs
                    assert out["embeddings"].shape == (len(seqs), dim)
                    done = int(out.attrs["completed_rows"])
                    assert 0 <= done <= len(seqs)
                    for start in range(done, len(seqs), chunk_size):
                        end = min(start + chunk_size, len(seqs))
                        values = np.asarray(embed(seqs[start:end]), dtype=np.float32)
                        assert values.shape == (end - start, dim)
                        assert np.isfinite(values).all()
                        out["embeddings"][start:end] = values
                        out.flush()
                        out.attrs["completed_rows"] = end
                        out.flush()
                        logging.info("%s L%d %s: %d/%d", fm, length, split, end, len(seqs))
                    if split == "train":
                        for start in range(0, len(seqs), 4096):
                            stats.update(out["embeddings"][start:start + 4096])
            train_stats = stats.finalize()
            assert train_stats["std"] > 0
            for path in paths:
                with h5py.File(path, "r+") as out:
                    for key, value in train_stats.items():
                        out.attrs[f"emb_{key}"] = value
                    out.attrs["generation_complete"] = True


def run_condition(index, workspace, device, workers):
    conditions = [(fm, layer) for fm, layers in LAYERS.items() for layer in layers]
    fm, layer = conditions[index]
    source = workspace / "subset.h5"
    assert source.is_file(), "Run prepare first"
    directory = workspace / fm / f"layer_{layer}"
    directory.mkdir(parents=True, exist_ok=True)
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="train", overrides=[
            f"data=hg38_multilen_seqdisjoint/{fm}_multi_hg38_multilen_seqdisjoint_mean",
            "model=query_decoder", "optim.epochs=20", f"optim.num_workers={workers}",
            f"train.device={device}", "train.seed=42", "train.use_wandb=false",
        ], return_hydra_config=True)
    cfg.data.source_corpus = str(source)
    cfg.data.artifact_dir = str(directory / "embeddings")
    cfg.data.seq_lengths = LENGTHS
    cfg.data.train_seq_lengths = LENGTHS
    for split in SPLITS:
        cfg.data[f"{split}_csvs"] = [str(directory / "embeddings" /
            f"{split}_{fm}_{length}_hg38_layer_sensitivity.h5") for length in LENGTHS]
    cfg.hydra.run.dir = str(directory / "train")
    # Resolve data interpolation only: Hydra runtime interpolation needs the child process.
    cfg.data = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    manifest = dict(foundation_model=fm, layer=extraction_name(fm, layer),
                    paper_layer=layer == ANCHORS[fm], source_sha256=file_sha256(str(source)),
                    training=OmegaConf.to_container(cfg, resolve=False),
                    code_sha256={p: file_sha256(str(ROOT / p)) for p in (
                        "scripts/layer_sensitivity.py", "scripts/layer_sensitivity_extract.py",
                        "train.py", "evaluate.py", "src/data.py", "src/model/query_decoder.py")})
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Study condition changed; use a new workspace"
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    set_determinism(42)
    extractor = importlib.import_module("scripts.layer_sensitivity_extract")
    embed = extractor.make_embedder(fm, layer, cfg.data.checkpoint, device)
    write_embeddings(source, directory / "embeddings", fm, layer, cfg.data.embedding_dim,
                     cfg.data.checkpoint, embed)
    # Release the foundation model before training the inversion decoder.
    del embed
    extractor.torch.cuda.empty_cache()
    config_path = directory / "study_train.yaml"
    OmegaConf.save(cfg, config_path)
    subprocess.run([sys.executable, str(ROOT / "train.py"), "--config-path", str(directory),
                    "--config-name", "study_train"], cwd=ROOT, check=True)
    for length in LENGTHS:
        subprocess.run([sys.executable, str(ROOT / "evaluate.py"),
                        f"run_dir={directory / 'train'}", f"eval_seq_length={length}",
                        f"device={device}", "identification_only=false", "identification.enabled=false",
                        f"hydra.run.dir={directory / f'eval_L{length}'}"], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--workspace", type=Path, default=Path("outputs/layer_sensitivity_v1"))
    parser.add_argument("--source", type=Path,
                        default=Path("data/hg38_multilen_seqdisjoint/hg38_multilen_seqdisjoint.h5"))
    parser.add_argument("--index", type=int, choices=range(12), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.action == "prepare":
        prepare(args.source.resolve(), args.workspace.resolve() / "subset.h5")
        audit = audit_subset(args.workspace.resolve() / "subset.h5")
        (args.workspace.resolve() / "dataset_audit_100k.json").write_text(json.dumps(audit, indent=2))
    else:
        run_condition(args.index, args.workspace.resolve(), args.device, args.workers)


if __name__ == "__main__":
    main()
