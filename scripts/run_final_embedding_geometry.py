"""Regenerate one geometry analysis from the sequence-disjoint test corpus."""

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embedding_analysis.embedding_analysis_mean import analyze_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args()
    models = ("dnabert2", "evo2", "ntv2")
    lengths = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100)
    assert 0 <= args.index < len(models) * len(lengths)
    fm = models[args.index // len(lengths)]
    length = lengths[args.index % len(lengths)]
    dataset = f"{fm}_{length}_hg38_multilen_seqdisjoint_mean"
    output = ROOT / "outputs" / f"analysis_mean_{fm}" / f"data={dataset}"
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    source = ROOT / "data/hg38_multilen_seqdisjoint/hg38_multilen_seqdisjoint.h5"
    data = {
        "mean": True,
        "embedding_dim": {"dnabert2": 768, "evo2": 4096, "ntv2": 1024}[fm],
        "seq_length": length,
        "skip_sha256_check": True,
        "require_sequence_disjoint": True,
        "source_schema": "dna_inversion_hg38_multilen_v2",
        "source_corpus": str(source),
    }
    for split in ("train", "val", "test"):
        data[f"{split}_csv"] = str(source.parent / f"{split}_{fm}_{length}_hg38_multilen_seqdisjoint.h5")
    settings = {"splits": ["test"], "subsample": 2000, "seed": 42, "n_neighbors": 15, "min_dist": 0.1}
    np.random.seed(settings["seed"])
    stats = analyze_dataset(OmegaConf.create(data), dataset, str(output), OmegaConf.create(settings))
    assert stats["n_samples"] == settings["subsample"]
    assert stats["avg_seq_len"] == length
    assert all(np.isfinite(stats[key]) for key in ("spearman_corr_cosine", "spearman_corr_euclidean"))
    with h5py.File(source, "r") as handle:
        digest = str(handle[f"lengths/{length}"].attrs["ordered_sequence_sha256"])
    provenance = {
        "data": data,
        "analysis": settings,
        "ordered_sequence_sha256": digest,
        "job_id": os.environ["SLURM_ARRAY_JOB_ID"] if "SLURM_ARRAY_JOB_ID" in os.environ else "local",
        "array_index": args.index,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "analysis_script_sha256": hashlib.sha256((ROOT / "embedding_analysis/embedding_analysis_mean.py").read_bytes()).hexdigest(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"COMPLETE {dataset}: {stats['n_samples']} test sequences", flush=True)


if __name__ == "__main__":
    main()
