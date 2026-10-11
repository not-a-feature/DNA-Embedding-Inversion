"""Exercise the real legacy-identification startup path on a tiny cohort slice."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import sys
import tempfile

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as evaluate_entry
from src.evaluate import levenshtein_similarity, reconstruct_sequences
from src.evaluation.locus_blind import _fetch_cohort_snps_at_windows


def _decode(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _single_training_run(foundation_model: str) -> Path:
    models = list((ROOT / "outputs" / f"train_multi_{foundation_model}").rglob("model.pt"))
    assert len(models) == 1, (
        f"Expected exactly one {foundation_model} legacy model, found {len(models)}: {models}"
    )
    return models[0].parent


def smoke_model(
    foundation_model: str,
    device: torch.device,
    num_samples: int,
    batch_size: int,
    logger: logging.Logger,
) -> list[str]:
    run_dir = _single_training_run(foundation_model)
    prep = evaluate_entry._prepare_run(
        str(run_dir), device=device, eval_seq_length=25, logger=logger
    )
    assert prep.foundation_model == foundation_model
    assert prep.train_stats_override

    h5_path = ROOT / "data" / "1000g" / (
        f"test_{foundation_model}_25_1000g_id_random.h5"
    )
    with h5py.File(h5_path, "r", swmr=True) as handle:
        assert bool(handle.attrs["identification_mode"])
        embeddings = np.asarray(handle["embeddings"][:num_samples], dtype=np.float32)
        true_sequences = [_decode(value) for value in handle["sequences"][:num_samples]]
        id_probe = handle["individual_ids"][: min(len(handle["individual_ids"]), 20_000)]
        sample_ids = list(dict.fromkeys(_decode(value) for value in id_probe))[:2]

    assert embeddings.shape == (num_samples, prep.data_cfg.embedding_dim)
    predicted = reconstruct_sequences(
        prep.model,
        {"embeddings": embeddings},
        device,
        prep.tokenizer,
        prep.mode,
        prep.data_cfg.seq_length,
        prep.data_cfg.embedding_dim,
        normalization_stats=prep.train_stats_override,
        normalization_method=prep.data_cfg.normalization_method,
        data_is_mean=prep.data_cfg.mean,
        inference_batch_size=batch_size,
    )
    assert len(predicted) == num_samples
    assert all(isinstance(sequence, str) for sequence in predicted)
    similarities = [
        levenshtein_similarity(true, pred)
        for true, pred in zip(true_sequences, predicted)
    ]
    logger.info(
        "%s smoke reconstruction: n=%d, predicted lengths=%s, mean Levenshtein=%.4f",
        foundation_model,
        num_samples,
        [len(sequence) for sequence in predicted],
        float(np.mean(similarities)),
    )
    return sample_ids


def smoke_bcftools(sample_ids: list[str], logger: logging.Logger) -> None:
    assert len(sample_ids) == 2, f"Expected two cohort samples, got {sample_ids}"
    bcftools = shutil.which("bcftools")
    assert bcftools, "bcftools is not available after project activation"
    windows = [("chr1", 1_000_000, 1_000_024), ("chr1", 2_000_000, 2_000_024)]
    with tempfile.TemporaryDirectory(prefix="legacy_id_smoke_", dir=ROOT / "outputs") as workdir:
        result = _fetch_cohort_snps_at_windows(
            bcftools=bcftools,
            vcf_dir=str(ROOT / "data" / "1000g_vcfs"),
            windows=windows,
            samples=sample_ids,
            workdir=workdir,
        )
    assert set(result) == set(windows)
    logger.info("bcftools -R smoke query succeeded for %d windows", len(windows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    assert args.num_samples > 0 and args.batch_size > 0

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("legacy-identification-smoke")
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA was requested but is unavailable"

    sample_ids: list[str] | None = None
    for foundation_model in ("evo2", "ntv2"):
        current_ids = smoke_model(
            foundation_model,
            device=device,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            logger=logger,
        )
        if sample_ids is None:
            sample_ids = current_ids
        else:
            assert current_ids == sample_ids, "FM identification cohorts use different samples"

    assert sample_ids is not None
    smoke_bcftools(sample_ids, logger)
    logger.info("LEGACY IDENTIFICATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
