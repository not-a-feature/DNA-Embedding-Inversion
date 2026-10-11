"""
Corrector Model Evaluation Logic.
"""

import os
import re
import logging
import time
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np
from omegaconf import OmegaConf
import seaborn as sns
from collections import defaultdict

from src.utils import load_run_config, save_json, NUCLEOTIDES
from src.plotting_utils import (
    get_series_color,
    plot_aggregate_metrics,
)
from src.tokenizers import CharacterTokenizer
from src.data import load_split_embeddings, create_dataset
from src.evaluate import (
    compute_sequence_accuracy,
    compare_nucleotide_distributions,
    compute_all_levenshtein_similarities,
    plot_nucleotide_frequencies,
    plot_accuracy_metrics,
    save_sequences_to_csv,
    generate_random_baseline_sequences,
    compute_per_position_accuracy,
    compute_nucleotide_confusion_matrix,
    plot_per_position_accuracy,
    plot_nucleotide_confusion_matrix,
    plot_levenshtein_comparison,
    plot_similarity_ridge,
    compute_nucleotide_frequencies,
)
from src.corrector_inference import reconstruct_iterative
from src.model.corrector import CorrectorReconstructor
from src.model.encoder import EncoderReconstructor

logger = logging.getLogger(__name__)


def load_corrector_models(
    run_dir: str, device: torch.device, config: Dict[str, Any], tokenizer: Any
) -> Tuple[CorrectorReconstructor, EncoderReconstructor]:
    """Load both the corrector model and the associated base model.

    Parameters
    ----------
    run_dir : str
        Path to the run directory.
    device : torch.device
        Device to load models on.
    config : Dict[str, Any]
        Run configuration.
    tokenizer : Any
        Tokenizer for base model output dim.

    Returns
    -------
    Tuple[CorrectorReconstructor, EncoderReconstructor]
        The loaded corrector and base models.
    """
    # 1. Load Corrector Model
    checkpoint_path = os.path.join(run_dir, "model.pt")
    assert os.path.exists(checkpoint_path), f"model.pt not found in {run_dir}"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    corrector_config = checkpoint["config"]

    # Verify it is a corrector
    assert checkpoint["mode"] == "corrector", f"Expected mode='corrector', got {checkpoint['mode']}"

    output_dim = checkpoint["output_dim"]
    effective_seq_length = checkpoint.get("effective_seq_length", config["data"]["seq_length"])

    model_cfg = corrector_config["model"]
    corrector_model = CorrectorReconstructor(
        input_dim=checkpoint["input_dim"],
        seq_length=effective_seq_length,
        output_dim=output_dim,
        d_model=model_cfg["d_model"],
        n_proj=checkpoint.get("n_proj", model_cfg.get("n_proj", 8)),
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["num_layers"],
        num_decoder_layers=model_cfg.get("num_decoder_layers", model_cfg["num_layers"]),
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
    )
    corrector_model.load_state_dict(checkpoint["state_dict"])
    corrector_model.to(device)
    corrector_model.eval()
    logger.info(f"Loaded Corrector model from {checkpoint_path}")

    # 2. Load Base Model
    base_model_path = os.path.join(run_dir, "base_model.pt")
    assert os.path.exists(base_model_path), f"base_model.pt not found in {run_dir}"

    # Base model is always an EncoderReconstructor (MLP) in this pipeline
    base_model = EncoderReconstructor(
        input_dim=config["data"]["embedding_dim"],
        mode="mean",  # Base model for corrector is always mean-based MLP
        seq_length=effective_seq_length,
        output_dim=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        dropout=config["model"]["dropout"],
    ).to(device)

    base_model.load_state_dict(torch.load(base_model_path, map_location=device))
    base_model.eval()
    logger.info(f"Loaded Base model from {base_model_path}")

    return corrector_model, base_model


def evaluate_corrector_run(
    run_dir: str,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger,
    max_samples: Optional[int] = None,
    iterations: int = 1,
) -> Dict[str, Any]:
    """Evaluate a single corrector run.

    Parameters
    ----------
    run_dir : str
        Path to the training run directory.
    device : torch.device
        Device for model inference.
    output_dir : str
        Directory to save evaluation results.
    logger : logging.Logger
        Logger instance.
    max_samples : int | None
        Maximum number of samples to evaluate.
    iterations : int
        Number of refinement iterations.

    Returns
    -------
    Dict[str, Any]
        Evaluation results.
    """
    logger.info(f"Evaluating corrector run: {run_dir}")

    # Load run configuration
    run_config = load_run_config(run_dir)

    # Infer foundation model for metadata
    foundation_model = "unknown"
    if "data" in run_config and "train_csv" in run_config["data"]:
        train_csv = run_config["data"]["train_csv"]
        basename = os.path.basename(train_csv)
        match = re.search(r"train_(.+?)_\d+_hg38", basename)
        if match:
            foundation_model = match.group(1)

    # Always use character-level tokenizer (A=0, C=1, G=2, T=3)
    tokenizer = CharacterTokenizer()

    # Load models
    corrector_model, base_model = load_corrector_models(run_dir, device, run_config, tokenizer)

    # Load test data
    data_cfg = OmegaConf.create(run_config["data"])
    data_dict, counts_dict, train_stats = load_split_embeddings(data_cfg)
    logger.info(f"Loaded test data: {data_cfg['test_csv']} with {counts_dict['test']} samples")

    if train_stats:
        logger.info(
            f"Training embedding stats - min: {train_stats['min']:.4f}, "
            f"max: {train_stats['max']:.4f}, mean: {train_stats['mean']:.4f}, "
            f"std: {train_stats['std']:.4f}"
        )

    # Create dataset (Corrector always uses mean embeddings as input)
    test_dataset = create_dataset(
        data=data_dict["test"],
        mode="mean",
        tokenizer=tokenizer,
        embedding_dim=data_cfg.embedding_dim,
        seq_length=data_cfg.seq_length,
        normalization_stats=train_stats,
        normalization_method=data_cfg.normalization_method,
        data_is_mean=data_cfg.get("mean", False),
        subset_fraction=None,
        max_samples=max_samples,
    )

    seq_length = data_cfg.seq_length
    true_sequences = []
    embeddings_list = []

    logger.info("Extracting sequences and embeddings...")
    start_time = time.time()

    for idx in range(len(test_dataset)):
        # Get sequence
        s = test_dataset.h5_file["sequences"][idx]
        if isinstance(s, bytes):
            true_sequences.append(s.decode("utf-8"))
        else:
            true_sequences.append(str(s))

        # Get embedding
        emb, _ = test_dataset[idx]
        embeddings_list.append(emb.numpy())

    target_embeddings = np.array(embeddings_list)

    # Get FM checkpoint for re-embedding
    fm_checkpoint = run_config["data"].get("checkpoint", "zhihan1996/DNABERT-2-117M")
    logger.info(f"Using Foundation Model: {fm_checkpoint}")

    # Reconstruct sequences iteratively
    logger.info(f"Reconstructing {len(true_sequences)} sequences with {iterations} iterations...")

    predicted_sequences = reconstruct_iterative(
        corrector_model=corrector_model,
        base_model=base_model,
        target_embeddings=target_embeddings,
        fm_checkpoint=fm_checkpoint,
        tokenizer=tokenizer,
        device=device,
        iterations=iterations,
        normalization_stats=train_stats,
        normalization_method=data_cfg.normalization_method,
    )

    assert len(predicted_sequences) == len(true_sequences)

    end_time = time.time()
    total_time = end_time - start_time
    time_per_sequence = total_time / len(predicted_sequences)
    logger.info(
        f"Reconstructed {len(predicted_sequences)} sequences in {total_time:.2f}s ({time_per_sequence:.4f}s/seq)"
    )

    # --- METRICS AND PLOTTING (Copied from evaluate.py) ---

    # Compute accuracy metrics
    logger.info("Computing accuracy metrics...")
    accuracy_metrics = compute_sequence_accuracy(true_sequences, predicted_sequences)
    logger.info("Accuracy metrics:")
    for metric, value in accuracy_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Compute per-sequence accuracy for std dev
    nucleotide_accuracies = []
    for t, p in zip(true_sequences, predicted_sequences):
        max_len = max(len(t), len(p))
        matches = sum(1 for i in range(min(len(t), len(p))) if t[i] == p[i])
        if max_len > 0:
            nucleotide_accuracies.append(matches / max_len)
        else:
            nucleotide_accuracies.append(1.0)

    accuracy_mean = np.mean(nucleotide_accuracies)
    accuracy_std = np.std(nucleotide_accuracies)
    logger.info(f"Accuracy stats: mean={accuracy_mean:.4f}, std={accuracy_std:.4f}")

    # Compute Levenshtein similarities
    logger.info("Computing Levenshtein similarities...")
    levenshtein_similarities = compute_all_levenshtein_similarities(
        true_sequences, predicted_sequences
    )
    logger.info(
        f"Levenshtein similarity stats: min={min(levenshtein_similarities):.3f}, "
        f"max={max(levenshtein_similarities):.3f}, mean={np.mean(levenshtein_similarities):.3f}"
    )

    # Generate random baseline
    logger.info("Generating random baseline sequences...")
    baseline_sequences = generate_random_baseline_sequences(true_sequences)

    # Compute baseline metrics
    logger.info("Computing baseline metrics...")
    baseline_accuracy = compute_sequence_accuracy(true_sequences, baseline_sequences)
    baseline_levenshtein = compute_all_levenshtein_similarities(true_sequences, baseline_sequences)

    # Compute improvement over baseline
    improvement = {
        key: (
            ((accuracy_metrics[key] - baseline_accuracy[key]) / baseline_accuracy[key] * 100)
            if baseline_accuracy[key] != 0
            else 0.0
        )
        for key in accuracy_metrics.keys()
    }
    logger.info("Model improvement over baseline (% increase):")
    for metric, value in improvement.items():
        logger.info(f"  {metric}: {value:.2f}%")

    # Compare nucleotide distributions
    logger.info("Comparing nucleotide distributions...")

    # Check if we have any valid predicted sequences
    if any(len(s) > 0 for s in predicted_sequences):
        freq_comparison = compare_nucleotide_distributions(true_sequences, predicted_sequences)
    else:
        logger.warning("All predicted sequences are empty! Skipping frequency comparison.")
        freq_comparison = {
            "true": compute_nucleotide_frequencies(true_sequences),
            "pred": {n: 0.0 for n in NUCLEOTIDES},
        }

    # Compute per-position accuracy
    logger.info("Computing per-position accuracy...")
    model_position_accuracy = compute_per_position_accuracy(true_sequences, predicted_sequences)
    baseline_position_accuracy = compute_per_position_accuracy(true_sequences, baseline_sequences)

    # Compute nucleotide confusion matrix
    logger.info("Computing nucleotide confusion matrix...")
    confusion_matrix = compute_nucleotide_confusion_matrix(true_sequences, predicted_sequences)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save numerical results
    results = {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "model_name": "corrector",
        "inversion_model": "corrector",
        "foundation_model": foundation_model,
        "mode": "corrector",
        "iterations": iterations,
        "seq_length": seq_length,
        "num_sequences": len(true_sequences),
        "accuracy_metrics": accuracy_metrics,
        "baseline_accuracy_metrics": baseline_accuracy,
        "improvement_over_baseline": improvement,
        "nucleotide_frequencies": freq_comparison,
        "levenshtein_mean": float(np.mean(levenshtein_similarities)),
        "levenshtein_std": float(np.std(levenshtein_similarities)),
        "baseline_levenshtein_mean": float(np.mean(baseline_levenshtein)),
        "baseline_levenshtein_std": float(np.std(baseline_levenshtein)),
        "accuracy_mean": float(accuracy_mean),
        "accuracy_std": float(accuracy_std),
        "total_inference_time": total_time,
        "time_per_sequence": time_per_sequence,
        "per_position_accuracy": model_position_accuracy.tolist(),
    }
    save_json(results, os.path.join(output_dir, "evaluation_results.json"))

    # Save plot data
    plot_data = {
        "freq_comparison": freq_comparison,
        "accuracy_metrics": accuracy_metrics,
        "levenshtein_similarities": levenshtein_similarities,
        "baseline_levenshtein": baseline_levenshtein,
        "model_position_accuracy": model_position_accuracy,
        "baseline_position_accuracy": baseline_position_accuracy,
        "confusion_matrix": confusion_matrix,
    }
    torch.save(plot_data, os.path.join(output_dir, "plot_data.pt"))

    # Save sequences to CSV
    save_sequences_to_csv(
        true_sequences,
        predicted_sequences,
        levenshtein_similarities,
        os.path.join(output_dir, "sequences.csv"),
    )

    # Save example reconstructions
    examples = []
    for i in range(len(true_sequences)):
        examples.append(
            {
                "index": i,
                "true_sequence": true_sequences[i],
                "predicted_sequence": predicted_sequences[i],
                "match": true_sequences[i] == predicted_sequences[i],
            }
        )
    save_json({"examples": examples}, os.path.join(output_dir, "example_reconstructions.json"))

    # --- PLOTTING ---
    logger.info("Creating plots...")

    # Use "corrector" plus dataset info for color
    display_model_name = f"corrector ({foundation_model})"
    model_color = get_series_color(display_model_name)

    plot_nucleotide_frequencies(
        freq_comparison,
        os.path.join(output_dir, "nucleotide_frequencies"),
        "Nucleotide Frequency Comparison (Test Set)",
        model_color=model_color,
    )

    plot_accuracy_metrics(
        accuracy_metrics,
        os.path.join(output_dir, "accuracy_metrics"),
        model_color=model_color,
    )

    plot_per_position_accuracy(
        model_position_accuracy,
        baseline_position_accuracy,
        os.path.join(output_dir, "per_position_accuracy"),
        model_color=model_color,
    )

    plot_nucleotide_confusion_matrix(
        confusion_matrix,
        NUCLEOTIDES,
        os.path.join(output_dir, "confusion_matrix"),
        "Nucleotide Confusion Matrix (Model Predictions)",
        cmap=sns.light_palette(model_color, as_cmap=True),
    )

    plot_levenshtein_comparison(
        levenshtein_similarities,
        baseline_levenshtein,
        os.path.join(output_dir, "levenshtein_comparison"),
        model_color=model_color,
    )

    # Ridge plot (manually prepare data as in evaluate.py)
    model_sims_by_len = defaultdict(list)
    baseline_sims_by_len = defaultdict(list)

    for i, seq in enumerate(true_sequences):
        l = len(seq)
        model_sims_by_len[l].append(levenshtein_similarities[i])
        baseline_sims_by_len[l].append(baseline_levenshtein[i])

    plot_similarity_ridge(
        model_sims_by_len,
        baseline_sims_by_len,
        os.path.join(output_dir, "similarity_ridge"),
        model_color=model_color,
    )

    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    return results
