"""
DNA Embedding Inversion Attack - Evaluation Entry Point
"""

from __future__ import annotations

import os
import re

# Disable tokenizer parallelism to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import logging
from typing import Dict, List, Any, Tuple

import time
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import (
    find_latest_run_dir,
    load_run_config,
    save_json,
    NUCLEOTIDES,
)
from src.plotting_utils import (
    configure_plot_style,
    get_fm_color,
    plot_nucleotide_frequencies,
    plot_accuracy_metrics,
    plot_per_position_accuracy,
    plot_nucleotide_confusion_matrix,
    plot_levenshtein_comparison,
)
from src.tokenizers import CharacterTokenizer, HuggingFaceTokenizer
from src.data import load_split_embeddings, create_dataset
from src.evaluate import (
    reconstruct_sequences,
    compute_sequence_accuracy,
    compare_nucleotide_distributions,
    compute_all_levenshtein_similarities,
    load_model_from_run,
    save_sequences_to_csv,
    generate_random_baseline_sequences,
    compute_per_position_accuracy,
    compute_nucleotide_confusion_matrix,
    compute_nucleotide_frequencies,
    compute_shannon_entropy,
    compute_repetitiveness,
)
from src.evaluation.multirun import (
    is_multirun_directory,
    get_multirun_subdirs,
    aggregate_and_plot_multirun,
)

# Use Agg backend for non-interactive plotting
matplotlib.use("Agg")

configure_plot_style()


def evaluate_single_run(
    run_dir: str,
    device: torch.device,
    output_dir: str,
    logger: logging.Logger,
    max_samples: int | None = None,
) -> Dict[str, Any]:
    """Evaluate a single training run.

    This performs the full evaluation pipeline for one model and saves
    all plots and metrics to the specified output directory.

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

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - run_dir: path to the run
        - seq_length: sequence length used
        - inversion_model: model architecture/name
        - foundation_model: foundation model name
        - metrics: evaluation metrics
        - levenshtein_mean: mean Levenshtein similarity
        - accuracy_mean: mean overall accuracy
        - time_per_sequence: average inference time per sequence (seconds)
    """
    logger.info(f"Evaluating run: {run_dir}")

    # Load run configuration to get data paths and mode
    run_config = load_run_config(run_dir)
    mode = run_config["model"]["mode"]

    # Extract base model name from config
    model_cfg = run_config["model"]
    base_model_name = model_cfg["model_name"]

    # Check for encoder parameters to distinguish grid search runs
    if model_cfg.get("model_type") == "encoder":
        d_model = model_cfg.get("d_model")
        dim_ff = model_cfg.get("dim_feedforward")
        n_layers = model_cfg.get("num_layers")

        # Only append if all parameters are present (to avoid cluttering non-grid runs if feasible)
        # or just append what is found.
        params = []
        if d_model:
            params.append(f"d={d_model}")
        if dim_ff:
            params.append(f"df={dim_ff}")
        if n_layers:
            params.append(f"L={n_layers}")

        if params:
            base_model_name = f"Encoder ({', '.join(params)})"

    logger.info(f"Base model name from config: {base_model_name}")

    # Infer foundation model from data config to distinguish runs
    foundation_model = "unknown"
    if "data" in run_config and "train_csv" in run_config["data"]:
        train_csv = run_config["data"]["train_csv"]
        # Extract foundation model from filename pattern: train_{foundation_model}_{seq_len}_hg38
        basename = os.path.basename(train_csv)
        match = re.search(r"train_(.+?)_\d+_hg38", basename)
        if match:
            foundation_model = match.group(1)

    # Append foundation model to make model name unique if not already present
    if foundation_model != "unknown" and foundation_model not in base_model_name:
        model_name = f"{base_model_name} ({foundation_model})"
    else:
        model_name = base_model_name

    inversion_model = base_model_name
    display_model_name = model_name

    logger.info(f"Final model name: {model_name}")
    logger.info(f"Using mode from config: {mode}")

    # Load the model
    model = load_model_from_run(run_dir, device)

    # Check if this is a corrector model - redirect to eval_corrector.py
    if model.__class__.__name__ == "CorrectorReconstructor":
        raise RuntimeError(
            "Detected CorrectorReconstructor model. Please use eval_corrector.py instead.\n"
            f"Run: python eval_corrector.py run_dir={run_dir}"
        )

    # Load test data with HDF5
    data_cfg = OmegaConf.create(run_config["data"])
    assert isinstance(data_cfg, DictConfig), "Failed to create DictConfig from data config"

    data_dict, counts_dict, train_stats = load_split_embeddings(data_cfg)
    logger.info(f"Loaded test data: {data_cfg['test_csv']} with {counts_dict['test']} samples")
    if train_stats:
        logger.info(
            f"Training embedding stats (used for normalization) - min: {train_stats['min']:.4f}, "
            f"max: {train_stats['max']:.4f}, mean: {train_stats['mean']:.4f}, "
            f"std: {train_stats['std']:.4f}"
        )
    # Setup Tokenizer
    # Prioritize tokenizer from data config to match training logic

    tokenizer_cfg = run_config["data"]["tokenizer"]
    tokenizer_type = tokenizer_cfg["type"]

    if tokenizer_type == "char":
        tokenizer = CharacterTokenizer()
    elif tokenizer_type == "huggingface":
        # Check if model_name is in config
        hf_model_name = tokenizer_cfg["model_name"]

        tokenizer = HuggingFaceTokenizer(hf_model_name)
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    test_dataset = create_dataset(
        data=data_dict["test"],
        mode=mode,
        tokenizer=tokenizer,
        embedding_dim=data_cfg.embedding_dim,
        seq_length=data_cfg.seq_length,
        normalization_stats=train_stats if train_stats else None,
        normalization_method=data_cfg.normalization_method,
        data_is_mean=data_cfg.mean,
        subset_fraction=data_cfg.subset_fraction,
        max_samples=max_samples,
    )

    base_dataset = test_dataset

    seq_length = data_cfg.seq_length
    true_sequences = []
    embeddings_list = []

    logger.info("Extracting sequences and embeddings...")
    start_time = time.time()

    for idx in range(len(test_dataset)):
        # Get sequence
        s = base_dataset.h5_file["sequences"][idx]
        if isinstance(s, bytes):
            true_sequences.append(s.decode("utf-8"))
        else:
            true_sequences.append(str(s))

        # Get embedding (already normalized by dataset)
        emb, _ = base_dataset[idx]
        embeddings_list.append(emb.numpy())

    # Prepare data for reconstruction
    # Pass as list directly to handle variable sequence lengths
    test_data_for_reconstruction = {"embeddings": embeddings_list}

    # Reconstruct sequences
    logger.info(f"Reconstructing {len(true_sequences)} sequences from embeddings...")

    # Standard reconstruction for per_token and mean modes
    predicted_sequences = reconstruct_sequences(
        model,
        test_data_for_reconstruction,
        device,
        tokenizer,
        mode,
        data_cfg.seq_length,
        data_cfg.embedding_dim,
        normalization_method=data_cfg.normalization_method,
        data_is_mean=data_cfg.mean,
    )

    assert len(predicted_sequences) == len(true_sequences)
    dataset_seq_length = (
        seq_length if seq_length is not None else max(len(s) for s in true_sequences)
    )
    target_seq_length = min(100, dataset_seq_length)
    predicted_sequences = [seq[:target_seq_length] for seq in predicted_sequences]
    logger.info(f"Truncated predicted sequences to length {target_seq_length}")
    end_time = time.time()
    total_time = end_time - start_time
    time_per_sequence = total_time / len(predicted_sequences)
    logger.info(
        f"Reconstructed {len(predicted_sequences)} sequences in {total_time:.2f}s ({time_per_sequence:.4f}s/seq)"
    )

    # Compute accuracy metrics
    logger.info("Computing accuracy metrics...")
    accuracy_metrics = compute_sequence_accuracy(true_sequences, predicted_sequences)
    logger.info("Accuracy metrics:")
    for metric, value in accuracy_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")

    # Compute per-sequence accuracy to get standard deviation
    nucleotide_accuracies = []
    for t, p in zip(true_sequences, predicted_sequences):
        # Matches / max(len_true, len_pred) to be consistent with overall accuracy logic
        max_len = max(len(t), len(p))
        matches = sum(1 for i in range(min(len(t), len(p))) if t[i] == p[i])
        if max_len > 0:
            nucleotide_accuracies.append(matches / max_len)
        else:
            nucleotide_accuracies.append(1.0)

    accuracy_mean = np.mean(nucleotide_accuracies)
    accuracy_std = np.std(nucleotide_accuracies)
    logger.info(f"Accuracy stats: mean={accuracy_mean:.4f}, std={accuracy_std:.4f}")

    # Compute all Levenshtein similarities for distribution analysis
    logger.info("Computing Levenshtein similarities...")
    levenshtein_similarities = compute_all_levenshtein_similarities(
        true_sequences, predicted_sequences
    )
    logger.info(
        f"Levenshtein similarity stats: min={min(levenshtein_similarities):.3f}, "
        f"max={max(levenshtein_similarities):.3f}, mean={np.mean(levenshtein_similarities):.3f}"
    )

    # Generate random baseline sequences for comparison
    logger.info("Generating random baseline sequences...")
    baseline_sequences = generate_random_baseline_sequences(true_sequences)
    logger.info(f"Generated {len(baseline_sequences)} random baseline sequences")

    # Compute baseline metrics
    logger.info("Computing baseline metrics...")
    baseline_accuracy = compute_sequence_accuracy(true_sequences, baseline_sequences)
    baseline_levenshtein = compute_all_levenshtein_similarities(true_sequences, baseline_sequences)
    logger.info("Baseline accuracy metrics:")
    for metric, value in baseline_accuracy.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info(
        f"Baseline Levenshtein similarity stats: min={min(baseline_levenshtein):.3f}, "
        f"max={max(baseline_levenshtein):.3f}, mean={np.mean(baseline_levenshtein):.3f}"
    )

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
    freq_comparison = compare_nucleotide_distributions(true_sequences, predicted_sequences)
    logger.info("True nucleotide frequencies:")
    for nuc, freq in freq_comparison["true"].items():
        logger.info(f"  {nuc}: {freq:.4f}")
    logger.info("Predicted nucleotide frequencies:")
    for nuc, freq in freq_comparison["pred"].items():
        logger.info(f"  {nuc}: {freq:.4f}")

    # Compute per-position accuracy
    logger.info("Computing per-position accuracy...")
    model_position_accuracy = compute_per_position_accuracy(true_sequences, predicted_sequences)
    baseline_position_accuracy = compute_per_position_accuracy(true_sequences, baseline_sequences)
    logger.info(
        f"Model mean position accuracy: {np.mean(model_position_accuracy):.4f}, "
        f"Baseline mean position accuracy: {np.mean(baseline_position_accuracy):.4f}"
    )

    # Compute nucleotide confusion matrix
    logger.info("Computing nucleotide confusion matrix...")
    confusion_matrix = compute_nucleotide_confusion_matrix(true_sequences, predicted_sequences)
    logger.info("Confusion matrix computed")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save numerical results
    # Compute difficulty metrics
    logger.info("Computing difficulty metrics (Entropy, Repetitiveness)...")
    shannon_entropies = compute_shannon_entropy(true_sequences)
    repetitiveness_scores = compute_repetitiveness(true_sequences)

    logger.info(
        f"Shannon Entropy stats (True): mean={np.mean(shannon_entropies):.3f}, std={np.std(shannon_entropies):.3f}"
    )

    # Compute metrics for predicted sequences
    logger.info("Computing metrics for predicted sequences...")
    predicted_shannon_entropies = compute_shannon_entropy(predicted_sequences)
    predicted_repetitiveness_scores = compute_repetitiveness(predicted_sequences)

    logger.info(
        f"Shannon Entropy stats (Pred): mean={np.mean(predicted_shannon_entropies):.3f}, std={np.std(predicted_shannon_entropies):.3f}"
    )

    # Save example reconstructions
    num_examples = len(true_sequences)
    examples = []
    for i in range(num_examples):
        examples.append(
            {
                "index": i,
                "true_sequence": true_sequences[i],
                "predicted_sequence": predicted_sequences[i],
                "match": true_sequences[i] == predicted_sequences[i],
                "shannon_entropy": float(shannon_entropies[i]),
                "repetitiveness": float(repetitiveness_scores[i]),
                "predicted_shannon_entropy": float(predicted_shannon_entropies[i]),
                "predicted_repetitiveness": float(predicted_repetitiveness_scores[i]),
            }
        )
    save_json({"examples": examples}, os.path.join(output_dir, "example_reconstructions.json"))
    logger.info(f"Saved {num_examples} example reconstructions")

    # Save numerical results
    results = {
        "run_dir": run_dir,
        "output_dir": output_dir,
        "seq_length": seq_length,
        "inversion_model": inversion_model,
        "foundation_model": foundation_model,
        "accuracy_metrics": accuracy_metrics,
        "baseline_accuracy_metrics": baseline_accuracy,
        "improvement_over_baseline": improvement,
        "nucleotide_frequencies": freq_comparison,
        "num_sequences": len(true_sequences),
        "levenshtein_mean": float(np.mean(levenshtein_similarities)),
        "levenshtein_std": float(np.std(levenshtein_similarities)),
        "baseline_levenshtein_mean": float(np.mean(baseline_levenshtein)),
        "baseline_levenshtein_std": float(np.std(baseline_levenshtein)),
        "accuracy_mean": float(accuracy_mean),
        "accuracy_std": float(accuracy_std),
        "total_inference_time": total_time,
        "time_per_sequence": time_per_sequence,
        "shannon_entropy_mean": float(np.mean(shannon_entropies)),
        "repetitiveness_mean": float(np.mean(repetitiveness_scores)),
        "predicted_shannon_entropy_mean": float(np.mean(predicted_shannon_entropies)),
        "predicted_repetitiveness_mean": float(np.mean(predicted_repetitiveness_scores)),
    }
    save_json(results, os.path.join(output_dir, "evaluation_results.json"))

    # Save plot data for regeneration
    plot_data = {
        "freq_comparison": freq_comparison,
        "accuracy_metrics": accuracy_metrics,
        "levenshtein_similarities": levenshtein_similarities,
        "baseline_levenshtein": baseline_levenshtein,
        "model_position_accuracy": model_position_accuracy,
        "baseline_position_accuracy": baseline_position_accuracy,
        "confusion_matrix": confusion_matrix,
        "shannon_entropies": shannon_entropies,
        "repetitiveness_scores": repetitiveness_scores,
        "predicted_shannon_entropies": predicted_shannon_entropies,
        "predicted_repetitiveness_scores": predicted_repetitiveness_scores,
    }
    torch.save(plot_data, os.path.join(output_dir, "plot_data.pt"))

    # Create plots
    logger.info("Creating plots...")

    model_color = get_fm_color(display_model_name)

    plot_nucleotide_frequencies(
        freq_comparison,
        os.path.join(output_dir, "nucleotide_frequencies"),
        "Nucleotide Frequency Comparison (Test Set)",
        model_color=model_color,
    )
    logger.info("Saved nucleotide frequency plot")

    plot_accuracy_metrics(
        accuracy_metrics,
        os.path.join(output_dir, "accuracy_metrics"),
        model_color=model_color,
    )
    logger.info("Saved accuracy metrics plot")

    plot_per_position_accuracy(
        model_position_accuracy,
        baseline_position_accuracy,
        os.path.join(output_dir, "per_position_accuracy"),
        model_color=model_color,
    )
    logger.info("Saved per-position accuracy plot")

    plot_nucleotide_confusion_matrix(
        confusion_matrix,
        NUCLEOTIDES,
        os.path.join(output_dir, "confusion_matrix"),
        "Nucleotide Confusion Matrix (Model Predictions)",
        cmap=sns.light_palette(model_color, as_cmap=True),
    )
    logger.info("Saved nucleotide confusion matrix plot")

    plot_levenshtein_comparison(
        levenshtein_similarities,
        baseline_levenshtein,
        os.path.join(output_dir, "levenshtein_comparison"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein similarity comparison plot")

    # New metrics plots
    from src.plotting_utils import plot_metric_vs_similarity

    plot_metric_vs_similarity(
        shannon_entropies,
        levenshtein_similarities,
        "Shannon Entropy (Bits) [True]",
        os.path.join(output_dir, "levenshtein_vs_entropy_true"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein vs Entropy (True) plot")

    plot_metric_vs_similarity(
        predicted_shannon_entropies,
        levenshtein_similarities,
        "Shannon Entropy (Bits) [Predicted]",
        os.path.join(output_dir, "levenshtein_vs_entropy_predicted"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein vs Entropy (Predicted) plot")

    plot_metric_vs_similarity(
        repetitiveness_scores,
        levenshtein_similarities,
        "4-mer Redundancy [True]",
        os.path.join(output_dir, "levenshtein_vs_repetitiveness_true"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein vs Repetitiveness (True) plot")

    plot_metric_vs_similarity(
        predicted_repetitiveness_scores,
        levenshtein_similarities,
        "4-mer Redundancy [Predicted]",
        os.path.join(output_dir, "levenshtein_vs_repetitiveness_predicted"),
        model_color=model_color,
    )
    logger.info("Saved Levenshtein vs Repetitiveness (Predicted) plot")

    return results


def regenerate_plots_single_run(
    run_dir: str,
    output_dir: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """Regenerate plots for a single run using saved data.

    Parameters
    ----------
    run_dir : str
        Path to the training run directory.
    output_dir : str
        Directory to save evaluation results.
    logger : logging.Logger
        Logger instance.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing evaluation results (loaded from json).
    """
    import json

    logger.info(f"Regenerating plots for run: {run_dir}")

    plot_data_path = os.path.join(run_dir, "plot_data.pt")
    results_path = os.path.join(run_dir, "evaluation_results.json")

    assert os.path.exists(plot_data_path), f"Plot data not found at {plot_data_path}"
    assert os.path.exists(results_path), f"Results file not found at {results_path}"

    # Load data
    plot_data = torch.load(plot_data_path, weights_only=False)

    freq_comparison = plot_data["freq_comparison"]
    accuracy_metrics = plot_data["accuracy_metrics"]
    levenshtein_similarities = plot_data["levenshtein_similarities"]
    baseline_levenshtein = plot_data.get("baseline_levenshtein")
    model_position_accuracy = plot_data["model_position_accuracy"]
    baseline_position_accuracy = plot_data["baseline_position_accuracy"]
    confusion_matrix = plot_data["confusion_matrix"]

    with open(results_path, "r") as f:
        res = json.load(f)
        inversion_model = res["inversion_model"]
        foundation_model = res["foundation_model"]

    # Construct display model name
    if foundation_model != "unknown" and foundation_model not in inversion_model:
        model_name = f"{inversion_model} ({foundation_model})"
    else:
        model_name = inversion_model

    # Create plots
    logger.info("Creating plots...")

    model_color = get_fm_color(model_name)

    plot_nucleotide_frequencies(
        freq_comparison,
        os.path.join(output_dir, "nucleotide_frequencies"),
        "Nucleotide Frequency Comparison (Test Set)",
        model_color=model_color,
    )
    logger.info("Saved nucleotide frequency plot")

    plot_accuracy_metrics(
        accuracy_metrics,
        os.path.join(output_dir, "accuracy_metrics"),
        model_color=model_color,
    )
    logger.info("Saved accuracy metrics plot")

    plot_per_position_accuracy(
        model_position_accuracy,
        baseline_position_accuracy,
        os.path.join(output_dir, "per_position_accuracy"),
        model_color=model_color,
    )
    logger.info("Saved per-position accuracy plot")

    plot_nucleotide_confusion_matrix(
        confusion_matrix,
        NUCLEOTIDES,
        os.path.join(output_dir, "confusion_matrix"),
        "Nucleotide Confusion Matrix (Model Predictions)",
        cmap=sns.light_palette(model_color, as_cmap=True),
    )
    logger.info("Saved nucleotide confusion matrix plot")

    if baseline_levenshtein is not None:
        plot_levenshtein_comparison(
            levenshtein_similarities,
            baseline_levenshtein,
            os.path.join(output_dir, "levenshtein_comparison"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein similarity comparison plot")

    # Regenerate new metric plots if data exists
    if "shannon_entropies" in plot_data and "repetitiveness_scores" in plot_data:
        from src.plotting_utils import plot_metric_vs_similarity

        shannon_entropies = plot_data["shannon_entropies"]
        repetitiveness_scores = plot_data["repetitiveness_scores"]

        plot_metric_vs_similarity(
            shannon_entropies,
            levenshtein_similarities,
            "Shannon Entropy (Bits) [True]",
            os.path.join(output_dir, "levenshtein_vs_entropy_true"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein vs Entropy (True) plot")

        plot_metric_vs_similarity(
            repetitiveness_scores,
            levenshtein_similarities,
            "4-mer Redundancy [True]",
            os.path.join(output_dir, "levenshtein_vs_repetitiveness_true"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein vs Repetitiveness (True) plot")

    if (
        "predicted_shannon_entropies" in plot_data
        and "predicted_repetitiveness_scores" in plot_data
    ):
        # Load if not already loaded (though it should be fine to re-import or use existing)
        from src.plotting_utils import plot_metric_vs_similarity

        predicted_shannon_entropies = plot_data["predicted_shannon_entropies"]
        predicted_repetitiveness_scores = plot_data["predicted_repetitiveness_scores"]

        plot_metric_vs_similarity(
            predicted_shannon_entropies,
            levenshtein_similarities,
            "Shannon Entropy (Bits) [Predicted]",
            os.path.join(output_dir, "levenshtein_vs_entropy_predicted"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein vs Entropy (Predicted) plot")

        plot_metric_vs_similarity(
            predicted_repetitiveness_scores,
            levenshtein_similarities,
            "4-mer Redundancy [Predicted]",
            os.path.join(output_dir, "levenshtein_vs_repetitiveness_predicted"),
            model_color=model_color,
        )
        logger.info("Saved Levenshtein vs Repetitiveness (Predicted) plot")

    # Add output_dir for aggregate function to find plot_data.pt
    res["output_dir"] = run_dir
    return res


@hydra.main(config_path="conf", config_name="evaluate", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute evaluation pipeline for trained DNA sequence reconstruction model.

    Supports both single-run and multi-run (Hydra multirun) evaluation modes.
    - Single run: evaluates one model and saves results
    - Multi-run: detects Hydra multirun directory, evaluates each sub-model,
      then creates aggregate comparison plots

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration for evaluation.
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting evaluation pipeline")
    logger.info("Config:\n" + OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.device)

    # Find the run directory to evaluate
    if cfg.run_dir is None:
        # Resolve runs_base_dir relative to original CWD
        runs_base_dir = hydra.utils.to_absolute_path(cfg.runs_base_dir)
        run_dir = find_latest_run_dir(runs_base_dir)
        logger.info(f"Auto-detected latest run: {run_dir}")
    else:
        # Resolve specified run_dir relative to original CWD
        run_dir = hydra.utils.to_absolute_path(cfg.run_dir)
        logger.info(f"Using specified run: {run_dir}")

    assert os.path.exists(run_dir), f"Run directory not found: {run_dir}"

    # Get Hydra output directory
    hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir  # type: ignore[attr-defined]

    # Check if this is a multirun directory
    if cfg.only_plots:
        logger.info("=" * 80)
        logger.info("ONLY PLOTS MODE - Regenerating plots from existing evaluation results")

        # Check for multirun evaluation structure (run_0, run_1)
        eval_subdirs = [
            d
            for d in os.listdir(run_dir)
            if os.path.isdir(os.path.join(run_dir, d))
            and d.startswith("run_")
            and d.split("_")[1].isdigit()
        ]

        if len(eval_subdirs) > 0:
            logger.info(f"Detected {len(eval_subdirs)} multirun subdirectories")
            # Sort by job number
            eval_subdirs.sort(key=lambda x: int(x.split("_")[1]))

            results_list = []
            for subdir_name in eval_subdirs:
                subdir = os.path.join(run_dir, subdir_name)

                if cfg.aggregate_only:
                    # Load results directly from JSON without regenerating plots
                    import json

                    results_path = os.path.join(subdir, "evaluation_results.json")
                    assert os.path.exists(results_path), f"Results file not found: {results_path}"
                    with open(results_path, "r") as f:
                        result = json.load(f)
                    result["output_dir"] = subdir
                    results_list.append(result)
                else:
                    logger.info(f"Re-plotting run {subdir_name}")

                    run_output_dir = os.path.join(hydra_output_dir, subdir_name)
                    os.makedirs(run_output_dir, exist_ok=True)

                    result = regenerate_plots_single_run(subdir, run_output_dir, logger)
                    if result:
                        results_list.append(result)

            # Create aggregate plots
            if results_list:
                logger.info("Creating aggregate comparison plots")
                aggregate_and_plot_multirun(results_list, hydra_output_dir, logger)
            else:
                logger.warning("No results found to aggregate")

        else:
            # Single run evaluation
            logger.info("Single run evaluation detected")
            regenerate_plots_single_run(run_dir, hydra_output_dir, logger)

        logger.info(f"Results saved to: {hydra_output_dir}")

    elif is_multirun_directory(run_dir):
        logger.info("=" * 80)
        logger.info("MULTIRUN DETECTED - Evaluating multiple models")

        # Get all subdirectories
        subdirs = get_multirun_subdirs(run_dir)
        logger.info(f"Found {len(subdirs)} runs to evaluate")

        # Evaluate each run
        results_list = []
        for i, subdir in enumerate(subdirs):
            job_num = os.path.basename(subdir)
            logger.info(f"Evaluating run {i+1}/{len(subdirs)} (job {job_num})")

            # Create output directory for this specific run
            run_output_dir = os.path.join(hydra_output_dir, f"run_{job_num}")

            # Evaluate this run
            result = evaluate_single_run(subdir, device, run_output_dir, logger, cfg.max_samples)
            results_list.append(result)

            logger.info(f"Completed evaluation for run {job_num}")

        # Create aggregate plots
        logger.info("Creating aggregate comparison plots")
        aggregate_and_plot_multirun(results_list, hydra_output_dir, logger)

        logger.info("MULTIRUN EVALUATION COMPLETE")
        logger.info(f"Results saved to: {hydra_output_dir}")

    else:
        # Single run mode - original behavior
        logger.info("=" * 80)
        logger.info("SINGLE RUN MODE")
        logger.info("=" * 80)

        evaluate_single_run(run_dir, device, hydra_output_dir, logger, cfg.max_samples)

        logger.info(f"\n{'='*80}")
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Results saved to: {hydra_output_dir}")
        logger.info(f"{'='*80}")


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore
