"""
Mean Embedding Analysis - Visualize embedding space and similarity metrics

This script performs analysis on mean-pooled embeddings.
It includes:
- Dimensionality reduction (UMAP/PCA)
- Embedding similarity vs Sequence similarity plots
- Pairwise distance statistics
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig
from umap import UMAP
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
import pandas as pd
import Levenshtein
from hydra.core.hydra_config import HydraConfig

# Add project root to path to allow imports from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_split_embeddings
from src.utils import save_json
from src.evaluate import compute_shannon_entropy, compute_repetitiveness
from src.plotting_utils import (
    configure_plot_style,
    get_series_colors,
    get_fm_color,
    PLOT_STYLE,
    plot_distribution,
)

# Use Agg backend for non-interactive plotting
matplotlib.use("Agg")

configure_plot_style()
# Enable font fallback to prevent crashes on systems without preferred fonts
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = [
    "DejaVu Sans",
    "Liberation Sans",
    "FreeSans",
    "Bitstream Vera Sans",
    "sans-serif",
]
matplotlib.rcParams["text.usetex"] = False

logger = logging.getLogger(__name__)


def plot_similarity_scatter(
    results_df: pd.DataFrame,
    output_path: str,
    x_col: str = "seq_sim",
    y_col: str = "emb_sim",
    y_label: str = "Embedding Similarity",
    title_suffix: str = "",
    color: str = None,
) -> None:
    """Plot scatter of embedding similarity vs sequence similarity."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use hexbin for density visualization if many points
    if len(results_df) > 1000:
        # Create a colormap from the base color
        if color:
            cmap = sns.light_palette(color, as_cmap=True)
        else:
            cmap = "Blues"

        hb = ax.hexbin(
            results_df[x_col], results_df[y_col], gridsize=50, cmap=cmap, mincnt=1, bins="log"
        )
        cb = fig.colorbar(hb, ax=ax, label="Log Count")
    else:
        sns.scatterplot(data=results_df, x=x_col, y=y_col, alpha=0.5, ax=ax, color=color)

    # Add 1:1 line for reference if applicable (optional, usually they are not 1:1)
    # plt.plot([0, 1], [0, 1], 'r--', alpha=0.3)

    ax.set_xlabel(
        "Sequence Similarity (1 - Norm. Levenshtein)", fontsize=PLOT_STYLE["label_fontsize"]
    )
    ax.set_ylabel(y_label, fontsize=PLOT_STYLE["label_fontsize"])

    # Calculate Spearman correlation
    corr = results_df[[x_col, y_col]].corr(method="spearman").iloc[0, 1]
    ax.text(
        0.05,
        0.95,
        f"Spearman Corr: {corr:.3f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=PLOT_STYLE["legend_fontsize"],
    )

    ax.grid(True, alpha=0.3)

    try:
        plt.tight_layout()
    except Exception as e:
        logger.warning(f"tight_layout failed (likely font issue): {e}")
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.pdf")
    plt.close()


def analyze_dataset(
    data_cfg: DictConfig,
    dataset_name: str,
    output_dir: str,
    analysis_cfg: DictConfig,
) -> Dict[str, Any]:
    """Execute analysis for a single dataset."""
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Analyzing dataset: {dataset_name}")

    if not data_cfg.get("mean", False):
        logger.warning(
            f"Dataset {dataset_name} is NOT configured as mean-pooled in data config. Proceeding anyway, interpreting as mean embeddings if possible."
        )

    logger.info("Loading embeddings...")
    # Load data - utilizing existing src.data functionality
    try:
        h5_files_dict, _, _ = load_split_embeddings(data_cfg)
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        return {}

    splits_to_analyze = analysis_cfg.splits if analysis_cfg.get("splits") else ["test"]
    all_sequences = []
    all_embeddings = []

    for split in splits_to_analyze:
        if split not in h5_files_dict:
            continue

        h5_file = h5_files_dict[split]
        sequences_dataset = h5_file["sequences"]
        embeddings_dataset = h5_file["embeddings"]

        logger.info(f"Processing {split} split: {len(sequences_dataset)} sequences")

        n_samples = len(sequences_dataset)
        indices = np.arange(n_samples)

        # Subsample if needed
        subsample_limit = analysis_cfg.subsample
        if n_samples > subsample_limit:
            logger.info(f"Subsampling {subsample_limit} from {split} for analysis.")
            rng = np.random.RandomState(analysis_cfg.get("seed", 42))
            indices = rng.choice(n_samples, size=subsample_limit, replace=False)
            indices.sort()

        for idx in indices:
            seq_bytes = sequences_dataset[idx]
            seq = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)

            emb = np.asarray(embeddings_dataset[idx], dtype=np.float32)

            all_sequences.append(seq)
            all_embeddings.append(emb)

    if not all_embeddings:
        logger.warning(f"No embeddings found for {dataset_name}")
        return {}

    all_embeddings = np.array(all_embeddings)
    logger.info(f"Total sequences collected: {len(all_sequences)}")
    logger.info(f"Embedding shape: {all_embeddings.shape}")

    # Subsample for pairwise comparison (N^2 complexity)
    pairwise_limit = analysis_cfg.subsample
    if len(all_sequences) > pairwise_limit:
        logger.info(f"Subsampling to {pairwise_limit} for pairwise similarity analysis.")
        rng = np.random.RandomState(analysis_cfg.get("seed", 42))
        sub_indices = rng.choice(len(all_sequences), size=pairwise_limit, replace=False)

        sub_embeddings = all_embeddings[sub_indices]
        sub_sequences = [all_sequences[i] for i in sub_indices]
    else:
        sub_embeddings = all_embeddings
        sub_sequences = all_sequences

    # Sequence Similarity (1 - Normalized Levenshtein)
    n_sub = len(sub_sequences)
    # We only need upper triangle
    pairs_indices = np.triu_indices(n_sub, k=1)

    # Extract pairs
    seqs_i = [sub_sequences[i] for i in pairs_indices[0]]
    seqs_j = [sub_sequences[j] for j in pairs_indices[1]]

    logger.info(f"Calculating Levenshtein distances for {len(seqs_i)} pairs...")

    # Levenshtein distance
    lev_dists = np.array([Levenshtein.distance(s1, s2) for s1, s2 in zip(seqs_i, seqs_j)])
    max_lens = np.array([max(len(s1), len(s2)) for s1, s2 in zip(seqs_i, seqs_j)])

    # Avoid div by zero
    max_lens[max_lens == 0] = 1

    norm_lev_dists = lev_dists / max_lens
    seq_similarities = 1.0 - norm_lev_dists

    # 2. Embedding Similarities
    # Use selected embeddings
    embs_i = sub_embeddings[pairs_indices[0]]
    embs_j = sub_embeddings[pairs_indices[1]]

    # Cosine Similarity
    from sklearn.metrics.pairwise import cosine_similarity

    logger.info("Computing Cosine Similarity...")
    cosine_sim_matrix = cosine_similarity(sub_embeddings)
    cosine_sims = cosine_sim_matrix[pairs_indices]

    # Normalized Euclidean Similarity
    logger.info("Computing Euclidean Distances...")
    euclidean_dist_matrix = pairwise_distances(sub_embeddings, metric="euclidean")
    euclidean_dists = euclidean_dist_matrix[pairs_indices]

    # Normalize: 1 - dist/max
    max_euc_dist = np.max(euclidean_dists)
    if max_euc_dist == 0:
        norm_euc_sims = np.ones_like(euclidean_dists)
    else:
        norm_euc_sims = 1.0 - (euclidean_dists / max_euc_dist)

    # Determine color for this dataset
    base_color = get_fm_color(dataset_name)
    logger.info(f"Using color {base_color} for dataset {dataset_name}")

    # Prepare DataFrame for plotting
    plot_df = pd.DataFrame(
        {"seq_sim": seq_similarities, "cosine_sim": cosine_sims, "euclidean_sim": norm_euc_sims}
    )

    # Plotting
    plot_similarity_scatter(
        plot_df,
        os.path.join(output_dir, "similarity_cosine"),
        x_col="seq_sim",
        y_col="cosine_sim",
        y_label="Cosine Similarity",
        title_suffix=f"({dataset_name})",
        color=base_color,
    )

    plot_similarity_scatter(
        plot_df,
        os.path.join(output_dir, "similarity_euclidean"),
        x_col="seq_sim",
        y_col="euclidean_sim",
        y_label="Norm. Euclidean Similarity (1 - d/max)",
        title_suffix=f"({dataset_name})",
        color=base_color,
    )

    # Plot distribution of normalized Euclidean distances (normalized by embedding dimension)
    embedding_dim = sub_embeddings.shape[1]
    norm_euc_dists = euclidean_dists / np.sqrt(embedding_dim)

    # Save NED data for cross-dataset merged plots
    np.save(os.path.join(output_dir, "norm_euclidean_dists.npy"), norm_euc_dists)

    plot_distribution(
        norm_euc_dists,
        os.path.join(output_dir, "euclidean_distance_distribution"),
        bins=50,
        kde=True,
        color=base_color,
        xlabel="Normalized Euclidean Distance (d / √dim)",
        ylabel="Count",
        xlim=(0, 0.5),
        ylim=(0, 250_000),
    )

    # UMAP visualization
    logger.info("Performing UMAP on mean embeddings...")
    umap_reducer = UMAP(
        n_components=2,
        n_neighbors=analysis_cfg.n_neighbors,
        min_dist=analysis_cfg.min_dist,
        n_jobs=-1,
    )

    # Subsample for UMAP
    umap_limit = analysis_cfg.subsample
    if len(all_embeddings) > umap_limit:
        rng = np.random.RandomState(analysis_cfg.seed)
        fit_idx = rng.choice(len(all_embeddings), size=umap_limit, replace=False)
        umap_embs = all_embeddings[fit_idx]
        umap_seqs = [all_sequences[i] for i in fit_idx]
    else:
        umap_embs = all_embeddings
        umap_seqs = all_sequences

    umap_2d = umap_reducer.fit_transform(umap_embs)

    # Plot UMAP
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=base_color, alpha=0.6, s=5)

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.savefig(os.path.join(output_dir, "umap_mean_embeddings.png"))
    plt.savefig(os.path.join(output_dir, "umap_mean_embeddings.pdf"))
    plt.close()

    # --- Sequence Complexity Analysis ---
    logger.info("Computing sequence complexity metrics...")
    entropies = np.array(compute_shannon_entropy(umap_seqs))
    repetitiveness = np.array(compute_repetitiveness(umap_seqs, k=4))

    # Distribution: Shannon Entropy
    plot_distribution(
        entropies,
        os.path.join(output_dir, "sequence_entropy_distribution"),
        bins=50,
        kde=True,
        color=base_color,
        xlabel="Shannon Entropy (bits)",
        ylabel="Count",
    )

    # Distribution: K-mer Repetitiveness (unique 4-mer ratio)
    plot_distribution(
        repetitiveness,
        os.path.join(output_dir, "kmer_repetitiveness_distribution"),
        bins=50,
        kde=True,
        color=base_color,
        xlabel="Unique 4-mer Ratio (lower = more repetitive)",
        ylabel="Count",
    )

    # Joint scatter: Shannon Entropy vs K-mer Uniqueness
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(entropies, repetitiveness, c=base_color, alpha=0.4, s=8)
    ax.set_xlabel("Shannon Entropy (bits)", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Unique 4-mer Ratio", fontsize=PLOT_STYLE["label_fontsize"])

    ax.grid(True, alpha=0.3)
    corr_er = pd.Series(entropies).corr(pd.Series(repetitiveness), method="spearman")
    ax.text(
        0.05,
        0.95,
        f"Spearman: {corr_er:.3f}",
        transform=ax.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        fontsize=PLOT_STYLE["legend_fontsize"],
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "entropy_vs_repetitiveness.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "entropy_vs_repetitiveness.pdf"), dpi=300)
    plt.close()

    # UMAP colored by Shannon Entropy
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        umap_2d[:, 0],
        umap_2d[:, 1],
        c=entropies,
        cmap="viridis",
        alpha=0.6,
        s=5,
    )
    fig.colorbar(sc, ax=ax, label="Shannon Entropy (bits)")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.savefig(os.path.join(output_dir, "umap_entropy.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "umap_entropy.pdf"), dpi=300)
    plt.close()

    # UMAP colored by repetitiveness
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        umap_2d[:, 0],
        umap_2d[:, 1],
        c=repetitiveness,
        cmap="viridis",
        alpha=0.6,
        s=5,
    )
    fig.colorbar(sc, ax=ax, label="Unique 4-mer Ratio")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    plt.savefig(os.path.join(output_dir, "umap_repetitiveness.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, "umap_repetitiveness.pdf"), dpi=300)
    plt.close()

    # Complexity statistics
    low_entropy_frac = float(np.mean(entropies < 1.5))
    low_kmer_frac = float(np.mean(repetitiveness < 0.3))
    logger.info(
        f"Complexity stats: {low_entropy_frac:.1%} sequences with entropy < 1.5 bits, "
        f"{low_kmer_frac:.1%} with unique 4-mer ratio < 0.3"
    )

    stats = {
        "dataset": dataset_name,
        "n_samples": len(all_sequences),
        "avg_seq_len": float(np.mean([len(s) for s in all_sequences])),
        "max_euclidean_dist": float(max_euc_dist) if "max_euc_dist" in locals() else 0.0,
        "embedding_dim": int(all_embeddings.shape[1]),
        "spearman_corr_cosine": float(
            plot_df["seq_sim"].corr(plot_df["cosine_sim"], method="spearman")
        ),
        "spearman_corr_euclidean": float(
            plot_df["seq_sim"].corr(plot_df["euclidean_sim"], method="spearman")
        ),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)),
        "entropy_min": float(np.min(entropies)),
        "entropy_max": float(np.max(entropies)),
        "repetitiveness_mean": float(np.mean(repetitiveness)),
        "repetitiveness_std": float(np.std(repetitiveness)),
        "frac_low_entropy_lt1.5": low_entropy_frac,
        "frac_low_kmer_lt0.3": low_kmer_frac,
    }

    save_json(stats, os.path.join(output_dir, "stats.json"))
    return stats


@hydra.main(config_path="../conf", config_name="embedding_analysis", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute embedding analysis pipeline."""
    logger.info("Starting mean embedding analysis")

    results = {}
    try:
        main_output_dir = os.getcwd()
        try:
            # We must detect the dataset choice to assign correct colors
            dataset_name = HydraConfig.get().runtime.choices.data
        except Exception as e:
            raise RuntimeError(f"Could not detect dataset name from HydraConfig: {e}")

        if dataset_name == "dataset":
            # This means fallback kicked in or hydra config was generic. We must fail.
            raise RuntimeError(
                f"Detailed dataset name not found (got '{dataset_name}'). Cannot assign correct colors."
            )

        results[dataset_name] = analyze_dataset(cfg.data, dataset_name, main_output_dir, cfg)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        raise e


if __name__ == "__main__":
    main()
