"""
Embedding Analysis - Visualize token distributions in embedding space

This script performs dimensionality reduction using UMAP and analyzes token embeddings.
It supports analyzing a single dataset or comparing multiple datasets.
It handles different tokenization strategies (Char-level, BPE, k-mer) by inferring
the tokenizer from the dataset name or config.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Any, Optional, Union

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig, OmegaConf
from umap import UMAP
from scipy.stats import gaussian_kde
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import hydra.utils as hy_utils
from hydra.core.hydra_config import HydraConfig
from transformers import AutoTokenizer

# Add project root to path to allow imports from src
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data import load_split_embeddings
from src.utils import save_json, configure_plot_style, get_series_colors, PLOT_STYLE

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


def get_tokenizer(dataset_name: str, data_cfg: DictConfig = None) -> Optional[Any]:
    """Infer and load tokenizer based on dataset name or config file paths."""

    # Helper to check string in various signals
    def detects(pattern: str) -> bool:
        if pattern in dataset_name:
            return True
        if data_cfg is not None:
            # Check paths in data config
            for key in ["train_csv", "val_csv", "test_csv"]:
                val = data_cfg.get(key)
                if val and pattern in str(val):
                    return True
        return False

    if detects("dnabert2"):
        logger.info(
            "Detected DNABERT-2 dataset. Loading tokenizer from 'zhihan1996/DNABERT-2-117M'..."
        )
        return AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    elif detects("ntv2") or detects("nucleotide-transformer"):
        logger.info(
            "Detected Nucleotide Transformer v2 dataset. Loading tokenizer from 'InstaDeepAI/nucleotide-transformer-v2-500m-multi-species'..."
        )
        return AutoTokenizer.from_pretrained(
            "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", trust_remote_code=True
        )

    elif detects("evo2"):
        logger.info("Detected Evo2 dataset. Assuming character-level tokenization for analysis.")
        return None  # Evo2 is char level

    logger.warning("Failed to infer tokenizer. Falling back to character-level.")
    return None


def tokenize_sequence(
    tokenizer: Any, sequence: str, dataset_name: str, data_cfg: DictConfig = None
) -> List[str]:
    """Tokenize sequence into a list of token strings, aligned with embeddings."""
    if tokenizer is None:
        return list(sequence)

    # Helper to check type
    def detects(pattern: str) -> bool:
        if pattern in dataset_name:
            return True
        if data_cfg is not None:
            for key in ["train_csv", "val_csv", "test_csv"]:
                val = data_cfg.get(key)
                if val and pattern in str(val):
                    return True
        return False

    # Tokenize without special tokens to match embedding generation
    if detects("dnabert2"):
        encoding = tokenizer(sequence, add_special_tokens=False)
        input_ids = encoding["input_ids"]
        return tokenizer.convert_ids_to_tokens(input_ids)

    if detects("ntv2") or detects("nucleotide-transformer"):
        encoding = tokenizer(sequence, add_special_tokens=False)
        input_ids = encoding["input_ids"]
        return tokenizer.convert_ids_to_tokens(input_ids)

    # Default fallback
    return tokenizer.tokenize(sequence)


def get_dataset_palette(dataset_name: str, n_colors: int) -> List[str]:
    """Get color palette based on dataset name."""
    if "dnabert" in dataset_name:
        return sns.color_palette("Greens_r", n_colors).as_hex()
    elif "ntv2" in dataset_name or "nucleotide" in dataset_name:
        return sns.color_palette("Oranges_r", n_colors).as_hex()
    elif "evo" in dataset_name:
        return sns.color_palette("Blues_r", n_colors).as_hex()
    else:
        return sns.color_palette("Purples_r", n_colors).as_hex()


def plot_token_scatter(
    embeddings_2d: np.ndarray,
    tokens: List[str],
    unique_tokens: List[str],
    method: str,
    output_path: str,
    token_colors: Dict[str, str] = None,
    dim_labels: Tuple[str, str] = ("Dimension 1", "Dimension 2"),
) -> None:
    """Plot 2D scatter plot of embeddings colored by token."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # If too many unique tokens, pick top N most frequent?
    tokens_to_plot = unique_tokens
    if len(unique_tokens) > 20:
        logger.info(
            f"Too many unique tokens ({len(unique_tokens)}). Plotting top 20 most frequent only."
        )
        from collections import Counter

        counts = Counter(tokens)
        tokens_to_plot = [t for t, _ in counts.most_common(20)]

    # Create a stable color mapping for tokens if not provided
    if token_colors is None:
        token_colors = get_series_colors(tokens_to_plot)

    # Plot each token class
    for token in tokens_to_plot:
        mask = np.array([t == token for t in tokens])
        color = token_colors.get(token, "#333333")
        ax.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[color],
            label=token,
            alpha=0.6,
            s=5,  # Increased size slightly for visibility
            edgecolors="none",
        )

    ax.set_xlabel(f"{method.upper()} {dim_labels[0]}", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel(f"{method.upper()} {dim_labels[1]}", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_title(
        f"Token Embeddings: {dim_labels[0]} vs {dim_labels[1]}",
        fontsize=PLOT_STYLE["title_fontsize"],
    )
    ax.legend(
        markerscale=5,
        loc="best",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0.0,
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


def compute_token_means(
    embeddings: np.ndarray,
    tokens: List[str],
    unique_tokens: List[str],
) -> Dict[str, np.ndarray]:
    """Compute mean embedding for each token."""
    token_means = {}
    tokens_arr = np.array(tokens)  # Optimize masking
    for token in unique_tokens:
        mask = tokens_arr == token
        if not np.any(mask):
            continue
        token_embeddings = embeddings[mask]
        token_means[token] = np.mean(token_embeddings, axis=0)
    return token_means


def plot_token_distributions(
    token_means_dict: Dict[str, np.ndarray],
    unique_tokens: List[str],
    output_path: str,
    token_colors: Dict[str, str] = None,
    title_suffix: str = "",
) -> None:
    """Plot KDE distributions of token mean embeddings."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Limit to top tokens if too many
    tokens_to_plot = unique_tokens

    if token_colors is None:
        token_colors = get_series_colors(tokens_to_plot)

    plotted_count = 0
    for token in tokens_to_plot:
        mean_embedding = token_means_dict.get(token)
        if mean_embedding is None:
            continue

        try:
            kde = gaussian_kde(mean_embedding)
            x_min, x_max = mean_embedding.min(), mean_embedding.max()
            x_range = x_max - x_min
            if x_range == 0:
                continue
            x_eval = np.linspace(x_min - 0.1 * x_range, x_max + 0.1 * x_range, 1000)
            density = kde(x_eval)

            color = token_colors.get(token, "#333333")
            ax.plot(x_eval, density, label=token, color=color, linewidth=2)
            ax.fill_between(x_eval, density, alpha=0.3, color=color)
            plotted_count += 1
            if plotted_count >= 15:  # Do not clutter plot too much
                break
        except Exception as e:
            logger.warning(f"Could not plot KDE for token {token}: {e}")

    ax.set_xlabel("Embedding Value", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Density", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_title(
        f"Distribution of Token Mean Embeddings {title_suffix}",
        fontsize=PLOT_STYLE["title_fontsize"],
    )
    # Only show legend if reasonable
    if plotted_count <= 20:
        ax.legend(loc="best", bbox_to_anchor=(1.05, 1), fontsize=PLOT_STYLE["legend_fontsize"])
    ax.grid(True, alpha=0.3)

    try:
        plt.tight_layout()
    except Exception as e:
        logger.warning(f"tight_layout failed (likely font issue): {e}")
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.pdf")
    plt.close()


def plot_token_pca(
    token_means_dict: Dict[str, np.ndarray],
    unique_tokens: List[str],
    output_path: str,
    title_suffix: str = "",
) -> None:
    """Plot PCA of token mean embeddings."""
    if len(token_means_dict) < 3:
        return

    # Prepare data
    tokens_present = [t for t in unique_tokens if t in token_means_dict]
    data_list = [token_means_dict[t] for t in tokens_present]
    X = np.array(data_list)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], alpha=0.9, s=100, ax=ax)

    # Annotate points
    for i, token in enumerate(tokens_present):
        ax.annotate(
            token,
            (X_pca[i, 0], X_pca[i, 1]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=PLOT_STYLE["annotation_fontsize"],
            alpha=0.8,
        )

    ax.set_xlabel("PC1", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("PC2", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_title(
        f"PCA of Token Embeddings {title_suffix}\nExplained Var: {pca.explained_variance_ratio_}",
        fontsize=PLOT_STYLE["title_fontsize"],
    )
    try:
        plt.tight_layout()
    except Exception as e:
        logger.warning(f"tight_layout failed (likely font issue): {e}")
    plt.savefig(os.path.join(output_path, "pca_tokens.png"))
    plt.savefig(os.path.join(output_path, "pca_tokens.pdf"))
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

    if data_cfg.get("mean", False):
        logger.warning(f"Dataset {dataset_name} is mean-pooled. Skipping per-token analysis.")
        return {}

    # Infer tokenizer using both name and config content
    tokenizer = get_tokenizer(dataset_name, data_cfg)
    if tokenizer is False:
        logger.error(f"Skipping dataset {dataset_name} due to tokenizer load failure.")
        return {}

    logger.info("Loading embeddings...")
    h5_files_dict, _, _ = load_split_embeddings(data_cfg)

    splits_to_analyze = (
        analysis_cfg.splits if analysis_cfg.get("splits") else ["train", "val", "test"]
    )

    all_tokens = []
    all_embeddings = []

    for split in splits_to_analyze:
        if split not in h5_files_dict:
            continue

        h5_file = h5_files_dict[split]
        sequences_dataset = h5_file["sequences"]
        embeddings_dataset = h5_file["embeddings"]

        logger.info(f"Processing {split} split: {len(sequences_dataset)} sequences")

        for idx in range(len(sequences_dataset)):
            seq_bytes = sequences_dataset[idx]
            seq = seq_bytes.decode("utf-8") if isinstance(seq_bytes, bytes) else str(seq_bytes)

            emb_flat = np.asarray(embeddings_dataset[idx], dtype=np.float32)
            embedding_dim = data_cfg.embedding_dim
            seq_length = len(emb_flat) // embedding_dim
            emb = emb_flat.reshape(seq_length, embedding_dim)

            # Tokenize
            seq_tokens = tokenize_sequence(tokenizer, seq, dataset_name, data_cfg)

            # Alignment check
            if len(seq_tokens) != emb.shape[0]:
                logger.error(
                    f"Tokens/Embeddings mismatch at idx {idx}: Tokens {len(seq_tokens)} vs Emb {emb.shape[0]}."
                )
                logger.error(f"Sequence: {seq}")
                logger.error(f"Tokens: {seq_tokens}")
                if abs(len(seq_tokens) - emb.shape[0]) > 0:
                    raise ValueError(
                        f"Mismatch between number of tokens ({len(seq_tokens)}) and embeddings ({emb.shape[0]}) for sequence {idx}."
                    )

            for token, token_emb in zip(seq_tokens, emb):
                all_tokens.append(token)
                all_embeddings.append(token_emb)

    if not all_embeddings:
        logger.warning(f"No embeddings found for {dataset_name}")
        return {}

    all_embeddings = np.array(all_embeddings)
    unique_tokens = sorted(set(all_tokens))
    logger.info(f"Collected {len(all_tokens)} tokens. Unique count: {len(unique_tokens)}")

    if len(unique_tokens) > 50:
        logger.info(f"First 50 unique tokens: {unique_tokens[:50]}")

    logger.info(f"Embedding shape: {all_embeddings.shape}")

    # UMAP
    logger.info("Performing UMAP...")
    umap_reducer = UMAP(
        n_components=3,
        n_neighbors=analysis_cfg.n_neighbors,
        min_dist=analysis_cfg.min_dist,
        n_jobs=-1,
    )

    # Subsample for UMAP
    umap_limit = analysis_cfg.subsample
    if len(all_embeddings) > umap_limit:
        logger.info(f"Subsampling for UMAP fit ({umap_limit})...")
        rng = np.random.RandomState(analysis_cfg.seed)
        fit_indices = rng.choice(len(all_embeddings), size=umap_limit, replace=False)
        umap_embeddings_fit = all_embeddings[fit_indices]
        umap_tokens_fit = [all_tokens[i] for i in fit_indices]
    else:
        umap_embeddings_fit = all_embeddings
        umap_tokens_fit = all_tokens

    umap_embeddings_3d = umap_reducer.fit_transform(umap_embeddings_fit)

    # Determine Palette
    palette_colors = get_dataset_palette(dataset_name, len(unique_tokens))
    # Map tokens to colors
    token_colors = {
        token: palette_colors[i % len(palette_colors)] for i, token in enumerate(unique_tokens)
    }

    # Plots
    plot_token_scatter(
        umap_embeddings_3d[:, [0, 1]],
        umap_tokens_fit,
        unique_tokens,
        "umap",
        os.path.join(output_dir, "umap_scatter_dim1_dim2"),
        token_colors=token_colors,
    )

    # Silhouette
    logger.info("Computing silhouette scores...")
    token_to_idx = {token: idx for idx, token in enumerate(unique_tokens)}
    token_labels = np.array([token_to_idx.get(t, -1) for t in umap_tokens_fit])

    valid_mask = token_labels != -1
    sil_emb_all = umap_embeddings_3d[valid_mask]
    sil_labels_all = token_labels[valid_mask]

    subsample = analysis_cfg.subsample
    if len(sil_labels_all) > subsample:
        rng = np.random.RandomState(analysis_cfg.seed)
        sil_indices = rng.choice(len(sil_labels_all), size=subsample, replace=False)
        sil_emb = sil_emb_all[sil_indices]
        sil_labels = sil_labels_all[sil_indices]
    else:
        sil_emb = sil_emb_all
        sil_labels = sil_labels_all

    if len(set(sil_labels)) > 1:
        sil_score = silhouette_score(sil_emb, sil_labels)
        logger.info(f"Silhouette Score (UMAP 3D): {sil_score}")
    else:
        sil_score = 0.0
        logger.warning("Silhouette score undefined: only 1 cluster found in subsample.")

    # Token Mean Distributions
    logger.info("Computing token means...")
    token_means = compute_token_means(all_embeddings, all_tokens, unique_tokens)

    from collections import Counter

    counts = Counter(all_tokens)

    # Plot PCA of token means (Top 100)
    top_tokens_pca = [t for t, _ in counts.most_common(100)]
    plot_token_pca(token_means, top_tokens_pca, output_dir, title_suffix=f"({dataset_name})")

    # Plot top frequent tokens only in distribution (Top 15)
    top_tokens_dist = [t for t, _ in counts.most_common(15)]

    plot_token_distributions(
        token_means,
        top_tokens_dist,
        os.path.join(output_dir, "token_mean_distributions"),
        token_colors=token_colors,
        title_suffix=f"({dataset_name})",
    )

    # Compute pairwise distances between token means
    logger.info("Computing token pairwise distances...")
    distance_stats = {}
    if len(unique_tokens) > 1:
        # Align means with unique_tokens order
        ordered_means = np.array([token_means[t] for t in unique_tokens])

        # Compute pairwise Euclidean distances
        try:
            # pdist returns condensed distance matrix
            distances_condensed = pdist(ordered_means, metric="euclidean")

            # Statistics
            distance_stats["min_distance"] = float(np.min(distances_condensed))
            distance_stats["max_distance"] = float(np.max(distances_condensed))
            distance_stats["mean_distance"] = float(np.mean(distances_condensed))
            distance_stats["std_distance"] = float(np.std(distances_condensed))

            # Full distance matrix (skip if vocabulary too large)
            if len(unique_tokens) <= 2000:
                distance_matrix = squareform(distances_condensed)
                distance_stats["distance_matrix_tokens"] = unique_tokens
                distance_stats["distance_matrix"] = distance_matrix.tolist()
            else:
                logger.warning(
                    f"Too many unique tokens ({len(unique_tokens)}) to save full distance matrix in JSON."
                )
                distance_stats["distance_matrix_note"] = "Skipped due to size (>2000 tokens)"

        except Exception as e:
            logger.error(f"Failed to compute distances: {e}")

    stats = {
        "dataset": dataset_name,
        "n_tokens": len(all_tokens),
        "n_unique_tokens": len(unique_tokens),
        "silhouette_score": float(sil_score),
        "embedding_dim": int(all_embeddings.shape[1]),
        **distance_stats,
    }

    save_json(stats, os.path.join(output_dir, "stats.json"))
    return stats


def compare_results(results: Dict[str, Any], output_path: str):
    """Generate comparison plots across datasets."""
    logger.info("Generating comparison plots...")

    # 1. Silhouette Score Comparison
    datasets = list(results.keys())
    scores = [results[d]["silhouette_score"] for d in datasets]

    plt.figure(figsize=(10, 6))
    dataset_colors = get_series_colors(datasets)
    bar_colors = [dataset_colors[d] for d in datasets]
    bars = plt.bar(datasets, scores, color=bar_colors)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=PLOT_STYLE["annotation_fontsize"],
        )

    plt.title("Silhouette Score Comparison", fontsize=PLOT_STYLE["title_fontsize"])
    plt.ylabel("Silhouette Score", fontsize=PLOT_STYLE["label_fontsize"])
    plt.xticks(rotation=45, fontsize=PLOT_STYLE["tick_fontsize"])
    try:
        plt.tight_layout()
    except Exception as e:
        logger.warning(f"tight_layout failed (likely font issue): {e}")
    plt.savefig(os.path.join(output_path, "comparison_silhouette.png"))
    plt.savefig(os.path.join(output_path, "comparison_silhouette.pdf"))
    plt.close()


@hydra.main(config_path="../conf", config_name="embedding_analysis", version_base=None)
def main(cfg: DictConfig) -> None:
    """Execute embedding analysis pipeline."""
    logger.info("Starting embedding analysis")

    results = {}

    # 1. Analyze default data
    try:
        # Save main config results one level up (in the root of the run folder)
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
        # Propagate the error!
        logger.error(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        # Do not continue if analysis fails
        raise e


if __name__ == "__main__":
    main()
