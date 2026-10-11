"""Tokenizer Analysis — Subword tokenizer behavior for DNABERT-2 and NTv2.

Evo2 uses a char-level tokenizer (1 token = 1 nucleotide), so it is excluded.
This script generates random DNA sequences and analyzes how the BPE (DNABERT-2)
and k-mer (NTv2) tokenizers segment them.

Produces two plots:
1. Token count vs. sequence length (line plot) — compression ratio.
2. Token length distribution (histogram) — k-mer / subword structure.
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import hydra
from omegaconf import DictConfig

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.plotting_utils import configure_plot_style, get_series_color, PLOT_STYLE
from src.utils import set_determinism

matplotlib.use("Agg")
configure_plot_style()

logger = logging.getLogger(__name__)

# Models to analyze: display_name -> HuggingFace checkpoint
MODELS = {
    "DNABERT-2": "zhihan1996/DNABERT-2-117M",
    "NTv2": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species",
}

NUCLEOTIDES = ["A", "C", "G", "T"]


def generate_random_sequences(n: int, length: int, rng: np.random.RandomState) -> list[str]:
    """Generate *n* random DNA sequences of *length* nucleotides."""
    indices = rng.randint(0, 4, size=(n, length))
    return ["".join(NUCLEOTIDES[i] for i in row) for row in indices]


def _color_for(name: str) -> str:
    """Map model display name to the project color palette."""
    # get_series_color detects 'dnabert' / 'ntv2' keywords
    return get_series_color(name)


def plot_token_counts(
    results: dict[str, dict[int, list[int]]],
    seq_lengths: list[int],
    output_path: str,
) -> None:
    """Line plot: mean token count (± std) vs. nucleotide sequence length."""
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = ["o", "s", "^", "D"]
    for idx, (model_name, length_to_counts) in enumerate(results.items()):
        lengths_sorted = sorted(length_to_counts.keys())
        means = [np.mean(length_to_counts[l]) for l in lengths_sorted]
        stds = [np.std(length_to_counts[l]) for l in lengths_sorted]
        color = _color_for(model_name)

        ax.plot(
            lengths_sorted,
            means,
            marker=markers[idx % len(markers)],
            linestyle="-",
            linewidth=PLOT_STYLE["line_width"],
            markersize=PLOT_STYLE["marker_size"],
            label=model_name,
            color=color,
        )
        ax.fill_between(
            lengths_sorted,
            np.array(means) - np.array(stds),
            np.array(means) + np.array(stds),
            color=color,
            alpha=0.2,
        )

    # Reference line: 1:1 (char-level, i.e. Evo2)
    ax.plot(
        seq_lengths,
        seq_lengths,
        linestyle="--",
        linewidth=1.5,
        color="#555555",
        label="Char-level (Evo2)",
    )

    ax.set_xlabel("Sequence Length (nucleotides)", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Number of Tokens", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_title(
        "Token Count vs. Sequence Length", fontsize=PLOT_STYLE["title_fontsize"], fontweight="bold"
    )
    ax.legend(fontsize=PLOT_STYLE["legend_fontsize"])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.pdf", dpi=300)
    plt.close()
    logger.info(f"Saved token count plot to {output_path}")


def plot_token_length_distribution(
    token_lengths: dict[str, list[int]],
    output_path: str,
) -> None:
    """Overlapping histogram / KDE of individual token lengths (in characters)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, lengths in token_lengths.items():
        color = _color_for(model_name)
        sns.histplot(
            lengths,
            kde=True,
            color=color,
            label=model_name,
            alpha=0.45,
            element="step",
            stat="density",
            ax=ax,
        )

    ax.set_xlabel("Token Length (characters)", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_ylabel("Density", fontsize=PLOT_STYLE["label_fontsize"])
    ax.set_title(
        "Token Length Distribution", fontsize=PLOT_STYLE["title_fontsize"], fontweight="bold"
    )
    ax.legend(fontsize=PLOT_STYLE["legend_fontsize"])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}.png", dpi=300)
    plt.savefig(f"{output_path}.pdf", dpi=300)
    plt.close()
    logger.info(f"Saved token length distribution to {output_path}")


@hydra.main(config_path="../conf", config_name="tokenizer_analysis", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run tokenizer analysis for DNABERT-2 and NTv2."""
    set_determinism(cfg.seed)
    rng = np.random.RandomState(cfg.seed)
    output_dir = os.getcwd()  # Hydra changes cwd to the run directory

    from transformers import AutoTokenizer

    # Instantiate tokenizers
    tokenizers = {
        name: AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        for name, ckpt in MODELS.items()
    }

    seq_lengths = list(cfg.seq_lengths)

    # --- Plot 1: Token count vs. sequence length ---
    # {model_name: {seq_length: [token_counts]}}
    count_results: dict[str, dict[int, list[int]]] = {name: {} for name in MODELS}

    for length in seq_lengths:
        sequences = generate_random_sequences(cfg.n_samples, length, rng)
        for model_name, tok in tokenizers.items():
            counts = [len(tok.tokenize(seq)) for seq in sequences]
            count_results[model_name][length] = counts
        logger.info(f"Tokenized {cfg.n_samples} sequences of length {length}")

    plot_token_counts(
        count_results, seq_lengths, os.path.join(output_dir, "token_counts_vs_length")
    )

    # --- Plot 2: Token length distribution at a fixed sequence length ---
    fixed_length = cfg.fixed_length
    fixed_sequences = generate_random_sequences(cfg.n_samples, fixed_length, rng)

    token_length_data: dict[str, list[int]] = {}
    for model_name, tok in tokenizers.items():
        all_lengths: list[int] = []
        for seq in fixed_sequences:
            tokens = tok.tokenize(seq)
            all_lengths.extend(len(t.replace("Ġ", "").replace("▁", "")) for t in tokens)
        token_length_data[model_name] = all_lengths
        logger.info(f"{model_name}: {len(all_lengths)} tokens at length {fixed_length}")

    plot_token_length_distribution(
        token_length_data, os.path.join(output_dir, "token_length_distribution")
    )

    logger.info("Tokenizer analysis complete.")


if __name__ == "__main__":
    main()
