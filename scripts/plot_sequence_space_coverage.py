"""Plot benchmark coverage of the theoretical DNA sequence space.

The default counts reproduce the current hg38 multi-length benchmark.  If the
data are regenerated, pass either ``--used-counts`` alongside ``--lengths`` or
a JSON mapping from sequence length to the total number of sampled windows. The
coordinate-aware revision can be read directly from its source HDF5:

    python scripts/plot_sequence_space_coverage.py
    python scripts/plot_sequence_space_coverage.py \
        --counts-json outputs/sequence_counts.json
    python scripts/plot_sequence_space_coverage.py \
        --source-h5 data/hg38_multilen/hg38_multilen.h5

The script writes both PDF and PNG versions and prints the plotted values so
the figure can be audited without opening it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LENGTHS = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100)


def current_benchmark_count(length: int) -> int:
    """Return the total train/validation/test count used at a given length."""
    if length <= 20:
        return 1_000_000
    if length <= 45:
        return 2_000_000
    if length <= 80:
        return 3_000_000
    return 4_000_000


def load_counts(
    lengths: Sequence[int],
    used_counts: Sequence[int] | None,
    counts_json: Path | None,
    source_h5: Path | None = None,
) -> list[int]:
    """Resolve counts from CLI values, a JSON mapping, or the benchmark defaults."""
    num_sources = sum(value is not None for value in (used_counts, counts_json, source_h5))
    assert num_sources <= 1, "Use only one of --used-counts, --counts-json, --source-h5"

    if source_h5:
        import h5py

        with h5py.File(source_h5, "r", swmr=True) as handle:
            assert handle.attrs.get("schema", "") == "dna_inversion_hg38_multilen_v1"
            counts = [len(handle[f"lengths/{length}/sequences"]) for length in lengths]
    elif counts_json:
        with counts_json.open(encoding="utf-8") as handle:
            raw = json.load(handle)
        counts = [int(raw[str(length)] if str(length) in raw else raw[length]) for length in lengths]
    elif used_counts:
        assert len(used_counts) == len(lengths), (
            "--used-counts must contain exactly one value per --lengths entry"
        )
        counts = [int(value) for value in used_counts]
    else:
        counts = [current_benchmark_count(length) for length in lengths]

    assert all(count > 0 for count in counts), "All used-sequence counts must be positive"
    for length, count in zip(lengths, counts):
        assert count <= 4**length, f"Count {count:,} exceeds 4^{length}"
    return counts


def plot_coverage(lengths: Sequence[int], counts: Sequence[int], output_base: Path) -> None:
    """Create count and fractional-coverage panels."""
    lengths_array = np.asarray(lengths, dtype=int)
    used = np.asarray(counts, dtype=float)
    theoretical = np.asarray([float(4**length) for length in lengths], dtype=float)
    coverage_percent = 100.0 * used / theoretical

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, (ax_count, ax_fraction) = plt.subplots(1, 2, figsize=(7.2, 3.05))

    ax_count.semilogy(
        lengths_array,
        theoretical,
        color="#303030",
        linewidth=2.0,
        label=r"All DNA strings ($4^l$)",
    )
    ax_count.semilogy(
        lengths_array,
        used,
        color="#2B6CB0",
        marker="o",
        markersize=4,
        linewidth=1.8,
        label="hg38 windows sampled",
    )
    ax_count.set_xlabel("Sequence length $l$ (nt)")
    ax_count.set_ylabel("Number of sequences")
    ax_count.grid(True, which="both", alpha=0.22)
    ax_count.legend(loc="upper left", frameon=False)

    ax_fraction.semilogy(
        lengths_array,
        coverage_percent,
        color="#B83280",
        marker="o",
        markersize=4,
        linewidth=1.8,
    )
    ax_fraction.set_xlabel("Sequence length $l$ (nt)")
    ax_fraction.set_ylabel(r"Coverage of $4^l$ (\%)")
    ax_fraction.grid(True, which="both", alpha=0.22)
    ax_fraction.annotate(
        f"{coverage_percent[0]:.1f}%",
        xy=(lengths_array[0], coverage_percent[0]),
        xytext=(8, 7),
        textcoords="offset points",
        color="#B83280",
    )
    if len(lengths_array) > 1:
        ax_fraction.annotate(
            f"{coverage_percent[1]:.3g}%",
            xy=(lengths_array[1], coverage_percent[1]),
            xytext=(8, -18),
            textcoords="offset points",
            color="#B83280",
        )

    for axis in (ax_count, ax_fraction):
        axis.set_xticks(lengths_array[::2])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=2.1)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("length\tused\ttheoretical\tcoverage_percent")
    for length, count, possible, percent in zip(
        lengths, counts, (4**length for length in lengths), coverage_percent
    ):
        print(f"{length}\t{count}\t{possible}\t{percent:.6g}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument("--used-counts", nargs="+", type=int)
    parser.add_argument("--counts-json", type=Path)
    parser.add_argument(
        "--source-h5",
        type=Path,
        help="Read per-length row counts from hg38_multilen.h5",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tex" / "figures" / "sequence_space_coverage",
        help="Output path without an extension (both .pdf and .png are written)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert args.lengths == sorted(set(args.lengths)), "Lengths must be unique and increasing"
    counts = load_counts(args.lengths, args.used_counts, args.counts_json, args.source_h5)
    plot_coverage(args.lengths, counts, args.output_base)


if __name__ == "__main__":
    main()
