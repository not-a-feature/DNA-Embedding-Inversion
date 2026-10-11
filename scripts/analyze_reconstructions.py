import json
import glob
import os
import sys
import math
import collections
import matplotlib.pyplot as plt
import Levenshtein
import numpy as np


def calculate_shannon_entropy(sequence):
    """Calculates the Shannon entropy of a DNA sequence."""
    if not sequence:
        return 0.0

    counts = collections.Counter(sequence)
    frequencies = [count / len(sequence) for count in counts.values()]

    entropy = -sum(p * math.log2(p) for p in frequencies)
    return entropy


def load_data(outputs_dir):
    """
    Loads data from evaluation results.
    Returns a dictionary structured as:
    data[model_name][seq_len] = list of (true_seq, pred_seq, match, index, similarity, entropy)
    """
    data_store = collections.defaultdict(lambda: collections.defaultdict(list))

    # Pattern to find all eval_mean runs
    # Assuming structure: outputs/eval_mean_{model}/run_{id}/
    search_pattern = os.path.join(outputs_dir, "eval_mean_*", "run_*", "evaluation_results.json")
    print(f"Searching for results in: {search_pattern}")

    files = glob.glob(search_pattern)
    print(f"Found {len(files)} evaluation result files.")

    for eval_file in files:
        run_dir = os.path.dirname(eval_file)
        reconstructions_file = os.path.join(run_dir, "example_reconstructions.json")

        if not os.path.exists(reconstructions_file):
            # print(f"Skipping {run_dir}: example_reconstructions.json not found.")
            continue

        try:
            with open(eval_file, "r") as f:
                eval_data = json.load(f)

            with open(reconstructions_file, "r") as f:
                recon_data = json.load(f)

            model = eval_data.get("foundation_model", "unknown")
            seq_len = eval_data.get("seq_length", 0)

            examples = recon_data.get("examples", [])

            for ex in examples:
                true_seq = ex["true_sequence"]
                pred_seq = ex["predicted_sequence"]
                match = ex["match"]
                idx = ex["index"]

                similarity = Levenshtein.ratio(true_seq, pred_seq)
                entropy = calculate_shannon_entropy(true_seq)

                data_item = {
                    "true_sequence": true_seq,
                    "predicted_sequence": pred_seq,
                    "match": match,
                    "index": idx,
                    "similarity": similarity,
                    "entropy": entropy,
                }

                data_store[model][seq_len].append(data_item)

        except Exception as e:
            print(f"Error processing {run_dir}: {e}")
            continue

    return data_store


def analyze_and_print_extremes(data_store):
    print("\n" + "=" * 80)
    print("EXTREME RECONSTRUCTIONS (BEST AND WORST)")
    print("=" * 80)

    for model in sorted(data_store.keys()):
        for seq_len in sorted(data_store[model].keys()):
            items = data_store[model][seq_len]

            # Sort by similarity
            # High similarity is good, Low similarity is bad
            items.sort(key=lambda x: x["similarity"], reverse=True)

            print(f"\nModel: {model}, Sequence Length: {seq_len}")
            print(f"Total examples: {len(items)}")

            print("-" * 40)
            print("TOP 10 BEST RECONSTRUCTIONS (Highest Levenshtein Similarity)")
            for i, item in enumerate(items[:10]):
                print(f"{i+1}. Index: {item['index']}")
                print(f"   True: {item['true_sequence']}")
                print(f"   Pred: {item['predicted_sequence']}")
                print(
                    f"   Sim:  {item['similarity']:.4f}, Entropy: {item['entropy']:.4f}, Match: {item['match']}"
                )

            print("-" * 40)
            print("TOP 10 WORST RECONSTRUCTIONS (Lowest Levenshtein Similarity)")
            for i, item in enumerate(items[-10:]):
                print(f"{i+1}. Index: {item['index']}")
                print(f"   True: {item['true_sequence']}")
                print(f"   Pred: {item['predicted_sequence']}")
                print(
                    f"   Sim:  {item['similarity']:.4f}, Entropy: {item['entropy']:.4f}, Match: {item['match']}"
                )
            print("-" * 40)


def plot_entropy_vs_similarity(data_store, output_dir="plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]
    # Use a colormap with enough colors
    cmap = plt.get_cmap("tab10")

    for model in sorted(data_store.keys()):
        plt.figure(figsize=(10, 6))

        seq_lens = sorted(data_store[model].keys())

        for idx, seq_len in enumerate(seq_lens):
            items = data_store[model][seq_len]

            entropies = [x["entropy"] for x in items]
            similarities = [x["similarity"] for x in items]

            # Define color and marker first
            color = cmap(idx % 10)
            marker = markers[idx % len(markers)]

            try:
                # Calculate correlation
                correlation = np.corrcoef(entropies, similarities)[0, 1]

                # Calculate regression line
                if len(entropies) > 1:
                    m, b = np.polyfit(entropies, similarities, 1)
                    reg_line = [m * x + b for x in entropies]

                    # Sort for cleaner line plotting if needed, but scatter x is not sorted.
                    # Actually for line plot we can just plot min/max or sorted.
                    # Let's plot the line segment covering the range.
                    min_x, max_x = min(entropies), max(entropies)
                    plt.plot(
                        [min_x, max_x],
                        [m * min_x + b, m * max_x + b],
                        color=color,
                        linestyle="--",
                        alpha=0.7,
                        linewidth=2,
                    )

                label = f"Seq Len {seq_len} (R={correlation:.2f})"
            except Exception as e:
                print(f"Error calculating regression for {model} {seq_len}: {e}")
                label = f"Seq Len {seq_len}"

            marker = markers[idx % len(markers)]

            plt.scatter(
                entropies, similarities, label=label, alpha=0.5, s=20, color=color, marker=marker
            )

        plt.title(f"{model}: Shannon Entropy vs Levenshtein Similarity")
        plt.xlabel("Shannon Entropy (bits)")
        plt.ylabel("Levenshtein Similarity")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, f"shannon_vs_levenshtein_{model}.png")
        plt.savefig(output_path, dpi=300)
        print(f"\nPlot saved to: {output_path}")
        plt.close()


def main():
    # Detect current directory and try to find 'outputs'
    cwd = os.getcwd()
    outputs_dir = os.path.join(cwd, "outputs")

    if not os.path.exists(outputs_dir):
        # Fallback for when running from scripts/ or something
        parent_outputs = os.path.join(os.path.dirname(cwd), "outputs")
        if os.path.exists(parent_outputs):
            outputs_dir = parent_outputs
        else:
            # Last resort: try absolute path if known, or error out
            # Assuming standard layout
            print(f"Error: 'outputs' directory not found in {cwd} or parent.")
            sys.exit(1)

    print(f"Using outputs directory: {outputs_dir}")
    data = load_data(outputs_dir)

    if not data:
        print("No data found.")
        # Don't exit error, just exit cleanly if empty
        sys.exit(0)

    analyze_and_print_extremes(data)
    plot_entropy_vs_similarity(data)


if __name__ == "__main__":
    main()
