import os
import json
import collections
import re

# Hardcoded list of eval mean directories
EVAL_DIRS = ["outputs/eval_mean_dnabert2", "outputs/eval_mean_evo2", "outputs/eval_mean_ntv2"]

# Hardcoded list of analysis mean directories
ANALYSIS_DIRS = [
    "outputs/analysis_mean_dnabert2",
    "outputs/analysis_mean_evo2",
    "outputs/analysis_mean_ntv2",
]

FOUNDATION_MODEL_NAMES = {
    "dnabert2": "DNABERT-2",
    "evo2": "Evo2",
    "ntv2": "NTv2",
}


def load_data():
    """
    Load evaluation results from the hardcoded directories.
    Returns a nested dictionary: data[dataset][method][seq_len] = {metrics}
    """
    data = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(dict))
    )

    for eval_dir in EVAL_DIRS:
        if not os.path.exists(eval_dir):
            pass

        if not os.path.exists(eval_dir):
            print(
                f"Warning: Directory {eval_dir} does not exist (checked relative to {os.getcwd()})."
            )
            continue

        # Iterate over run_X subdirectories
        for run_name in os.listdir(eval_dir):
            run_path = os.path.join(eval_dir, run_name)
            if not os.path.isdir(run_path) or not run_name.startswith("run_"):
                continue

            json_path = os.path.join(run_path, "evaluation_results.json")
            if not os.path.exists(json_path):
                continue

            try:
                with open(json_path, "r") as f:
                    results = json.load(f)
            except Exception as e:
                print(f"Error reading {json_path}: {e}")
                continue

            # Extract keys
            foundation_model = results.get("foundation_model", "unknown")
            inversion_model = results.get("inversion_model", "unknown")
            seq_length = results.get("seq_length", 0)

            # Clean up names
            # Map foundation model to printable dataset name
            if "dnabert2" in foundation_model:
                dataset = "DNABERT-2"
            elif "evo2" in foundation_model:
                dataset = "Evo2"
            elif "ntv2" in foundation_model:
                dataset = "NTv2"
            else:
                dataset = foundation_model

            # Clean up method name
            # e.g. transformer_mean -> Transformer
            method = inversion_model.replace("_mean", "")
            if method.lower() == "resnet":
                method = "ResNet"
            elif method.lower() == "knn":
                method = "Nearest Neighbor"
            else:
                method = method.capitalize()

            # Extract metrics
            lev_mean = results.get("levenshtein_mean", 0.0)
            lev_std = results.get("levenshtein_std", 0.0)

            # Accuracy might be nested
            # Use accuracy_mean (macro-average) to be consistent with accuracy_std (also macro)
            acc_val = results.get("accuracy_mean", 0.0)
            # Check for accuracy std, default to 0.0 if missing (older runs might miss it)
            acc_std = results.get("accuracy_std", 0.0)

            data[dataset][method][seq_length] = {
                "lev_mean": lev_mean,
                "lev_std": lev_std,
                "acc_mean": acc_val,
                "acc_std": acc_std,
            }

    return data


def generate_latex_tables(data):
    """
    Generate LaTeX tables from the loaded data.
    """
    datasets = sorted(data.keys())

    for dataset in datasets:
        print(f"\n% Table for {dataset}")
        print("\\begin{table}[h]")
        print("\\centering")
        # Determine columns based on sequence lengths present
        # Find all sequence lengths for this dataset
        methods = data[dataset]
        all_seq_lens = set()
        for m in methods:
            all_seq_lens.update(methods[m].keys())

        sorted_seq_lens = sorted(list(all_seq_lens))

        # Column format: Method, Metric, then one col per seq len
        # l c c...
        col_spec = "lc" + "c" * len(sorted_seq_lens)
        print(f"\\begin{{tabular}}{{{col_spec}}}")
        print("\\toprule")

        # Header row
        header_seqs = " & ".join([f"\\textbf{{{sl}}}" for sl in sorted_seq_lens])
        print(f"\\textbf{{Method}} & \\textbf{{Metric}} & {header_seqs} \\\\")
        print("\\midrule")

        # Sort methods: Transformer, Decoder, ResNet, KNN, others
        def sort_key(name):
            order = {"Encoder": 1, "Decoder": 2, "ResNet": 3, "KNN": 4}
            return order.get(name, 99), name

        sorted_methods = sorted(methods.keys(), key=sort_key)

        first_row = True
        for method in sorted_methods:
            if not first_row:
                print("\\midrule")
            first_row = False

            # Prepare rows for Levenshtein and Accuracy
            lev_row_data = []
            acc_row_data = []

            for sl in sorted_seq_lens:
                if sl in methods[method]:
                    d = methods[method][sl]
                    # Format: value \pm std
                    lev_str = f"${d['lev_mean']:.2f} \\pm {d['lev_std']:.2f}$"
                    # For accuracy, only show std if non-zero?
                    if d["acc_std"] > 0:
                        acc_str = f"${d['acc_mean']:.2f} \\pm {d['acc_std']:.2f}$"
                    else:
                        acc_str = f"${d['acc_mean']:.2f}$"

                    lev_row_data.append(lev_str)
                    acc_row_data.append(acc_str)
                else:
                    lev_row_data.append("-")
                    acc_row_data.append("-")

            print(
                f"\\multirow{{2}}{{*}}{{{method}}} & Levenshtein & "
                + " & ".join(lev_row_data)
                + " \\\\"
            )
            print(f" & Accuracy & " + " & ".join(acc_row_data) + " \\\\")

        print("\\bottomrule")
        print("\\end{tabular}")
        print(f"\\caption{{Reconstruction results for {dataset} dataset.}}")
        print(f"\\label{{tab:results_{dataset.lower()}}}")
        print("\\end{table}")
        print()


def load_spearman_data():
    """
    Load Spearman correlation data from analysis_mean directories.
    Returns a nested dict: spearman[dataset][seq_len] = {"cosine": float, "euclidean": float}
    """
    spearman = collections.defaultdict(dict)

    for analysis_dir in ANALYSIS_DIRS:
        if not os.path.exists(analysis_dir):
            print(f"Warning: Directory {analysis_dir} does not exist.")
            continue

        for subdir in os.listdir(analysis_dir):
            stats_path = os.path.join(analysis_dir, subdir, "stats.json")
            if not os.path.isfile(stats_path):
                continue

            with open(stats_path, "r") as f:
                stats = json.load(f)

            dataset_str = stats["dataset"]  # e.g. "dnabert2_10_hg38_mean"
            # Extract foundation model key and seq_length from dataset string
            match = re.match(r"^(dnabert2|evo2|ntv2)_(\d+)_", dataset_str)
            assert match, f"Cannot parse dataset string: {dataset_str}"

            fm_key = match.group(1)
            seq_len = int(match.group(2))
            dataset_name = FOUNDATION_MODEL_NAMES[fm_key]

            spearman[dataset_name][seq_len] = {
                "cosine": stats["spearman_corr_cosine"],
                "euclidean": stats["spearman_corr_euclidean"],
            }

    return spearman


def generate_spearman_table(spearman):
    """
    Generate a horizontal LaTeX table of Spearman correlations.
    Rows: (dataset, metric) pairs. Columns: sequence lengths.
    The highest value per column is bolded.
    """
    datasets = sorted(spearman.keys())
    metrics = ["cosine", "euclidean"]
    metric_labels = {"cosine": "Cosine", "euclidean": "Euclidean"}

    # Collect all sequence lengths across all datasets
    all_seq_lens = set()
    for ds in datasets:
        all_seq_lens.update(spearman[ds].keys())
    sorted_seq_lens = sorted(all_seq_lens)

    # Build the value matrix: rows[(dataset, metric)][seq_len] = value
    rows = []
    for ds in datasets:
        for metric in metrics:
            row_values = {}
            for sl in sorted_seq_lens:
                if sl in spearman[ds]:
                    row_values[sl] = spearman[ds][sl][metric]
            rows.append((ds, metric, row_values))

    # Find the max value per column independently for each metric
    col_max = {}
    for metric in metrics:
        col_max[metric] = {}
        metric_rows = [(ds, m, rv) for ds, m, rv in rows if m == metric]
        for sl in sorted_seq_lens:
            vals = [rv[sl] for _, _, rv in metric_rows if sl in rv]
            col_max[metric][sl] = max(vals) if vals else None

    # Generate LaTeX
    print("\n% Spearman correlation table")
    print("\\begin{table}[h]")
    print("\\centering")
    col_spec = "ll" + "c" * len(sorted_seq_lens)
    print(f"\\begin{{tabular}}{{{col_spec}}}")
    print("\\toprule")

    header_seqs = " & ".join([f"\\textbf{{{sl}}}" for sl in sorted_seq_lens])
    print(f"\\textbf{{Model}} & \\textbf{{Similarity}} & {header_seqs} \\\\")
    print("\\midrule")

    prev_ds = None
    for ds, metric, row_values in rows:
        if prev_ds is not None and ds != prev_ds:
            print("\\midrule")
        prev_ds = ds

        cells = []
        for sl in sorted_seq_lens:
            if sl in row_values:
                val = row_values[sl]
                is_max = col_max[metric][sl] is not None and val == col_max[metric][sl]
                formatted = f"{val:.4f}"
                if is_max:
                    cells.append(f"$\\mathbf{{{formatted}}}$")
                else:
                    cells.append(f"${formatted}$")
            else:
                cells.append("-")

        ds_label = f"\\multirow{{2}}{{*}}{{{ds}}}" if metric == metrics[0] else ""
        print(f"{ds_label} & {metric_labels[metric]} & " + " & ".join(cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print(
        "\\caption{Spearman correlation between embedding similarity (cosine / Euclidean) and sequence similarity (Levenshtein) for different models and sequence lengths.}"
    )
    print("\\label{tab:spearman_correlation}")
    print("\\end{table}")
    print()


if __name__ == "__main__":
    # Ensure we are in project root if running from scripts/ does not work
    # But user will likely run from project root: python scripts/generate_latex_tables.py
    data = load_data()
    generate_latex_tables(data)

    spearman = load_spearman_data()
    generate_spearman_table(spearman)
