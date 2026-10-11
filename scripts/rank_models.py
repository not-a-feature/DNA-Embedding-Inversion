#!/usr/bin/env python3
"""
Script to valid evaluation results and rank models based on performance.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
import glob

def load_results(base_dir):
    """Recursively find and load evaluation_results.json files."""
    results_files = glob.glob(os.path.join(base_dir, "**", "evaluation_results.json"), recursive=True)
    all_results = []
    
    for f in results_files:
        try:
            with open(f, "r") as fh:
                data = json.load(fh)
                
            # Extract key metrics
            # If model name logic in evaluate.py changed, we might need to reconstruct it here
            # or rely on what's in the json if we re-ran evaluation.
            
            # We want to aggregate by "Model Configuration". 
            # The 'inversion_model' in json might be the old short name if not re-evaluated.
            # But we can look at the config.yaml too if needed. 
            # For now let's try to use what's in the json, assuming user re-runs evaluation 
            # or we parse the path.
            
            res_entry = {
                "path": f,
                "inversion_model": data.get("inversion_model", "Unknown"),
                "foundation_model": data.get("foundation_model", "Unknown"),
                "seq_length": data.get("seq_length", 0),
                "accuracy": data.get("accuracy_mean", 0.0),
                "levenshtein": data.get("levenshtein_mean", 0.0),
            }
            
            # Attempt to construct a richer model ID if possible from the run directory config
            run_dir = data.get("run_dir")
            if run_dir and os.path.isdir(run_dir):
                config_path = os.path.join(run_dir, ".hydra", "config.yaml")
                # We can perform similar logic to evaluate.py here if we want to rank 
                # existing runs without re-evaluating them.
                # But let's stick to the JSON for speed first.
                pass

            all_results.append(res_entry)
            
        except Exception as e:
            print(f"Error loading {f}: {e}")
            
    return pd.DataFrame(all_results)

def rank_models(df):
    """Rank models by aggregating performance across sequence lengths."""
    if df.empty:
        print("No results found.")
        return

    # Group by Model identifier. 
    # Use inversion_model as the key.
    
    # We aggregate by taking the mean of metrics across available sequence lengths
    grouped = df.groupby(["inversion_model", "foundation_model"]).agg({
        "accuracy": ["mean", "std", "count"],
        "levenshtein": ["mean", "std"],
        "seq_length": lambda x: list(sorted(unique_vals(x)))
    }).reset_index()
    
    # Flatten columns
    grouped.columns = [
        "Model", "Foundation Model", 
        "Acc Mean", "Acc Std", "Count",
        "Lev Mean", "Lev Std",
        "Lengths"
    ]
    
    # Sort by Accuracy Mean descending
    grouped = grouped.sort_values("Acc Mean", ascending=False)
    
    return grouped

def unique_vals(x):
    return list(set(x))

def main():
    parser = argparse.ArgumentParser(description="Rank DNA Inversion Models")
    parser.add_argument("--dir", type=str, default="outputs", help="Base directory to search")
    parser.add_argument("--out", type=str, default=None, help="Output CSV file")
    
    args = parser.parse_args()
    
    print(f"Searching for results in {args.dir}...")
    df = load_results(args.dir)
    print(f"Found {len(df)} evaluation result files.")
    
    ranking = rank_models(df)
    
    if ranking is not None and not ranking.empty:
        print("\nModel Ranking (averaged over all found sequence lengths):")
        print("="*100)
        # Format for nicer printing
        print_df = ranking.copy()
        print_df["Acc Mean"] = print_df["Acc Mean"].apply(lambda x: f"{x:.4f}")
        print_df["Lev Mean"] = print_df["Lev Mean"].apply(lambda x: f"{x:.4f}")
        print(print_df.to_string(index=False))
        print("="*100)
        
        if args.out:
            ranking.to_csv(args.out, index=False)
            print(f"\nSaved ranking to {args.out}")
    else:
        print("No ranking generated.")

if __name__ == "__main__":
    main()
