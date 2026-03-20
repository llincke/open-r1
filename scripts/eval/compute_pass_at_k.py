#!/usr/bin/env python3
# scripts/eval/compute_pass_at_k.py
import argparse
import os
import glob
import json
import numpy as np
import pandas as pd
import wandb

from datasets import load_dataset

def load_from_parquet_dir(parquet_dir):
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {parquet_dir}")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df

def pass_at_k(n, c, k):
    """
    Estimator for pass@k metric.
    
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def compute_pass_at_k_bool_matrix(bool_matrix, ks):
    # bool_matrix: N x G numpy array of booleans
    N, G = bool_matrix.shape
    results = {}
    for k in ks:
        k = min(k, G)
        # For each problem, count how many correct solutions out of G generations
        num_correct = bool_matrix.sum(axis=1)  # Shape: (N,)
        # Use the pass_at_k estimator for each problem
        pass_at_k_vals = np.array([pass_at_k(G, int(c), k) for c in num_correct])
        results[f"pass@{k}"] = float(np.mean(pass_at_k_vals))
    return results

def normalize_rewards_to_bool(rewards, threshold=1.0):
    # rewards: list of lists or array; interpret >= threshold as pass
    arr = np.array(rewards, dtype=float)
    return arr >= threshold

def main():
    p = argparse.ArgumentParser()
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--parquet-file", help="Single parquet file produced by generate_local.py")
    grp.add_argument("--parquet-dir", help="Directory with LightEval details parquet files (details/*.parquet)")
    grp.add_argument("--hf-dataset", help="HF dataset repo or local dataset path that contains per-generation fields")
    p.add_argument("--gen-field", default="pass_rate_generations",
                   help="Column name with list-of-generations per example (default: pass_rate_generations)")
    p.add_argument("--reward-field", default="pass_rate_rewards",
                   help="Optional column with per-generation reward values (shape N x G); if present used directly")
    p.add_argument("--ks", nargs="+", type=int, default=[1,5,10,16], help="List of k to compute pass@k")
    p.add_argument("--reward-threshold", type=float, default=1.0, help="Threshold to consider a generation a pass (if using rewards)")
    p.add_argument("--wandb-project", type=str, default=None, help="WandB project to log results to (optional)")
    p.add_argument("--wandb-run", type=str, default=None, help="WandB run name (optional)")
    p.add_argument("--output", type=str, default=None, help="Optional JSON file to write the pass@k results")
    p.add_argument("--metadata", type=str, default=None, help="Optional JSON file with generation parameters to include in output")
    args = p.parse_args()

    if args.parquet_file:
        df = pd.read_parquet(args.parquet_file)
    elif args.parquet_dir:
        df = load_from_parquet_dir(args.parquet_dir)
    else:
        ds = load_dataset(args.hf_dataset)
        # assume the split is 'train' or 'test'; try 'test' first
        if isinstance(ds, dict):
            split = "test" if "test" in ds else list(ds.keys())[0]
            df = ds[split].to_pandas()
        else:
            df = ds.to_pandas()

    # Determine boolean pass matrix N x G
    bool_matrix = None
    if args.reward_field and args.reward_field in df.columns:
        # rewards present as lists per row
        rewards = df[args.reward_field].tolist()
        bool_matrix = normalize_rewards_to_bool(rewards, threshold=args.reward_threshold)
    elif args.gen_field and args.gen_field in df.columns:
        # We only have generations; attempt to find a per-generation 'pass' field in details.
        # LightEval details often include a 'passed' or 'score' per generation; try to detect common names
        # Fall back to simple heuristic: if details contain 'scores' or 'is_correct' column; else fail.
        # Here we attempt keys commonly produced by LightEval: 'scores', 'is_correct', 'correct'
        if "is_correct" in df.columns:
            bool_matrix = np.array(df["is_correct"].tolist(), dtype=bool)
        elif "scores" in df.columns:
            bool_matrix = np.array(df["scores"].tolist(), dtype=float) >= args.reward_threshold
        else:
            raise ValueError("No explicit reward/pass column found; please provide --reward-field or use a details parquet with pass flags.")
    else:
        raise ValueError("Could not find generation or reward columns in inputs.")

    # Ensure shape N x G
    bool_matrix = np.asarray(bool_matrix)
    if bool_matrix.ndim != 2:
        raise ValueError(f"Expected 2D boolean array; got shape {bool_matrix.shape}")

    ks = sorted(args.ks)
    results = compute_pass_at_k_bool_matrix(bool_matrix, ks)

    # Load and merge metadata if provided
    metadata = {}
    if args.metadata:
        with open(args.metadata, "r") as f:
            metadata = json.load(f)
    
    # Combine results and metadata
    final_output = {
        "results": results,
        "metadata": metadata,
        "stats": {
            "num_examples": bool_matrix.shape[0],
            "num_generations": bool_matrix.shape[1]
        }
    }

    # Log to wandb if requested
    if args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config={"ks": ks})
        wandb.log(results)
        wandb.finish()

    # Write results to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(final_output, f, indent=2)

    print("Computed pass@k:", results)

if __name__ == "__main__":
    main()