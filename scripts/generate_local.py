#!/usr/bin/env python3
"""Generate per-example completions using vLLM and save to a parquet file.

The output parquet contains one row per prompt with columns:
  - prompt                : the input prompt string
  - solution              : ground-truth answer (if found in the dataset)
  - pass_rate_generations : list of G generation strings
  - pass_rate_rewards     : list of G binary rewards (1.0 = correct, 0.0 = wrong, None = unverifiable)

The pass_rate_rewards are computed in a separate post-processing step to avoid
vLLM engine conflicts. These columns are consumed directly by
scripts/eval/compute_pass_at_k.py (--reward-field pass_rate_rewards).

Usage:
    python scripts/generate_local.py \\
        --model-dir data/Qwen2.5-1.5B-GRPO-4offpolicy-math12k \\
        --dataset hiyouga/math12k --split test --num-generations 16
"""
import argparse
import traceback

from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def choose_prompt_column(ds):
    lower = [c.lower() for c in ds.column_names]
    for candidate in ("prompt", "question", "input", "text", "problem"):
        if candidate in lower:
            return ds.column_names[lower.index(candidate)]
    return ds.column_names[0]


def choose_solution_column(ds):
    """Return the ground-truth answer column name, or None if not present."""
    lower = [c.lower() for c in ds.column_names]
    for candidate in ("solution", "answer", "target", "label", "output"):
        if candidate in lower:
            return ds.column_names[lower.index(candidate)]
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out-parquet", default=None)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7,
                   help="Sampling temperature (default: 0.7)")
    p.add_argument("--top-p", type=float, default=0.8,
                   help="Nucleus sampling parameter (default: 0.8)")
    p.add_argument("--system-prompt", type=str, default=None,
                   help="Optional system prompt to include in the chat template")
    p.add_argument("--no-score", action="store_true",
                   help="Skip reward scoring (for faster generation-only runs)")
    args = p.parse_args()

    print("Loading dataset", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split)

    prompt_col = choose_prompt_column(ds)
    solution_col = choose_solution_column(ds)
    print(f"Using prompt column: {prompt_col!r},  solution column: {solution_col!r}")

    prompts = [row[prompt_col] for row in ds]
    solutions = [row[solution_col] for row in ds] if solution_col else None
    print(f"Loaded {len(prompts)} prompts")

    print("Initializing vLLM model...")
    llm = LLM(model=args.model_dir, trust_remote_code=True)

    # Apply chat template so the model knows when to stop generating
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    prompts = [
        tokenizer.apply_chat_template(
            (
                [{"role": "system", "content": args.system_prompt}] 
                if args.system_prompt else []
            ) + [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    print(f"Applied chat template. Example prompt[:200]: {prompts[0][:200]!r}")

    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=20,
        n=args.num_generations,
        max_tokens=args.max_tokens,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    # --- generation ---
    all_prompts_out = []
    all_solutions_out = []
    all_generations = []   # list of N lists of G strings

    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        batch_solutions = solutions[i:i + args.batch_size] if solutions else None
        print(f"Generating prompts {i}–{i + len(batch_prompts) - 1}")
        try:
            outputs = llm.generate(batch_prompts, sampling_params=sampling, use_tqdm=False)
            for j, out in enumerate(outputs):
                gens = [c.text for c in out.outputs]
                all_prompts_out.append(out.prompt)
                all_solutions_out.append(batch_solutions[j] if batch_solutions else None)
                all_generations.append(gens)
        except Exception:
            print("Error generating batch starting at", i)
            traceback.print_exc()

    # --- scoring (optional, in separate Python process to avoid vLLM conflicts) ---
    all_rewards = []
    if args.no_score:
        # Skip reward scoring
        all_rewards = [[None] * args.num_generations for _ in all_generations]
        print("Skipping reward scoring (--no-score flag set)")
    else:
        print("Scoring generations with accuracy_reward …")
        # Import reward function in a fresh context to avoid vLLM engine issues
        import sys
        sys.path.insert(0, "src")
        from open_r1.rewards import accuracy_reward

        for i, (gens, sol) in enumerate(zip(all_generations, all_solutions_out)):
            if i % 100 == 0:
                print(f"  Scoring {i}/{len(all_generations)}")
            if sol is None:
                all_rewards.append([None] * len(gens))
            else:
                try:
                    # accuracy_reward expects completions as list[list[dict]]
                    completions = [[{"content": g}] for g in gens]
                    rewards = accuracy_reward(completions, solution=[sol] * len(gens))
                    all_rewards.append(rewards)
                except Exception as e:
                    print(f"    Error scoring generation {i}: {e}")
                    all_rewards.append([None] * len(gens))

    # --- save ---
    rows = []
    for prompt, sol, gens, rews in zip(all_prompts_out, all_solutions_out, all_generations, all_rewards):
        rows.append({
            "prompt": prompt,
            "solution": sol,
            "pass_rate_generations": gens,
            "pass_rate_rewards": rews,
        })

    out_parquet = (
        args.out_parquet
        or f"gen_results_{args.dataset.replace('/', '_')}_{args.split}_g{args.num_generations}.parquet"
    )
    print("Writing parquet to", out_parquet)
    df = pd.DataFrame(rows)
    df.to_parquet(out_parquet, index=False)
    print("Done. Wrote:", out_parquet)
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows: {len(df)},  Generations per row: {args.num_generations}")


if __name__ == "__main__":
    main()
