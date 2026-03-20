#!/bin/bash
#SBATCH --job-name=gen_local
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu_a100_short
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --time=00:30:00

module load devel/cuda/12.8
set -x -e

cd $HOME/open-r1
source openr1/bin/activate

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CUDA_VISIBLE_DEVICES=0

# Configuration (edit if needed)
#MODEL_DIR="/pfs/work9/workspace/scratch/ma_llincke-Results_Qwen2.5-1.5B-Instruct/GRPO-MATH12k-0313-constlr/checkpoint-1350"
MODEL_DIR="Qwen/Qwen2.5-1.5B-Instruct" #Baseline
DATASET="hiyouga/math12k"
SPLIT="test"
NUM_GENERATIONS=16
#OUT_PARQUET="gen_results_qwen1.5B_grpo_clr_0313_1350_g${NUM_GENERATIONS}.parquet"
OUT_PARQUET="gen_results_qwen1.5B_baseline_g${NUM_GENERATIONS}.parquet"
BATCH_SIZE=16
MAX_TOKENS=2048  # increase for reasoning models that need longer outputs
TEMPERATURE=0.3
TOP_P=0.95
SYSTEM_PROMPT="You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"

# Step 1 – generate completions without scoring (vLLM only, no dependencies)
python scripts/generate_local.py \
  --model-dir "$MODEL_DIR" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --num-generations $NUM_GENERATIONS \
  --batch-size $BATCH_SIZE \
  --max-tokens $MAX_TOKENS \
  --out-parquet "$OUT_PARQUET" \
  --temperature $TEMPERATURE \
  --top-p $TOP_P \
  --no-score \
  --system-prompt "$SYSTEM_PROMPT"

echo "Wrote parquet: $OUT_PARQUET"

# Step 1b – score the generations (in separate process, after vLLM is shut down)
python3 << SCORE_EOF
import sys
import json
sys.path.insert(0, "src")
import pandas as pd
from open_r1.rewards import accuracy_reward

df = pd.read_parquet("$OUT_PARQUET")
print(f"Scoring {len(df)} examples...")

all_rewards = []
parse_errors = []
parse_error_count = 0

for idx, row in df.iterrows():
    if idx % 50 == 0:
        print(f"  Scoring {idx}/{len(df)}")
    sol = row["solution"]
    gens = row["pass_rate_generations"]
    if sol is None:
        all_rewards.append([None] * len(gens))
    else:
        try:
            completions = [[{"content": g}] for g in gens]
            rewards = accuracy_reward(completions, solution=[sol] * len(gens))
            all_rewards.append(rewards)
        except Exception as e:
            error_str = str(e)
            print(f"    Error at row {idx}: {e}")
            # Track parsing errors
            if "parse" in error_str.lower():
                parse_error_count += 1
                parse_errors.append({
                    "idx": idx,
                    "solution": str(sol)[:100],  # store first 100 chars
                    "error": error_str[:100]
                })
            all_rewards.append([None] * len(gens))

df["pass_rate_rewards"] = all_rewards
df.to_parquet("$OUT_PARQUET", index=False)

# Save error statistics
error_stats = {
    "total_parse_errors": parse_error_count,
    "total_examples": len(df),
    "parse_error_rate": parse_error_count / len(df) if len(df) > 0 else 0,
    "parse_error_examples": parse_errors[:20]  # save first 20 examples
}
with open("${OUT_PARQUET%.parquet}_scoring_errors.json", "w") as f:
    json.dump(error_stats, f, indent=2)

print(f"Wrote rewards to $OUT_PARQUET")
print(f"Parse errors: {parse_error_count}/{len(df)} ({100*parse_error_count/len(df):.1f}%)")
SCORE_EOF

# Step 2 – compute pass@k from the rewards stored in the parquet
# Create metadata file with generation parameters and error stats
METADATA_FILE="${OUT_PARQUET%.parquet}_metadata.json"
ERROR_FILE="${OUT_PARQUET%.parquet}_scoring_errors.json"
python3 << METADATA_EOF
import json

# Load error stats if they exist
error_stats = {}
try:
    with open("$ERROR_FILE", "r") as f:
        error_stats = json.load(f)
except FileNotFoundError:
    pass

metadata = {
    "model_dir": "$MODEL_DIR",
    "dataset": "$DATASET",
    "split": "$SPLIT",
    "num_generations": $NUM_GENERATIONS,
    "batch_size": $BATCH_SIZE,
    "max_tokens": $MAX_TOKENS,
    "temperature": $TEMPERATURE,
    "top_p": $TOP_P,
    "system_prompt": "$SYSTEM_PROMPT",
    "scoring_errors": error_stats
}
with open("$METADATA_FILE", "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Wrote metadata with error stats to $METADATA_FILE")
METADATA_EOF

# Run compute_pass_at_k with metadata
python scripts/eval/compute_pass_at_k.py \
  --parquet-file "$OUT_PARQUET" \
  --reward-field pass_rate_rewards \
  --ks 1 2 4 8 16 \
  --metadata "$METADATA_FILE" \
  --output "pass_at_k_results_${OUT_PARQUET%.parquet}.json"
  #--wandb-project "open-r1-pass@k" \
  #--wandb-run "Gwen2.5-1.5B-Instruct-grpo-math12k-0310"

echo "Pass@k results written to pass_at_k_results_${OUT_PARQUET%.parquet}.json. Job finished."