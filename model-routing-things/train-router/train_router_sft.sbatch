#!/bin/bash
# ───── Slurm resource requests ─────────────────────────────────────────
#SBATCH --job-name=1gpu-router-sft-train-model-5-instance-100-pruned-4-by-example        # appears in squeue
#SBATCH --partition=general         
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:1    
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                 
#SBATCH --time=0-48:00                  # D-HH:MM  
#SBATCH --output=/home/sophiapi/batch-logs/%x-%j.out # stdout  (%x=job-name  %j=job-ID)
#SBATCH --error=/home/sophiapi/batch-logs/%x-%j.err  # stderr   (can be same as output)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiapi@andrew.cmu.edu

# ───── Export any environment variables ───────────────────────────────
export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ───── Training configuration ─────────────────────────────────────────
MODEL_NAME="model-5-instance-100-pruned-4-with-ids-by-example"
TRAIN_FILE="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_train_model-5_instance-100_pruned-4_with-ids_by-example_sample400.jsonl"
VALID_FILE="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_valid_model-5_instance-100_pruned-4_with-ids_by-example_sample100.jsonl"
OUTPUT_DIR="/data/user_data/sophiapi/checkpoints/qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example"

# ───── Make the checkpoints directory if it doesn't exist ───────────────
ls /data/user_data/$USER
mkdir -p /data/user_data/$USER/checkpoints

# ───── Move to the directory that holds the code ─────────────────────
cd ~/model-routing/OpenHands

# ───── Activate Poetry environment and run training ───────────────────
# Training parameters:
# - batch-size 1: Conservative batch size for memory management
# - epochs 2: Standard fine-tuning duration
# - learning-rate 2e-5: Conservative learning rate for stable training
# - max-length 8192: Maximum sequence length (matches Qwen3-0.6B context window)
# - max-filter-tokens 32000: Coarse filter threshold for extremely long examples
poetry run python ~/model-routing/train_router_sft_fixed.py \
    --model-name "$MODEL_NAME" \
    --train-file "$TRAIN_FILE" \
    --valid-file "$VALID_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size 1 \
    --epochs 2 \
    --learning-rate 2e-5 \
    --max-length 8192 \
    --max-filter-tokens 32000

# ───── Post-run sync / cleanup (optional) ──────────────────────────────
# The model will be saved to checkpoints/qwen3_router_json
# wandb logs will be automatically uploaded 