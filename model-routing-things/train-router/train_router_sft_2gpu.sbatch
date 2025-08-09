#!/bin/bash
# ───── Slurm resource requests ─────────────────────────────────────────
#SBATCH --job-name=2gpu-router-sft-train-model-5-instance-100-pruned-4-by-example        # appears in squeue
#SBATCH --partition=general         
#SBATCH --nodes=1
#SBATCH --gres=gpu:A100_80GB:2    # Use 2 GPUs instead of 1
#SBATCH --cpus-per-task=16        # Increase CPUs for multi-GPU
#SBATCH --mem=256G                # Increase memory for multi-GPU
#SBATCH --time=0-48:00                  # D-HH:MM  
#SBATCH --output=/home/sophiapi/batch-logs/%x-%j.out # stdout  (%x=job-name  %j=job-ID)
#SBATCH --error=/home/sophiapi/batch-logs/%x-%j.err  # stderr   (can be same as output)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiapi@andrew.cmu.edu

# ───── Export any environment variables ───────────────────────────────
export CUDA_VISIBLE_DEVICES=0,1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ───── Training configuration ─────────────────────────────────────────
MODEL_NAME="model-5-instance-100-pruned-4-with-ids-by-example"
TRAIN_FILE="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_train_model-5_instance-100_pruned-4_with-ids_by-example.jsonl"
VALID_FILE="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_valid_model-5_instance-100_pruned-4_with-ids_by-example.jsonl"
OUTPUT_DIR="/data/user_data/sophiapi/checkpoints/qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example"

# ───── Make the checkpoints directory if it doesn't exist ───────────────
ls /data/user_data/$USER
mkdir -p /data/user_data/$USER/checkpoints

# ───── Move to the directory that holds the code ─────────────────────
cd ~/model-routing/OpenHands

# ───── Activate Poetry environment and run training ───────────────────
# Use torchrun for distributed training
torchrun --nproc_per_node=2 --master_port=29500 ~/model-routing/train_router_sft.py \
    --model-name "$MODEL_NAME" \
    --train-file "$TRAIN_FILE" \
    --valid-file "$VALID_FILE" \
    --output-dir "$OUTPUT_DIR"

# ───── Post-run sync / cleanup (optional) ──────────────────────────────
# The model will be saved to checkpoints/qwen3_router_json
# wandb logs will be automatically uploaded 