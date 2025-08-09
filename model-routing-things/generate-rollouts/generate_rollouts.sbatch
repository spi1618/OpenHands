#!/bin/bash
# ───── Slurm resource requests ─────────────────────────────────────────
#SBATCH --job-name=haiku-rollouts-5           # appears in squeue
#SBATCH --partition=general             # up to 2 days; preempt for longer
#SBATCH --nodes=1
#SBATCH --gres=gpu:0               # no gpus needed for remote runtime
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=0-48:00                  # D-HH:MM   (48 h here)
#SBATCH --output=batch-logs/%x-%j.out # stdout  (%x=job-name  %j=job-ID)
#SBATCH --error=batch-logs/%x-%j.err  # stderr   (can be same as output)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiapi@andrew.cmu.edu

echo "=== Job started at $(date) ==="
echo "=== Current directory: $(pwd) ==="
echo "=== User: $(whoami) ==="
echo "=== Hostname: $(hostname) ==="

# Export environment variables
export RUNTIME=remote
export SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev"
export EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"

# Load environment variables
source /home/sophiapi/.env
echo "=== Environment loaded ==="
echo "=== ALLHANDS_API_KEY: ${ALLHANDS_API_KEY:0:10}... ==="

# Move to the directory
cd ~/model-routing/OpenHands
echo "=== Changed to directory: $(pwd) ==="
echo "=== Python version: $(python3 --version) ==="

# Check if the script exists
echo "=== Checking if batch script exists ==="
ls -la evaluation/benchmarks/swe_bench/scripts/batch_evaluate_models.py

# Check if config exists
echo "=== Checking if config exists ==="
ls -la evaluation/benchmarks/swe_bench/configs/models_config.yaml

# Run the Python script with unbuffered output
echo "=== Starting batch evaluation ==="
python3 -u evaluation/benchmarks/swe_bench/scripts/batch_evaluate_models.py \
  --config evaluation/benchmarks/swe_bench/configs/models_config.yaml \
  --num-workers 2

echo "=== Job completed at $(date) ==="

