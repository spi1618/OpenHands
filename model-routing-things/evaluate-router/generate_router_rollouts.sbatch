#!/bin/bash
# ───── Slurm resource requests ─────────────────────────────────────────
#SBATCH --job-name=router-rollout           # appears in squeue
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
echo "=== LITELLM_API_KEY: ${LITELLM_API_KEY:0:10}... ==="

# Move to the directory
cd ~/model-routing/OpenHands
echo "=== Changed to directory: $(pwd) ==="
echo "=== Python version: $(python3 --version) ==="

# Check if the script exists
echo "=== Checking if batch script exists ==="
ls -la evaluation/benchmarks/swe_bench/scripts/evaluate_router.py

# Configuration (can be overridden by environment variables)
EVAL_LIMIT=${EVAL_LIMIT:-100}
MAX_ITER=${MAX_ITER:-100}
NUM_WORKERS=${NUM_WORKERS:-1}
SKIP_EVAL=${SKIP_EVAL:-false}
START_ROUTER=${START_ROUTER:-false}
ANALYZE_DECISIONS=${ANALYZE_DECISIONS:-true}
ROUTER_URL=${ROUTER_URL:-"http://localhost:8000"}
ROUTER_CHECKPOINT=${ROUTER_CHECKPOINT:-"checkpoint-11500"}

echo "=== Configuration ==="
echo "  Eval Limit: ${EVAL_LIMIT}"
echo "  Max Iterations: ${MAX_ITER}"
echo "  Num Workers: ${NUM_WORKERS}"
echo "  Skip Evaluation: ${SKIP_EVAL}"
echo "  Start Router: ${START_ROUTER}"
echo "  Analyze Decisions: ${ANALYZE_DECISIONS}"
echo "  Router URL: ${ROUTER_URL}"
echo "  Router Checkpoint: ${ROUTER_CHECKPOINT}"

# Build command arguments
CMD_ARGS=(
    "python3" "-u" "evaluation/benchmarks/swe_bench/scripts/evaluate_router.py"
    "--eval-limit" "${EVAL_LIMIT}"
    "--max-iter" "${MAX_ITER}"
    "--num-workers" "${NUM_WORKERS}"
    "--router-url" "${ROUTER_URL}"
)

if [ "$SKIP_EVAL" = "true" ]; then
    CMD_ARGS+=("--skip-evaluation")
fi

if [ "$START_ROUTER" = "true" ]; then
    CMD_ARGS+=("--start-router")
fi

if [ "$ANALYZE_DECISIONS" = "true" ]; then
    CMD_ARGS+=("--analyze-decisions")
fi

# Run the Python script with unbuffered output
echo "=== Starting router evaluation ==="
echo "=== Command: ${CMD_ARGS[*]} ==="
echo ""

"${CMD_ARGS[@]}"

echo "=== Job completed at $(date) ==="

