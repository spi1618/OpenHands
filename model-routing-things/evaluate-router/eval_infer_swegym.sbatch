#!/bin/bash
# ───── Slurm resource requests ─────────────────────────────────────────
#SBATCH --job-name=eval-infer-swegym           # appears in squeue
#SBATCH --partition=general      
#SBATCH --nodes=1
#SBATCH --gres=gpu:1          
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-48:00                  # D-HH:MM   
#SBATCH --output=/home/sophiapi/batch-logs/%x-%j.out # stdout  (%x=job-name  %j=job-ID)
#SBATCH --error=/home/sophiapi/batch-logs/%x-%j.err  # stderr   (can be same as output)
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

# Move to the OpenHands directory
cd /home/sophiapi/model-routing/OpenHands
echo "=== Changed to directory: $(pwd) ==="

# Initialize conda and activate the mr-env environment
echo "=== Initializing conda ==="
source /home/sophiapi/.bashrc
conda activate mr-env
echo "=== Activated conda environment: mr-env ==="
echo "=== Python version: $(python3 --version) ==="
echo "=== Python path: $(which python3) ==="

# Check if the input file exists
INPUT_FILE="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/gpt-3.5-turbo_maxiter_100_N_v0.43.0-no-hint-run_1_20250915_125316/output.jsonl"
echo "=== Checking if input file exists ==="
if [ ! -f "$INPUT_FILE" ]; then
    echo "=== ERROR: Input file not found: $INPUT_FILE ==="
    exit 1
fi
echo "=== Input file found: $INPUT_FILE ==="

# Check if the eval_infer.py script exists
EVAL_SCRIPT="/home/sophiapi/model-routing/OpenHands/evaluation/benchmarks/swe_bench/eval_infer.py"
echo "=== Checking if eval script exists ==="
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "=== ERROR: Eval script not found: $EVAL_SCRIPT ==="
    exit 1
fi
echo "=== Eval script found: $EVAL_SCRIPT ==="

# Run the evaluation command
echo "=== Starting evaluation inference ==="
echo "=== Command: poetry run python evaluation/benchmarks/swe_bench/eval_infer.py --eval-num-workers 2 --input-file $INPUT_FILE --dataset SWE-Gym/SWE-Gym --split train ==="

poetry run python evaluation/benchmarks/swe_bench/eval_infer.py \
    --eval-num-workers 2 \
    --input-file "$INPUT_FILE" \
    --dataset SWE-Gym/SWE-Gym \
    --split train

EVAL_EXIT_CODE=$?

echo "=== Evaluation completed with exit code: $EVAL_EXIT_CODE ==="
echo "=== Job completed at $(date) ==="

exit $EVAL_EXIT_CODE
