#!/bin/bash
# ───── Slurm resource requests ─────────────────────────────────────────
#SBATCH --job-name=modal-eval-sweb     # appears in squeue
#SBATCH --partition=general         
#SBATCH --nodes=1
#SBATCH --gres=gpu:L40S:1    
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G                 
#SBATCH --time=0-48:00                  # D-HH:MM  
#SBATCH --output=/home/sophiapi/batch-logs/%x-%j.out # stdout  (%x=job-name  %j=job-ID)
#SBATCH --error=/home/sophiapi/batch-logs/%x-%j.err  # stderr   (can be same as output)
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sophiapi@andrew.cmu.edu

# ───── Define output jsonl path ──────────────────────────────────────
# OUTPUT_JSONL_PATH="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-3.5-turbo_maxiter_100_N_v0.43.0-no-hint-run_1_20250928_221252/output.jsonl"
OUTPUT_JSONL_PATH="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/gpt-3.5-turbo_maxiter_100_N_v0.43.0-no-hint-run_1_20250928_210601/output.jsonl"

# ───── Export any environment variables ───────────────────────────────
export CUDA_VISIBLE_DEVICES=0

export RUNTIME=remote
export SANDBOX_REMOTE_RUNTIME_API_URL="https://runtime.eval.all-hands.dev"
export EVAL_DOCKER_IMAGE_PREFIX="us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"
export LD_LIBRARY_PATH="/home/sophiapi/.cache/pypoetry/virtualenvs/openhands-ai-5xnPD_gY-py3.12/lib/python3.12/site-packages/cusparselt/lib:$LD_LIBRARY_PATH"

# Activate modal-eval-new conda environment
conda activate modal-eval-new

cd /home/sophiapi/model-routing/OpenHands-clean

# Run the command:
# conda run -n modal-eval-new evaluation/benchmarks/swe_bench/scripts/eval_infer.sh $OUTPUT_JSONL_PATH "" "princeton-nlp/SWE-bench_Verified" "test" modal
conda run -n modal-eval-new evaluation/benchmarks/swe_bench/scripts/eval_infer.sh $OUTPUT_JSONL_PATH "" "SWE-Gym/SWE-Gym" "train" modal