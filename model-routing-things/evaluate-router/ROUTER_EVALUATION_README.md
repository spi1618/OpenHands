# Router Evaluation for SWE-Bench

This directory contains scripts to automate router-based evaluation on SWE-Bench, eliminating the need for manual terminal management.

## Quick Start

### 1. Set up environment variables
```bash
# In your ~/.env file or export directly
export LITELLM_API_KEY="sk-..."  # Your LiteLLM API key
export ALLHANDS_API_KEY="..."    # Your AllHands API key
```

### 2. Run router evaluation
```bash
# From the model-routing directory
cd ~/model-routing

# Configure parameters (see "Configuration" below)

# Submit batch job
sbatch batch-scripts/generate_router_rollouts.sbatch
```

## Configuration

You can customize the evaluation by setting environment variables:

```bash
# Number of instances to evaluate (default: 100)
export EVAL_LIMIT=50

# Maximum iterations per instance (default: 100)
export MAX_ITER=50

# Number of parallel workers (default: 2)
export NUM_WORKERS=4

# Skip evaluation step (default: false)
export SKIP_EVAL=false

# Start router server automatically (default: true)
export START_ROUTER=true

# Analyze router decisions (default: true)
export ANALYZE_DECISIONS=true

# Router server URL (default: http://localhost:8000)
export ROUTER_URL="http://babel-2-25:8000"  # Use actual hostname in cluster
```

## Cluster Usage

For cluster environments where the router runs on a different node:

### 1. Start router server on a compute node
```bash
# Get interactive node for router
salloc -p debug --gres=gpu:1 --time=0-2:00 --mem=64G
srun --pty bash -l

# Start router (note the hostname)
cd ~/model-routing
export LITELLM_API_KEY="sk-..."
# Export other environment variables...
python3 swe_bench_router.py
# Note: Server starts on http://babel-2-25:8000 (or whatever hostname)
```

## 2. Test Router Connectivity (optional)
```bash
# 1. Test health endpoint
curl http://YOUR_HOSTNAME:8000/health
# Example: curl http://babel-0-23:8000/health

# 2. Test chat endpoint
curl -X POST http://YOUR_HOSTNAME:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
# Example: curl -X POST http://babel-0-23:8000/v1/chat/completions ...
```

### 3. Update OpenHands config
```bash
# Update the OpenHands config with the actual hostname
cd ~/model-routing/OpenHands
sed -i "/\[llm\.router\]/,/^\[/ s/base_url = \"http:\/\/[^:]*:8000\"/base_url = \"http:\/\/babel-2-25:8000\"/" config.toml
```

### 4. Run evaluation with correct router URL
```bash
# Export environment variables...

# Set the router URL to match your router server
export ROUTER_URL="http://babel-2-25:8000"

# Configure parameters (see "Configuration" below)

# Submit batch job
sbatch batch-scripts/generate_router_rollouts.sbatch
```

## Manual Usage

If you prefer to run manually instead of using the batch script:

```bash
cd ~/model-routing/OpenHands

# Run with default settings
python3 evaluation/benchmarks/swe_bench/scripts/evaluate_router.py --num-workers 2

# Run with custom settings
python3 evaluation/benchmarks/swe_bench/scripts/evaluate_router.py \
  --eval-limit 50 \
  --max-iter 50 \
  --num-workers 4 \
  --router-url "http://babel-2-25:8000" \
  --start-router \
  --analyze-decisions
```

## What the Script Does

1. **Checks router health** - Verifies the router server is running
2. **Starts router** (if needed) - Automatically starts the router server
3. **Runs inference** - Executes SWE-Bench tasks using the router
4. **Analyzes decisions** - Shows which models the router selected
5. **Runs evaluation** - Evaluates the generated patches
6. **Cleans up** - Stops the router server if it was started

## Output Files

- **Inference results**: `evaluation/evaluation_outputs/outputs/.../output.jsonl`
- **Evaluation results**: `evaluation/evaluation_outputs/outputs/.../output.swebench_eval.jsonl`
- **Batch logs**: `batch-logs/router-rollout-<job-id>.out`

## Troubleshooting

### Router server not starting
- Check that `LITELLM_API_KEY` is set correctly
- Verify the router model path exists: `/home/sophiapi/model-routing/OpenHands/checkpoints/qwen3_router_json`

### Network connectivity issues
- Set `ROUTER_URL` to the correct hostname (e.g., `http://babel-2-25:8000`)
- Update `config.toml` with the correct hostname
- Ensure the router server is running on the specified hostname

### Missing dependencies
- Ensure you're in the correct conda environment
- Check that all required packages are installed

## Files

- `evaluate_router.py` - Main evaluation script
- `generate_router_rollouts.sbatch` - SLURM batch script
- `config.toml` - OpenHands configuration with router settings
- `swe_bench_router.py` - Router server implementation
- `router_inference.py` - Router model inference logic 