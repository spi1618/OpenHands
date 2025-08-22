#!/usr/bin/env python3
"""
Batch evaluation script for router on SWE-Bench_Verified test set.
Usage: python evaluation/benchmarks/swe_bench/scripts/evaluate_router.py --num-workers 2
"""

import subprocess
import json
import random
from pathlib import Path
import yaml
import os
import argparse
from datetime import datetime
from typing import List, Dict
import re
import requests
import time

def check_router_health(router_url: str) -> bool:
    """Check if the router server is running and healthy."""
    try:
        response = requests.get(f"{router_url}/health", timeout=10)
        return response.status_code == 200
    except requests.RequestException:
        return False

def start_router_server(router_dir: str, api_key: str, random_mode: bool = False) -> subprocess.Popen:
    """Start the router server in the background."""
    env = os.environ.copy()
    env["LITELLM_API_KEY"] = api_key
    
    # Pass through the ROUTER_CHECKPOINT environment variable
    router_checkpoint = os.environ.get("ROUTER_CHECKPOINT")
    if router_checkpoint:
        env["ROUTER_CHECKPOINT"] = router_checkpoint
        print(f"[INFO] Passing ROUTER_CHECKPOINT={router_checkpoint} to router server")
    
    # Pass through the RANDOM_MODE environment variable
    if random_mode:
        env["RANDOM_MODE"] = "true"
        print(f"[INFO] Passing RANDOM_MODE=true to router server")
    
    cmd = [
        "python3", "swe_bench_router.py"
    ]
    
    print(f"[INFO] Starting router server in {router_dir}")
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        cwd=router_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Wait a bit for the server to start
    time.sleep(5)
    
    return process

def run_router_inference(eval_limit: int, max_iter: int, num_workers: int = 1):
    """Run inference using the router."""
    # [ignore] Use the router LLM config from the model-routing directory
    # router_llm_config = "../router_llm_config.toml"
    
    # This function calls run_infer.sh, which calls run_infer.py
    # The actual router API call (/v1/chat/completions) happens in run_infer.py
    # OpenHands uses the router as a standard LLM endpoint via /v1/chat/completions
    
    cmd = [
        "./evaluation/benchmarks/swe_bench/scripts/run_infer.sh",
        "llm.router",           # MODEL_CONFIG (router config)
        "HEAD",                      # COMMIT_HASH
        "CodeActAgent",              # AGENT
        str(eval_limit),             # EVAL_LIMIT
        str(max_iter),               # MAX_ITER
        str(num_workers),            # NUM_WORKERS
        "princeton-nlp/SWE-bench_Verified",   # DATASET
        "test",                      # SPLIT
        "1",                         # N_RUNS (default is 1)
        "swe"                        # MODE (default is swe)
    ]
    env = os.environ.copy()
    env.update({
        "ALLHANDS_API_KEY": os.environ.get("ALLHANDS_API_KEY", ""),
        "RUNTIME": "remote",
        "SANDBOX_REMOTE_RUNTIME_API_URL": "https://runtime.eval.all-hands.dev",
        "EVAL_DOCKER_IMAGE_PREFIX": "us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"
    })
    print(f"[INFO] Running router inference")
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    # Run the command and show output in real-time while capturing it
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Collect output while showing it in real-time
    stdout_lines = []
    for line in process.stdout:
        print(line.rstrip())  # Print to terminal in real-time
        stdout_lines.append(line)  # Collect for processing
    
    # Wait for process to complete
    process.wait()
    result = subprocess.CompletedProcess(
        args=cmd,
        returncode=process.returncode,
        stdout=''.join(stdout_lines),
        stderr=''
    )
    
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
    
    # Extract the output file path from stdout
    match = re.search(r'### OUTPUT FILE: (.+) ###', result.stdout)
    if match:
        output_file = match.group(1)
        print(f"[INFO] Router inference completed")
        print(f"[INFO] Results saved to: {output_file}")
        return output_file
    else:
        print(f"[WARNING] Could not find output file path in stdout for router")
        print(f"[INFO] Router inference completed")
        return None

def run_router_evaluation(output_file: str, num_workers: int = 1):
    """Run evaluation for router results."""
    if not output_file or not os.path.exists(output_file):
        print(f"[ERROR] Output file {output_file} does not exist. Skipping evaluation for router")
        return None
    
    cmd = [
        "./evaluation/benchmarks/swe_bench/scripts/eval_infer_remote.sh",
        output_file,                 # INPUT_FILE
        str(num_workers),            # NUM_WORKERS
        "princeton-nlp/SWE-bench_Verified",   # DATASET
        "test"                      # SPLIT
    ]
    env = os.environ.copy()
    env.update({
        "ALLHANDS_API_KEY": os.environ.get("ALLHANDS_API_KEY", ""),
        "RUNTIME": "remote",
        "SANDBOX_REMOTE_RUNTIME_API_URL": "https://runtime.eval.all-hands.dev",
        "EVAL_DOCKER_IMAGE_PREFIX": "us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"
    })
    print(f"[INFO] Running router evaluation")
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    # Run the command and show output in real-time
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Show output in real-time
    for line in process.stdout:
        print(line.rstrip())
    
    # Wait for process to complete
    process.wait()
    
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, cmd)
    
    print(f"[INFO] Router evaluation completed")
    return process

def analyze_router_decisions(output_file: str):
    """Analyze router decisions from the output file."""
    if not output_file or not os.path.exists(output_file):
        print(f"[ERROR] Output file {output_file} does not exist. Cannot analyze router decisions.")
        return
    
    print(f"[INFO] Analyzing router decisions from {output_file}")
    
    model_counts = {}
    total_steps = 0
    
    with open(output_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # Look for router decisions in the trajectory
                if 'trajectory' in data:
                    for step in data['trajectory']:
                        if 'model_used' in step:
                            model = step['model_used']
                            model_counts[model] = model_counts.get(model, 0) + 1
                            total_steps += 1
            except json.JSONDecodeError:
                continue
    
    print(f"[INFO] Router Decision Analysis:")
    print(f"  Total steps: {total_steps}")
    for model, count in sorted(model_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_steps * 100) if total_steps > 0 else 0
        print(f"  {model}: {count} times ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Evaluate router on SWE-Bench Verified.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--eval-limit", type=int, default=100, help="Number of instances to evaluate")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum iterations per instance")
    parser.add_argument("--skip-evaluation", default=False, action="store_true", help="Skip evaluation step")
    parser.add_argument("--start-router", default=False, action="store_true", help="Start router server automatically")
    parser.add_argument("--router-dir", default="/home/sophiapi/model-routing", help="Directory containing router server")
    parser.add_argument("--analyze-decisions", default=False, action="store_true", help="Analyze router decisions after inference")
    parser.add_argument("--router-url", default="http://localhost:8123", help="URL of the router server")
    parser.add_argument("--random-mode", default=False, action="store_true", help="Enable random mode for router server")
    args = parser.parse_args()
    
    router_process = None
    
    try:
        # Check if router is running
        if not check_router_health(args.router_url):
            if args.start_router:
                print(f"[INFO] Router not running at {args.router_url}. Starting router server...")
                api_key = os.environ.get("LITELLM_API_KEY")
                if not api_key:
                    raise RuntimeError("LITELLM_API_KEY environment variable is required to start router server")
                
                router_process = start_router_server(args.router_dir, api_key, args.random_mode)
                
                # Wait for router to be ready
                max_retries = 10
                for i in range(max_retries):
                    if check_router_health(args.router_url):
                        print(f"[INFO] Router server is ready at {args.router_url}")
                        break
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Router server failed to start within {max_retries * 2} seconds")
            else:
                raise RuntimeError(f"Router server is not running at {args.router_url}. Use --start-router to start it automatically.")
        else:
            print(f"[INFO] Router server is already running at {args.router_url}")
        
        timestamp = datetime.now().isoformat(timespec="seconds")
        print(f"[INFO] Running router inference at timestamp: {timestamp}\n")
        
        # Run inference with router
        output_file = run_router_inference(args.eval_limit, args.max_iter, args.num_workers)
        
        # Analyze router decisions if requested
        if args.analyze_decisions and output_file:
            analyze_router_decisions(output_file)
        
        # Run evaluation if not skipped
        if not args.skip_evaluation and output_file:
            print(f"[INFO] Running router evaluation at timestamp: {timestamp}\n")
            run_router_evaluation(output_file, args.num_workers)
        
    except Exception as e:
        print(f"[ERROR] Failed for router: {e}")
        raise
    finally:
        # Clean up router process if we started it
        if router_process:
            print(f"[INFO] Stopping router server...")
            router_process.terminate()
            try:
                router_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                router_process.kill()
            print(f"[INFO] Router server stopped")

if __name__ == "__main__":
    main()
