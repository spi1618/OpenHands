#!/usr/bin/env python3
"""
Batch evaluation script for multiple models on SWE-Bench.
Usage: python evaluation/benchmarks/swe_bench/scripts/batch_evaluate_models.py --config models_config.yaml --output-dir evaluation/evaluation_outputs/datasets/
"""

import subprocess
import json
import random
# import pandas as pd
from pathlib import Path
import yaml
import os
import argparse
from datetime import datetime
from typing import List, Dict
import re

def run_inference(model_config: Dict, experiment_name: str, eval_limit: int, max_iter: int, num_workers: int = 1):
    """Run inference for a single model using run_infer.sh."""
    cmd = [
        "./evaluation/benchmarks/swe_bench/scripts/run_infer.sh",
        model_config["llm_config"],  # MODEL_CONFIG
        "HEAD",                      # COMMIT_HASH
        "CodeActAgent",              # AGENT
        str(eval_limit),             # EVAL_LIMIT
        str(max_iter),               # MAX_ITER
        str(num_workers),            # NUM_WORKERS
        "SWE-Gym/SWE-Gym",           # DATASET (default is SWE-Bench lite)
        "train",                     # SPLIT
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
    print(f"[INFO] Running inference for model: {model_config['name']}")
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
        print(f"[INFO] Inference completed for model: {model_config['name']}")
        print(f"[INFO] Results saved to: {output_file}")
        return output_file
    else:
        print(f"[WARNING] Could not find output file path in stdout for model {model_config['name']}")
        print(f"[INFO] Inference completed for model: {model_config['name']}")
        return None

def run_evaluation(output_file: str, model: Dict, num_workers: int = 1):
    """Run evaluation for a single model using eval_infer_remote.sh."""
    if not output_file or not os.path.exists(output_file):
        print(f"[ERROR] Output file {output_file} does not exist. Skipping evaluation for model {model['name']}")
        return None
    
    cmd = [
        "./evaluation/benchmarks/swe_bench/scripts/eval_infer_remote.sh",
        output_file,                 # INPUT_FILE
        str(num_workers),            # NUM_WORKERS
        "SWE-Gym/SWE-Gym",           # DATASET
        "train"                      # SPLIT
    ]
    env = os.environ.copy()
    env.update({
        "ALLHANDS_API_KEY": os.environ.get("ALLHANDS_API_KEY", ""),
        "RUNTIME": "remote",
        "SANDBOX_REMOTE_RUNTIME_API_URL": "https://runtime.eval.all-hands.dev",
        "EVAL_DOCKER_IMAGE_PREFIX": "us-central1-docker.pkg.dev/evaluation-092424/swe-bench-images"
    })
    print(f"[INFO] Running evaluation for model: {model['name']}")
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
    
    print(f"[INFO] Evaluation completed for model: {model['name']}")
    return process

def main():
    
    # # Load the verified instances  
    # swegym_verified_path = Path("/home/sophiapi/model-routing/OpenHands/evaluation/benchmarks/swe_bench/split/swegym_verified_instances.json")  
    # with open(swegym_verified_path, 'r') as f:  
    #     verified_instances = json.load(f)  
    
    # # Select 100 random instances  
    # random_instances = random.sample(verified_instances, 100)  
    # print("Selected SWE-Gym instances:")  
    # for instance_id in random_instances:  
    #     print(f"- {instance_id}")
        
    # # Write the selected instances to the config.toml file
    # config_path = Path("/home/sophiapi/model-routing/OpenHands/evaluation/benchmarks/swe_bench/config.toml")
    # with open(config_path, 'w') as f:
    #     f.write(f"selected_ids = {random_instances}")
    
    parser = argparse.ArgumentParser(description="Batch evaluate multiple models on SWE-Bench.")
    parser.add_argument("--config", required=True, help="Path to models_config.yaml")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--skip-evaluation", default=False, action="store_true", help="Skip evaluation step")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    models = config["models"]
    eval_limit = config.get("eval_limit", 100)
    max_iter = config.get("max_iter", 100)

    for model in models:
        try:
            # for i in range(len(random_instances)):
            timestamp = datetime.now().isoformat(timespec="seconds")
            print(f"[INFO] Running inference for model: {model['name']} at timestamp: {timestamp}\n")
            
            # Run inference
            output_file = run_inference(model, f"experiment_{timestamp}", eval_limit, max_iter, args.num_workers)
            # output_file = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-3-5-haiku-20241022_maxiter_100_N_v0.43.0-no-hint-run_1_20250801_042435/output.jsonl"
            
            # Run evaluation if not skipped
            if not args.skip_evaluation and output_file:
                print(f"[INFO] Running evaluation for model: {model['name']} at timestamp: {timestamp}\n")
                run_evaluation(output_file, model, args.num_workers)
            
        except Exception as e:
            print(f"[ERROR] Failed for model {model['name']}: {e}")


if __name__ == "__main__":
    main()
