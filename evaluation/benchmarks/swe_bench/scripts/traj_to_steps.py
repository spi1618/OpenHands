#!/usr/bin/env python3
"""
Script to convert multiple output jsonl files to a single step-wise jsonl file (where each line is a step in the trajectory).
Usage: python evaluation/benchmarks/swe_bench/scripts/traj_to_steps.py --output-dir evaluation/evaluation_outputs/datasets/
"""

### WARNING: THIS IS HARDCODED TO THE OUTPUT JSONL FILES

import json
from pathlib import Path
import argparse
from datetime import datetime

def extract_stepwise_trajectories_to_jsonl(model_results, output_file: Path):
    """
    For each trajectory, break it into partial trajectories as described:
    - 0th step: all jsons up to (not including) the second "agent" json.
    - 1st step: all jsons after 0th step up to and including the first json with "observation".
    - 2nd step: all jsons after 1st step up to and including the second json with "observation".
    - ...
    - Last step: all jsons remaining in history.
    
    Each partial trajectory uses a sliding window approach with at most 4 steps:
    - First 4 partial trajectories: include all steps up to that point (1, 2, 3, 4 steps respectively)
    - Subsequent partial trajectories: include only the last 4 steps using a sliding window
    """
    with open(output_file, 'w') as out_f:
        for record in model_results:
            model = record.get("model")
            successfully_patched = record.get("successfully_patched", False)
            history = record.get("history", [])

            # Skip failed instances where history is None or not a list
            if history is None or not isinstance(history, list):
                print(f"[WARNING] Skipping instance {record.get('instance_id', 'unknown')} for model {model} - history is {type(history)}")
                continue

            # Find indices for step boundaries
            agent_indices = [i for i, h in enumerate(history) if h.get("source") == "agent"]
            obs_indices = [i for i, h in enumerate(history) if h.get("source") == "agent" and "observation" in h]

            step_boundaries = []

            # 0th step: up to (not including) the second "agent" json
            if len(agent_indices) >= 2:
                step_boundaries.append(agent_indices[1])
            else:
                step_boundaries.append(len(history))

            # Steps for each "observation"
            for obs_idx in obs_indices:
                step_boundaries.append(obs_idx + 1)

            # Ensure the last step is the full history
            if step_boundaries[-1] < len(history):
                step_boundaries.append(len(history))

            # Remove duplicates and sort
            step_boundaries = sorted(set(step_boundaries))

            # Create sliding window partial trajectories with at most 4 steps each
            for i in range(len(step_boundaries)):
                # For each step boundary, create a partial trajectory that includes
                # at most 4 steps using a sliding window approach
                start_step_idx = max(0, i - 3)  # Start from 3 steps back, but not before 0
                end_step_idx = i + 1
                
                # Get the history up to the start boundary of the current window
                if start_step_idx > 0:
                    start_boundary = step_boundaries[start_step_idx - 1]
                else:
                    start_boundary = 0
                
                # Get the history up to the end boundary of the current step
                end_boundary = step_boundaries[i]
                partial_trajectory = history[start_boundary:end_boundary]
                
                out_f.write(json.dumps({
                    "model": model,
                    "instance_id": record.get("instance_id"),
                    "successfully_patched": successfully_patched,
                    "partial_trajectory": partial_trajectory
                }) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Create step-wise trajectory jsonl dataset from output jsonl files of multiple models on SWE-Bench.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the step-wise trajectory jsonl file")
    args = parser.parse_args()

    all_trajectories = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    
    # Model to output directories mapping - allows multiple directories per model
    model_output_dirs = {
        "claude-3-5-haiku": [
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-3-5-haiku-20241022_maxiter_100_N_v0.43.0-no-hint-run_1_20250726_182956/output.jsonl",
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-3-5-haiku-20241022_maxiter_100_N_v0.43.0-no-hint-run_1_20250801_000045/output.jsonl",
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-3-5-haiku-20241022_maxiter_100_N_v0.43.0-no-hint-run_1_20250801_022525/output.jsonl",
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-3-5-haiku-20241022_maxiter_100_N_v0.43.0-no-hint-run_1_20250801_042435/output.jsonl",
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-3-5-haiku-20241022_maxiter_100_N_v0.43.0-no-hint-run_1_20250804_231650/output.jsonl",
        ],
        "claude-sonnet-4": [
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-sonnet-4-20250514_maxiter_100_N_v0.43.0-no-hint-run_1_20250725_232903/output.jsonl",
        ],
        "deepseek-v3": [
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/deepseek-v3_maxiter_100_N_v0.43.0-no-hint-run_1_20250726_180617/output.jsonl",
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/deepseek-v3_maxiter_100_N_v0.43.0-no-hint-run_1_20250731_170336/output.jsonl",
        ],
        "devstral-small": [
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/devstral-small-2505_maxiter_100_N_v0.43.0-no-hint-run_1_20250731_170650/output.jsonl"
        ],
        "kimi-k2": [
            "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/kimi-k2-0711-preview_maxiter_100_N_v0.43.0-no-hint-run_1_20250731_170858/output.jsonl"
        ]
    }

    # Process each model and its associated output directories
    for model, output_dirs in model_output_dirs.items():
        for output_jsonl in output_dirs:
            try:
                with open(output_jsonl, 'r') as f:
                    for line in f:
                        result = json.loads(line)
                        trajectory_record = {
                            "model": model,
                            "instance_id": result.get("instance_id"),
                            "successfully_patched": result.get("report", {}).get("resolved", False),
                            # "instruction": result.get("instruction"), 
                            "history": result.get("history"),
                            # "git_patch": result.get("test_result", {}).get("git_patch", ""), # irrelevant, gives away final answer
                            # "error": result.get("error"), # irrelevant, only stores the most recent error
                            # "metrics": result.get("metrics"), # irrelevant
                            # "metadata": result.get("metadata"), # irrelevant
                        }
                        all_trajectories.append(trajectory_record)
                print(f"[INFO] Trajectories for model {model} from {output_jsonl} extracted.")
            except Exception as e:
                print(f"[ERROR] Failed for model {model} from {output_jsonl}: {e}")

    # Save all steps to a single JSONL file (one line per step)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # HARDCODED
    stepwise_traj_jsonl = output_dir / f"model-5_instance-100_pruned-4_with-ids_swe_gym_stepwise_trajectories_{timestamp.replace(':', '-')}.jsonl"
    extract_stepwise_trajectories_to_jsonl(all_trajectories, stepwise_traj_jsonl)
    print(f"[INFO] All step-wise trajectories saved to {stepwise_traj_jsonl}")

if __name__ == "__main__":
    main()
