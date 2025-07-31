#!/usr/bin/env python3
"""
Script to convert multiple output jsonl files to a single step-wise jsonl file (where each line is a step in the trajectory).
Usage: python evaluation/benchmarks/swe_bench/scripts/traj_to_steps.py --output-dir evaluation/evaluation_outputs/datasets/
"""

### WARNING: THIS IS HARDCODED TO THE OUTPUT JSONL FILES
### WARNING: ALSO, THE MODEL NAMING DEPENDS ON THE ORDER OF THE OUTPUT JSONL FILES - BE CAREFUL, SORRY

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
                    "successfully_patched": successfully_patched,
                    "partial_trajectory": partial_trajectory
                }) + '\n')

def main():
    parser = argparse.ArgumentParser(description="Create step-wise trajectory jsonl dataset from output jsonl files of multiple models on SWE-Bench.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the step-wise trajectory jsonl file")
    args = parser.parse_args()

    all_trajectories = []
    timestamp = datetime.now().isoformat(timespec="seconds")
    
    output_dirs = [
        "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-3-5-haiku-20241022_maxiter_100_N_v0.43.0-no-hint-experiment_2025-07-05T19:07:17-run_1/output.jsonl",
        "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/claude-sonnet-4-20250514_maxiter_100_N_v0.43.0-no-hint-experiment_2025-07-05T21:27:57-run_1/output.jsonl",
        "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/deepseek-v3_maxiter_100_N_v0.43.0-no-hint-experiment_2025-07-05T22:21:10-run_1/output.jsonl",
        "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/devstral-small-2505_maxiter_100_N_v0.43.0-no-hint-experiment_2025-07-05T17:12:16-run_1/output.jsonl"
    ]
    i = 0

    # this basically copies the output jsonl files of each model, combines them into a all_trajectories list, 
    #     and then calls extract_stepwise_trajectories_to_jsonl to save the step-wise trajectories to a single jsonl file
    #     seems like a lot of unnecessary copying, but it's basically one time use so it's fine
    for model in ["claude-3-5-haiku", "claude-sonnet-4", "deepseek-v3","devstral-small"]:
        try:
            output_jsonl = output_dirs[i]
            i += 1
            with open(output_jsonl, 'r') as f:
                for line in f:
                    result = json.loads(line)
                    trajectory_record = {
                        # "instance_id": result.get("instance_id"),
                        "model": model,
                        "successfully_patched": result.get("report", {}).get("resolved", False),
                        # "instruction": result.get("instruction"), 
                        "history": result.get("history"),
                        # "git_patch": result.get("test_result", {}).get("git_patch", ""), # irrelevant, gives away final answer
                        # "error": result.get("error"), # irrelevant, only stores the most recent error
                        # "metrics": result.get("metrics"), # irrelevant
                        # "metadata": result.get("metadata"), # irrelevant
                    }
                    all_trajectories.append(trajectory_record)
            print(f"[INFO] Trajectories for model {model} extracted.")
        except Exception as e:
            print(f"[ERROR] Failed for model {model}: {e}")

    # Save all steps to a single JSONL file (one line per step)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stepwise_traj_jsonl = output_dir / f"model-5_pruned-4_swe_gym_stepwise_trajectories_{timestamp.replace(':', '-')}.jsonl"
    extract_stepwise_trajectories_to_jsonl(all_trajectories, stepwise_traj_jsonl)
    print(f"[INFO] All step-wise trajectories saved to {stepwise_traj_jsonl}")

if __name__ == "__main__":
    main()
