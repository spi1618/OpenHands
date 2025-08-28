#!/usr/bin/env python3
"""
Script to convert multiple output jsonl files to a single step-wise jsonl file (where each line is a step in the trajectory).
Usage: python3 evaluation/benchmarks/swe_bench/scripts/traj_to_steps_cleaned.py --output-dir evaluation/evaluation_outputs/datasets/
"""

### WARNING: THIS IS HARDCODED TO THE OUTPUT JSONL FILES

import json
from pathlib import Path
import argparse
from datetime import datetime

def extract_partial_trajectory_from_history(history, event_number) -> list[dict]:
    """
    Extracts a partial trajectory of the first event_number events from the history, where each step is a dictionary
    """
    partial_trajectory = []
    current_event = 1
    while current_event < event_number:
        event = history[current_event]
        cleaned_event = {}
        # Extract the source of the event
        source = event.get("source") # "user", "environment", "agent"
        cleaned_event["source"] = source
        # If the source is "agent" and the "args" field exists (usually indicating the event is an agent action) has a "thought" field, extract the thought
        if source == "agent" and "args" in event and "thought" in event["args"]:
            thought = event["args"]["thought"]
            cleaned_event["thought"] = thought
        # Extract the message of the event
        message = event.get("message")
        cleaned_event["message"] = message
        # Add the cleaned event to the partial trajectory
        partial_trajectory.append(cleaned_event)
        current_event += 1
    return partial_trajectory
        

def main():
    parser = argparse.ArgumentParser(description="Create cleaned partial trajectory jsonl dataset from output jsonl files of multiple models on SWE-Bench.")
    parser.add_argument("--output-dir", required=True, help="Directory to save the cleaned partial trajectory jsonl file")
    args = parser.parse_args()

    timestamp = datetime.now().isoformat(timespec="seconds")
    
    # Save all steps to a single JSONL file (one line per step)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Create a subdirectory within the output directory to store the cleaned partial trajectories and the text file of metadata
    subdir = output_dir / f"model-5_instance-100_with-ids_swe-gym_cleaned_no-oh-prompt_partial-trajectories_{timestamp.replace(':', '-')}"
    subdir.mkdir(parents=True, exist_ok=True)
    # HARDCODED
    stepwise_traj_jsonl = subdir / f"model-5_instance-100_with-ids_swe-gym_cleaned_no-oh-prompt_partial-trajectories_{timestamp.replace(':', '-')}.jsonl"
    metadata_txt = subdir / f"model-5_instance-100_with-ids_swe-gym_cleaned_no-oh-prompt_partial-trajectories_{timestamp.replace(':', '-')}.txt"
    
    all_partial_trajectories = []
    
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
        ],
        # "qwen3-coder": [
        #     "TO BE FILLED IN"
        # ]
    }
    
    skipped_instances = {model: [] for model in model_output_dirs.keys()}
    successfully_patched_instances = {model: [] for model in model_output_dirs.keys()}
    failed_instances = {model: [] for model in model_output_dirs.keys()}

    # Process each model and its associated output directories
    for model, output_dirs in model_output_dirs.items():
        for output_jsonl in output_dirs:
            # For each output jsonl file...
            try:
                with open(output_jsonl, 'r') as f:
                    # ...and for each line in the file...
                    for line in f:
                        result = json.loads(line)
                        # If history is null, skip and add it to the dictionary of skipped instances
                        if result.get("history") is None:
                            skipped_instances[model].append(result.get("instance_id"))
                            continue
                        # If the instance is successfully patched, add it to the dictionary of successfully patched instances
                        if result.get("report", {}).get("resolved", False):
                            successfully_patched_instances[model].append(result.get("instance_id"))
                        # If the instance is not successfully patched, add it to the dictionary of failed instances
                        else:
                            failed_instances[model].append(result.get("instance_id"))
                        total_events = len(result.get("history"))
                        # ...extract the partial trajectory of the first 1 non-oh-system-prompt event, the first 2 non-oh-system-prompt events, ..., the first total_events non-oh-system-prompt events...
                        for i in range(2, total_events + 1):
                            trajectory_record = {
                                "model": model,
                                "instance_id": result.get("instance_id"),
                                "successfully_patched": result.get("report", {}).get("resolved", False),
                                # "instruction": result.get("instruction"), 
                                "partial_trajectory": extract_partial_trajectory_from_history(result.get("history"), i)
                                # "git_patch": result.get("test_result", {}).get("git_patch", ""), # irrelevant, gives away final answer
                                # "error": result.get("error"), # irrelevant, only stores the most recent error
                                # "metrics": result.get("metrics"), # irrelevant
                                # "metadata": result.get("metadata"), # irrelevant
                            }
                            all_partial_trajectories.append(trajectory_record)
                print(f"[INFO] Trajectories for model {model} from {output_jsonl} extracted.")
            except Exception as e:
                print(f"[ERROR] Failed for model {model} from {output_jsonl}: {e}")
                raise e
    
    # Save all partial trajectories to a single JSONL file
    with open(stepwise_traj_jsonl, 'w') as f:
        for partial_trajectory in all_partial_trajectories:
            f.write(json.dumps(partial_trajectory) + '\n')
    print(f"[INFO] All partial trajectories saved to {stepwise_traj_jsonl}")
    
    # Save the skipped instances to the metadata text file
    with open(metadata_txt, 'w') as f:
        f.write(f"# Skipped instances:\n")
        for model in model_output_dirs.keys():
            f.write(f"{model} skipped {len(skipped_instances[model])}\n")
        for model, instances in skipped_instances.items():
            f.write(f"{model}: {instances}\n")
    print(f"[INFO] Skipped instances saved to {metadata_txt}")

    # Save the successfully patched instances to the metadata text file
    with open(metadata_txt, 'a') as f:
        f.write(f"\n# Successfully patched instances:\n")
        for model in model_output_dirs.keys():
            f.write(f"{model} successfully patched {len(successfully_patched_instances[model])}\n")
        for model, instances in successfully_patched_instances.items():
            f.write(f"{model}: {instances}\n")
    print(f"[INFO] Successfully patched instances saved to {metadata_txt}")

    # Save the failed instances to the metadata text file
    with open(metadata_txt, 'a') as f:
        f.write(f"\n# Failed instances:\n")
        for model in model_output_dirs.keys():
            f.write(f"{model} failed {len(failed_instances[model])}\n")
        for model, instances in failed_instances.items():
            f.write(f"{model}: {instances}\n")
    print(f"[INFO] Failed instances saved to {metadata_txt}")
    
    # Save the total number of instances to the metadata text file
    with open(metadata_txt, 'a') as f:
        f.write(f"\n# Total saved instances (successfully patched + failed):\n")
        for model in model_output_dirs.keys():
            f.write(f"{model}: {len(successfully_patched_instances[model]) + len(failed_instances[model])}\n")
    print(f"[INFO] Total saved instances (successfully patched + failed) saved to {metadata_txt}")

if __name__ == "__main__":
    main()
