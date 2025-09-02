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

def extract_partial_trajectory_from_history(history, event_number, total_events, model_names) -> list[dict]:
    """
    Extracts a partial trajectory of the first event_number events from the history, where each step is a dictionary
    Rules:
    - We ignore the first event since it's the openhands system prompt
    - We ignore all environment events (eg. "Added workspace context") <-- this doesn't even always appear??
    - We ignore all user events except for the first one (the one that contains the string "Consider the following issue description")
    - We extract only the messages associated with actions of the agent (NO THOUGHTS)
    - We extract only the messages from the user <-- sometimes the user shows up more than just the two times in the beginning though so we might scratch it out of the inference time history
    - We extract the name of the model (LLM) that the agent used to take each action
    - So basically we ignore all events not from the agent EXCEPT FOR the first user event; also ignore agent observations
    """    
    map_raw_model_name_to_model_name = {
        "litellm_proxy/claude-3-5-haiku-20241022": "claude-3-5-haiku",
        "litellm_proxy/claude-sonnet-4-20250514": "claude-sonnet-4",
        "litellm_proxy/fireworks_ai/accounts/fireworks/models/deepseek-v3": "deepseek-v3",
        "litellm_proxy/mistral/devstral-small-2505": "devstral-small",
        "litellm_proxy/kimi-k2-0711-preview": "kimi-k2",
        # "litellm_proxy/qwen3-coder-480b-a35b-instruct": "qwen3-coder",
    }
    partial_trajectory = []
    current_event = 1
    user_event_found = False
    while current_event < event_number:
        event = history[current_event]
        cleaned_event = {}
        # Extract the source of the event
        source = event.get("source") # "user", "environment", "agent"
        # print(f"SOURCE: {source}\n")
        cleaned_event["source"] = source
        # ================================
        # If the event a user event but not the instructional user event, skip it
        if source == "user" and "Consider the following issue description" in event.get("message") and not user_event_found:
            user_event_found = True
            current_accumulated_output_tokens = 0
        # ================================
        # If the source is "agent" and the "action" field exists and the "llm_metrics" field exists...
        elif source == "agent" and "action" in event and "llm_metrics" in event.keys():
            # print(f"Agent action detected (current event {current_event}, step id {event.get('id')}), message preview: {event.get('message')[:200]}...\n")
            # ...then extract model name that the agent used to take the action (default to "")
            model_name = ""
            if "tool_call_metadata" in event.keys():
                model_raw = event["tool_call_metadata"]["model_response"]["model"]
                if model_raw not in map_raw_model_name_to_model_name.keys():
                    raise ValueError(f"Model {model_raw} not found in map_raw_model_name_to_model_name")
                model_name = map_raw_model_name_to_model_name[model_raw]
                # Check that the model name is consistent with the model names list given by main function
                if model_name not in model_names:
                    raise ValueError(f"Model {model_name} not found in model_names")
            cleaned_event["model_name"] = model_name
            # ...also extract the current accumulated output tokens
            if "accumulated_token_usage" not in event["llm_metrics"].keys():
                raise ValueError(f"Accumulated token usage not found in LLM metrics in event {event.get('id')}")
            if "completion_tokens" not in event["llm_metrics"]["accumulated_token_usage"].keys():
                raise ValueError(f"Completion tokens not found in accumulated token usage in LLM metrics in event {event.get('id')}")
            current_accumulated_output_tokens = event["llm_metrics"]["accumulated_token_usage"]["completion_tokens"]
        # ================================
        # If the current event is not the instructional user event or an llm agent action event, skip it
        else:
            current_event += 1
            continue
        # ================================
        # Compute the number of output tokens that the agent will generate on the immediate next step
        # --------------------------------
        # Find the next action event
        next_action_event = current_event + 1
        next_action_event_found = False
        while next_action_event < total_events:
            next_event = history[next_action_event]
            if next_event.get("source") == "agent" and "action" in next_event.keys() and "llm_metrics" in next_event.keys():
                next_action_event_found = True
                break
            next_action_event += 1
        if not next_action_event_found:
            # This should only happen if the current event is the last llm agent action event in the history
            # Note to future self: originally I tried to detect this, but sometimes there were the shenanigans were too shenanigan-y and I gave up
            # if current_event < total_events - 3: # -3 instead of -1 because sometimes shenanigans i don't understand happen and the last event is not an action event
            #     raise ValueError(f"No next action event found in history (total events {total_events}) for event {event.get('id')}")
            next_step_output_tokens = current_accumulated_output_tokens
        else:
            if "llm_metrics" not in history[next_action_event].keys():
                print(f"LLM metrics not found in history[next_action_event] where next_action_event = {next_action_event} and total events = {total_events}")
                print(f"history[next_action_event] where next_action_event = {next_action_event}: {history[next_action_event]}")
                raise ValueError(f"LLM metrics not found in history[next_action_event] where next_action_event = {next_action_event}")
            next_step_accumulated_output_tokens = history[next_action_event]["llm_metrics"]["accumulated_token_usage"]["completion_tokens"]
        # --------------------------------
        # Compute the difference between the next step accumulated output tokens and the current accumulated output tokens
        next_step_output_tokens = str(next_step_accumulated_output_tokens - current_accumulated_output_tokens)
        cleaned_event["next_step_output_tokens"] = next_step_output_tokens
        # ================================
        # Extract the message of the event
        cleaned_event["message"] = event.get("message")
        # ================================
        # Add the cleaned event to the partial trajectory
        partial_trajectory.append(cleaned_event)
        current_event += 1
        
    # Check that the user's first message with the string "Consider the following issue description" is in the partial trajectory
    if len(partial_trajectory) == 0:
        raise ValueError("No user's first message with the string 'Consider the following issue description' found in the partial trajectory")
    if partial_trajectory[0]["source"] != "user" or "Consider the following issue description" not in partial_trajectory[0]["message"]:
        raise ValueError("The user's first message with the string 'Consider the following issue description' is not in the partial trajectory")
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
    subdir = output_dir / f"model-5_instance-100_with-ids_swe-gym_consistent_cleaned_no-oh-prompt_partial-trajectories_{timestamp.replace(':', '-')}"
    subdir.mkdir(parents=True, exist_ok=True)
    # HARDCODED
    stepwise_traj_jsonl = subdir / f"model-5_instance-100_with-ids_swe-gym_consistent_cleaned_no-oh-prompt_partial-trajectories_{timestamp.replace(':', '-')}.jsonl"
    metadata_txt = subdir / f"model-5_instance-100_with-ids_swe-gym_consistent_cleaned_no-oh-prompt_partial-trajectories_{timestamp.replace(':', '-')}.txt"
    
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
    
    # List of all model names
    model_names_list = list(model_output_dirs.keys())
    
    skipped_instances = {model: [] for model in model_output_dirs.keys()}
    zero_nontrivial_events_instances = {model: [] for model in model_output_dirs.keys()}
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
                        total_events = len(result.get("history"))
                        # If the instance has 0 events, add it to the dictionary of zero events instances
                        if total_events <= 1:
                            zero_nontrivial_events_instances[model].append(result.get("instance_id"))
                            continue
                        # If the instance is successfully patched, add it to the dictionary of successfully patched instances
                        if result.get("report", {}).get("resolved", False):
                            successfully_patched_instances[model].append(result.get("instance_id"))
                        # If the instance is not successfully patched, add it to the dictionary of failed instances
                        else:
                            failed_instances[model].append(result.get("instance_id"))
                        # ...extract the partial trajectory of the first 1 non-oh-system-prompt event, the first 2 non-oh-system-prompt events, ..., the first total_events non-oh-system-prompt events...
                        for i in range(2, total_events + 1):
                            extracted_partial_traj = extract_partial_trajectory_from_history(result.get("history"), i, total_events, model_names_list)
                            trajectory_record = {
                                "model": model,
                                "instance_id": result.get("instance_id"),
                                "successfully_patched": result.get("report", {}).get("resolved", False),
                                # "instruction": result.get("instruction"), 
                                "partial_trajectory": extracted_partial_traj,
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
    
    # Remove duplicates from all_partial_trajectories
    seen = set()
    unique_trajectories = []
    for trajectory in all_partial_trajectories:
        # Create a tuple representation that can be hashed
        trajectory_tuple = (
            trajectory["model"],
            trajectory["instance_id"],
            trajectory["successfully_patched"],
            tuple(tuple(step.items()) for step in trajectory["partial_trajectory"])
        )
        if trajectory_tuple not in seen:
            seen.add(trajectory_tuple)
            unique_trajectories.append(trajectory)
    all_partial_trajectories = unique_trajectories
    
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

    # Save the zero nontrivial events instances to the metadata text file
    with open(metadata_txt, 'a') as f:
        f.write(f"\n# Zero nontrivial events instances:\n")
        for model in model_output_dirs.keys():
            f.write(f"{model} zero events {len(zero_nontrivial_events_instances[model])}\n")
        for model, instances in zero_nontrivial_events_instances.items():
            f.write(f"{model}: {instances}\n")
    print(f"[INFO] Zero nontrivial events instances saved to {metadata_txt}")

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
