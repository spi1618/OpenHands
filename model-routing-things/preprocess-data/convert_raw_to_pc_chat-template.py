### Script to convert raw data to prompt completion format
### 
### This script now accepts output directories instead of specific filenames.
### It automatically generates timestamped filenames in the format: YYYYMMDD_HHMMSS_originalname.extension

import os
import json
import argparse
from typing import Dict, List, Any
from datetime import datetime

import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import SFTTrainer, SFTConfig, apply_chat_template


# Example usage:

# python3 convert_raw_to_pc_chat-template.py \
# --original-file /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_consistent_cleaned_no-oh-prompt_partial-trajectories_2025-08-30T19-46-44/20000-samples/train.jsonl \
# --pc-output-dir /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_consistent_cleaned_no-oh-prompt_partial-trajectories_2025-08-30T19-46-44/20000-samples/ \
# --hf-output-dir /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_consistent_cleaned_no-oh-prompt_partial-trajectories_2025-08-30T19-46-44/20000-samples/ \
# --base-model Qwen/Qwen2.5-0.5B-Instruct \
# --max-length 16384 \
# --max-filter-tokens 32000



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train router model with SFT')
    parser.add_argument('--original-file', required=True, 
                       help='Path to original data JSONL file')
    parser.add_argument('--pc-output-dir', required=True, 
                       help='Directory to save prompt-completion format JSONL file')
    parser.add_argument('--hf-output-dir', required=True, 
                       help='Directory to save HuggingFace dataset format')
    parser.add_argument('--base-model', default="Qwen/Qwen2.5-0.5B-Instruct",
                       help='Base model to fine-tune')
    parser.add_argument('--max-length', type=int, default=8192,
                       help='Maximum sequence length for training')
    parser.add_argument('--max-filter-tokens', type=int, default=32000,
                       help='Coarse filter threshold for extremely long examples')
    return parser.parse_args()


def safe_json_dumps(obj: Any) -> str:
    # TODO: set sort_keys to false, regenerate data, retrain router model,
    # THEN ALSO set it to false in router_inference_stupid.py
    """Safely serialize objects to JSON, handling non-serializable types."""
    try:
        return json.dumps(obj, sort_keys=False, default=str)
    except:
        return str(obj)


def generate_timestamped_filename(base_name: str, max_length: int, extension: str = "") -> str:
    """Generate a filename with timestamp prefix.
    
    Args:
        base_name: Base name for the file (without extension)
        extension: File extension (with or without leading dot)
    
    Returns:
        Filename in format: YYYYMMDD_HHMMSS_basename.extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if extension and not extension.startswith('.'):
        extension = '.' + extension
    return f"{timestamp}_{base_name}_chat-template-v2_max-len-{max_length}{extension}"

# Function that takes a partial trajectory and returns (the partial trajectory sans token count, the next token count of the last step)
def scalp_token_counts_from_partial_trajectory(partial_trajectory: List[Dict]) -> tuple[List[Dict], int]:
    next_token_count = int(partial_trajectory[-1]["next_step_output_tokens"]) # this is a string
    partial_trajectory_scalped = []
    for i in range(len(partial_trajectory)):
        event = {}
        event["source"] = partial_trajectory[i]["source"]
        event["message"] = partial_trajectory[i]["message"]
        if "model" in event.keys():
            event["model"] = partial_trajectory[i]["model"]
        partial_trajectory_scalped.append(event)
    return partial_trajectory_scalped, next_token_count

def bucket_from_tokens(n: int) -> int:
    # Lower bound inclusive, upper bound exclusive, bucket 8 unbounded
    if n < 64: return 1
    if n < 128: return 2
    if n < 256: return 3
    if n < 384: return 4
    if n < 512: return 5
    if n < 896: return 6
    if n < 1024: return 7
    return 8

def make_assistant_json(successfully_patched: bool, next_token_count: int) -> str:
    payload = {
        "success": "YES" if successfully_patched else "NO",
        "output_tokens_bucket": f"Bucket {bucket_from_tokens(next_token_count)}"
    }
    # One-line JSON, no trailing newline
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

def build_partial_trajectory_with_truncation(partial_trajectory: List[Dict], max_traj_tokens: int, tokenizer=None):
    """
    Literally just cuts down on the trajectory in case the trajectory is too long.
    """
    was_truncated = False
    
    # Calculate token lengths for essential parts (if tokenizer available)
    if tokenizer:
        # Convert trajectory to text and truncate if needed
        trajectory_text = "\n".join(safe_json_dumps(step) for step in partial_trajectory)
        # Count the number of tokens in the trajectory
        trajectory_tokens = tokenizer.encode(trajectory_text)
        # Debug logging for truncation decisions
        if len(trajectory_tokens) > max_traj_tokens:    
            was_truncated = True
            # Truncate trajectory tokens and reconstruct
            # Truncate the trajectory to (roughly)the first max_traj_tokens / 2 tokens + last max_traj_tokens / 2 tokens
            temp = max_traj_tokens // 2 - 20 # 20 tokens for the ...(omitted for brevity)... buffer
            truncated_trajectory_tokens_beginning = trajectory_tokens[:temp]
            truncated_trajectory_tokens_end = trajectory_tokens[-temp:]
            truncated_trajectory = tokenizer.decode(truncated_trajectory_tokens_beginning) + "\n...(omitted for brevity)...\n" + tokenizer.decode(truncated_trajectory_tokens_end)
        else:
            truncated_trajectory =  trajectory_text
    else:
        raise ValueError("No tokenizer provided")
    
    return truncated_trajectory, was_truncated
    
    
def convert_dataset_format(raw_data: List[Dict], max_length: int = 8192, max_filter_tokens: int = 32000, tokenizer=None, output_file=None) -> int:
    """
    Convert raw dataset format to TRL-compatible prompt-completion format.
    Saves the result in PC format to output_file.
    
    Raw format: (model, instance_id, successfully_patched, partial_trajectory)
    Target format: (prompt, completion)
    
    Includes sophisticated truncation logic to preserve essential structure while fitting within max_length.
    """
    converted_count = 0
    skipped_count = 0
    truncated_count = 0
    coarse_filtered_count = 0
    debug_long_examples = 0
    
    print(f"DEBUG: Starting dataset conversion with max_length={max_length}, max_filter_tokens={max_filter_tokens}")
    
    # Compute the number of tokens in my (yes me, long-suffering summer intern) part of the prompt
    with open("/home/sophiapi/model-routing/system_prompt.txt", "r", encoding="utf-8") as f:
        system_part = f.read()
    # system_part = (
    #     "You predict whether the agent or assistant will ultimately solve the SWE issue successfully given the partial trajectory so far and the candidate model that will be used to attempt the rest of the task.\n"
    #     "You also predict how many output tokens the candidate model will generate on the immediate next step by\n"
    #     "Respond with YES or NO followed by ... For example, if you predict that the candidate model will be successful and will generate 100 output tokens, you should respond with 'Success: YES, Output tokens: Bucket 8'.\n"
    #     # TODO: FIX THIS AND SET UP THE BUCKETS
    #     "The partial trajectory contains information about the agent or assistant's actions, and the names of the models that they used to take the actions.\n"
    #     "The intermediate steps in the partial trajectory may be omitted for brevity.\n"
    #     "The user will provide the partial trajectory.\n"
    # )
    trajectory_header = "### Partial trajectory:\n\n\n"
    model_part = f"### Candidate model\n[M] FILLER-TEXT-TO-UPPER-BOUND-LENGTH-OF-MODEL-NAME\n\n"
    question_part = "### Will this agent eventually succeed if the rest of the task is attempted with the candidate model? Into which of the 8 output token buckets will the candidate model's immediate next step fall?"
    
    system_tokens = len(tokenizer.encode(system_part))
    trajectory_header_tokens = len(tokenizer.encode(trajectory_header))
    model_tokens = len(tokenizer.encode(model_part))
    question_tokens = len(tokenizer.encode(question_part))
    
    # Reserve space for essential parts
    reserved_tokens = system_tokens + trajectory_header_tokens + model_tokens + question_tokens
    
    for i, item in enumerate(raw_data):
        # Extract fields
        model_name = item["model"]
        successfully_patched = item["successfully_patched"]
        partial_trajectory = item["partial_trajectory"]
        
        # Scalp the partial trajectory
        partial_trajectory, next_token_count = scalp_token_counts_from_partial_trajectory(partial_trajectory)
        
        # Construct model_part
        model_part = f"### Candidate model\n[M] {model_name}\n\n"
        
        # Coarse filter: skip extremely long examples that would lose too much context
        if tokenizer:
            raw_trajectory_text = "\n".join(safe_json_dumps(step) for step in partial_trajectory)
            raw_tokens = len(tokenizer.encode(raw_trajectory_text))
            
            if raw_tokens > max_filter_tokens:
                coarse_filtered_count += 1
                if coarse_filtered_count < 5:  # Only show first 5 for debugging
                    print(f"DEBUG: Coarse filtered example {i}: {raw_tokens} tokens > {max_filter_tokens}")
                continue
        else:
            raise ValueError("No tokenizer provided")
        
        
        # Calculate how much space we have for trajectory content
        max_trajectory_tokens = max_length - reserved_tokens - 100  # Leave some buffer
    
        
        # Build prompt with smart truncation
        truncated_trajectory, was_truncated = build_partial_trajectory_with_truncation(
            partial_trajectory, max_trajectory_tokens, tokenizer
        )
        
        # Build completion
        completion = make_assistant_json(successfully_patched, next_token_count)
        
        if was_truncated:
            truncated_count += 1
        
        # Build the prompts:
        prompt_part1 = {
            "role": "system",
            "content": system_part
        }
        prompt_part2 = {
            "role": "user",
            "content": trajectory_header + truncated_trajectory
        }
        prompt_part3 = {
            "role": "system",
            "content": model_part + question_part
        }
                
        item = {
            "prompt": [prompt_part1, prompt_part2, prompt_part3],
            "completion": [{"role": "assistant", "content": completion}]
        }
        
        # Save to PC format - APPEND DON'T OVERWRITE
        if output_file:
            with open(output_file, "a") as f:
                f.write(json.dumps(item) + "\n")
                
        converted_count += 1
        
    print(f"DEBUG: Final conversion stats:")
    print(f"  - Converted: {converted_count} examples")
    print(f"  - Truncated: {truncated_count} examples")
    print(f"  - Coarse filtered: {coarse_filtered_count} examples")
    print(f"  - Total processed: {len(raw_data)} examples")
    
    
    return converted_count
    

def load_and_prepare_dataset(original_file: str, pc_output_dir: str, hf_output_dir: str, max_length: int, max_filter_tokens: int, tokenizer=None):
    """Load and prepare dataset in the correct format for TRL."""
    print("Loading raw dataset...")

    # Load raw data
    raw_data = [json.loads(line) for line in open(original_file)]
    
    print(f"Raw data: {len(raw_data)} examples")
    
    # Generate timestamped filenames
    base_name = os.path.splitext(os.path.basename(original_file))[0]  # Remove extension
    pc_output_file = os.path.join(pc_output_dir, generate_timestamped_filename(base_name, max_length, "jsonl"))
    # hf_output_file = os.path.join(hf_output_dir, generate_timestamped_filename(base_name, max_length, "hf"))
    
    print(f"Generated filenames:")
    print(f"  - PC output: {pc_output_file}")
    # print(f"  - HF output: {hf_output_file}")
    
    # Convert to TRL format with tokenizer for smart truncation
    converted_data_count = convert_dataset_format(raw_data, max_length, max_filter_tokens, tokenizer, pc_output_file)
    
    return


def main():
    args = parse_args()
    
    # Load tokenizer first (needed for smart truncation)
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
            
    print(f"DEBUG: Final tokenizer vocab size: {len(tokenizer)}")
    
    # Load and prepare datasets with tokenizer for smart truncation
    print(f"DEBUG: Dataset preparation parameters:")
    print(f"  - max_length: {args.max_length}")
    print(f"  - max_filter_tokens: {args.max_filter_tokens}")
    print(f"  - Model context window: idk what this is")
    
    # Create output directories if they don't exist
    os.makedirs(args.pc_output_dir, exist_ok=True)
    os.makedirs(args.hf_output_dir, exist_ok=True)    
    
    datasets = load_and_prepare_dataset(args.original_file, args.pc_output_dir, args.hf_output_dir, args.max_length, args.max_filter_tokens, tokenizer)
    
if __name__ == "__main__":
    main()