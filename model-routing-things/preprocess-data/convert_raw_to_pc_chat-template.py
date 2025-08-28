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

# poetry run python3 ../convert_raw_to_pc_chat-template.py \
# --original-file /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/train.jsonl \
# --pc-output-dir /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples \
# --hf-output-dir /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples \
# --base-model Qwen/Qwen2.5-0.5B-Instruct \
# --max-length 8192 \
# --max-filter-tokens 32000

# python3 convert_raw_to_pc_chat-template.py \
# --original-file /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/train.jsonl \
# --pc-output-dir /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples \
# --hf-output-dir /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples \
# --base-model Qwen/Qwen2.5-0.5B-Instruct \
# --max-length 8192 \
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



def build_prompt_with_truncation(partial_trajectory: List[Dict], model_name: str, max_length: int, tokenizer=None):
    """
    Build prompt with intelligent truncation that preserves essential structure.
    
    This function ensures that:
    1. System instructions are always preserved
    2. Model information is always preserved  
    3. Question is always preserved
    4. Trajectory is truncated intelligently to fit within max_length
    """
    was_truncated = False
    
    # Define the essential prompt parts that must be preserved
    system_part = (
        "<|system|>\n"
        "You predict whether the agent or assistant will ultimately solve the SWE issue successfully given the partial trajectory so far and the candidate model that will be used to attempt the rest of the task.\n"
        "Respond with YES or NO only.\n"
        "The partial trajectory contains information about the agent or assistant's actions, observations, and interactions with the user or environment.\n"
        "The partial trajectory may be truncated to the most recent information.\n"
        "</|system|>\n\n"
    )
    trajectory_header = "### Partial trajectory\n"
    model_part = f"### Candidate model\n[M] {model_name}\n\n"
    question_part = "### Will this agent eventually succeed if the rest of the task is attempted with the candidate model?\n"
    
    # Calculate token lengths for essential parts (if tokenizer available)
    if tokenizer:
        system_tokens = len(tokenizer.encode(system_part))
        trajectory_header_tokens = len(tokenizer.encode(trajectory_header))
        model_tokens = len(tokenizer.encode(model_part))
        question_tokens = len(tokenizer.encode(question_part))
        
        # Reserve space for essential parts
        reserved_tokens = system_tokens + trajectory_header_tokens + model_tokens + question_tokens
        
        # Calculate how much space we have for trajectory content
        max_trajectory_tokens = max_length - reserved_tokens - 50  # Leave some buffer
        
        # Convert trajectory to text and truncate if needed
        trajectory_text = "\n".join(safe_json_dumps(step) for step in partial_trajectory)
        
        # Count the number of tokens in the trajectory
        trajectory_tokens = tokenizer.encode(trajectory_text)
        
        # Debug logging for truncation decisions
        if len(trajectory_tokens) > max_trajectory_tokens:
            was_truncated = True
            # Truncate trajectory tokens and reconstruct
            # Truncate the trajectory to (roughly)the first max_trajectory_tokens / 2 tokens + last max_trajectory_tokens / 2 tokens
            temp = max_trajectory_tokens // 2 - 20 # 20 tokens for the ...(omitted for brevity)... buffer
            truncated_trajectory_tokens_beginning = trajectory_tokens[:temp]
            truncated_trajectory_tokens_end = trajectory_tokens[-temp:]
            truncated_trajectory = tokenizer.decode(truncated_trajectory_tokens_beginning) + "\n...(omitted for brevity)...\n" + tokenizer.decode(truncated_trajectory_tokens_end)
        else:
            truncated_trajectory =  trajectory_text
    else:
        raise ValueError("No tokenizer provided")
    
    # Reconstruct the full prompt with all essential parts
    prompt = system_part + trajectory_header + truncated_trajectory + model_part + question_part
    
    # Final debug check and safety enforcement
    if tokenizer:
        final_tokens = len(tokenizer.encode(prompt))
        
        if final_tokens > max_length:
            raise ValueError(f"Final prompt exceeds max_length! {final_tokens} > {max_length}")
    
    return prompt, was_truncated
    
    
def convert_dataset_format(raw_data: List[Dict], max_length: int = 8192, max_filter_tokens: int = 32000, tokenizer=None) -> List[Dict]:
    """
    Convert raw dataset format to TRL-compatible prompt-completion format.
    
    Raw format: (model, instance_id, successfully_patched, partial_trajectory)
    Target format: (prompt, completion)
    
    Includes sophisticated truncation logic to preserve essential structure while fitting within max_length.
    """
    converted_data = []
    skipped_count = 0
    truncated_count = 0
    coarse_filtered_count = 0
    debug_long_examples = 0
    
    print(f"DEBUG: Starting dataset conversion with max_length={max_length}, max_filter_tokens={max_filter_tokens}")
    
    for i, item in enumerate(raw_data):
        # Extract fields
        model_name = item["model"]
        successfully_patched = item["successfully_patched"]
        partial_trajectory = item["partial_trajectory"]
        
        # Coarse filter: skip extremely long examples that would lose too much context
        if tokenizer:
            raw_trajectory_text = "\n".join(safe_json_dumps(step) for step in partial_trajectory)
            raw_tokens = len(tokenizer.encode(raw_trajectory_text))
            
            if raw_tokens > max_filter_tokens:
                coarse_filtered_count += 1
                if coarse_filtered_count < 5:  # Only show first 5 for debugging
                    print(f"DEBUG: Coarse filtered example {i}: {raw_tokens} tokens > {max_filter_tokens}")
                continue
        
        # Build prompt with smart truncation
        prompt, was_truncated = build_prompt_with_truncation(
            partial_trajectory, model_name, max_length, tokenizer
        )
        completion = "YES" if successfully_patched else "NO"
        
        if was_truncated:
            truncated_count += 1
        
        # Final token length check
        if tokenizer:
            prompt_tokens = len(tokenizer.encode(prompt))
            completion_tokens = len(tokenizer.encode(completion))
            total_tokens = prompt_tokens + completion_tokens
            
            if total_tokens > max_length:
                raise ValueError(f"Final prompt exceeds max_length! {total_tokens} > {max_length}")
        
        converted_data.append({
            "prompt": [{"role": "user", "content": prompt}],
            "completion": [{"role": "assistant", "content": completion}]
        })
        
    print(f"DEBUG: Final conversion stats:")
    print(f"  - Converted: {len(converted_data)} examples")
    print(f"  - Truncated: {truncated_count} examples")
    print(f"  - Coarse filtered: {coarse_filtered_count} examples")
    print(f"  - Total processed: {len(raw_data)} examples")
    
    return converted_data
    

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
    converted_data = convert_dataset_format(raw_data, max_length, max_filter_tokens, tokenizer)
    
    # Save to PC format
    with open(pc_output_file, "w") as f:
        for item in converted_data:
            f.write(json.dumps(item) + "\n")
    
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