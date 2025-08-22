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

# python3 convert_raw_to_pc.py \
# --original-file /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_train_model-5_instance-100_pruned-4_with-ids_by-example_sample400.jsonl \
# --pc-output-dir /home/sophiapi/model-routing/debug \
# --hf-output-dir /home/sophiapi/model-routing/debug \
# --base-model Qwen/Qwen3-0.6B-Base \
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
    parser.add_argument('--base-model', default="Qwen/Qwen3-0.6B-Base",
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
    return f"{timestamp}_{base_name}_max-length-{max_length}{extension}"
    

def build_prompt(partial_trajectory: List[Dict], model_name: str) -> str:
    """Build the prompt for the router model."""
    # Convert trajectory steps to readable format
    trajectory_text = "\n".join(safe_json_dumps(step) for step in partial_trajectory)
    
    prompt = (
        "<|system|>\n"
        "You predict whether the agent will ultimately solve the SWE issue given the trajectory so far. "
        "Respond with [YES] or [NO] only.\n"
        "</|system|>\n\n"
        "### Partial trajectory\n" + trajectory_text + "\n\n"
        "### Candidate model\n[M] " + model_name + "\n\n"
        "### Will this agent eventually succeed?\n"
    )
    
    return prompt



def build_prompt_with_truncation(partial_trajectory: List[Dict], model_name: str, max_length: int, tokenizer=None) -> str:
    """
    Build prompt with intelligent truncation that preserves essential structure.
    
    This function ensures that:
    1. System instructions are always preserved
    2. Model information is always preserved  
    3. Question is always preserved
    4. Trajectory is truncated intelligently to fit within max_length
    """
    # Define the essential prompt parts that must be preserved
    system_part = (
        "<|system|>\n"
        "You predict whether the agent or assistant will ultimately solve the SWE issue successfully given the partial trajectory so far.\n"
        "Respond with [YES] or [NO] only.\n"
        "The partial trajectory contains information about the agent or assistant's actions, observations, and interactions with the user or environment.\n"
        "The partial trajectory may be truncated to the most recent information.\n"
        "</|system|>\n\n"
    )
    trajectory_header = "### Partial trajectory\n"
    model_part = f"### Candidate model\n[M] {model_name}\n\n"
    question_part = "### Will this agent eventually succeed?\n"
    
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
        # TODO: consider adding a coarse filter to remove the most egregiously long trajectories
        #     eg. if the trajectory is longer than 400,000 characters, truncate it to the last 400,000 characters
        if len(trajectory_text) > 400000:
            trajectory_text = trajectory_text[-400000:]
        # Count the number of tokens in the trajectory
        trajectory_tokens = tokenizer.encode(trajectory_text)
        
        # Debug logging for truncation decisions
        if len(trajectory_tokens) > max_trajectory_tokens:
            # print(f"DEBUG: Truncating trajectory: {len(trajectory_tokens)} > {max_trajectory_tokens} tokens")
            # print(f"  - Reserved tokens: {reserved_tokens}")
            # print(f"  - Max length: {max_length}")
            # print(f"  - Buffer: 50")
            
            # Truncate trajectory tokens and reconstruct
            # Truncate the trajectory to the last max_trajectory_tokens tokens, NOT the first max_trajectory_tokens tokens
            truncated_trajectory_tokens = trajectory_tokens[-max_trajectory_tokens:]
            truncated_trajectory = tokenizer.decode(truncated_trajectory_tokens)
            
            # print(f"  - Truncated to: {len(truncated_trajectory_tokens)} tokens")
        else:
            truncated_trajectory =  trajectory_text
            # print(f"DEBUG: No truncation needed: {len(trajectory_tokens)} <= {max_trajectory_tokens} tokens")
    else:
        raise ValueError("No tokenizer provided")
        # # Fallback to character-based truncation if no tokenizer
        # trajectory_text = "\n".join(safe_json_dumps(step) for step in partial_trajectory)
        # truncated_trajectory = trajectory_text + "\n\n"
    
    # Reconstruct the full prompt with all essential parts
    prompt = system_part + trajectory_header + truncated_trajectory + model_part + question_part
    
    # Final debug check and safety enforcement
    if tokenizer:
        final_tokens = len(tokenizer.encode(prompt))
        # print(f"DEBUG: Final prompt length: {final_tokens} tokens (max: {max_length})")
        
        if final_tokens > max_length:
            raise ValueError(f"Final prompt exceeds max_length! {final_tokens} > {max_length}")
            # print(f"ERROR: Final prompt exceeds max_length! {final_tokens} > {max_length}")
            # print(f"  - This should not happen with proper truncation!")
            # print(f"  - Forcing additional truncation...")
            
            # # Emergency truncation: take only the first max_length tokens
            # prompt_tokens = tokenizer.encode(prompt)
            # if len(prompt_tokens) > max_length:
            #     prompt_tokens = prompt_tokens[:max_length]
            #     prompt = tokenizer.decode(prompt_tokens)
            #     final_tokens = len(tokenizer.encode(prompt))
            #     print(f"  - Emergency truncation applied: {final_tokens} tokens")
    
    return prompt
    
    
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
        prompt = build_prompt_with_truncation(
            partial_trajectory, model_name, max_length, tokenizer
        )
        completion = "[YES]" if successfully_patched else "[NO]"
        
        # Final token length check
        if tokenizer:
            prompt_tokens = len(tokenizer.encode(prompt))
            completion_tokens = len(tokenizer.encode(completion))
            total_tokens = prompt_tokens + completion_tokens
            
            if total_tokens > max_length:
                skipped_count += 1
                if skipped_count < 5:  # Only show first 5 for debugging
                    print(f"DEBUG: Skipped example {i}: {total_tokens} tokens > {max_length}")
                    print(f"  - Prompt tokens: {prompt_tokens}")
                    print(f"  - Completion tokens: {completion_tokens}")
                    print(f"  - Model: {model_name}")
                continue
            
            # Check if truncation actually happened
            original_prompt = build_prompt(partial_trajectory, model_name)
            original_tokens = len(tokenizer.encode(original_prompt))
            if prompt_tokens < original_tokens:
                truncated_count += 1
                # if debug_long_examples < 5:
                #     print(f"DEBUG: Truncated example {i}: {original_tokens} -> {prompt_tokens} tokens")
        
        converted_data.append({
            "prompt": prompt,
            "completion": completion
        })
        
    print(f"DEBUG: Final conversion stats:")
    print(f"  - Converted: {len(converted_data)} examples")
    print(f"  - Skipped: {skipped_count} examples") 
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
    hf_output_file = os.path.join(hf_output_dir, generate_timestamped_filename(base_name, max_length, "hf"))
    
    print(f"Generated filenames:")
    print(f"  - PC output: {pc_output_file}")
    print(f"  - HF output: {hf_output_file}")
    
    # Convert to TRL format with tokenizer for smart truncation
    converted_data = convert_dataset_format(raw_data, max_length, max_filter_tokens, tokenizer)
    
    # Save to PC format
    with open(pc_output_file, "w") as f:
        for item in converted_data:
            f.write(json.dumps(item) + "\n")
    
    # Convert to the format expected by Dataset.from_dict()  
    dataset_dict = {  
        "prompt": [item["prompt"] for item in converted_data],  
        "completion": [item["completion"] for item in converted_data]  
    }
    
    # Save to HuggingFace dataset format
    dataset = Dataset.from_dict(dataset_dict)
    dataset.save_to_disk(hf_output_file)
    
    return
    

def main():
    args = parse_args()
    
    # Load tokenizer first (needed for smart truncation)
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Add special tokens for the task
    # (the tokenizer here is just for counting purposes and is not actually used for training)
    special_tokens = {"additional_special_tokens": ["[YES]", "[NO]"]}
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    
    print(f"DEBUG: Added {num_added_tokens} special tokens to tokenizer")
    print(f"DEBUG: Special tokens: {special_tokens['additional_special_tokens']}")
    
    # Verify special tokens are actually in the tokenizer
    for token in special_tokens['additional_special_tokens']:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id == tokenizer.unk_token_id:
            print(f"⚠️  WARNING: Token '{token}' maps to UNK token ID!")
        elif token_id == tokenizer.pad_token_id:
            print(f"⚠️  WARNING: Token '{token}' maps to PAD token ID!")
            
    print(f"DEBUG: Final tokenizer vocab size: {len(tokenizer)}")
    
    # Load and prepare datasets with tokenizer for smart truncation
    print(f"DEBUG: Dataset preparation parameters:")
    print(f"  - max_length: {args.max_length}")
    print(f"  - max_filter_tokens: {args.max_filter_tokens}")
    print(f"  - Model context window: 131072 (Qwen3-0.6B)")
    
    # Create output directories if they don't exist
    os.makedirs(args.pc_output_dir, exist_ok=True)
    os.makedirs(args.hf_output_dir, exist_ok=True)
    
    datasets = load_and_prepare_dataset(args.original_file, args.pc_output_dir, args.hf_output_dir, args.max_length, args.max_filter_tokens, tokenizer)
    
if __name__ == "__main__":
    main()