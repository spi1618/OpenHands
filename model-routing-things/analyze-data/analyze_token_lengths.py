import json
import argparse
import os
from datetime import datetime
from transformers import AutoTokenizer
import numpy as np
from collections import Counter

# Load the same tokenizer used in training
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
SEP_MODEL = "[M]"
TOK_SPECIALS = {"additional_special_tokens": [SEP_MODEL, "[YES]", "[NO]"]}
tok.add_special_tokens(TOK_SPECIALS)

def build_prompt(traj, cand):
    # Handle datetime objects and other non-serializable types
    def safe_json_dumps(obj):
        try:
            return json.dumps(obj, sort_keys=True, default=str)
        except:
            return str(obj)
    
    steps = "\n".join(safe_json_dumps(s) for s in traj)

    return (
        "<|system|>\n"
        "You predict whether the agent will ultimately solve the SWE issue given the trajectory so far.  Respond with [YES] or [NO] only."
        "\n</|system|>\n\n"
        "### Partial trajectory\n" + steps + "\n\n"
        "### Candidate model\n" + SEP_MODEL + " " + cand + "\n\n"
        "### Will this agent eventually succeed?\n"
    )

def row_to_text(ex):
    prompt = build_prompt(ex["partial_trajectory"], ex["model"])
    answer = " [YES]" if ex["successfully_patched"] else " [NO]"
    return {"text": prompt + answer}

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Analyze token lengths in training and validation datasets')
    parser.add_argument('--train-file', required=True, help='Path to the training dataset file')
    parser.add_argument('--valid-file', required=True, help='Path to the validation dataset file')
    parser.add_argument('--output-dir', default='/home/sophiapi/model-routing/token_length_reports', 
                       help='Directory to save output files (default: /home/sophiapi/model-routing/token_length_reports)')
    return parser.parse_args()

def extract_dataset_name(file_path):
    """Extract dataset name from file path after router_*_"""
    filename = os.path.basename(file_path)
    # Look for pattern router_*_ where * is train or valid
    if 'router_train_' in filename:
        # Extract everything after router_train_
        return filename.split('router_train_')[1].split('.')[0]
    elif 'router_valid_' in filename:
        # Extract everything after router_valid_
        return filename.split('router_valid_')[1].split('.')[0]
    else:
        # Fallback: use filename without extension
        return os.path.splitext(filename)[0]

# Load datasets
args = parse_args()
train_file = args.train_file
valid_file = args.valid_file
output_dir = args.output_dir

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Extract dataset name from train file (assuming train and valid have same dataset name)
dataset_name = extract_dataset_name(train_file)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"{dataset_name}_{timestamp}.txt"
output_file = os.path.join(output_dir, output_filename)

# Open output file for writing
output_f = open(output_file, 'w')

# Write header with file info
print(f"Saving analysis results to: {output_file}")
output_f.write(f"Token Length Analysis Results\n")
output_f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
output_f.write(f"Dataset: {dataset_name}\n")
output_f.write(f"Train file: {train_file}\n")
output_f.write(f"Valid file: {valid_file}\n")
output_f.write("=" * 80 + "\n\n")

def write_output(text):
    """Write text to both console and output file"""
    print(text)
    output_f.write(text + '\n')
    output_f.flush()  # Ensure immediate writing

def analyze_dataset(file_path, dataset_name):
    write_output(f"\n=== Analyzing {dataset_name} ===")
    
    token_lengths = []
    trajectory_lengths = []
    examples_with_long_trajectories = []
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            # if i >= 100:  # Analyze first 100 examples for speed # nah screw that I want to see it all
            #     break
                
            ex = json.loads(line.strip())
            
            # Count trajectory steps
            traj_length = len(ex["partial_trajectory"])
            trajectory_lengths.append(traj_length)
            
            # Convert to training format and count tokens
            formatted = row_to_text(ex)
            tokens = tok.encode(formatted["text"])
            token_length = len(tokens)
            token_lengths.append(token_length)
            
            # Track examples with very long trajectories
            if traj_length > 50:
                examples_with_long_trajectories.append({
                    'index': i,
                    'traj_length': traj_length,
                    'token_length': token_length,
                    'model': ex["model"],
                    'success': ex["successfully_patched"]
                })
    
    write_output(f"Analyzed {len(token_lengths)} examples")
    write_output(f"Token length statistics:")
    write_output(f"  Mean: {np.mean(token_lengths):.1f}")
    write_output(f"  Median: {np.median(token_lengths):.1f}")
    write_output(f"  Min: {np.min(token_lengths)}")
    write_output(f"  Max: {np.max(token_lengths)}")
    write_output(f"  Std: {np.std(token_lengths):.1f}")
    
    write_output(f"\nTrajectory length statistics:")
    write_output(f"  Mean: {np.mean(trajectory_lengths):.1f}")
    write_output(f"  Median: {np.median(trajectory_lengths):.1f}")
    write_output(f"  Min: {np.min(trajectory_lengths)}")
    write_output(f"  Max: {np.max(trajectory_lengths)}")
    
    # Show distribution
    write_output(f"\nToken length distribution:")
    for threshold in [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]:
        count = sum(1 for x in token_lengths if x > threshold)
        write_output(f"  > {threshold:,} tokens: {count} examples ({count/len(token_lengths)*100:.1f}%)")
    
    if examples_with_long_trajectories:
        write_output(f"\nExamples with >50 trajectory steps:")
        for ex in examples_with_long_trajectories[:5]:  # Show first 5
            write_output(f"  Example {ex['index']}: {ex['traj_length']} steps, {ex['token_length']} tokens, model={ex['model']}, success={ex['success']}")
    
    return token_lengths, trajectory_lengths

# Analyze both datasets
train_tokens, train_trajs = analyze_dataset(train_file, "Training")
valid_tokens, valid_trajs = analyze_dataset(valid_file, "Validation")

# Check for the extremely long example mentioned in the error
write_output(f"\n=== Looking for extremely long examples ===")
write_output(f"Error mentioned: 146,015 tokens > 131,072 max")

def find_extremely_long_examples(file_path, dataset_name):
    write_output(f"\nSearching {dataset_name} for examples > 100k tokens...")
    
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            ex = json.loads(line.strip())
            formatted = row_to_text(ex)
            tokens = tok.encode(formatted["text"])
            token_length = len(tokens)
            
            if token_length > 100000:
                write_output(f"  Found extremely long example {i}:")
                write_output(f"    Token length: {token_length:,}")
                write_output(f"    Trajectory steps: {len(ex['partial_trajectory'])}")
                write_output(f"    Instance ID: {ex['instance_id']}")
                write_output(f"    Model: {ex['model']}")
                write_output(f"    Success: {ex['successfully_patched']}")
                
                # Show first few trajectory steps
                write_output(f"    First 3 trajectory steps:")
                for j, step in enumerate(ex['partial_trajectory'][:3]):
                    step_str = str(step)[:200] + "..." if len(str(step)) > 200 else str(step)
                    write_output(f"      Step {j}: {step_str}")
                write_output("")

find_extremely_long_examples(train_file, "Training")
find_extremely_long_examples(valid_file, "Validation")

# Close the output file
output_f.close() 