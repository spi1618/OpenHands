import json
import random
import math
import os

# Script that balances the datasets by randomly sampling 800 successes and 200 failures
# Run convert_raw_to_pc.py after this script 

max_total_train_samples = 800

train_valid_ratio = 4 # (80/20 train/val split)

success_ratio = 0.4
failure_ratio = 0.6

full_raw_train_file = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_train_model-5_instance-100_pruned-4_with-ids_by-example.jsonl"
full_raw_valid_file = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_valid_model-5_instance-100_pruned-4_with-ids_by-example.jsonl"

# First count the number of successes and failures in both the train and valid sets
# Let num_full_train_successes = number of lines in full_raw_train_file where successfully_patched is true
# Let num_full_train_failures = number of lines in full_raw_train_file where successfully_patched is false
# Let num_full_valid_successes = number of lines in full_raw_valid_file where successfully_patched is true
# Let num_full_valid_failures = number of lines in full_raw_valid_file where successfully_patched is false

def count_successes_failures(file_path):
    """Count the number of successes and failures in a JSONL file."""
    successes = 0
    failures = 0
    
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('successfully_patched', False):
                    successes += 1
                else:
                    failures += 1
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line in {file_path}")
                continue
    
    return successes, failures

print("Counting successes and failures in datasets...")
num_full_train_successes, num_full_train_failures = count_successes_failures(full_raw_train_file)
num_full_valid_successes, num_full_valid_failures = count_successes_failures(full_raw_valid_file)

print(f"Train set: {num_full_train_successes} successes, {num_full_train_failures} failures")
print(f"Valid set: {num_full_valid_successes} successes, {num_full_valid_failures} failures")

# Set total_train_samples = floor(min(max_total_train_samples, num_full_train_successes (1 + failure_ratio/success_ratio)))
# Set total_valid_samples = floor(min(max_total_train_samples / train_valid_ratio, num_full_valid_successes (1 + failure_ratio/success_ratio)))
# Set total_train_samples = floor(min(total_train_samples, total_valid_samples * train_valid_ratio))
# Set total_valid_samples = floor(min(total_valid_samples, total_train_samples / train_valid_ratio))

total_train_samples = math.floor(min(max_total_train_samples, num_full_train_successes * (1 + failure_ratio/success_ratio)))
total_valid_samples = math.floor(min(max_total_train_samples / train_valid_ratio, num_full_valid_successes * (1 + failure_ratio/success_ratio)))
total_train_samples = math.floor(min(total_train_samples, total_valid_samples * train_valid_ratio))
total_valid_samples = math.floor(min(total_valid_samples, total_train_samples / train_valid_ratio))

print(f"Target train samples: {total_train_samples}")
print(f"Target valid samples: {total_valid_samples}")

output_train_file = f"/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_train_model-5_instance-100_pruned-4_with-ids_by-example_sample{total_train_samples}_balanced-{int(success_ratio*100)}-{int(failure_ratio*100)}.jsonl"
output_valid_file = f"/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_valid_model-5_instance-100_pruned-4_with-ids_by-example_sample{total_valid_samples}_balanced-{int(success_ratio*100)}-{int(failure_ratio*100)}.jsonl"

# Generate a random permutation [index1, index2, ...] of the train set
# Set added_train_successes = 0
# Set added_train_failures = 0
# Go through the permutation
# If the line is a success and added_train_successes < total_train_samples, add the line (which should be a single json) to output_train_file jsonl
#     Increment added_train_successes
# If the line is a failure and added_train_failures < total_train_samples, add the line (which should be a single json) to output_train_file jsonl
#     Increment added_train_failures

def sample_balanced_dataset(input_file, output_file, target_samples, success_ratio, failure_ratio):
    """Sample a balanced dataset from the input file."""
    # Read all lines and separate successes and failures
    successes = []
    failures = []
    
    with open(input_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if data.get('successfully_patched', False):
                    successes.append(line.strip())
                else:
                    failures.append(line.strip())
            except json.JSONDecodeError:
                continue
    
    # Calculate target counts
    target_successes = math.floor(target_samples * success_ratio)
    target_failures = math.floor(target_samples * failure_ratio)
    
    # Adjust if we don't have enough samples
    if len(successes) < target_successes:
        print(f"Warning: Only {len(successes)} successes available, using all")
        target_successes = len(successes)
    
    if len(failures) < target_failures:
        print(f"Warning: Only {len(failures)} failures available, using all")
        target_failures = len(failures)
    
    # Randomly sample
    sampled_successes = random.sample(successes, target_successes) if target_successes > 0 else []
    sampled_failures = random.sample(failures, target_failures) if target_failures > 0 else []
    
    # Combine and shuffle
    all_samples = sampled_successes + sampled_failures
    random.shuffle(all_samples)
    
    # Write to output file
    with open(output_file, 'w') as f:
        for line in all_samples:
            f.write(line + '\n')
    
    return len(sampled_successes), len(sampled_failures)

print(f"\nGenerating balanced train dataset...")
train_successes, train_failures = sample_balanced_dataset(
    full_raw_train_file, output_train_file, total_train_samples, success_ratio, failure_ratio
)

# Generate a random permutation [index1, index2, ...] of the valid set
# Set added_valid_successes = 0
# Set added_valid_failures = 0
# Go through the permutation
# If the line is a success and added_valid_successes < total_valid_samples, add the line (which should be a single json) to output_valid_file jsonl
#     Increment added_valid_successes
# If the line is a failure and added_valid_failures < total_valid_samples, add the line (which should be a single json) to output_valid_file jsonl
#     Increment added_valid_failures

print(f"Generating balanced valid dataset...")
valid_successes, valid_failures = sample_balanced_dataset(
    full_raw_valid_file, output_valid_file, total_valid_samples, success_ratio, failure_ratio
)

print(f"\nTrain dataset created: {train_successes} successes, {train_failures} failures")
print(f"Valid dataset created: {valid_successes} successes, {valid_failures} failures")

# Verify that the number of lines in output_train_file is added_train_successes + added_train_failures
# Verify that the number of lines in output_valid_file is added_valid_successes + added_valid_failures

print(f"\nVerifying output files...")
with open(output_train_file, 'r') as f:
    train_lines = sum(1 for _ in f)
with open(output_valid_file, 'r') as f:
    valid_lines = sum(1 for _ in f)

print(f"Output train file has {train_lines} lines (expected: {train_successes + train_failures})")
print(f"Output valid file has {valid_lines} lines (expected: {valid_successes + valid_failures})")

print(f"\nBalanced datasets created successfully!")
print(f"Train file: {output_train_file}")
print(f"Valid file: {output_valid_file}")