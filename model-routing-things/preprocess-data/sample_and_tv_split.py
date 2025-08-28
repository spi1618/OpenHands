# Script to take the full dataset and split it into train and validation sets
# Arguments:
# --input-file: path to the full dataset
# --sample-size: number of examples to sample
# --train-proportion: proportion of instances to put in the train set (eg. 0.8)
# --success-rate: OPTIONAL - target success rate for both train and validation sets (float between 0 and 1, inclusive)
#                 If provided, the script will create balanced datasets with this success rate
#                 If not provided, the script will randomly sample as before

# Example usage:
# python3 sample_and_tv_split.py \
#     --input-file /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned_no-oh-prompt_partial-trajectories_2025-08-28T00-49-09/model-5_instance-100_with-ids_swe-gym_cleaned_no-oh-prompt_partial-trajectories_2025-08-28T00-49-09.jsonl \
#     --sample-size 20000 \
#     --train-proportion 0.8 \
#     --success-rate 0.4 
# This will sample 1000 examples from the full dataset, putting 80% of the instances in the train set and 20% in the validation set, and aiming for a 50% success rate in both sets.
# This will create a relatively balanced dataset where about 40% of the examples in both sets are successful and about 60% are unsuccessful.

# Ensures that examples with the same instance_id are in the same set (no two examples with the same instance_id in different sets)

# Assumes that the input file is a jsonl file 
# and that each line of the jsonl file is a json object with the following fields:
# - model: string
# - instance_id: string
# - successfully_patched: boolean
# - partial_trajectory: list of dicts

# Writes the train and validation sets to jsonl files in the same directory as the input file
# Also in the same directory, writes a tv_metadata.txt file with the following information:
# - sample_size: number of instances sampled
# - train_proportion: proportion of instances put in the train set
# - train_size: number of instances put in the train set
# - val_size: number of instances put in the validation set
# - train_file: path to the train set
# - val_file: path to the validation set
# - train_success_rate: proportion of successes in the train set
# - val_success_rate: proportion of successes in the validation set
# - target_success_rate: target success rate (if --success-rate was provided)
# - balanced_mode: boolean indicating if balanced sampling was used

# Logical flow of this script:
# 1. Load the data from the input file
# 2. Group the data by instance_id and store the line numbers of the examples that belong to each instance_id
#     The output of step 2 is a dictionary of the form {instance_id: [example_line_number, ...], ...}
# 3. Randomly divide the instances into train and validation sets according to the train_proportion argument (ensuring that examples with the same instance_id are in the same set)
#     The output of step 3 is a dictionary of the form {"train": [instance_id, ...], "val": [instance_id, ...]}
# 4. Construct the train set:
#     If --success-rate is NOT provided (random sampling mode):
#         4a. While the size of the train set is less than sample_size*train_proportion...
#             4a1. Randomly select an instance from the train set
#             4a2. Randomly select an example line number that belongs to the instance
#             4a3. Add the example to the train set
#             4a4. Remove the example line number from the instance's list of line numbers
#     If --success-rate IS provided (balanced sampling mode):
#         4b. Calculate target number of successful and failed examples for train set
#         4b1. First, validate that there are enough successes and failures available in the train instances
#              - Count total available successes and failures across all train instances
#              - If either count is insufficient, throw an error with details
#         4b2. Then, sample examples in a single loop:
#              - Keep counters: successes_added and failures_added
#              - While either counter is below target:
#                * Randomly select an instance from train set
#                * Randomly select an example line number from that instance
#                * If example is success AND successes_added < target_successes:
#                  - Add example to train set, increment successes_added
#                  - Remove line number from instance's list
#                  - If instance has no more examples, remove from train set
#                * If example is failure AND failures_added < target_failures:
#                  - Add example to train set, increment failures_added
#                  - Remove line number from instance's list
#                  - If instance has no more examples, remove from train set
#                * If example type is already at target, skip it
#              - Continue until both targets are reached
# 5. Construct the validation set:
#     If --success-rate is NOT provided (random sampling mode):
#         5a. While the size of the validation set is less than sample_size*(1-train_proportion)...
#             5a1. Randomly select an instance from the validation set
#             5a2. Randomly select an example line number that belongs to the instance
#             5a3. Add the example to the validation set
#             5a4. Remove the example line number from the instance's list of line numbers
#     If --success-rate IS provided (balanced sampling mode):
#         5b. Calculate target number of successful and failed examples for validation set
#         5b1. First, validate that there are enough successes and failures available in the validation instances
#              - Count total available successes and failures across all validation instances
#              - If either count is insufficient, throw an error with details
#         5b2. Then, sample examples in a single loop:
#              - Keep counters: successes_added and failures_added
#              - While either counter is below target:
#                * Randomly select an instance from validation set
#                * Randomly select an example line number from that instance
#                * If example is success AND successes_added < target_successes:
#                  - Add example to validation set, increment successes_added
#                  - Remove line number from instance's list
#                  - If instance has no more examples, remove from validation set
#                * If example is failure AND failures_added < target_failures:
#                  - Add example to validation set, increment failures_added
#                  - Remove line number from instance's list
#                  - If instance has no more examples, remove from validation set
#                * If example type is already at target, skip it
#              - Continue until both targets are reached
# 6. Write the train and validation sets to jsonl files
# 7. Write a metadata file with the metadata information above

# Important notes for balanced sampling:
# - The script will attempt to achieve the target success rate but may not be able to if there aren't enough examples
# - Instance grouping is still respected (all examples with same instance_id go to same set)
# - If target success rate cannot be achieved, the script will throw an error with details about what's available
# - The script will validate that sufficient examples exist before starting the sampling process


import json
import random
import os
import argparse
from collections import defaultdict

def load_data(input_file):
    """Load data from jsonl file and return list of examples with line numbers."""
    examples = []
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line.strip():
                example = json.loads(line.strip())
                example['line_number'] = line_num
                examples.append(example)
    return examples

def group_by_instance_id(examples):
    """Group examples by instance_id and store line numbers."""
    instance_groups = defaultdict(list)
    for example in examples:
        instance_id = example['instance_id']
        instance_groups[instance_id].append(example['line_number'])
    return instance_groups

def split_instances(instance_groups, train_proportion):
    """Randomly divide instances into train and validation sets."""
    instance_ids = list(instance_groups.keys())
    random.shuffle(instance_ids)
    
    split_point = int(len(instance_ids) * train_proportion)
    train_instances = instance_ids[:split_point]
    val_instances = instance_ids[split_point:]
    
    return {"train": train_instances, "val": val_instances}

def construct_dataset(examples, instance_groups, selected_instances, target_size):
    """Construct dataset by sampling examples from selected instances."""
    dataset = []
    available_instances = selected_instances.copy()
    
    while len(dataset) < target_size and available_instances:
        # Randomly select an instance
        instance_id = random.choice(available_instances)
        
        # Get available line numbers for this instance
        available_lines = instance_groups[instance_id]
        
        if not available_lines:
            # No more examples available for this instance
            available_instances.remove(instance_id)
            continue
        
        # Randomly select an example line number
        line_number = random.choice(available_lines)
        
        # Find the example with this line number
        example = next(ex for ex in examples if ex['line_number'] == line_number)
        
        # Add to dataset (remove line_number from example)
        example_copy = example.copy()
        del example_copy['line_number']
        dataset.append(example_copy)
        
        # Remove the used line number
        available_lines.remove(line_number)
        
        # If no more examples for this instance, remove it from available
        if not available_lines:
            available_instances.remove(instance_id)
    
    return dataset

def write_jsonl(data, output_file):
    """Write data to jsonl file."""
    with open(output_file, 'w') as f:
        for example in data:
            f.write(json.dumps(example) + '\n')

def calculate_success_rate(dataset):
    """Calculate success rate (proportion of successfully_patched=True)."""
    if not dataset:
        return 0.0
    successful = sum(1 for example in dataset if example['successfully_patched'])
    return successful / len(dataset)

def count_available_examples_by_type(examples, instance_groups, selected_instances):
    """Count available successful and failed examples for given instances."""
    success_count = 0
    failure_count = 0
    
    for instance_id in selected_instances:
        for line_number in instance_groups[instance_id]:
            example = next(ex for ex in examples if ex['line_number'] == line_number)
            if example['successfully_patched']:
                success_count += 1
            else:
                failure_count += 1
    
    return success_count, failure_count

def construct_balanced_dataset(examples, instance_groups, selected_instances, target_successes, target_failures):
    """Construct balanced dataset with specified number of successes and failures."""
    dataset = []
    available_instances = selected_instances.copy()
    successes_added = 0
    failures_added = 0
    
    # Create a copy of instance_groups to modify during sampling
    working_instance_groups = {instance_id: line_numbers.copy() for instance_id, line_numbers in instance_groups.items()}
    
    while (successes_added < target_successes or failures_added < target_failures) and available_instances:
        # Randomly select an instance
        instance_id = random.choice(available_instances)
        
        # Get available line numbers for this instance
        available_lines = working_instance_groups[instance_id]
        
        if not available_lines:
            # No more examples available for this instance
            available_instances.remove(instance_id)
            continue
        
        # Randomly select an example line number
        line_number = random.choice(available_lines)
        
        # Find the example with this line number
        example = next(ex for ex in examples if ex['line_number'] == line_number)
        
        # Check if this example is a success or failure and if we still need that type
        if example['successfully_patched'] and successes_added < target_successes:
            # Add successful example
            example_copy = example.copy()
            del example_copy['line_number']
            dataset.append(example_copy)
            successes_added += 1
            
            # Remove the used line number
            available_lines.remove(line_number)
            
            # If no more examples for this instance, remove it from available
            if not available_lines:
                available_instances.remove(instance_id)
                
        elif not example['successfully_patched'] and failures_added < target_failures:
            # Add failed example
            example_copy = example.copy()
            del example_copy['line_number']
            dataset.append(example_copy)
            failures_added += 1
            
            # Remove the used line number
            available_lines.remove(line_number)
            
            # If no more examples for this instance, remove it from available
            if not available_lines:
                available_instances.remove(instance_id)
        
        # If example type is already at target, skip it (don't remove line number)
        # This allows the loop to continue until both targets are reached
    
    return dataset

def write_metadata(metadata, output_dir):
    """Write metadata to tv_metadata.txt file."""
    metadata_file = os.path.join(output_dir, 'tv_metadata.txt')
    with open(metadata_file, 'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

def main():
    parser = argparse.ArgumentParser(description='Split dataset into train and validation sets')
    parser.add_argument('--input-file', required=True, help='Path to the full dataset')
    parser.add_argument('--sample-size', type=int, required=True, help='Number of examples to sample')
    parser.add_argument('--train-proportion', type=float, required=True, help='Proportion of instances for train set (e.g., 0.8)')
    parser.add_argument('--success-rate', type=float, help='Target success rate for balanced datasets (float between 0 and 1, inclusive)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if args.train_proportion <= 0 or args.train_proportion >= 1:
        raise ValueError("train_proportion must be between 0 and 1 (exclusive)")
    
    if args.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    
    if args.success_rate is not None:
        if args.success_rate < 0 or args.success_rate > 1:
            raise ValueError("success_rate must be between 0 and 1 (inclusive)")
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Load the data from the input file
    print(f"Loading data from {args.input_file}...")
    examples = load_data(args.input_file)
    print(f"Loaded {len(examples)} examples")
    
    # Step 2: Group the data by instance_id
    print("Grouping examples by instance_id...")
    instance_groups = group_by_instance_id(examples)
    print(f"Found {len(instance_groups)} unique instances")
    
    # Step 3: Randomly divide instances into train and validation sets
    print("Splitting instances into train and validation sets...")
    instance_splits = split_instances(instance_groups, args.train_proportion)
    train_instances = instance_splits["train"]
    val_instances = instance_splits["val"]
    print(f"Train instances: {len(train_instances)}, Validation instances: {len(val_instances)}")
    
    # Calculate target sizes
    train_target_size = int(args.sample_size * args.train_proportion)
    val_target_size = args.sample_size - train_target_size
    
    # Check if we're using balanced sampling
    balanced_mode = args.success_rate is not None
    
    # Initialize variables for balanced sampling (will be set if used)
    train_target_successes = None
    train_target_failures = None
    val_target_successes = None
    val_target_failures = None
    
    if balanced_mode:
        print(f"Using balanced sampling with target success rate: {args.success_rate:.1f}")
        
        # Calculate target counts for each set
        train_target_successes = int(train_target_size * args.success_rate)
        train_target_failures = train_target_size - train_target_successes
        val_target_successes = int(val_target_size * args.success_rate)
        val_target_failures = val_target_size - val_target_successes
        
        print(f"Train targets: {train_target_successes} successes, {train_target_failures} failures")
        print(f"Validation targets: {val_target_successes} successes, {val_target_failures} failures")
        
        # Validate that we have enough examples available
        print("Validating available examples...")
        train_available_successes, train_available_failures = count_available_examples_by_type(
            examples, instance_groups, train_instances)
        val_available_successes, val_available_failures = count_available_examples_by_type(
            examples, instance_groups, val_instances)
        
        print(f"Train available: {train_available_successes} successes, {train_available_failures} failures")
        print(f"Validation available: {val_available_successes} successes, {val_available_failures} failures")
        
        # Check if we have enough examples
        if train_available_successes < train_target_successes:
            raise ValueError(f"Not enough successful examples for train set. Need {train_target_successes}, have {train_available_successes}")
        if train_available_failures < train_target_failures:
            raise ValueError(f"Not enough failed examples for train set. Need {train_target_failures}, have {train_available_failures}")
        if val_available_successes < val_target_successes:
            raise ValueError(f"Not enough successful examples for validation set. Need {val_target_successes}, have {val_available_successes}")
        if val_available_failures < val_target_failures:
            raise ValueError(f"Not enough failed examples for validation set. Need {val_target_failures}, have {val_available_failures}")
        
        # Step 4: Construct the balanced train set
        print(f"Constructing balanced train set...")
        train_set = construct_balanced_dataset(examples, instance_groups, train_instances, 
                                             train_target_successes, train_target_failures)
        print(f"Train set size: {len(train_set)}")
        
        # Step 5: Construct the balanced validation set
        print(f"Constructing balanced validation set...")
        val_set = construct_balanced_dataset(examples, instance_groups, val_instances, 
                                           val_target_successes, val_target_failures)
        print(f"Validation set size: {len(val_set)}")
        
    else:
        # Step 4: Construct the train set (random sampling)
        print(f"Constructing train set (target size: {train_target_size})...")
        train_set = construct_dataset(examples, instance_groups, train_instances, train_target_size)
        print(f"Train set size: {len(train_set)}")
        
        # Step 5: Construct the validation set (random sampling)
        print(f"Constructing validation set (target size: {val_target_size})...")
        val_set = construct_dataset(examples, instance_groups, val_instances, val_target_size)
        print(f"Validation set size: {len(val_set)}")
    
    # Step 6: Write the train and validation sets to jsonl files
    input_dir = os.path.dirname(args.input_file)
    if balanced_mode:
        # add success-rate to the file name
        train_file = os.path.join(input_dir, f'train-balanced-{args.success_rate:.1f}.jsonl')    
        val_file = os.path.join(input_dir, f'val-balanced-{args.success_rate:.1f}.jsonl')
    else:
        train_file = os.path.join(input_dir, 'train.jsonl')
        val_file = os.path.join(input_dir, 'val.jsonl')
    
    print(f"Writing train set to {train_file}...")
    write_jsonl(train_set, train_file)
    
    print(f"Writing validation set to {val_file}...")
    write_jsonl(val_set, val_file)
    
    # Step 7: Write metadata file
    print("Writing metadata file...")
    metadata = {
        'sample_size': args.sample_size,
        'train_proportion': args.train_proportion,
        'train_size': len(train_set),
        'val_size': len(val_set),
        'train_file': train_file,
        'val_file': val_file,
        'train_success_rate': calculate_success_rate(train_set),
        'val_success_rate': calculate_success_rate(val_set),
        'balanced_mode': balanced_mode
    }
    
    # Add balanced sampling specific metadata
    if balanced_mode:
        metadata['target_success_rate'] = args.success_rate
        metadata['train_target_successes'] = train_target_successes
        metadata['train_target_failures'] = train_target_failures
        metadata['val_target_successes'] = val_target_successes
        metadata['val_target_failures'] = val_target_failures
    
    write_metadata(metadata, input_dir)
    
    print("Dataset splitting completed successfully!")
    if balanced_mode:
        print(f"Balanced sampling mode with target success rate: {args.success_rate}")
        print(f"Train set: {len(train_set)} examples ({train_target_successes} successes, {train_target_failures} failures), actual success rate: {metadata['train_success_rate']:.3f}")
        print(f"Validation set: {len(val_set)} examples ({val_target_successes} successes, {val_target_failures} failures), actual success rate: {metadata['val_success_rate']:.3f}")
    else:
        print(f"Random sampling mode")
        print(f"Train set: {len(train_set)} examples, success rate: {metadata['train_success_rate']:.3f}")
        print(f"Validation set: {len(val_set)} examples, success rate: {metadata['val_success_rate']:.3f}")

if __name__ == "__main__":
    main()
