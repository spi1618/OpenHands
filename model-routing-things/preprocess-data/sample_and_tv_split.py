# Script to take the full dataset and split it into train and validation sets
# Arguments:
# --input-file: path to the full dataset
# --sample-size: number of examples to sample
# --train-proportion: proportion of instances to put in the train set (eg. 0.8)

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

# Logical flow of this script:
# 1. Load the data from the input file
# 2. Group the data by instance_id and store the line numbers of the examples that belong to each instance_id
#     The output of step 2 is a dictionary of the form {instance_id: [example_line_number, ...], ...}
# 3. Randomly divide the instances into train and validation sets according to the train_proportion argument (ensuring that examples with the same instance_id are in the same set)
#     The output of step 3 is a dictionary of the form {"train": [instance_id, ...], "val": [instance_id, ...]}
# 4. Construct the train set:
#     4a. While the size of the train set is less than sample_size*train_proportion...
#         4a1. Randomly select an instance from the train set
#         4a2. Randomly select an example line number that belongs to the instance
#         4a3. Add the example to the train set
#         4a4. Remove the example line number from the instance's list of line numbers
# 5. Construct the validation set:
#     5a. While the size of the validation set is less than sample_size*(1-train_proportion)...
#         5a1. Randomly select an instance from the validation set
#         5a2. Randomly select an example line number that belongs to the instance
#         5a3. Add the example to the validation set
#         5a4. Remove the example line number from the instance's list of line numbers
# 6. Write the train and validation sets to jsonl files
# 7. Write a metadata file with the metadata information above:


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
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    if args.train_proportion <= 0 or args.train_proportion >= 1:
        raise ValueError("train_proportion must be between 0 and 1 (exclusive)")
    
    if args.sample_size <= 0:
        raise ValueError("sample_size must be positive")
    
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
    
    # Step 4: Construct the train set
    print(f"Constructing train set (target size: {train_target_size})...")
    train_set = construct_dataset(examples, instance_groups, train_instances, train_target_size)
    print(f"Train set size: {len(train_set)}")
    
    # Step 5: Construct the validation set
    print(f"Constructing validation set (target size: {val_target_size})...")
    val_set = construct_dataset(examples, instance_groups, val_instances, val_target_size)
    print(f"Validation set size: {len(val_set)}")
    
    # Step 6: Write the train and validation sets to jsonl files
    input_dir = os.path.dirname(args.input_file)
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
        'val_success_rate': calculate_success_rate(val_set)
    }
    
    write_metadata(metadata, input_dir)
    
    print("Dataset splitting completed successfully!")
    print(f"Train set: {len(train_set)} examples, success rate: {metadata['train_success_rate']:.3f}")
    print(f"Validation set: {len(val_set)} examples, success rate: {metadata['val_success_rate']:.3f}")

if __name__ == "__main__":
    main()
