# Checks that the train and validation sets have disjoint instance ids
# Assumes that the data files are jsonl files where each line is a json object with key "instance_id"

import json
import sys

TRAIN_PATH = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned_no-oh-prompt_partial-trajectories_2025-08-28T00-49-09/20000-samples/train.jsonl"
VAL_PATH = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned_no-oh-prompt_partial-trajectories_2025-08-28T00-49-09/20000-samples/val.jsonl"

def load_instance_ids(file_path):
    """Load instance IDs from a JSONL file."""
    instance_ids = set()
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                instance_ids.add(data['instance_id'])
        return instance_ids
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        sys.exit(1)
    except KeyError:
        print(f"Error: File {file_path} missing 'instance_id' key")
        sys.exit(1)

def main():
    # Load instance IDs from both files
    train_ids = load_instance_ids(TRAIN_PATH)
    val_ids = load_instance_ids(VAL_PATH)
    
    # Check for intersection
    intersection = train_ids & val_ids
    
    if intersection:
        print(f"ERROR: Found {len(intersection)} overlapping instance IDs:")
        for instance_id in sorted(intersection)[:10]:  # Show first 10
            print(f"  {instance_id}")
        if len(intersection) > 10:
            print(f"  ... and {len(intersection) - 10} more")
        sys.exit(1)
    else:
        print(f"✓ Train and validation sets are disjoint")
        print(f"  Train instances: {len(train_ids)}")
        print(f"  Validation instances: {len(val_ids)}")

if __name__ == "__main__":
    main()



