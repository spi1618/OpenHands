#!/bin/bash

# Input file
INPUT_FILE="//home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_pruned-4_with-ids_swe_gym_stepwise_trajectories_2025-08-05T15-00-53.jsonl"

# Output directory (same as input)
OUTPUT_DIR="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets"

# Output files
TRAIN_FILE="$OUTPUT_DIR/router_train_model-5_instance-100_pruned-4_with-ids_by-example.jsonl"
VALID_FILE="$OUTPUT_DIR/router_valid_model-5_instance-100_pruned-4_with-ids_by-example.jsonl"

echo "=== DATA ANALYSIS ==="
echo "Total lines in input file: $(wc -l < "$INPUT_FILE")"

echo ""
echo "Lines per model:"
awk -F'"' '
    /"model":/ {
        model = $4
        model_counts[model]++
    }
    END {
        for (model in model_counts) {
            print "  " model ": " model_counts[model] " lines"
        }
    }
' "$INPUT_FILE"

echo ""
echo "Lines for python__mypy-9577 instance:"
awk -F'"' '
    /"instance_id":/ && $8 == "python__mypy-9577" {
        count++
    }
    END {
        print "  Total: " count " lines"
    }
' "$INPUT_FILE"

echo ""
echo "Lines for python__mypy-9577 instance by model:"
awk -F'"' '
    /"model":/ {
        model = $4
        models[NR] = model
    }
    /"instance_id":/ && $8 == "python__mypy-9577" {
        instance_lines[NR] = 1
    }
    END {
        for (line_num in instance_lines) {
            if (line_num in models) {
                model_counts[models[line_num]]++
            }
        }
        for (model in model_counts) {
            print "  " model ": " model_counts[model] " lines"
        }
    }
' "$INPUT_FILE"

echo ""
echo "=== FILTERING AND SPLITTING ==="

# Create temporary directory for processing
TEMP_DIR="/tmp/split_data_$$"
mkdir -p "$TEMP_DIR"

echo "Processing data to group by instance_id (excluding python__mypy-9577)..."

# Extract instance_id from each line and create a mapping
# This creates files like: instance_id_1.jsonl, instance_id_2.jsonl, etc.
# Filter out python__mypy-9577 instance
awk -F'"' '
    /"instance_id":/ {
        instance_id = $8
        if (instance_id != "python__mypy-9577") {
            if (instance_id in seen) {
                print $0 >> "'$TEMP_DIR'/instance_" seen[instance_id] ".jsonl"
            } else {
                seen[instance_id] = ++count
                print $0 >> "'$TEMP_DIR'/instance_" count ".jsonl"
            }
        }
    }
' "$INPUT_FILE"

# Get list of instance files and shuffle them
INSTANCE_FILES=($(ls "$TEMP_DIR"/instance_*.jsonl | shuf))

# Calculate split points (80% for train)
TOTAL_INSTANCES=${#INSTANCE_FILES[@]}
TRAIN_INSTANCES=$((TOTAL_INSTANCES * 80 / 100))
VALID_INSTANCES=$((TOTAL_INSTANCES - TRAIN_INSTANCES))

echo "Total instances: $TOTAL_INSTANCES"
echo "Train instances: $TRAIN_INSTANCES"
echo "Valid instances: $VALID_INSTANCES"

# Create train file
echo "Creating train file..."
for ((i=0; i<TRAIN_INSTANCES; i++)); do
    cat "${INSTANCE_FILES[$i]}" >> "$TRAIN_FILE"
done

# Create validation file
echo "Creating validation file..."
for ((i=TRAIN_INSTANCES; i<TOTAL_INSTANCES; i++)); do
    cat "${INSTANCE_FILES[$i]}" >> "$VALID_FILE"
done

# Clean up temporary files
rm -rf "$TEMP_DIR"

echo "Split complete!"
echo "Train file: $TRAIN_FILE"
echo "Valid file: $VALID_FILE"
echo "Train file lines: $(wc -l < "$TRAIN_FILE")"
echo "Valid file lines: $(wc -l < "$VALID_FILE")"