#!/bin/bash

# Input file
INPUT_FILE="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/pruned-3_swe_gym_stepwise_trajectories_2025-07-26T23-45-14.jsonl"

# Output directory (same as input)
OUTPUT_DIR="/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets"

# Output files
TRAIN_FILE="$OUTPUT_DIR/router_train_pruned-3.jsonl"
VALID_FILE="$OUTPUT_DIR/router_valid_pruned-3.jsonl"

# Get total number of lines
TOTAL_LINES=$(wc -l < "$INPUT_FILE")

# Calculate split points (80% for train)
TRAIN_LINES=$((TOTAL_LINES * 80 / 100))
VALID_LINES=$((TOTAL_LINES - TRAIN_LINES))

echo "Total lines: $TOTAL_LINES"
echo "Train lines: $TRAIN_LINES"
echo "Valid lines: $VALID_LINES"

# Shuffle and split
shuf "$INPUT_FILE" | head -n "$TRAIN_LINES" > "$TRAIN_FILE"
shuf "$INPUT_FILE" | tail -n "$VALID_LINES" > "$VALID_FILE"

echo "Split complete!"
echo "Train file: $TRAIN_FILE"
echo "Valid file: $VALID_FILE"