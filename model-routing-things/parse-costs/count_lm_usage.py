#!/usr/bin/env python3
"""
Script to count language model usage from OpenHands evaluation output JSONL files.

This script analyzes JSONL files containing evaluation outputs and generates a report
showing how many times each language model was used across all steps in the history.

To use this script, modify the INPUT_JSONL_FILE_PATH variable in the main() function
to point to your desired JSONL file, then run: python count_lm_usage.py
"""

import json
import sys
import os
from collections import defaultdict
from pathlib import Path


def count_lm_usage(jsonl_file_path):
    """
    Count language model usage from a JSONL file.
    
    Args:
        jsonl_file_path (str): Path to the input JSONL file
        
    Returns:
        tuple: (model_counts, skipped_instances, total_instances)
    """
    model_counts = defaultdict(int)
    skipped_instances = 0
    total_instances = 0
    
    print(f"Processing file: {jsonl_file_path}")
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                total_instances += 1
                
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse JSON on line {line_num}: {e}")
                    skipped_instances += 1
                    continue
                
                # Check if history field exists
                if 'history' not in data:
                    print(f"Warning: No 'history' field found in instance on line {line_num}, skipping")
                    skipped_instances += 1
                    continue
                
                history = data['history']
                if not isinstance(history, list):
                    print(f"Warning: 'history' field is not a list in instance on line {line_num}, skipping")
                    skipped_instances += 1
                    continue
                
                # Process each step in the history
                for step in history:
                    if not isinstance(step, dict):
                        continue
                        
                    # Check if tool_call_metadata exists
                    if 'tool_call_metadata' not in step:
                        continue
                        
                    tool_call_metadata = step['tool_call_metadata']
                    if not isinstance(tool_call_metadata, dict):
                        continue
                    
                    # Check if model_response exists
                    if 'model_response' not in tool_call_metadata:
                        continue
                        
                    model_response = tool_call_metadata['model_response']
                    if not isinstance(model_response, dict):
                        continue
                    
                    # Extract model name
                    if 'model' in model_response:
                        model_name = model_response['model']
                        model_counts[model_name] += 1
                        
    except FileNotFoundError:
        print(f"Error: File not found: {jsonl_file_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    return model_counts, skipped_instances, total_instances


def generate_report(model_counts, skipped_instances, total_instances, output_file_path):
    """
    Generate a text report of language model usage.
    
    Args:
        model_counts (dict): Dictionary mapping model names to usage counts
        skipped_instances (int): Number of instances that were skipped
        total_instances (int): Total number of instances processed
        output_file_path (str): Path where the report should be saved
    """
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write("Language Model Usage Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total instances processed: {total_instances}\n")
        f.write(f"Instances skipped: {skipped_instances}\n")
        f.write(f"Instances with valid history: {total_instances - skipped_instances}\n\n")
        
        if not model_counts:
            f.write("No language model usage found in the data.\n")
            return
        
        f.write("Model Usage Counts:\n")
        f.write("-" * 30 + "\n")
        
        # Sort models by usage count (descending)
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        
        total_model_calls = sum(model_counts.values())
        
        for model_name, count in sorted_models:
            percentage = (count / total_model_calls) * 100 if total_model_calls > 0 else 0
            f.write(f"{model_name:<50} {count:>6} calls ({percentage:>5.1f}%)\n")
        
        f.write("-" * 30 + "\n")
        f.write(f"{'Total model calls':<50} {total_model_calls:>6}\n")


def main():
    """Main function to execute the analysis."""
    # Set the input file path here
    INPUT_JSONL_FILE_PATH = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/SWE-Gym__SWE-Gym-train/CodeActAgent/gpt-3.5-turbo_maxiter_100_N_v0.43.0-no-hint-run_1_20250928_210601/output.jsonl"
    
    jsonl_file_path = INPUT_JSONL_FILE_PATH
    
    # Validate input file
    if not os.path.exists(jsonl_file_path):
        print(f"Error: Input file does not exist: {jsonl_file_path}")
        sys.exit(1)
    
    if not jsonl_file_path.endswith('.jsonl'):
        print("Warning: Input file does not have .jsonl extension")
    
    # Count language model usage
    model_counts, skipped_instances, total_instances = count_lm_usage(jsonl_file_path)
    
    # Generate output file path
    input_path = Path(jsonl_file_path)
    output_file_path = input_path.parent / f"{input_path.stem}_lm_usage_report.txt"
    
    # Generate report
    generate_report(model_counts, skipped_instances, total_instances, output_file_path)
    
    print(f"\nAnalysis complete!")
    print(f"Processed {total_instances} instances")
    print(f"Skipped {skipped_instances} instances")
    print(f"Found {len(model_counts)} unique language models")
    print(f"Report saved to: {output_file_path}")
    
    # Print summary to console
    if model_counts:
        print(f"\nTop 5 most used models:")
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, count) in enumerate(sorted_models[:5], 1):
            print(f"{i}. {model_name}: {count} calls")


if __name__ == "__main__":
    main()
