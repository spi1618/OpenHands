#!/usr/bin/env python3
"""
Script to analyze success rates from the OpenHands evaluation dataset.
Counts successful and failed patches by model.
Supports both single file analysis and train/validation file pair analysis.
"""

import json
import sys
import os
import argparse
from datetime import datetime
from collections import defaultdict
from pathlib import Path

def analyze_success_rates(jsonl_file_path):
    """
    Analyze success rates from a JSONL file containing evaluation results.
    NOTE: assumes that the JSONL file contains the following fields:
        - model
        - successfully_patched
        - instance_id
        - partial_trajectory (this one is not used)
    
    Args:
        jsonl_file_path (str): Path to the JSONL file to analyze
        
    Returns:
        dict: Analysis results with counts by model
    """
    # If the path is not absolute, assume it's relative to the datasets directory
    if not jsonl_file_path.startswith('/'):
        jsonl_file_path = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/" + jsonl_file_path
    
    results = defaultdict(lambda: {"successful": 0, "failed": 0, "total": 0, "successful_instances": set(), "total_instances": set()})
    total_examples = 0
    
    try:
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    data = json.loads(line)
                    
                    # Extract key fields
                    model = data.get('model', 'unknown')
                    successfully_patched = data.get('successfully_patched', False)
                    instance_id = data.get('instance_id', 'unknown')
                    
                    # Update counts
                    results[model]["total"] += 1
                    results[model]["total_instances"].add(instance_id)
                    
                    if successfully_patched:
                        results[model]["successful"] += 1
                        results[model]["successful_instances"].add(instance_id)
                    else:
                        results[model]["failed"] += 1
                    
                    total_examples += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON on line {line_num}: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File '{jsonl_file_path}' not found.")
        return None
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    return results, total_examples

def extract_dataset_name(file_path):
    """Extract dataset name from file path"""
    filename = os.path.basename(file_path)
    # Remove file extension
    return os.path.splitext(filename)[0]

def write_output(output_f, text):
    """Write text to both console and output file"""
    print(text)
    output_f.write(text + '\n')
    output_f.flush()  # Ensure immediate writing

def print_analysis(results, total_examples, output_f, dataset_name=""):
    """
    Print formatted analysis results.
    
    Args:
        results (dict): Analysis results from analyze_success_rates
        total_examples (int): Total number of examples processed
        output_f: File object to write output to
        dataset_name (str): Optional name for the dataset (e.g., "Train", "Validation")
    """
    if not results:
        return
    
    # Add dataset name to header if provided
    header = "SUCCESS RATE ANALYSIS"
    if dataset_name:
        header += f" - {dataset_name}"
    
    write_output(output_f, "=" * 80)
    write_output(output_f, header)
    write_output(output_f, "=" * 80)
    write_output(output_f, f"Total examples processed: {total_examples}")
    write_output(output_f, "")
    
    # Overall statistics
    total_successful = sum(model_data["successful"] for model_data in results.values())
    total_failed = sum(model_data["failed"] for model_data in results.values())
    overall_success_rate = (total_successful / total_examples * 100) if total_examples > 0 else 0
    
    write_output(output_f, f"OVERALL STATISTICS:")
    write_output(output_f, f"  Successful patches: {total_successful}")
    write_output(output_f, f"  Failed patches: {total_failed}")
    write_output(output_f, f"  Overall success rate: {overall_success_rate:.2f}%")
    write_output(output_f, "")
    
    # Per-model breakdown
    write_output(output_f, "BREAKDOWN BY MODEL:")
    write_output(output_f, "-" * 121)
    write_output(output_f, f"{'Model':<30} {'Successful':<12} {'Failed':<8} {'Total':<7} {'Success Rate':<14} {'Success Inst':<14} {'Total Inst':<12} {'Inst Success Rate':<17}")
    write_output(output_f, "-" * 121)
    
    for model, data in sorted(results.items()):
        success_rate = (data["successful"] / data["total"] * 100) if data["total"] > 0 else 0
        successful_instances = len(data["successful_instances"])
        total_instances = len(data["total_instances"])
        instance_success_rate = (successful_instances / total_instances * 100) if total_instances > 0 else 0
        write_output(output_f, f"{model:<30} {data['successful']:<12} {data['failed']:<8} {data['total']:<7} {success_rate:<11.2f}%   {successful_instances:<14} {total_instances:<12} {instance_success_rate:<16.2f}%")
    
    write_output(output_f, "-" * 121)
    write_output(output_f, "")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Analyze success rates from evaluation datasets')
    parser.add_argument('--output-dir', default='/home/sophiapi/model-routing/success_rate_reports', 
                       help='Directory to save output files (default: /home/sophiapi/model-routing/success_rate_reports)')
    parser.add_argument('files', nargs='+', help='JSONL file(s) to analyze (1 for single file, 2 for train/validation pair)')
    return parser.parse_args()

def main():
    """Main function to run the analysis."""
    args = parse_args()
    
    if len(args.files) < 1 or len(args.files) > 2:
        print("Usage: python3 analyze_success_rates.py [--output-dir <output_dir>] <jsonl_file_path> [validation_jsonl_file_path]")
        print("\nExamples:")
        print("  # Single file analysis:")
        print("  python3 analyze_success_rates.py model-4_instance-5_pruned-4_with-ids_swe_gym_stepwise_trajectories_2025-08-02T19-23-01.jsonl")
        print("  python3 analyze_success_rates.py --output-dir ./results model-4_instance-5_pruned-4_with-ids_swe_gym_stepwise_trajectories_2025-08-02T19-23-01.jsonl")
        print("  python3 analyze_success_rates.py --output-dir ./results /full/path/to/your/file.jsonl")
        print()
        print("  # Train/Validation pair analysis:")
        print("  python3 analyze_success_rates.py train_file.jsonl validation_file.jsonl")
        print("  python3 analyze_success_rates.py --output-dir ./results train_file.jsonl validation_file.jsonl")
        print("  python3 analyze_success_rates.py --output-dir ./results /path/to/train.jsonl /path/to/validation.jsonl")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract dataset name from first file
    dataset_name = extract_dataset_name(args.files[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"success_rates_{dataset_name}_{timestamp}.txt"
    output_file = os.path.join(args.output_dir, output_filename)
    
    # Open output file for writing
    output_f = open(output_file, 'w')
    
    # Write header with file info
    print(f"Saving analysis results to: {output_file}")
    output_f.write(f"Success Rate Analysis Results\n")
    output_f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_f.write(f"Dataset: {dataset_name}\n")
    
    if len(args.files) == 1:
        # Single file analysis
        jsonl_file_path = args.files[0]
        output_f.write(f"File: {jsonl_file_path}\n")
        output_f.write("=" * 80 + "\n\n")
        
        print(f"Analyzing success rates from: {jsonl_file_path}")
        print("This may take a moment for large files...")
        print()
        
        results, total_examples = analyze_success_rates(jsonl_file_path)
        
        if results is not None:
            print_analysis(results, total_examples, output_f)
        else:
            output_f.close()
            sys.exit(1)
    
    else:
        # Train/Validation pair analysis
        train_file_path = args.files[0]
        validation_file_path = args.files[1]
        output_f.write(f"Train file: {train_file_path}\n")
        output_f.write(f"Validation file: {validation_file_path}\n")
        output_f.write("=" * 80 + "\n\n")
        
        print(f"Analyzing success rates from train file: {train_file_path}")
        print(f"Analyzing success rates from validation file: {validation_file_path}")
        print("This may take a moment for large files...")
        print()
        
        # Analyze train file
        train_results, train_total = analyze_success_rates(train_file_path)
        if train_results is None:
            print("Failed to analyze train file.")
            output_f.close()
            sys.exit(1)
        
        # Analyze validation file
        validation_results, validation_total = analyze_success_rates(validation_file_path)
        if validation_results is None:
            print("Failed to analyze validation file.")
            output_f.close()
            sys.exit(1)
        
        # Print results for both datasets
        print_analysis(train_results, train_total, output_f, "Train")
        print_analysis(validation_results, validation_total, output_f, "Validation")
    
    # Close the output file
    output_f.close()
    print(f"Analysis complete. Results saved to: {output_file}")

if __name__ == "__main__":
    main() 