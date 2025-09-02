#!/usr/bin/env python3
"""
Output Cost Analysis Script

Parses evaluation output JSONL files to generate cost reports broken down by instance and requests.
Extracts token usage from the actual LLM responses in the output files.
"""

"""
Usage:
python parse_output_costs.py <output_file> -o <output_file> --no-summary

Example:
python3 parse_output_costs.py /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-3.5-turbo_maxiter_100_N_v0.43.0-no-hint-run_1_20250812_001159/output.jsonl -o /home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-3.5-turbo_maxiter_100_N_v0.43.0-no-hint-run_1_20250812_001159/output_costs.json --no-summary
"""

import json
import argparse
import os
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path


class OutputCostParser:
    def __init__(self, output_file_path: str):
        self.output_file_path = output_file_path
        self.instance_data = {}
        
    def parse_output(self) -> Dict[str, Any]:
        """Parse the JSONL output file and extract cost information."""
        print(f"Parsing JSONL output file: {self.output_file_path}")
        
        line_count = 0
        instance_count = 0
        
        with open(self.output_file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                line_count += 1
                
                try:
                    data = json.loads(line)
                    instance_count += 1
                    self._extract_cost_data(data, line_num)
                    
                    # Progress indicator
                    if line_count % 100 == 0:
                        print(f"Processed {line_count} lines, {instance_count} instances...")
                        
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        print(f"Completed parsing: {line_count} lines, {instance_count} instances")
        return self._generate_report()
    
    def _extract_cost_data(self, data: Dict[str, Any], line_num: int):
        """Extract cost information from a single JSON line."""
        instance_id = data.get("instance_id", f"unknown_{line_num}")
        
        # Initialize instance data
        self.instance_data[instance_id] = {
            "instance_id": instance_id,
            "cost_history": {},
            "metadata": {
                "instruction": data.get("instruction", "")[:200] + "..." if data.get("instruction") else "",
                "test_result": data.get("test_result", {}),
                "agent_class": data.get("metadata", {}).get("agent_class", "unknown"),
                "llm_config": data.get("metadata", {}).get("llm_config", {}),
                "max_iterations": data.get("metadata", {}).get("max_iterations", 0),
                "start_time": data.get("metadata", {}).get("start_time", ""),
                "dataset": data.get("metadata", {}).get("dataset", ""),
                "data_split": data.get("metadata", {}).get("data_split", ""),
                "line_number": line_num
            }
        }
        
        # Process history entries
        history = data.get("history", [])
        request_counter = 0
        
        for entry in history:
            # Check if this entry has LLM response data
            if self._has_llm_response(entry):
                request_counter += 1
                cost_info = self._extract_cost_from_entry(entry, request_counter)
                if cost_info:
                    request_key = f"request {request_counter}"
                    self.instance_data[instance_id]["cost_history"][request_key] = cost_info
        
        if request_counter > 0:
            print(f"Instance {instance_id}: extracted {request_counter} requests with cost information")
    
    def _has_llm_response(self, entry: Dict[str, Any]) -> bool:
        """Check if an entry contains LLM response data."""
        # Check for tool_call_metadata with model_response
        if "tool_call_metadata" in entry:
            tool_metadata = entry["tool_call_metadata"]
            if "model_response" in tool_metadata:
                model_response = tool_metadata["model_response"]
                return "usage" in model_response
        
        # Also check for direct usage in the entry
        return "usage" in entry
    
    def _extract_cost_from_entry(self, entry: Dict[str, Any], request_id: int) -> Optional[Dict[str, Any]]:
        """Extract cost information from a single history entry."""
        cost_info = {
            "request_id": request_id,
            "timestamp": entry.get("timestamp", ""),
            "source": entry.get("source", "unknown"),
            "message": entry.get("message", "")[:100] + "..." if entry.get("message") else "",
            "model": None,
            "input_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "cached_tokens": 0,
            "content_length": None,
            "action": entry.get("action", ""),
            "args": entry.get("args", {})
        }
        
        # Try to get usage from tool_call_metadata first
        if "tool_call_metadata" in entry:
            tool_metadata = entry["tool_call_metadata"]
            if "model_response" in tool_metadata:
                model_response = tool_metadata["model_response"]
                
                # Extract model name
                if "model" in model_response:
                    model_name = model_response["model"]
                    # Remove neulab/ prefix if present
                    if "/" in model_name:
                        model_name = model_name.split("/")[-1]
                    cost_info["model"] = model_name
                
                # Extract usage information
                if "usage" in model_response:
                    usage = model_response["usage"]
                    cost_info["input_tokens"] = usage.get("prompt_tokens")
                    cost_info["completion_tokens"] = usage.get("completion_tokens")
                    cost_info["total_tokens"] = usage.get("total_tokens")
                    
                    # Check for cached tokens in prompt_tokens_details
                    if "prompt_tokens_details" in usage and usage["prompt_tokens_details"]:
                        prompt_details = usage["prompt_tokens_details"]
                        if "cached_tokens" in prompt_details:
                            cost_info["cached_tokens"] = prompt_details["cached_tokens"]
        
        # Fallback: check for direct usage in the entry
        elif "usage" in entry:
            usage = entry["usage"]
            cost_info["input_tokens"] = usage.get("prompt_tokens")
            cost_info["completion_tokens"] = usage.get("completion_tokens")
            cost_info["total_tokens"] = usage.get("total_tokens")
        
        # Only return if we have meaningful cost information
        if cost_info["total_tokens"] is not None:
            return cost_info
        
        return None
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate the final cost report."""
        # Add summary statistics
        total_requests = sum(len(inst["cost_history"]) for inst in self.instance_data.values())
        total_tokens = sum(
            sum(req["total_tokens"] for req in inst["cost_history"].values())
            for inst in self.instance_data.values()
        )
        
        # Model usage summary
        model_usage = {}
        for inst in self.instance_data.values():
            for req in inst["cost_history"].values():
                model = req["model"] or "unknown"
                if model not in model_usage:
                    model_usage[model] = {
                        "total_requests": 0,
                        "total_input_tokens": 0,
                        "total_completion_tokens": 0,
                        "total_tokens": 0
                    }
                
                model_usage[model]["total_requests"] += 1
                model_usage[model]["total_input_tokens"] += req["input_tokens"] or 0
                model_usage[model]["total_completion_tokens"] += req["completion_tokens"] or 0
                model_usage[model]["total_tokens"] += req["total_tokens"] or 0
        
        report = {
            "summary": {
                "total_instances": len(self.instance_data),
                "total_requests": total_requests,
                "total_tokens": total_tokens,
                "model_usage": model_usage
            },
            "instances": self.instance_data
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: str):
        """Save the report to a JSON file."""
        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Report saved to: {output_file}")
    
    def print_summary(self, report: Dict[str, Any]):
        """Print a human-readable summary of the report."""
        summary = report["summary"]
        
        print("\n" + "="*60)
        print("OUTPUT COST ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total Instances: {summary['total_instances']}")
        print(f"Total Requests: {summary['total_requests']}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print()
        
        print("MODEL USAGE BREAKDOWN:")
        print("-" * 40)
        for model, stats in summary["model_usage"].items():
            print(f"{model:25} | {stats['total_requests']:3d} reqs | "
                  f"{stats['total_tokens']:8,} tokens | "
                  f"{stats['total_input_tokens']:8,} input | "
                  f"{stats['total_completion_tokens']:8,} output")
        
        print("\nINSTANCES:")
        print("-" * 40)
        for instance_id, instance_data in report["instances"].items():
            req_count = len(instance_data["cost_history"])
            total_tokens = sum(req["total_tokens"] for req in instance_data["cost_history"].values())
            metadata = instance_data["metadata"]
            agent_class = metadata.get("agent_class", "unknown")
            dataset = metadata.get("dataset", "unknown")
            print(f"{instance_id:30} | {req_count:3d} reqs | {total_tokens:8,} tokens | {agent_class:15} | {dataset}")


def generate_output_filename(output_file_path: str) -> str:
    """Generate output filename based on input filename."""
    # Extract the timestamp from the input file path
    # Example path: .../outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/gpt-3.5-turbo_maxiter_100_N_v0.43.0-no-hint-run_1_20250809_232336/output.jsonl
    # We want to extract: 20250809_232336
    
    # Look for timestamp pattern in the path (YYYYMMDD_HHMMSS)
    timestamp_match = re.search(r'(\d{8}_\d{6})', output_file_path)
    if timestamp_match:
        timestamp = timestamp_match.group(1)
        # Extract the base filename without extension
        output_filename = os.path.basename(output_file_path)
        base_name = output_filename.replace('.jsonl', '')
        
        return f"router_cost_reports/cost_report_{base_name}_{timestamp}.json"
    else:
        # Fallback if no timestamp found in path
        output_filename = os.path.basename(output_file_path)
        base_name = output_filename.replace('.jsonl', '')
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"router_cost_reports/cost_report_{base_name}_{current_timestamp}.json"


def main():
    parser = argparse.ArgumentParser(description="Parse evaluation output JSONL files and generate cost reports")
    parser.add_argument("output_file", help="Path to the evaluation output JSONL file")
    parser.add_argument("-o", "--output", 
                       help="Output JSON file (auto-generated if not specified)")
    parser.add_argument("--no-summary", action="store_true", 
                       help="Don't print summary to console")
    
    args = parser.parse_args()
    
    # Check if output file exists
    if not Path(args.output_file).exists():
        print(f"Error: Output file '{args.output_file}' not found")
        return 1
    
    # Generate output filename if not specified
    if args.output:
        output_file = args.output
    else:
        output_file = generate_output_filename(args.output_file)
    
    print(f"Output will be saved to: {output_file}")
    
    try:
        # Parse the output file
        parser = OutputCostParser(args.output_file)
        report = parser.parse_output()
        
        # Save the report
        parser.save_report(report, output_file)
        
        # Print summary unless disabled
        if not args.no_summary:
            parser.print_summary(report)
        
        return 0
        
    except Exception as e:
        print(f"Error parsing output file: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
