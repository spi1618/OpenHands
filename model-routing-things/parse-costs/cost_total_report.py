# Takes in a cost output jsonl files and computes + prints the average cost per instance

import json
import sys

def compute_average_cost(jsonl_file_path):
    total_cost = 0
    total_instances = 0
    with open(jsonl_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            cost = data['total_est_cost']
            total_cost += cost
            total_instances += 1
    average_cost = total_cost / total_instances
    print(f"Average cost per instance: {average_cost}")

def main():
    jsonl_file_path = "/home/sophiapi/model-routing/outlog-cost-reports/cost_report_router_log_20250928_210535.jsonl"
    compute_average_cost(jsonl_file_path)

if __name__ == "__main__":
    main()
