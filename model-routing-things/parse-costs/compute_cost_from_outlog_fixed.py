import json
import os
import re
import sys
import ast

AVAILABLE_MODELS = json.load(open("/home/sophiapi/model-routing/static_things/model_name_mapping.json", "r"))
AVAILABLE_MODELS = list(AVAILABLE_MODELS.values())

TOKEN_COSTS = json.load(open("/home/sophiapi/model-routing/static_things/token_costs.json", "r"))

OUTPUT_DIR = "/home/sophiapi/model-routing/outlog-cost-reports/"

# Here is an overview of the logic:

# This script takes in a log file and computes the cost of each instance in the log file.
# The output should be a JSONL file where each line is a JSON object containing the extracted information for a single instance

# See cost_report_template.json for an example of what any single instance's JSON object should look like; each instance's JSON should have the following fields:
# instance_id: the instance id that the instance belongs to
# history: a list of JSON objects, each representing a single request corresponding to the instance
# total_est_cost: the total estimated cost of the instance
# model_usage: a dictionary of model names and the number of times each model was used for the instance

# The outlog contains information about each request. Every line of output that is related to a single request is tagged with a uuid unique to that request
# The first line of output that is related to a request begins with the string "[DEBUG - REQUEST ID] Request ID: {uuid}" - see line 13 of router_log_2025913_215337_first500.log for an example of this
# The last line of output that is related to a request begins with the string "[DEBUG - {uuid} {timestamp}] Response usage: Usage({usage info here})" - see line 90 of router_log_2025913_215337_first500.log for an example of this

# You will have to identify the instance id that each request belongs to by examining the first line of output with the request's uuid that has the format:
# "[DEBUG - {uuid} {timestamp}] Message: {'role': 'user', 'content': {*}\n\n--------------------- NEW TASK DESCRIPTION ---------------------\n<uploaded_files>\n/workspace/{instance_id}\n</uploaded_files>{*}"
# (where {*} is any additional information that may be present) - see line 21 of router_log_2025913_215337_first500.log for an example of this

# You will also have to identify the selected model for each request by examining the line of output with the request's uuid that has the format:
# "[DEBUG - a1c414f6-47cd-4e2e-9cff-c048abd386a1 2025-09-13T21:54:51.666900] Final selected model: {selected_model}" - see line 80 of router_log_2025913_215337_first500.log for an example of this
# You should get exactly one line of output with the request's uuid that has this format

# You will also have to identify the estimated cache read tokens and the estimated cache write tokens for the selected model for each request
# Get selected_model_nickname by taking the selected_model string and removing the "litellm_proxy/" prefix
# First, identify the line of output with the request's uuid that has the format:
# "[DEBUG - {uuid} {timestamp}] Counting cache read and write tokens for model: {selected_model_nickname}"
# You should get exactly one line of output with the request's uuid that has this format
# Then, identify the *FIRST* line of output with the request's uuid that has the format:
# "[DEBUG - 36ec2c0d-2e5b-4db0-bf4b-7fdc263e613b] Manually computed cache read tokens: {estimated_cache_read_tokens}"
# See line 34 of router_log_2025913_215337_first500.log for an example of this
# Then, identify the *FIRST* line of output with the request's uuid that has the format:
# "[DEBUG - 36ec2c0d-2e5b-4db0-bf4b-7fdc263e613b] Manually computed cache write tokens: {estimated_cache_write_tokens}"
# See line 35 of router_log_2025913_215337_first500.log for an example of this

# You will also have to identify the actual output tokens for the selected model for each request
# First, identify the line of output with the request's uuid that has the format:
# "[DEBUG - {uuid} {timestamp}] Response usage: Usage(completion_tokens={actual_output_tokens}{*})"
# (where {*} is any additional information that may be present)
# You should get exactly one line of output with the request's uuid that has this format
# See line 90 of router_log_2025913_215337_first500.log for an example of this

# Finally, compute the estimated cost for each request by multiplying the estimated cache read tokens, the estimated cache write tokens, and the actual output tokens by the cost per token for the selected model
# Refer to token_costs.json for the cost per token for each model

# Save everything in a JSONL file where each line is a JSON object containing the extracted information for a single instance; save to OUTPUT_DIR

# def extract_instance_id_from_message(message_content):
    # """Extract instance ID from message content looking for the NEW TASK DESCRIPTION pattern."""
    # import re
    
    # # Look for the exact literal pattern:
    # # \n\n--------------------- NEW TASK DESCRIPTION ---------------------\n<uploaded_files>\n/workspace/{instance_id}\n</uploaded_files>\n\n
    # pattern = r'\\n\\n-{21} NEW TASK DESCRIPTION -{21}\\n<uploaded_files>\\n/workspace/([^/\\n]+)\\n</uploaded_files>\\n\\n'
    # match = re.search(pattern, message_content)
    # if match:
    #     instance_id = match.group(1).strip()
    #     if instance_id:
    #         return instance_id
    
    # return "unknown_instance"

def extract_instance_id_from_message(text: str) -> str | None:
    """Return the first '<uploaded_files>...'</uploaded_files>' block (inclusive).
    Exact tags only; no regex; works with real or literal '\n's."""
    start_tag = "<uploaded_files>"
    end_tag   = "</uploaded_files>"
    i = text.find(start_tag)
    if i == -1:
        return None
    j = text.find(end_tag, i + len(start_tag))
    if j == -1:
        return None
    return text[i : j + len(end_tag)]


def parse_log_file(log_file_path):
    """Parse the log file and extract request information."""
    requests = {}
    current_request_uuid = None
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Check for request ID start
            if "[DEBUG - REQUEST ID] Request ID:" in line:
                current_request_uuid = line.split("Request ID: ")[1].strip()
                requests[current_request_uuid] = {
                    'request_uuid': current_request_uuid,
                    'instance_id': None,
                    'selected_model': None,
                    'estimated_cache_read_tokens': None,
                    'estimated_cache_write_tokens': None,
                    'actual_output_tokens': None,
                    'est_cost': 0.0
                }
                continue
            
            if current_request_uuid is None:
                continue
                
            # Extract instance ID from message content
            if f"[DEBUG - {current_request_uuid}" in line and "Message:" in line and requests[current_request_uuid]['instance_id'] is None:
                try:
                    # Extract the message content
                    message_start = line.find("Message: {")
                    if message_start != -1:
                        message_content = line[message_start + 9:]  # Skip "Message: "
                        requests[current_request_uuid]['instance_id'] = extract_instance_id_from_message(message_content)
                except:
                    pass
            
            # Extract selected model
            if f"[DEBUG - {current_request_uuid}" in line and "Final selected model:" in line:
                model = line.split("Final selected model: ")[1].strip()
                requests[current_request_uuid]['selected_model'] = model
            
            # Extract cache read tokens (first occurrence)
            if f"[DEBUG - {current_request_uuid}" in line and "Manually computed cache read tokens:" in line:
                if requests[current_request_uuid]['estimated_cache_read_tokens'] is None:
                    tokens = line.split("Manually computed cache read tokens: ")[1].strip()
                    try:
                        requests[current_request_uuid]['estimated_cache_read_tokens'] = int(tokens)
                    except ValueError:
                        pass
            
            # Extract cache write tokens (first occurrence)
            if f"[DEBUG - {current_request_uuid}" in line and "Manually computed cache write tokens:" in line:
                if requests[current_request_uuid]['estimated_cache_write_tokens'] is None:
                    tokens = line.split("Manually computed cache write tokens: ")[1].strip()
                    try:
                        requests[current_request_uuid]['estimated_cache_write_tokens'] = int(tokens)
                    except ValueError:
                        pass
            
            # Extract actual output tokens from response usage
            if f"[DEBUG - {current_request_uuid}" in line and "Response usage: Usage(" in line:
                try:
                    # Extract completion_tokens from the Usage string
                    usage_str = line.split("Response usage: Usage(")[1]
                    # Look for completion_tokens=X
                    import re
                    match = re.search(r'completion_tokens=(\d+)', usage_str)
                    if match:
                        requests[current_request_uuid]['actual_output_tokens'] = int(match.group(1))
                except:
                    pass
    
    return requests

def calculate_cost(selected_model, cache_read_tokens, cache_write_tokens, output_tokens, token_costs):
    """Calculate the estimated cost for a request."""
    if not selected_model:
        return 0.0
    
    # Remove litellm_proxy/neulab/ prefix to get the model name for cost lookup
    model_name = selected_model.replace("litellm_proxy/neulab/", "").replace("litellm_proxy/", "")
    
    # Map model names to cost keys
    cost_key_mapping = {
        "claude-3-5-haiku-20241022": "claude-3-5-haiku",
        "claude-sonnet-4-20250514": "claude-sonnet-4",
        "kimi-k2-0711-preview": "kimi-k2",
        "deepseek-v3": "deepseek-v3",
        "devstral-small-2505": "devstral-small"
    }
    
    cost_key = cost_key_mapping.get(model_name, model_name)
    
    if cost_key not in token_costs:
        print(f"Warning: Cost key '{cost_key}' not found in token_costs for model '{selected_model}'")
        return 0.0
    
    costs = token_costs[cost_key]
    
    # Calculate total cost
    cache_read_cost = (cache_read_tokens or 0) * costs.get('cached_input_cost', 0)
    cache_write_cost = (cache_write_tokens or 0) * costs.get('uncached_input_cost', 0)
    output_cost = (output_tokens or 0) * costs.get('output_cost', 0)
    
    return cache_read_cost + cache_write_cost + output_cost

def group_requests_by_instance(requests):
    """Group requests by instance ID."""
    instances = {}
    
    for request_uuid, request_data in requests.items():
        instance_id = request_data['instance_id'] or 'unknown_instance'
        
        if instance_id not in instances:
            instances[instance_id] = {
                'instance_id': instance_id,
                'history': [],
                'total_est_cost': 0.0,
                'model_usage': {}
            }
        
        # Calculate cost for this request
        cost = calculate_cost(
            request_data['selected_model'],
            request_data['estimated_cache_read_tokens'],
            request_data['estimated_cache_write_tokens'],
            request_data['actual_output_tokens'],
            TOKEN_COSTS
        )
        
        request_data['est_cost'] = cost
        
        # Add to instance
        instances[instance_id]['history'].append(request_data)
        instances[instance_id]['total_est_cost'] += cost
        
        # Update model usage
        model = request_data['selected_model']
        if model:
            if model not in instances[instance_id]['model_usage']:
                instances[instance_id]['model_usage'][model] = 0
            instances[instance_id]['model_usage'][model] += 1
    
    return instances

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 compute_cost_from_outlog_fixed.py <log_file_path>")
        sys.exit(1)
    
    log_file_path = sys.argv[1]
    
    if not os.path.exists(log_file_path):
        print(f"Error: Log file {log_file_path} does not exist")
        sys.exit(1)
    
    # Parse the log file
    print(f"Parsing log file: {log_file_path}")
    requests = parse_log_file(log_file_path)
    print(f"Found {len(requests)} requests")
    
    # Group by instance
    instances = group_requests_by_instance(requests)
    print(f"Found {len(instances)} instances")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate output filename
    log_filename = os.path.basename(log_file_path)
    output_filename = f"cost_report_{log_filename.replace('.log', '.jsonl')}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    # Write JSONL output
    with open(output_path, 'w') as f:
        for instance_id, instance_data in instances.items():
            f.write(json.dumps(instance_data) + '\n')
    
    print(f"Cost report saved to: {output_path}")
    
    # Print summary
    total_cost = sum(instance['total_est_cost'] for instance in instances.values())
    print(f"Total estimated cost: ${total_cost:.6f}")
    
    # Print detailed summary
    print("\nSummary by instance:")
    for instance_id, instance_data in instances.items():
        print(f"  {instance_id}: ${instance_data['total_est_cost']:.6f} ({len(instance_data['history'])} requests)")
        for model, count in instance_data['model_usage'].items():
            model_short = model.replace("litellm_proxy/neulab/", "")
            print(f"    - {model_short}: {count} requests")

if __name__ == "__main__":
    main()

