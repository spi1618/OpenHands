from transformers import pipeline, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import json
import random
import os
from datetime import datetime
import torch
import numpy as np
from typing import Dict

def find_model_from_prompt(prompt):
    # Determine the model from the prompt
    # The model name will be the first word after the string "### Candidate model\n[M] ", delimited by spaces and newlines
    # For example: "### Candidate model\n[M] deepseek-v3\n\n" --> "deepseek-v3"
    # If the string is not found, return None
    if "### Candidate model\n[M] " in prompt:
        return prompt.split("### Candidate model\n[M] ")[1].split("\n")[0]
    else:
        return None


def get_model_logits(prompt: str, tokenizer: AutoTokenizer, model: AutoModelForCausalLM, yes_token_id: int, no_token_id: int) -> Dict[str, float]:
    """
    Take the truncated prompt string and give it to the model.
    Return the logits for the next token (only consider [YES] and [NO]).
    Note this returns the raw logits, NOT the probabilities.
    """
    # Take the truncated prompt string and give it to the model
    #     Tokenize the input string
    inputs = tokenizer(prompt, return_tensors="pt")
    #     Do a forward pass through the model and extract the logits
    with torch.no_grad():
        outputs = model(**inputs) # Tensor of shape (batch_size, sequence_length, vocab_size)
        next_token_logits = outputs.logits[0,-1, :] # Shape: [vocab_size]
        # DEBUG: check the shape of next_token_logits
        print(f"next_token_logits shape: {next_token_logits.shape}")
        if next_token_logits.shape[0] != len(tokenizer):
            raise ValueError("next_token_logits shape does not match tokenizer vocab size")
    # Get the logits for the next token (only consider [YES] and [NO])
        yes_logit = next_token_logits[yes_token_id]
        no_logit = next_token_logits[no_token_id]
    # Store the logits in the logits dictionary {"[YES]": logit, "[NO]": logit}
    logits = {"[YES]": yes_logit, "[NO]": no_logit}
    # Return the logits dictionary
    return logits

# Load your trained model  
path_to_checkpoint = "/data/user_data/sophiapi/checkpoints/stupid_withtokens_qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example_balanced-40-60_max-length-4096_binary/checkpoint-1588"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoint) 

# Load the model
model = AutoModelForCausalLM.from_pretrained(path_to_checkpoint)

# Get the yes and no token ids
yes_token_id = tokenizer.convert_tokens_to_ids("[YES]")
no_token_id = tokenizer.convert_tokens_to_ids("[NO]")

# Set up pipeline
pipe = pipeline("text-generation", model=path_to_checkpoint, tokenizer=tokenizer)

# Create conversation  
# Give examples from the validation set
path_to_validation_set = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/20250822_011509_router_valid_model-5_instance-100_pruned-4_with-ids_by-example_sample200_balanced-40-60_max-length-4096.jsonl"

# Create report file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "/home/sophiapi/model-routing/post_train_eval_reports"
output_filename = f"post_train_eval_{timestamp}.txt"
output_file = os.path.join(output_dir, output_filename)

# Open output file for writing
output_f = open(output_file, 'w')

# Write header to output file
output_f.write(f"Post-train evaluation report\n")
output_f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
output_f.write(f"Model: {path_to_checkpoint}\n")
output_f.write(f"Dataset: {path_to_validation_set}\n")
output_f.write(f"=" * 80 + "\n\n")


with open(path_to_validation_set, "r") as f:
    # Read all lines and randomly select num_examples
    num_examples = 50
    stats = {}
    all_lines = f.readlines()
    selected_lines = random.sample(all_lines, min(num_examples, len(all_lines)))
    
    for i, line in enumerate(selected_lines, 1):
        example = json.loads(line)
        prompt = [{"role": "user", "content": example["prompt"]}]  
        
        model_name = find_model_from_prompt(example['prompt'])
        if model_name is None:
            continue
        if model_name not in stats:
            stats[model_name] = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0, "not_yes_or_no": 0, "total": 0}
        
        
        output_f.write(f"\n\n=========== EXAMPLE {i} ===========\n")
        # output_f.write(f"Prompt: {example['prompt']}")
        
        # Generate with token limit  
        # TODO: not sure the tokenizer is strictly necessary here since it was already set up in the creation of pipe
        response = pipe(prompt, max_new_tokens=1, tokenizer=tokenizer)
        
        # output_f.write(f"\n=========== MODEL RESPONSE {i} ===========\n")
        # output_f.write(response)
        model_next_token = response[0]['generated_text'][1]['content']
        output_f.write(f"Model's next token: {model_next_token}\n") # Print the model's yes or no verdict
        
        # Get the model's logits
        model_logits = get_model_logits(example['prompt'], tokenizer, model, yes_token_id, no_token_id)
        output_f.write(f"Model logits: {model_logits}\n")
        
        #     Compute yes probability = exp(logit[YES]) / (exp(logit[YES]) + exp(logit[NO]))
        yes_prob = np.exp(model_logits["[YES]"]) / (np.exp(model_logits["[YES]"]) + np.exp(model_logits["[NO]"]))
        #     Also compute no probability (a bit redundant, but just in case)
        no_prob = np.exp(model_logits["[NO]"]) / (np.exp(model_logits["[YES]"]) + np.exp(model_logits["[NO]"]))
        # Check that the probabilities sum to 1
        if abs(yes_prob + no_prob - 1) > 1e-6:
            raise ValueError("yes_prob and no_prob do not sum to 1")
        output_f.write(f"Yes probability: {yes_prob}\n")
        output_f.write(f"No probability: {no_prob}\n")
        
        # Update stats
        if yes_prob > no_prob and example['completion'] == "[YES]":
            stats[model_name]["true_positive"] += 1
            
        elif yes_prob < no_prob and example['completion'] == "[NO]":
            stats[model_name]["true_negative"] += 1
            
        elif yes_prob > no_prob and example['completion'] == "[NO]":
            stats[model_name]["false_positive"] += 1
            
        elif yes_prob < no_prob and example['completion'] == "[YES]":
            stats[model_name]["false_negative"] += 1
            
        else:
            stats[model_name]["not_yes_or_no"] += 1
        stats[model_name]["total"] += 1
        # print("\n" + "="*50)
        
        output_f.write(f"=== END OF EXAMPLE {i} ===\n\n")
    
    for model_name, model_stats in stats.items():
        # Write to output file
        output_f.write(f"Model: {model_name}\n")
        output_f.write(f"Number of positive examples: {model_stats['true_positive'] + model_stats['false_negative']}\n")
        output_f.write(f"Number of negative examples: {model_stats['true_negative'] + model_stats['false_positive']}\n")
        output_f.write(f"Number of true positive predictions: {model_stats['true_positive']}\n")
        output_f.write(f"Number of true negative predictions: {model_stats['true_negative']}\n")
        output_f.write(f"Number of false positive predictions: {model_stats['false_positive']}\n")
        output_f.write(f"Number of false negative predictions: {model_stats['false_negative']}\n")
        output_f.write(f"Number of not yes or no predictions: {model_stats['not_yes_or_no']}\n")
        output_f.write(f"Number of total predictions: {model_stats['total']}\n")
        output_f.write(f"Accuracy: {(model_stats['true_positive'] + model_stats['true_negative'])/model_stats['total']*100:.2f}%\n\n")
        
        # # Print to console
        # print(f"Model: {model_name}")
        # print(f"Number of correct predictions: {model_stats['correct']}")
        # print(f"Number of false positive predictions: {model_stats['false_positive']}")
        # print(f"Number of false negative predictions: {model_stats['false_negative']}")
        # print(f"Number of not yes or no predictions: {model_stats['not_yes_or_no']}")
        # print(f"Number of total predictions: {model_stats['total']}")
        # print(f"Accuracy: {model_stats['correct']/model_stats['total']*100:.2f}%\n\n")
    
    # report overall stats
    output_f.write(f"Total number of positive examples: {sum(model_stats['true_positive'] for model_stats in stats.values()) + sum(model_stats['false_negative'] for model_stats in stats.values())}\n")
    output_f.write(f"Total number of negative examples: {sum(model_stats['true_negative'] for model_stats in stats.values()) + sum(model_stats['false_positive'] for model_stats in stats.values())}\n")
    output_f.write(f"Overall true positive rate: {sum(model_stats['true_positive'] for model_stats in stats.values())/num_examples*100:.2f}%\n")
    output_f.write(f"Overall true negative rate: {sum(model_stats['true_negative'] for model_stats in stats.values())/num_examples*100:.2f}%\n")
    output_f.write(f"Overall false positive rate: {sum(model_stats['false_positive'] for model_stats in stats.values())/num_examples*100:.2f}%\n")
    output_f.write(f"Overall false negative rate: {sum(model_stats['false_negative'] for model_stats in stats.values())/num_examples*100:.2f}%\n")
    output_f.write(f"Overall not yes or no rate: {sum(model_stats['not_yes_or_no'] for model_stats in stats.values())/num_examples*100:.2f}%\n")
    output_f.write(f"Overall accuracy: {(sum(model_stats['true_positive'] for model_stats in stats.values()) + sum(model_stats['true_negative'] for model_stats in stats.values()))/num_examples*100:.2f}%\n")
    
    # # Print to console
    # print(f"Overall accuracy: {sum(model_stats['correct'] for model_stats in stats.values())/num_examples*100:.2f}%")
    # print(f"Overall false positive rate: {sum(model_stats['false_positive'] for model_stats in stats.values())/num_examples*100:.2f}%")
    # print(f"Overall false negative rate: {sum(model_stats['false_negative'] for model_stats in stats.values())/num_examples*100:.2f}%")
    # print(f"Overall not yes or no rate: {sum(model_stats['not_yes_or_no'] for model_stats in stats.values())/num_examples*100:.2f}%")
  
print(f"Report written to {output_file}")