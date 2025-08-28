from transformers import pipeline, GenerationConfig, AutoTokenizer, AutoModelForCausalLM
import json
import random
import os
from datetime import datetime
import torch
import numpy as np
from typing import Dict

# Load your trained model  
path_to_checkpoint = "/data/user_data/sophiapi/checkpoints/stupid_chat-template_qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example_max-length-16384/checkpoint-1600"
# Give examples from the validation set
path_to_validation_set = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/20250827_172632_val_chat-template_max-len-16384.jsonl"


def find_model_from_prompt(prompt):
    # Determine the model from the prompt
    # The model name will be the first word after the string "### Candidate model\n[M] ", delimited by spaces and newlines
    # For example: "### Candidate model\n[M] deepseek-v3\n\n" --> "deepseek-v3"
    # If the string is not found, return None
    prompt = prompt[0]['content']
    if "### Candidate model\n[M] " in prompt:
        return prompt.split("### Candidate model\n[M] ")[1].split("\n")[0]
    else:
        return None

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoint) 

# Load the model
model = AutoModelForCausalLM.from_pretrained(path_to_checkpoint).to("cuda") 

# Get the yes and no token ids
yes_token_id = tokenizer.convert_tokens_to_ids("YES")
print(f"Yes token id: {yes_token_id}")
no_token_id = tokenizer.convert_tokens_to_ids("NO")
print(f"No token id: {no_token_id}")

# Set up pipeline
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Create conversation  

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
    num_examples = 20
    stats = {}
    all_lines = f.readlines()
    selected_lines = random.sample(all_lines, min(num_examples, len(all_lines)))
    
    # print(f"selected_lines: {selected_lines}\n")
    # output_f.write(f"selected_lines: {selected_lines}\n")
    
    # random_stuff = ["lock in", "unlock", "open", "close", "turn on", "turn off", "increase", "decrease", "move", "stop"]
    
    for i, line in enumerate(selected_lines):
        # print("++++ HI ++++++\n")
        example = json.loads(line)
        # print(f"example['prompt']: {example['prompt']}\n")
        # print(f"example['completion']: {example['completion']}\n")
        prompt = [{"role": "user", "content": example['prompt']}]  # CHANGEBACK # IS THIS STILL NECESSARY? the taakenizer already applies the chat template to input messages.
        
        model_name = find_model_from_prompt(example['prompt'])
        if model_name is None:
            continue
        if model_name not in stats:
            stats[model_name] = {"true_positive": 0, "true_negative": 0, "false_positive": 0, "false_negative": 0, "not_yes_or_no": 0, "total": 0}
        
        print(f"\n\n=========== EXAMPLE {i} ===========\n")
        output_f.write(f"\n\n=========== EXAMPLE {i} ===========\n")
        
        # Generate with token limit  
        # TODO: not sure the tokenizer is strictly necessary here since it was already set up in the creation of pipe
        # also i think this doesn't need the chat template?
        response = pipe(example['prompt'], max_new_tokens=1, tokenizer=tokenizer)
        
        # # print(f"response: {response}\n\n")
        # output_f.write(f"response: {response}\n\n")
        # print(f"response shape: {len(response)}")
        # print(f"response[0] keys: {response[0].keys()}")
        # print(f"response[0]['generated_text']: {response[0]['generated_text']}")
        # print(f"response[0]['generated_text'] length: {len(response[0]['generated_text'])}")
        
        model_yes_or_no = response[0]['generated_text'][1]['content']
        
        print(f"\nTrue completion: '{example['completion']}'\n")
        print(f"\nModel's response: '{model_yes_or_no}'\n")
        output_f.write(f"\nTrue completion: '{example['completion']}'\n")
        output_f.write(f"\nModel's response: {model_yes_or_no}\n") # Print the model's yes or no verdict
        
        inputs = tokenizer.apply_chat_template(example['prompt'], tokenize=True, return_tensors="pt", add_generation_prompt=True).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=1, do_sample=False, output_logits=True, return_dict_in_generate=True) # KEEP THIS
            # # UNCOMMENT TO CHECK OUTPUT DECODING
            # outputs = model.generate(inputs, max_new_tokens=5, do_sample=False) 
        
        # # UNCOMMENT TO CHECK OUTPUT DECODING
        # print(f"\noutputs: {outputs}")
        # print(f"\noutputs shape: {outputs.shape}")
        # decoded_outputs = tokenizer.decode(outputs[0], skip_special_tokens=False)
        # print(f"Decoded outputs: {decoded_outputs}\n\n")
        
        # print(f"\noutputs: {outputs}\n")
        # print(f"\noutputs logits: {outputs.logits}\n")
        # print(f"\noutputs logits length: {len(outputs.logits)}\n")
        # print(f"\nshape of outputs.logits[0]: {outputs.logits[0].shape}\n")
        # output_f.write(f"\noutputs: {outputs}\n")
        # output_f.write(f"\noutputs.logits: {outputs.logits}\n")
        # output_f.write(f"\noutputs logits length: {len(outputs.logits)}\n")
        # output_f.write(f"\nshape of outputs.logits[0]: {outputs.logits[0].shape}\n")
        
        yes_logit = outputs.logits[0][0][yes_token_id].cpu().numpy()
        no_logit = outputs.logits[0][0][no_token_id].cpu().numpy()
        print(f"yes_logit: {yes_logit}")
        print(f"no_logit: {no_logit}")
        output_f.write(f"\nyes_logit: {yes_logit}")
        output_f.write(f"\nno_logit: {no_logit}\n")
        
        #     Compute yes probability = exp(logit(YES)) / (exp(logit(YES)) + exp(logit(NO)))
        yes_prob = np.exp(yes_logit) / (np.exp(yes_logit) + np.exp(no_logit))
        #     Also compute no probability (a bit redundant, but just in case)
        no_prob = np.exp(no_logit) / (np.exp(yes_logit) + np.exp(no_logit))
        # Check that the probabilities sum to 1
        if abs(yes_prob + no_prob - 1) > 1e-6:
            raise ValueError("yes_prob and no_prob do not sum to 1")
        print(f"Yes probability: {yes_prob}")
        print(f"No probability: {no_prob}")
        output_f.write(f"Yes probability: {yes_prob}\n")
        output_f.write(f"No probability: {no_prob}\n")
        
        # Update stats
        if model_yes_or_no == "YES" and example['completion'] == "YES":
            stats[model_name]["true_positive"] += 1
            
        elif model_yes_or_no == "NO" and example['completion'] == "NO":
            stats[model_name]["true_negative"] += 1
            
        elif model_yes_or_no == "YES" and example['completion'] == "NO":
            stats[model_name]["false_positive"] += 1
            
        elif model_yes_or_no == "NO" and example['completion'] == "YES":
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