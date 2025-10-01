#!/usr/bin/env python3
"""
Router inference pipeline for SWE-Bench tasks.
Uses the fine-tuned model to predict which LLM is most likely to succeed.
"""

import json
import torch
import numpy as np
import re
import os
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, FalconForSequenceClassification

# Model configuration

BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "/data/user_data/sophiapi/checkpoints/stupid_consistent_qwen3_router_model-5_instance-100_max-length-16384_samples-20000")
CHECKPOINT = os.getenv("ROUTER_CHECKPOINT", "checkpoint-4500")  # Default to latest
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
MODEL_PATH = f"{BASE_MODEL_PATH}/{CHECKPOINT}"

MODEL_MAPPING = json.load(open("/home/sophiapi/model-routing/static_things/model_name_mapping.json", "r"))
AVAILABLE_MODELS = list(MODEL_MAPPING.keys())

with open("/home/sophiapi/model-routing/static_things/system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()

TOKEN_COSTS = json.load(open("/home/sophiapi/model-routing/static_things/token_costs.json", "r"))
BUCKETS = json.load(open("/home/sophiapi/model-routing/static_things/buckets.json", "r"))
LAMBDA = float(os.getenv("LAMBDA", 30))
EPSILON = float(os.getenv("EPSILON", 0))


class RouterInference:
    def __init__(self):
        """Initialize the router with the fine-tuned model."""
        print(f"Loading router model from: {MODEL_PATH}")
        print(f"Using checkpoint: {CHECKPOINT}")
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # Print the vocab size
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        ).to("cuda")
        
        # Set to evaluation mode
        self.model.eval()
        
        # Get token IDs for [YES] and [NO]
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("YES")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("NO")
        
        # Get token IDs for Bucket 1 through 8
        self.bucket_token_ids = {}
        for bucket_number in range(1, 9):
            token_id = self.tokenizer.convert_tokens_to_ids(f"{bucket_number}")
            self.bucket_token_ids[token_id] = bucket_number
        
        print(f"Router loaded on device: {next(self.model.parameters()).device}")
        print(f"YES token ID: {self.yes_token_id}")
        print(f"NO token ID: {self.no_token_id}")
        print(f"LAMBDA: {LAMBDA}")
        print(f"EPSILON: {EPSILON}")
    
       
    # ============== SOME HELPER FUNCTIONS ==============
    
    # this is copied from convert_raw_to_pc_chat-template.py
    def safe_json_dumps(self, obj):
        try:
            return json.dumps(obj, sort_keys=False, default=str)
        except:
            return str(obj)
        
    # def prune_to_last_four_steps(self, partial_trajectory: List[Dict]) -> List[Dict]:
        """
        For each trajectory, break it into partial trajectories as described:
        - 0th step: all dicts up to (not including) the second "agent" dict.
        - 1st step: all dicts after 0th step up to and including the first dict with "observation".
        - 2nd step: all dicts after 1st step up to and including the second dict with "observation".
        - ...
        - Last step: all dicts remaining in history.
        
        Prune the trajectory to at most 4 steps:
        - If the partial trajectory contains 4 steps or less: include all steps up to that point (1, 2, 3, 4 steps respectively)
        - If the partial trajectory contains more than 4 steps: include only the last 4 steps using a sliding window
        
        Return the pruned trajectory.
        """
        # TODO: implement this
        # Return the last four steps of the partial trajectory
        # return partial_trajectory[-4:] <-- NO NO NO NOT THIS, THIS IS WRONG
        # See traj_to_steps.py for the correct way to do this
        return [{"I": "want", "to": "jump"}, {"off": "a", "bridge": "expeditiously"}]
    
    def validate_model_response(self, model_response: str, request_id: str) -> Tuple[str, int]:
        """
        Validate the model response and return the verdict and the bucket number.
        Validate and parse model output of the form:
          {"success":"YES","output_tokens_bucket":"Bucket 4"}
        """
        print(f"[DEBUG - {request_id}] In validate_model_response...")
        print(f"[DEBUG - {request_id}] Model response: {model_response}")
        pattern = re.compile(r'^\{"success":"(YES|NO)","output_tokens_bucket":"Bucket ([1-8])"\}$')
        stripped_response = model_response.strip()
        match = pattern.match(stripped_response)
        if not match:
            raise ValueError(f"[DEBUG - {request_id}] Invalid model response: {model_response}")
        verdict = match.group(1)
        bucket = int(match.group(2))
        print(f"[DEBUG - {request_id}] Verdict: {verdict}")
        print(f"[DEBUG - {request_id}] Bucket: {bucket}")
        return verdict, bucket
    
    def parse_generated_tokens(self, generated_tokens: List[int], request_id: str) -> Tuple[str, int, int]:
        """
        Parse the generated tokens and return the verdict, the index of the YES/NO verdict, and the bucket number.
        """
        print(f"[DEBUG - {request_id}] In parse_generated_tokens...")
        print(f"[DEBUG - {request_id}] Generated tokens: {generated_tokens}")
        # Find the index of the YES/NO verdict
        yes_index = generated_tokens.index(self.yes_token_id) if self.yes_token_id in generated_tokens else -1
        no_index = generated_tokens.index(self.no_token_id) if self.no_token_id in generated_tokens else -1
        # Check that exactly one of the indices is not -1
        if yes_index == -1 and no_index == -1:
            raise ValueError(f"[DEBUG - {request_id}] Neither YES nor NO was generated")
        elif yes_index == -1:
            verdict = "NO"
            verdict_index = no_index
        elif no_index == -1:
            verdict = "YES"
            verdict_index = yes_index
        else:
            raise ValueError(f"[DEBUG - {request_id}] Both YES and NO were generated")
        
        # Find the bucket number
        bucket_found = False
        for i, token_id in enumerate(generated_tokens):
            if token_id in self.bucket_token_ids.keys():
                if bucket_found:
                    raise ValueError(f"[DEBUG - {request_id}] Multiple potential bucket numbers were generated")
                bucket_found = True
                bucket = int(self.bucket_token_ids[token_id])
        if not bucket_found:
            raise ValueError(f"[DEBUG - {request_id}] No bucket number was generated")
        
        return verdict, verdict_index, bucket
    
    def get_logit_at_index(self, outputs, index, token_id, request_id: str) -> float:
        """
        Get the logit at the given index.
        """
        print(f"[DEBUG - {request_id}] In get_logit_at_index...")
        return outputs.logits[index][0][token_id]
    
    # ============== SOME HELPER FUNCTIONS ==============
    
    
    def get_model_logits_and_bucket(self, truncated_prompt: List[Dict], request_id: str) -> Tuple[Dict[str, float], int]:
        """
        Take the truncated prompt list and give it to the model.
        Return the logits for the successful verdict and the unsuccessful verdict.
        Return the predicted bucket number.
        Note this returns the raw logits, NOT the probabilities.
        """
        print(f"[DEBUG - {request_id}] In get_model_logits_and_bucket...")
        logits_dict = {}
        bucket_number = 0
        
        # Put the truncated prompt list into the tokenizer or whatever black magic this is
        inputs = self.tokenizer.apply_chat_template(truncated_prompt, tokenize=True, return_tensors="pt", add_generation_prompt=True).to("cuda")
        
        # Stick it into the model
        with torch.no_grad():
            outputs = self.model.generate(inputs, 
                                          max_new_tokens=50, 
                                          do_sample=False,  # VERY IMPORTANT TO KEEP IF YOU WANT TO DO GREEDY DECODING
                                          output_logits=True, 
                                          return_dict_in_generate=True,
                                          )
        
        # # Debug because I'm frankly petrified of this whole pipeline
        # print(f"outputs: {outputs}")
        # print(f"outputs keys: {outputs.keys()}")
        
        # Use GREEDY DECODING to get the sequence of generated tokens
        # !!!!! CAUTION: DO NOT USE THIS IF do_sample IS FALSE, THIS WILL NOT WORK !!!!!
        generated_tokens = [torch.argmax(logits, dim=-1).item() for logits in outputs.logits]
        #     ^ this automatically ensures that the number of generated tokens is equal to the output.logits length
        # Decode the generated tokens
        model_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Validate the model response
        verdict_from_validate, bucket_from_validate = self.validate_model_response(model_response, request_id)
        print(f"[DEBUG - {request_id}] Verdict from validate_model_response: {verdict_from_validate}")
        print(f"[DEBUG - {request_id}] Bucket from validate_model_response: {bucket_from_validate}")
        
        # Parse the generated tokens
        verdict_from_parse, verdict_index, bucket_from_parse = self.parse_generated_tokens(generated_tokens, request_id)
        print(f"[DEBUG - {request_id}] Verdict from parse_generated_tokens: {verdict_from_parse}")
        print(f"[DEBUG - {request_id}] Verdict index from parse_generated_tokens: {verdict_index}")
        print(f"[DEBUG - {request_id}] Bucket from parse_generated_tokens: {bucket_from_parse}")
        
        # Check that the verdict and bucket number from the validate_model_response and parse_generated_tokens functions match
        if verdict_from_validate != verdict_from_parse:
            raise ValueError(f"[DEBUG - {request_id}] Verdict from validate_model_response and parse_generated_tokens do not match")
        if bucket_from_validate != bucket_from_parse:
            raise ValueError(f"[DEBUG - {request_id}] Bucket from validate_model_response and parse_generated_tokens do not match")
        
        # Get the logits for the successful verdict and the unsuccessful verdict
        yes_logit = self.get_logit_at_index(outputs, verdict_index, self.yes_token_id, request_id)
        no_logit = self.get_logit_at_index(outputs, verdict_index, self.no_token_id, request_id)
        
        # Store the logits in the logits dictionary {"YES": logit, "NO": logit}
        logits_dict = {"YES": yes_logit, "NO": no_logit}
        # Convert the logits to plain Python floats - yes this code is a dumb way to do it but im tired i don't really care atp
        logits_dict = {k: v.item() for k, v in logits_dict.items()}
        # Store the bucket number (this is kinda redundant but just for organizational symmetry and my sanity)
        bucket_number = bucket_from_validate # (should be the same as bucket_from_parse)
        
        return logits_dict, bucket_number
    
    def estimate_results(self, model_name: str, 
                         candidate_logits_dict: Dict[str, float], 
                         est_token_bucket: int, 
                         cached_input_tokens: Dict[str, int], 
                         uncached_input_tokens: Dict[str, int],
                         request_id: str) -> Tuple[float, float, float]:
        """
        Estimate the reward for the given logits and estimated token bucket.
        """
        print(f"[DEBUG - {request_id}] In estimate_results for model: {model_name}...")
        
        neulab_model_name = MODEL_MAPPING[model_name]
        # check that the neulab model name is in the cached_input_tokens and uncached_input_tokens dictionaries
        if neulab_model_name not in cached_input_tokens or neulab_model_name not in uncached_input_tokens:
            raise ValueError(f"[DEBUG - {request_id}] estimate_results: Neulab model name: {neulab_model_name} not found in cached_input_tokens or uncached_input_tokens")
        
        # Return the reward, the estimated probability of success, and the estimated cost
        
        print(f"[DEBUG - {request_id}] candidate_logits_dict: {candidate_logits_dict}")
        print(f"[DEBUG - {request_id}] candidate_logits_dict value types: {type(candidate_logits_dict['YES'])}, {type(candidate_logits_dict['NO'])}")
        
        # # dict -> plain Python floats
        # nums = {k: v.item() for k, v in candidate_logits_dict.items()}
        # print(f"[DEBUG - {request_id}] nums: {nums}")  # {'YES': 19.9373, 'NO': 27.3696}
        
        # Compute yes probability = exp(logit[YES]) / (exp(logit[YES]) + exp(logit[NO]))
        yes_prob = np.exp(candidate_logits_dict["YES"]) / (np.exp(candidate_logits_dict["YES"]) + np.exp(candidate_logits_dict["NO"]))
        # Also compute no probability (a bit redundant, but just in case)
        no_prob = np.exp(candidate_logits_dict["NO"]) / (np.exp(candidate_logits_dict["YES"]) + np.exp(candidate_logits_dict["NO"]))
        
        print(f"[DEBUG - {request_id}] yes_prob: {yes_prob}")
        print(f"[DEBUG - {request_id}] no_prob: {no_prob}")
        
        # Check that the probabilities sum to 1
        if abs(yes_prob + no_prob - 1) > 1e-6:
            raise ValueError(f"[DEBUG - {request_id}] yes_prob and no_prob do not sum to 1")
        
        # Compute the estimated cost
        cached_input_token_cost = cached_input_tokens[neulab_model_name] * TOKEN_COSTS[model_name]["cached_input_cost"]
        print(f"[DEBUG - {request_id}] Estimated cached input token cost: {cached_input_token_cost}")
        uncached_input_token_cost = uncached_input_tokens[neulab_model_name] * TOKEN_COSTS[model_name]["uncached_input_cost"]
        print(f"[DEBUG - {request_id}] Estimated uncached input token cost: {uncached_input_token_cost}")
        output_token_cost = BUCKETS[str(est_token_bucket)]["representative"] * TOKEN_COSTS[model_name]["output_cost"]
        print(f"[DEBUG - {request_id}] Estimated output token cost: {output_token_cost}")
        est_cost = cached_input_token_cost + uncached_input_token_cost + output_token_cost
        print(f"[DEBUG - {request_id}] Estimated total cost: {est_cost}")
        
        # Compute the estimated reward
        est_reward = yes_prob - LAMBDA * est_cost
        print(f"[DEBUG - {request_id}] Estimated reward: {est_reward}")
        
        return est_reward, yes_prob, est_cost
    
    # THIS WAS COPIED FROM convert_raw_to_pc_chat-template.py
    def build_partial_trajectory_with_truncation(self, partial_trajectory: List[Dict], max_traj_tokens: int, request_id: str):
        """
        Literally just cuts down on the trajectory in case the trajectory is too long.
        """
        print(f"[DEBUG - {request_id}] In build_partial_trajectory_with_truncation...")
        was_truncated = False
        
        if self.tokenizer:
            # Convert trajectory to text and truncate if needed
            trajectory_text = "\n".join(self.safe_json_dumps(step) for step in partial_trajectory)
            # Count the number of tokens in the trajectory
            trajectory_tokens = self.tokenizer.encode(trajectory_text)
            # Debug logging for truncation decisions
            if len(trajectory_tokens) > max_traj_tokens:    
                was_truncated = True
                # Truncate trajectory tokens and reconstruct
                # Truncate the trajectory to (roughly)the first max_traj_tokens / 2 tokens + last max_traj_tokens / 2 tokens
                temp = max_traj_tokens // 2 - 20 # 20 tokens for the ...(omitted for brevity)... buffer
                truncated_trajectory_tokens_beginning = trajectory_tokens[:temp]
                truncated_trajectory_tokens_end = trajectory_tokens[-temp:]
                truncated_trajectory = self.tokenizer.decode(truncated_trajectory_tokens_beginning) + "\n...(omitted for brevity)...\n" + self.tokenizer.decode(truncated_trajectory_tokens_end)
            else:
                truncated_trajectory =  trajectory_text
        else:
            raise ValueError(f"[DEBUG - {request_id}] No tokenizer provided")
    
        return truncated_trajectory, was_truncated    
    
    def build_truncated_prompt(self, partial_trajectory: List[Dict], candidate_model: str, request_id: str) -> List[Dict]:
        print(f"[DEBUG - {request_id}] In build_truncated_prompt...")
        # Require that the trajectory is a list
        if not isinstance(partial_trajectory, list):
            raise ValueError(f"[DEBUG - {request_id}] Trajectory must be a list")
        
        # Define essential parts of the prompt
        system_part = SYSTEM_PROMPT
        trajectory_header = "### Partial trajectory:\n\n\n"
        model_part = f"### Candidate model\n[M] {candidate_model}\n\n"
        question_part = "### Will this agent eventually succeed if the rest of the task is attempted with the candidate model? Into which of the 8 output token buckets will the candidate model's immediate next step fall?"
        
        # Tit's fine to leave this as is, we're using the tokenizer specific to the router model, not the candidate model
        # Compute the number of tokens in the essential parts
        system_tokens = len(self.tokenizer.encode(system_part))
        trajectory_header_tokens = len(self.tokenizer.encode(trajectory_header))
        model_tokens = len(self.tokenizer.encode(model_part))
        question_tokens = len(self.tokenizer.encode(question_part))
        
        # Reserve space for essential parts
        reserved_tokens = system_tokens + trajectory_header_tokens + model_tokens + question_tokens
        
        # Calculate how much space we have left for the actual partial trajectory content
        max_trajectory_tokens = MAX_TOKENS - reserved_tokens - 100  # Leave some buffer
        
        # Truncate the partial trajectory
        truncated_trajectory, was_truncated = self.build_partial_trajectory_with_truncation(partial_trajectory, max_trajectory_tokens, request_id)
        
        # Build the prompt with the truncated trajectory
        prompt_part1 = {
            "role": "system",
            "content": system_part
        }
        prompt_part2 = {
            "role": "user", "content": trajectory_header + truncated_trajectory
        }
        prompt_part3 = {
            "role": "system",
            "content": model_part + question_part
        }
        
        truncated_prompt_list = [prompt_part1, prompt_part2, prompt_part3]
        
        return truncated_prompt_list
        
    def select_best_model(self, partial_trajectory: List[Dict], cached_input_tokens: Dict[str, int], uncached_input_tokens: Dict[str, int], request_id: str) -> Tuple[str, float, float, float]:
        print(f"[DEBUG - {request_id}] In select_best_model...")
        # Assume that the trajectory is not truncated
        
        # Check that the trajectory is a list
        if not isinstance(partial_trajectory, list):
            raise ValueError(f"[DEBUG - {request_id}] Trajectory must be a list")
        
        # Instantiate a dictionary to store the logits for each model
        logits_dict = {}
        # Instantiate a dictionary to store the bucket numbers for each model
        bucket_numbers = {}
        # Instantiate a dictionary of dictionaries for each model to store {"estimated reward": float, "estimated probability of success": float, "estimated cost": float} for each model
        estimated_results_dict = {}
        
        # For each model, 
        for candidate_model in AVAILABLE_MODELS:
            
            # Pass the partial trajectory and candidate model name to build_truncated_prompt (get the truncated prompt string back)
            truncated_prompt_list = self.build_truncated_prompt(partial_trajectory, candidate_model, request_id)        
            
            # Take the truncated prompt string and pass it to get_model_logits_and_bucket (get the logits dictionary back)
            # Store the logits dictionary as the value for the model name key in the logits dictionary
            # So that eg. the logits dictionary looks like {"claude-3-5-haiku": {"YES": logit, "NO": logit}, "claude-sonnet-4": {"YES": logit, "NO": logit}, ...}
            # Bucket numbers dictionary looks like {"claude-3-5-haiku": bucket, "claude-sonnet-4": bucket, ...}
            logits_dict[candidate_model], bucket_numbers[candidate_model] = self.get_model_logits_and_bucket(truncated_prompt_list, request_id)
                        
            # Compute the estimated reward 
            est_reward, est_p_success, est_cost = self.estimate_results(candidate_model, 
                                                                        logits_dict[candidate_model], 
                                                                        bucket_numbers[candidate_model], 
                                                                        cached_input_tokens, 
                                                                        uncached_input_tokens,
                                                                        request_id)
            # Store the estimated results in the estimated_results_dict
            estimated_results_dict[candidate_model] = {"estimated reward": est_reward, 
                                                       "estimated probability of success": est_p_success, 
                                                       "estimated cost": est_cost}
        
        # Figure out which model has the highest estimated reward
        most_successful_model = max(estimated_results_dict, key=lambda x: estimated_results_dict[x]["estimated probability of success"])
        most_successful_reward = estimated_results_dict[most_successful_model]["estimated reward"]
        most_successful_p_success = estimated_results_dict[most_successful_model]["estimated probability of success"]
        most_successful_cost = estimated_results_dict[most_successful_model]["estimated cost"]
        
        print(f"[DEBUG - {request_id}] Most successful model: {most_successful_model}")
        print(f"[DEBUG - {request_id}] Most successful reward: {most_successful_reward}")
        print(f"[DEBUG - {request_id}] Most successful probability of success: {most_successful_p_success}")
        print(f"[DEBUG - {request_id}] Most successful cost: {most_successful_cost}")
        
        # Figure out which model has the lowest estimated cost
        least_costly_model = min(estimated_results_dict, key=lambda x: estimated_results_dict[x]["estimated cost"])
        least_costly_reward = estimated_results_dict[least_costly_model]["estimated reward"]
        least_costly_p_success = estimated_results_dict[least_costly_model]["estimated probability of success"]
        least_costly_cost = estimated_results_dict[least_costly_model]["estimated cost"]
        
        print(f"[DEBUG - {request_id}] Least costly model: {least_costly_model}")
        print(f"[DEBUG - {request_id}] Least costly reward: {least_costly_reward}")
        print(f"[DEBUG - {request_id}] Least costly probability of success: {least_costly_p_success}")
        print(f"[DEBUG - {request_id}] Least costly cost: {least_costly_cost}")
        
        # Figure out which model is the best
        best_model = max(estimated_results_dict, key=lambda x: estimated_results_dict[x]["estimated reward"])
        best_reward = estimated_results_dict[best_model]["estimated reward"]
        best_model_p_success = estimated_results_dict[best_model]["estimated probability of success"]
        best_model_cost = estimated_results_dict[best_model]["estimated cost"]
        
        print(f"[DEBUG - {request_id}] Best model: {best_model}")
        print(f"[DEBUG - {request_id}] Best reward: {best_reward}")
        print(f"[DEBUG - {request_id}] Best model probability of success: {best_model_p_success}")
        print(f"[DEBUG - {request_id}] Best model cost: {best_model_cost}")
        
        # Epsilon greedy part
        if np.random.random() < EPSILON:
            print(f"[DEBUG - {request_id}] ***Jackpot! You got epsilon greedy! Choosing a random model...***")
            random_model = np.random.choice(AVAILABLE_MODELS)
            random_reward = estimated_results_dict[random_model]["estimated reward"]
            random_model_p_success = estimated_results_dict[random_model]["estimated probability of success"]
            random_model_cost = estimated_results_dict[random_model]["estimated cost"]
            print(f"[DEBUG - {request_id}] Random model: {random_model}")
            print(f"[DEBUG - {request_id}] Random model's reward: {random_reward}")
            print(f"[DEBUG - {request_id}] Random model's probability of success: {random_model_p_success}")
            print(f"[DEBUG - {request_id}] Random model's cost: {random_model_cost}")
            return random_model, random_reward, random_model_p_success, random_model_cost
        
        # Return the model with the highest reward, its reward, its estimated probability of success, and its estimated cost
        return best_model, best_reward, best_model_p_success, best_model_cost
    
    # # this lowkey isn't even used
    # def route_step(self, partial_trajectory: List[Dict]) -> str:
    #     """Route to the best model for the next step."""
    #     print(f"\nRouting step with {len(partial_trajectory)} trajectory steps...")
    #     best_model, best_reward, best_model_p_success, best_model_cost = self.select_best_model(partial_trajectory)
    #     return best_model

# def test_router():
#     # TODO: this is like, wildly wrong. don't use this. please.
#     """Test the router with a sample trajectory."""
#     router = RouterInference()
    
#     # Sample partial trajectory (you would replace this with real SWE-Bench trajectory data)
#     sample_trajectory = [
#         {"source": "user", "content": "Fix the bug in the login function"},
#         {"source": "agent", "content": "I'll help you fix the login function. Let me first examine the code."},
#         {"source": "agent", "observation": "Found login.py file with authentication logic"}
#     ]
    
#     print("Testing router with sample trajectory:")
#     print(json.dumps(sample_trajectory, indent=2))
    
#     selected_model = router.route_step(sample_trajectory)
#     print(f"\nRouter selected: {selected_model}")
    
#     return router

if __name__ == "__main__":
    # router = test_router() 
    print("LMAO what are you doing?? how did you get here?? why are you doing this?? please stop.")