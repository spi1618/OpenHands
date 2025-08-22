#!/usr/bin/env python3
"""
Router inference pipeline for SWE-Bench tasks.
Uses the fine-tuned model to predict which LLM is most likely to succeed.
"""

import json
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configuration
import os
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "/data/user_data/sophiapi/checkpoints/stupid_withtokens_qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example")
CHECKPOINT = os.getenv("ROUTER_CHECKPOINT", "checkpoint-796")  # Default to latest
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
MODEL_PATH = f"{BASE_MODEL_PATH}/{CHECKPOINT}"
AVAILABLE_MODELS = [
    "claude-3-5-haiku",
    "claude-sonnet-4", 
    "deepseek-v3",
    "devstral-small",
    "kimi-k2"
]


class RouterInference:
    def __init__(self):
        """Initialize the router with the fine-tuned model."""
        print(f"Loading router model from: {MODEL_PATH}")
        print(f"Using checkpoint: {CHECKPOINT}")
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        # Print the vocab size
        print(f"Tokenizer vocab size (with custom tokens): {len(self.tokenizer)}")
        
        # Compare to vocab size without custom tokens
        tokenizer_no_custom = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        print(f"Tokenizer vocab size (without custom tokens): {len(tokenizer_no_custom)}")
        # if (len(self.tokenizer) != len(tokenizer_no_custom) + 2):
        #     raise ValueError("Tokenizer vocab size does not match expected size")
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True
        )
        
        # Set to evaluation mode
        self.model.eval()
        
        # Get token IDs for [YES] and [NO]
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("[YES]")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("[NO]")
        
        print(f"Router loaded on device: {next(self.model.parameters()).device}")
        print(f"[YES] token ID: {self.yes_token_id}")
        print(f"[NO] token ID: {self.no_token_id}")
    
       
    # ============== SOME HELPER FUNCTIONS ==============
    
    def safe_json_dumps(self, obj):
        # TODO: set sort_keys to False once you've also set it to false in convert_raw_to_pc.py
        try:
            return json.dumps(obj, sort_keys=False, default=str)
        except:
            return str(obj)
        
    def prune_to_last_four_steps(self, partial_trajectory: List[Dict]) -> List[Dict]:
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
    
    # ============== SOME HELPER FUNCTIONS ==============
    
    
    def get_model_logits(self, truncated_prompt: str) -> Dict[str, float]:
        """
        Take the truncated prompt string and give it to the model.
        Return the logits for the next token (only consider [YES] and [NO]).
        Note this returns the raw logits, NOT the probabilities.
        """
        # Take the truncated prompt string and give it to the model
        #     Tokenize the input string
        inputs = self.tokenizer(truncated_prompt, return_tensors="pt")
        #     Do a forward pass through the model and extract the logits
        with torch.no_grad():
            outputs = self.model(**inputs) # Tensor of shape (batch_size, sequence_length, vocab_size)
            next_token_logits = outputs.logits[0,-1, :] # Shape: [vocab_size]
            # DEBUG: check the shape of next_token_logits
            print(f"next_token_logits shape: {next_token_logits.shape}")
            if next_token_logits.shape[0] != len(self.tokenizer):
                raise ValueError("next_token_logits shape does not match tokenizer vocab size")
        # Get the logits for the next token (only consider [YES] and [NO])
            yes_logit = next_token_logits[self.yes_token_id]
            no_logit = next_token_logits[self.no_token_id]
        # Store the logits in the logits dictionary {"[YES]": logit, "[NO]": logit}
        logits = {"[YES]": yes_logit, "[NO]": no_logit}
        # Return the logits dictionary
        return logits
    
    def build_truncated_prompt(self, partial_trajectory: List[Dict], candidate_model: str) -> str:
        # Require that the trajectory is a list
        if not isinstance(partial_trajectory, list):
            raise ValueError("Trajectory must be a list")
        # Define essential parts of the prompt
        system_part = (
            "<|system|>\n"
            "You predict whether the agent or assistant will ultimately solve the SWE issue successfullygiven the partial trajectory so far.\n"
            "Respond with [YES] or [NO] only.\n"
            "The partial trajectory contains information about the agent or assistant's actions, observations, and interactions with the user or environment.\n"
            "The partial trajectory may be truncated to the most recent information.\n"
            "</|system|>\n\n"
        )
        trajectory_header = "### Partial trajectory\n"
        model_part = f"### Candidate model\n[M] {candidate_model}\n\n"
        question_part = "### Will this agent eventually succeed?\n"
        # #     then truncate it to the last four "steps"
        # #     TODO: this is too much of a headache, I'm just going to skip pruning for now
        # partial_trajectory_pruned = self.prune_to_last_four_steps(partial_trajectory)
        partial_trajectory_pruned = partial_trajectory
        #     then use the tokenizer and smart truncation to assemble the full prompt
        # Calculate token lengths for essential parts (if tokenizer available)
        if not self.tokenizer:
            raise ValueError("Tokenizer not found :(")
        system_tokens = len(self.tokenizer.encode(system_part))
        trajectory_header_tokens = len(self.tokenizer.encode(trajectory_header))
        model_tokens = len(self.tokenizer.encode(model_part))
        question_tokens = len(self.tokenizer.encode(question_part))
        # Reserve space for essential parts
        reserved_tokens = system_tokens + trajectory_header_tokens + model_tokens + question_tokens
        # Calculate how much space we have for trajectory content
        max_trajectory_tokens = MAX_TOKENS - reserved_tokens - 50  # Leave some buffer
        # Convert trajectory to text
        trajectory_text = "\n".join(self.safe_json_dumps(step) for step in partial_trajectory_pruned)
        # TODO: consider adding a coarse filter to remove the most egregiously long trajectories
        #     eg. if the trajectory is longer than 400,000 characters, truncate it to the last 400,000 characters
        if len(trajectory_text) > 400000:
            trajectory_text = trajectory_text[-400000:]
        # Count the number of tokens in the trajectory
        trajectory_tokens = self.tokenizer.encode(trajectory_text)
        # Truncate trajectory if needed
        if len(trajectory_tokens) > max_trajectory_tokens:
            # Truncate the trajectory to the last max_trajectory_tokens tokens, NOT the first max_trajectory_tokens tokens
            truncated_trajectory_tokens = trajectory_tokens[-max_trajectory_tokens:]
            truncated_trajectory = self.tokenizer.decode(truncated_trajectory_tokens)
        else:
            truncated_trajectory = trajectory_text
        # Assemble the full prompt
        truncated_prompt = system_part + trajectory_header + truncated_trajectory + model_part + question_part
        # Return the truncated prompt string
        return truncated_prompt
        #         the logic should be the same as the logic used to generate the training data (see convert_raw_to_pc.py)
    
    def select_best_model(self, partial_trajectory: List[Dict]) -> Tuple[str, float]:
        # Assume that the trajectory is not truncated
        # Check that the trajectory is a list
        if not isinstance(partial_trajectory, list):
            raise ValueError("Trajectory must be a list")
        # Instantiate a dictionary to store the logits for each model
        logits_dict = {}
        # Instantiate a dictionary to store the probabilities for each model
        yes_probabilities = {}
        # For each model, 
        for candidate_model in AVAILABLE_MODELS:
            #     Pass the partial trajectory and candidate model name to build_truncated_prompt (get the truncated prompt string back)
            truncated_prompt_string = self.build_truncated_prompt(partial_trajectory, candidate_model)        
            #     Take the truncated prompt string and pass it to get_model_logits (get the logits dictionary back)
            #     Store the logits dictionary as the value for the model name key in the logits dictionary
            #         So that eg. the logits dictionary looks like {"claude-3-5-haiku": {"[YES]": logit, "[NO]": logit}, "claude-sonnet-4": {"[YES]": logit, "[NO]": logit}, ...}
            logits_dict[candidate_model] = self.get_model_logits(truncated_prompt_string)
            #     Compute yes probability = exp(logit[YES]) / (exp(logit[YES]) + exp(logit[NO]))
            yes_prob = np.exp(logits_dict[candidate_model]["[YES]"]) / (np.exp(logits_dict[candidate_model]["[YES]"]) + np.exp(logits_dict[candidate_model]["[NO]"]))
            #     Also compute no probability (a bit redundant, but just in case)
            no_prob = np.exp(logits_dict[candidate_model]["[NO]"]) / (np.exp(logits_dict[candidate_model]["[YES]"]) + np.exp(logits_dict[candidate_model]["[NO]"]))
            # Check that the probabilities sum to 1
            if abs(yes_prob + no_prob - 1) > 1e-6:
                raise ValueError("yes_prob and no_prob do not sum to 1")
            #     Store the probability in the yes_probabilities dictionary
            #         So that eg. the yes_probabilities dictionary looks like {"claude-3-5-haiku": prob, "claude-sonnet-4": prob, ...}
            yes_probabilities[candidate_model] = yes_prob
        # Return (the model with the highest yes probability, its yes probability) as a tuple
        best_model = max(yes_probabilities.items(), key=lambda x: x[1])
        print(f"\nSelected model: {best_model[0]} (P(YES)={best_model[1]:.3f})")
        return best_model # dw this is a tuple, not just the model name
    
    def route_step(self, partial_trajectory: List[Dict]) -> str:
        """Route to the best model for the next step."""
        print(f"\nRouting step with {len(partial_trajectory)} trajectory steps...")
        best_model, probability = self.select_best_model(partial_trajectory)
        return best_model

def test_router():
    # TODO: this is like, wildly wrong. don't use this. please.
    """Test the router with a sample trajectory."""
    router = RouterInference()
    
    # Sample partial trajectory (you would replace this with real SWE-Bench trajectory data)
    sample_trajectory = [
        {"source": "user", "content": "Fix the bug in the login function"},
        {"source": "agent", "content": "I'll help you fix the login function. Let me first examine the code."},
        {"source": "agent", "observation": "Found login.py file with authentication logic"}
    ]
    
    print("Testing router with sample trajectory:")
    print(json.dumps(sample_trajectory, indent=2))
    
    selected_model = router.route_step(sample_trajectory)
    print(f"\nRouter selected: {selected_model}")
    
    return router

if __name__ == "__main__":
    router = test_router() 