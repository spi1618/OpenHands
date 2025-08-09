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
BASE_MODEL_PATH = "/data/user_data/sophiapi/checkpoints/qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example"
CHECKPOINT = os.getenv("ROUTER_CHECKPOINT", "checkpoint-11500")  # Default to latest
MODEL_PATH = f"{BASE_MODEL_PATH}/{CHECKPOINT}"
AVAILABLE_MODELS = [
    "claude-3-5-haiku-20241022",
    "claude-sonnet-4-20250514", 
    "deepseek-v3",
    "devstral-small-2505",
    "kimi-k2-0711-preview"
]

# Special tokens from training
SEP_MODEL = "[M]"
TOK_SPECIALS = {"additional_special_tokens": [SEP_MODEL, "[YES]", "[NO]"]}

class RouterInference:
    def __init__(self):
        """Initialize the router with the fine-tuned model."""
        print(f"Loading router model from: {MODEL_PATH}")
        print(f"Using checkpoint: {CHECKPOINT}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        self.tokenizer.add_special_tokens(TOK_SPECIALS)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True
        )
        
        # Get token IDs for [YES] and [NO]
        self.yes_token_id = self.tokenizer.convert_tokens_to_ids("[YES]")
        self.no_token_id = self.tokenizer.convert_tokens_to_ids("[NO]")
        
        print(f"Router loaded on device: {next(self.model.parameters()).device}")
        print(f"[YES] token ID: {self.yes_token_id}")
        print(f"[NO] token ID: {self.no_token_id}")
    
    def build_prompt(self, partial_trajectory: List[Dict], candidate_model: str) -> str:
        """Build the prompt in the same format as training."""
        # Handle datetime objects and other non-serializable types
        def safe_json_dumps(obj):
            try:
                return json.dumps(obj, sort_keys=True, default=str)
            except:
                return str(obj)
        
        steps = "\n".join(safe_json_dumps(s) for s in partial_trajectory)
        
        return (
            "<|system|>\n"
            "You predict whether the agent will ultimately solve the SWE issue given the trajectory so far.  Respond with [YES] or [NO] only."
            "\n</|system|>\n\n"
            "### Partial trajectory\n" + steps + "\n\n"
            "### Candidate model\n" + SEP_MODEL + " " + candidate_model + "\n\n"
            "### Will this agent eventually succeed?\n"
        )
    
    def get_model_probabilities(self, partial_trajectory: List[Dict]) -> Dict[str, float]:
        """Get success probabilities for all available models."""
        probabilities = {}
        
        for candidate_model in AVAILABLE_MODELS:
            # Build prompt for this candidate
            prompt = self.build_prompt(partial_trajectory, candidate_model)
            
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Get logits for the next token (should be [YES] or [NO])
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0, -1, :]  # Last token logits
                
                # Get logits for [YES] and [NO] tokens
                yes_logit = logits[self.yes_token_id].item()
                no_logit = logits[self.no_token_id].item()
                
                # Calculate probability: P([YES]) / (P([YES]) + P([NO]))
                # Using softmax to get proper probabilities
                logits_tensor = torch.tensor([yes_logit, no_logit])
                probs = torch.softmax(logits_tensor, dim=0)
                yes_prob = probs[0].item()
                
                probabilities[candidate_model] = yes_prob
                
                print(f"  {candidate_model}: YES logit={yes_logit:.3f}, NO logit={no_logit:.3f}, P(YES)={yes_prob:.3f}")
        
        return probabilities
    
    def select_best_model(self, partial_trajectory: List[Dict]) -> Tuple[str, float]:
        """Select the model with highest success probability."""
        probabilities = self.get_model_probabilities(partial_trajectory)
        
        # Find the model with highest probability
        best_model = max(probabilities.items(), key=lambda x: x[1])
        
        print(f"\nSelected model: {best_model[0]} (P={best_model[1]:.3f})")
        return best_model
    
    def route_step(self, partial_trajectory: List[Dict]) -> str:
        """Route to the best model for the next step."""
        print(f"\nRouting step with {len(partial_trajectory)} trajectory steps...")
        best_model, probability = self.select_best_model(partial_trajectory)
        return best_model

def test_router():
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