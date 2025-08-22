#!/usr/bin/env python3
"""
SWE-Bench Router with Fine-tuned Model Integration
Combines the router inference with the existing FastAPI router for SWE-Bench tasks.
"""

import os
import json
# import json
# import torch
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI, BadRequestError
from transformers import AutoTokenizer

# Import the router inference
from router_inference_stupid import RouterInference

# Configure available models + client
# DO NOT SIMPLIFY THE NAMES HERE, THEY MUST MATCH THE CONFIG SO WE CAN USE THE LITELLM PROXY SERVER; THEY SHOULD MATCH THE CONFIG FILE
AVAILABLE_MODELS: List[str] = [
    "neulab/claude-3-5-haiku-20241022",
    "neulab/claude-sonnet-4-20250514",
    "neulab/devstral-small-2505",
    "neulab/deepseek-v3",
    "neulab/kimi-k2-0711-preview"
]

# One global OpenAI client configured for the proxy
_client = OpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url="https://cmu.litellm.ai",
)

if _client.api_key is None:
    raise RuntimeError("Set LITELLM_API_KEY environment variable before running.")

# Initialize the router inference
print(f"[DEBUG] Starting SWE-Bench Router")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "/data/user_data/sophiapi/checkpoints/stupid_withtokens_qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example")
ROUTER_CHECKPOINT = os.getenv("ROUTER_CHECKPOINT", "checkpoint-796")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 4096))
RANDOM_MODE = os.getenv("RANDOM_MODE", "false").lower() == "true"

print(f"[DEBUG] Environment BASE_MODEL_PATH: {BASE_MODEL_PATH}")
print(f"[DEBUG] Environment ROUTER_CHECKPOINT: {ROUTER_CHECKPOINT}")
print(f"[DEBUG] Environment MAX_TOKENS: {MAX_TOKENS}")
print(f"[DEBUG] Environment RANDOM_MODE: {RANDOM_MODE}")

if RANDOM_MODE:
    print(f"[INFO] Random mode enabled - will use random model selection instead of router model")
    router_inference = None
else:
    print(f"[INFO] Router mode enabled - will use fine-tuned router model")
    router_inference = RouterInference()

# FastAPI schema
class TrajectoryStep(BaseModel):
    source: str  # "user", "agent", "system"
    content: Optional[str] = None
    observation: Optional[str] = None
    action: Optional[str] = None

class SWEBenchRoutingRequest(BaseModel):
    partial_trajectory: List[TrajectoryStep]
    user_query: Optional[str] = Field(default=None, description="Current user query/instruction")

class SWEBenchRoutingResponse(BaseModel):
    selected_model: str
    confidence: float
    all_probabilities: Dict[str, float]
    reasoning: Optional[str] = None

class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = Field(default=None, description="(optional) override to force a particular back‑end LLM")
    max_tokens: Optional[int] = 1024

class ChatResponse(BaseModel):
    model: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

# Truncation functions
def debug_message_content(message: str, max_preview_length: int = 200) -> str:
    """
    Debug helper to show message content with smart truncation.
    """
    if len(message) <= max_preview_length:
        return message
    
    # Show first and last parts with ellipsis
    preview_length = max_preview_length // 2
    return f"{message[:preview_length]}...{message[-preview_length:]}"

def debug_json_content(data, max_preview_length: int = 200) -> str:
    """
    Debug helper to show JSON content with smart truncation.
    """
    if isinstance(data, list):
        if not data:
            return "[]"
        result = []
        for i, item in enumerate(data):
            if isinstance(item, dict):
                # For dictionaries, show a preview of each key-value pair
                item_preview = []
                for key, value in item.items():
                    if isinstance(value, str):
                        if len(value) <= max_preview_length:
                            item_preview.append(f"{key}: {value}")
                        else:
                            preview_length = max_preview_length // 2
                            item_preview.append(f"{key}: {value[:preview_length]}...{value[-preview_length:]}")
                    elif isinstance(value, dict):
                        item_preview.append(f"{key}: {{...}} (dict with {len(value)} keys)")
                    elif isinstance(value, list):
                        item_preview.append(f"{key}: [...] (list with {len(value)} items)")
                    else:
                        item_preview.append(f"{key}: {value}")
                result.append("{" + ", ".join(item_preview) + "}")
            elif hasattr(item, 'model_dump'):
                # For Pydantic models like ChatMessage, convert to dict and show preview
                try:
                    item_dict = item.model_dump()
                    item_preview = []
                    for key, value in item_dict.items():
                        if isinstance(value, str):
                            if len(value) <= max_preview_length:
                                item_preview.append(f"{key}: {value}")
                            else:
                                preview_length = max_preview_length // 2
                                item_preview.append(f"{key}: {value[:preview_length]}...{value[-preview_length:]}")
                        elif isinstance(value, dict):
                            item_preview.append(f"{key}: {{...}} (dict with {len(value)} keys)")
                        elif isinstance(value, list):
                            item_preview.append(f"{key}: [...] (list with {len(value)} items)")
                        else:
                            item_preview.append(f"{key}: {value}")
                    result.append(f"{item.__class__.__name__}({', '.join(item_preview)})")
                except Exception:
                    # Fallback to string representation if model_dump fails
                    result.append(str(item))
            else:
                result.append(str(item))
        return "[" + ", ".join(result) + "]"
    elif isinstance(data, dict):
        result = []
        for key, value in data.items():
            if isinstance(value, str):
                if len(value) <= max_preview_length:
                    result.append(f"{key}: {value}")
                else:
                    preview_length = max_preview_length // 2
                    result.append(f"{key}: {value[:preview_length]}...{value[-preview_length:]}")
            elif isinstance(value, dict):
                result.append(f"{key}: {{...}} (dict with {len(value)} keys)")
            elif isinstance(value, list):
                result.append(f"{key}: [...] (list with {len(value)} items)")
            else:
                result.append(f"{key}: {value}")
        return "{" + ", ".join(result) + "}"
    else:
        return str(data)

# FastAPI app
app = FastAPI(title="SWE-Bench Router", version="0.1")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def route_chat(req: ChatRequest):
    """Proxy the chat completion to the routed model."""
    # Extract trajectory from messages for routing
    
    print(f"######## [DEBUG] RECEIVED REQUEST ########")
    
    # Debug: Print the request's messages
    print(f"[DEBUG] Request messages: {debug_json_content(req.messages)}")
    
    trajectory_dicts = []
    for msg in req.messages:
        # TODO: fix this, it's not detecting the sources properly, should be user/assistant/system
        # im telling you this is some openai bullshit, we need to fix this
        # I think this is fine now??? We're just replacing "assistant" with "agent"
        if msg.role == "user":
            trajectory_dicts.append({"source": "user", "content": msg.content})
        elif msg.role == "assistant":
            trajectory_dicts.append({"source": "agent", "content": msg.content})
        elif msg.role == "system":
            trajectory_dicts.append({"source": "system", "content": msg.content})
            
    # Debug: Print of trajectory_dicts
    print(f"[DEBUG] Raw trajectory: {debug_json_content(trajectory_dicts)}")
    # Debug: Print the length of trajectory_dicts
    print(f"[DEBUG] Raw trajectory length: {len(trajectory_dicts)}")
    
    # Use router to select the best model
    if RANDOM_MODE:
        # Random mode: select a random model
        print(f"[DEBUG] Random mode enabled, selecting a random model")
        import random
        selected_model = random.choice(AVAILABLE_MODELS)
        print(f"[DEBUG] Random selected: {selected_model}")
    elif trajectory_dicts:
        print(f"[DEBUG] Using router inference with {len(trajectory_dicts)} trajectory events")
        # Pass in the raw trajectory to router_inference_stupid.py
        best_model = router_inference.select_best_model(trajectory_dicts) # best_model is a tuple (model_name, yes_probability)
        print(f"[DEBUG] Router response - best_model: {best_model[0]}, confidence: {best_model[1]}")
        
        # Map internal model names to LiteLLM names
        # The keys should match the model names that router_inference_stupid.py uses
        # The values should match the model names that LiteLLM expects (same as AVAILABLE_MODELS)
        model_mapping = {
            "claude-3-5-haiku": "neulab/claude-3-5-haiku-20241022",
            "claude-sonnet-4": "neulab/claude-sonnet-4-20250514",
            "deepseek-v3": "neulab/deepseek-v3",
            "devstral-small": "neulab/devstral-small-2505",
            "kimi-k2": "neulab/kimi-k2-0711-preview"
        }
        selected_model = model_mapping.get(best_model[0], best_model[0])
        print(f"[DEBUG] Router selected: {best_model[0]} -> {selected_model}")
    else:
        # Fallback to random selection if no trajectory
        print(f"[DEBUG] No trajectory, using random fallback")
        import random
        selected_model = random.choice(AVAILABLE_MODELS)
        print(f"[DEBUG] Random fallback selected: {selected_model}")
    
    print(f"[DEBUG] Final selected model: {selected_model}")
    
    try:
        print(f"[DEBUG] Calling LiteLLM proxy with model: {selected_model}")
        response = _client.chat.completions.create(
            model=selected_model,
            messages=[m.model_dump() for m in req.messages],
            max_tokens=req.max_tokens,
        )
    except BadRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))

    print(f"[DEBUG] Response type: {type(response)}")
    # print(f"[DEBUG] Response attributes: {dir(response)}")
    print(f"[DEBUG] Response choices: {response.choices}")
    print(f"[DEBUG] Response usage: {response.usage}")
    
    choice = response.choices[0]
    usage = response.usage

    # Ensure we have the correct response format
    content = choice.message.content or ""
    if not content and hasattr(response, 'content'):
        content = response.content or ""
    
    print(f"[DEBUG] Final content: {content[:100]}...")
    
    # Return the response in the format that LiteLLM expects
    from fastapi.responses import JSONResponse
    
    # TODO: check this
    response_data = {
        "id": response.id,
        "object": "chat.completion",
        "created": response.created,
        "model": selected_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "finish_reason": "stop" # TODO: check this (what does finish_reason mean?)
            }
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0
        }
    }
    
    print(f"[DEBUG] Sending response: model={selected_model}, content_length={len(content)}, total_tokens={usage.total_tokens if usage else 0}")
    
    return JSONResponse(content=response_data)

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "models": AVAILABLE_MODELS,
        "router_model": "qwen3_router_json" if not RANDOM_MODE else "random",
        "mode": "random" if RANDOM_MODE else "router",
        "random_mode_enabled": RANDOM_MODE
    }

# Test endpoint
@app.post("/test_router")
async def test_router():
    # TODO: this is broken. please don't use it.
    """Test the router with a sample trajectory."""
    sample_trajectory = [
        TrajectoryStep(source="user", content="Fix the bug in the login function"),
        TrajectoryStep(source="agent", content="I'll help you fix the login function. Let me first examine the code."),
        TrajectoryStep(source="agent", observation="Found login.py file with authentication logic")
    ]
    
    req = SWEBenchRoutingRequest(partial_trajectory=sample_trajectory)
    return await route_swe_bench(req)

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or default to 8123
    port = int(os.getenv("ROUTER_PORT", "8123"))
    uvicorn.run(app, host="0.0.0.0", port=port) 