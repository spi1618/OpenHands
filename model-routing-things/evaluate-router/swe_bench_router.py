#!/usr/bin/env python3
"""
SWE-Bench Router with Fine-tuned Model Integration
Combines the router inference with the existing FastAPI router for SWE-Bench tasks.
"""

import os
# import json
# import torch
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI, BadRequestError

# Import the router inference
from router_inference import RouterInference

# Configure available models + client
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
print(f"[DEBUG] Environment ROUTER_CHECKPOINT: {os.getenv('ROUTER_CHECKPOINT', 'not set')}")
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

# FastAPI app
app = FastAPI(title="SWE-Bench Router", version="0.1")

@app.post("/route_swe_bench", response_model=SWEBenchRoutingResponse)
async def route_swe_bench(req: SWEBenchRoutingRequest):
    """Route a SWE-Bench step using the fine-tuned router model."""
    try:
        # Convert trajectory steps to the format expected by router
        trajectory_dicts = []
        for step in req.partial_trajectory:
            step_dict = {"source": step.source}
            if step.content:
                step_dict["content"] = step.content
            if step.observation:
                step_dict["observation"] = step.observation
            if step.action:
                step_dict["action"] = step.action
            trajectory_dicts.append(step_dict)
        
        # Get probabilities for all models
        probabilities = router_inference.get_model_probabilities(trajectory_dicts)
        
        # Select the best model
        best_model, confidence = router_inference.select_best_model(trajectory_dicts)
        
        # Map internal model names to LiteLLM names
        model_mapping = {
            "claude-3-5-haiku-20241022": "neulab/claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514": "neulab/claude-sonnet-4-20250514",
            "deepseek-v3": "neulab/deepseek-v3",
            "devstral-small-2505": "neulab/devstral-small-2505",
            "kimi-k2-0711-preview": "neulab/kimi-k2-0711-preview"
        }
        
        selected_litellm_model = model_mapping.get(best_model, best_model)
        
        return SWEBenchRoutingResponse(
            selected_model=selected_litellm_model,
            confidence=confidence,
            all_probabilities=probabilities
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def route_chat(req: ChatRequest):
    """Proxy the chat completion to the routed model."""
    # Extract trajectory from messages for routing
    trajectory_dicts = []
    for msg in req.messages:
        if msg.role == "user":
            trajectory_dicts.append({"source": "user", "content": msg.content})
        elif msg.role == "assistant":
            trajectory_dicts.append({"source": "agent", "content": msg.content})
    
    # Use router to select the best model
    if trajectory_dicts:
        print(f"[DEBUG] Using router inference with {len(trajectory_dicts)} trajectory steps")
        # probabilities = router_inference.get_model_probabilities(trajectory_dicts)
        best_model, confidence = router_inference.select_best_model(trajectory_dicts)
        
        # Map internal model names to LiteLLM names
        model_mapping = {
            "claude-3-5-haiku-20241022": "neulab/claude-3-5-haiku-20241022",
            "claude-sonnet-4-20250514": "neulab/claude-sonnet-4-20250514",
            "deepseek-v3": "neulab/deepseek-v3",
            "devstral-small-2505": "neulab/devstral-small-2505",
            "kimi-k2-0711-preview": "neulab/kimi-k2-0711-preview"
        }
        selected_model = model_mapping.get(best_model, best_model)
        print(f"[DEBUG] Router selected: {best_model} -> {selected_model}")
    else:
        # Fallback to round-robin if no trajectory
        print(f"[DEBUG] No trajectory, using round-robin fallback")
        import itertools
        _round_robin = itertools.cycle(AVAILABLE_MODELS)
        selected_model = next(_round_robin)
        print(f"[DEBUG] Round-robin selected: {selected_model}")
    
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
    print(f"[DEBUG] Response attributes: {dir(response)}")
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
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0
        }
    }
    
    return JSONResponse(content=response_data)

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "models": AVAILABLE_MODELS,
        "router_model": "qwen3_router_json"
    }

# Test endpoint
@app.post("/test_router")
async def test_router():
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
    uvicorn.run(app, host="0.0.0.0", port=8000) 