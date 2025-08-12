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
print(f"[DEBUG] Environment RANDOM_MODE: {os.getenv('RANDOM_MODE', 'not set')}")

# Check if random mode is enabled
RANDOM_MODE = os.getenv("RANDOM_MODE", "false").lower() == "true"
if RANDOM_MODE:
    print(f"[INFO] Random mode enabled - will use random model selection instead of router model")
    router_inference = None
else:
    print(f"[INFO] Router mode enabled - will use fine-tuned router model")
    router_inference = RouterInference()

# Initialize the tokenizer for accurate token counting
print(f"[DEBUG] Initializing tokenizer for accurate token counting")
try:
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    print(f"[DEBUG] Tokenizer initialized successfully")
except Exception as e:
    print(f"[WARNING] Failed to initialize tokenizer: {e}")
    print(f"[WARNING] Falling back to character-based estimation")
    tokenizer = None

def count_tokens(text: str) -> int:
    """
    Count tokens using the actual tokenizer, or fall back to character estimation.
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text))
        except Exception as e:
            print(f"[WARNING] Tokenizer failed, falling back to character estimation: {e}")
            return int(len(text) * 0.75)
    else:
        return int(len(text) * 0.75)

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

def truncate_trajectory_to_last_4_steps(trajectory_dicts: List[Dict]) -> List[Dict]:
    """
    Truncate trajectory to contain at most the last 4 steps as defined in traj_to_steps.py.
    A "step" is defined as an action/observation pair or the initial user query.
    """
    # Debug: Print a message to say that we're entering truncate_trajectory_to_last_4_steps
    print(f"[DEBUG - TTL4] Entering truncate_trajectory_to_last_4_steps")
    
    ### TODO: Remove this, I think it's wrong
    # if len(trajectory_dicts) <= 4:
    #     print(f"[DEBUG] trajectory_dicts has 4 or fewer steps, returning as is")
    #     return trajectory_dicts
    
    # Find step boundaries (similar to traj_to_steps.py logic)
    step_boundaries = []
    
    # Find agent indices (these mark potential step boundaries)
    agent_indices = [i for i, step in enumerate(trajectory_dicts) 
                     if step.get("source") == "agent"]
    # Debug: Print agent_indices
    print(f"[DEBUG - TTL4] agent_indices: {agent_indices}")
    
    # Find observation indices (these mark step completions)
    obs_indices = [i for i, step in enumerate(trajectory_dicts) 
                   if step.get("source") == "agent" and "observation" in step]
    # Debug: Print obs_indices
    print(f"[DEBUG - TTL4] obs_indices: {obs_indices}")
    
    # Build step boundaries similar to training data
    if len(agent_indices) >= 2:
        step_boundaries.append(agent_indices[1])
    
    for obs_idx in obs_indices:
        step_boundaries.append(obs_idx + 1)
        
    # Debug: Print step_boundaries
    print(f"[DEBUG - TTL4] step_boundaries: {step_boundaries}")
    
    # Ensure we have the full history if needed
    if step_boundaries and step_boundaries[-1] < len(trajectory_dicts):
        step_boundaries.append(len(trajectory_dicts))
    
    # If we have more than 4 steps, use sliding window
    if len(step_boundaries) > 4:
        start_idx = step_boundaries[-5]  # Start from 5th-to-last step boundary
        # Debug: Print start_idx
        print(f"[DEBUG - TTL4] start_idx: {start_idx}")
        return trajectory_dicts[start_idx:]
    
    return trajectory_dicts

def build_prompt_template() -> str:
    """Build the prompt template without trajectory content."""
    return (
        "<|system|>\n"
        "You predict whether the agent will ultimately solve the SWE issue given the trajectory so far.  Respond with [YES] or [NO] only."
        "\n</|system|>\n\n"
        "### Partial trajectory\n{trajectory}\n\n"
        "### Candidate model\n[M] {model}\n\n"
        "### Will this agent eventually succeed?\n"
    )

def smart_truncate_trajectory(trajectory_dicts: List[Dict], max_tokens: int) -> List[Dict]:
    """
    Intelligently truncate trajectory to fit within token limit while preserving step structure.
    """
    print(f"[DEBUG] Smart truncation starting with {len(trajectory_dicts)} events, max_tokens: {max_tokens}")
    
    # Start from the most recent events and work backwards
    truncated_events = []
    current_tokens = 0
    
    for i, event in enumerate(reversed(trajectory_dicts)):
        event_text = json.dumps(event)
        event_tokens = count_tokens(event_text)  # Use tokenizer
        
        print(f"[DEBUG] Event {len(trajectory_dicts) - i - 1}: size={event_tokens} tokens, current_total={current_tokens}")
        print(f"[DEBUG] Event {len(trajectory_dicts) - i - 1} content: {debug_json_content(event)}")
        
        # Check if adding this event would exceed the limit
        if current_tokens + event_tokens <= max_tokens:
            print(f"[DEBUG] Event {len(trajectory_dicts) - i - 1} fits, adding full event")
            truncated_events.insert(0, event)  # Add to beginning (preserve order)
            current_tokens += event_tokens
        else:
            print(f"[DEBUG] Event {len(trajectory_dicts) - i - 1} too large, trying partial event")
            # If we can't fit the full event, try to fit part of it
            remaining_tokens = max_tokens - current_tokens
            print(f"[DEBUG] Remaining space: {remaining_tokens} tokens")
            
            if remaining_tokens > 50:  # Only if we have meaningful space left (50 tokens)
                partial_event = truncate_event_content(event, remaining_tokens)
                if partial_event:
                    print(f"[DEBUG] Created partial event: {debug_json_content(partial_event)}")
                    truncated_events.insert(0, partial_event)
                else:
                    print(f"[DEBUG] Could not create partial event from remaining {remaining_tokens} tokens")
            else:
                print(f"[DEBUG] Not enough space for partial event (need >50 tokens, have {remaining_tokens})")
            break
    
    print(f"[INFO] Smart truncation: {len(trajectory_dicts)} → {len(truncated_events)} events, ~{current_tokens} tokens")
    return truncated_events

def truncate_event_content(event: Dict, max_chars: int) -> Optional[Dict]:
    """
    Truncate individual event content if needed, preserving essential information.
    """
    print(f"[DEBUG] truncate_step_content: max_tokens={max_chars}, event_keys={list(event.keys())}")
    
    # Prioritize different fields based on importance
    priority_fields = ["source", "action", "observation", "content"]
    
    truncated_event = {"source": event["source"]}  # Always preserve source
    current_chars = count_tokens(json.dumps({"source": event["source"]}))
    
    print(f"[DEBUG] Starting with source field: {current_chars} tokens")
    
    for field in priority_fields[1:]:  # Skip source (already added)
        if field in event:
            field_content = event[field]
            if field_content:
                # Calculate how much of this field we can fit
                field_json = json.dumps({field: field_content})
                field_chars = count_tokens(field_json)
                
                print(f"[DEBUG] Field '{field}': content_length={len(str(field_content))}, json_tokens={field_chars}")
                
                if current_chars + field_chars <= max_chars:
                    print(f"[DEBUG] Field '{field}' fits completely")
                    truncated_event[field] = field_content
                    current_chars += field_chars
                else:
                    print(f"[DEBUG] Field '{field}' too large, trying to truncate")
                    # Truncate the field content itself
                    remaining_chars = max_chars - current_chars - count_tokens(field_json.replace(field_content, ""))
                    print(f"[DEBUG] Remaining tokens for field '{field}': {remaining_chars}")
                    
                    if remaining_chars > 10:  # Only if we have meaningful space left (10 tokens)
                        if isinstance(field_content, str):
                            truncated_content = field_content[:remaining_chars] + "..."
                            truncated_event[field] = truncated_content
                            print(f"[DEBUG] Created truncated field '{field}': {count_tokens(truncated_content)} tokens")
                        else:
                            print(f"[DEBUG] Field '{field}' is not a string, skipping truncation")
                    else:
                        print(f"[DEBUG] Not enough space for field '{field}' (need >10 tokens, have {remaining_chars})")
                    break
            else:
                print(f"[DEBUG] Field '{field}' has no content, skipping")
        else:
            print(f"[DEBUG] Field '{field}' not present in event")
    
    result = truncated_event if len(truncated_event) > 1 else None
    print(f"[DEBUG] truncate_step_content result: {result}")
    return result

def truncate_trajectory_with_token_limit(trajectory_dicts: List[Dict], max_tokens: int = 8192) -> List[Dict]:
    """
    First truncate to last 4 steps, then apply token-level truncation if needed.
    """
    # Debug: Print a message to say that we're entering truncate_trajectory_with_token_limit
    print(f"[DEBUG - TTTL] Entering truncate_trajectory_with_token_limit")
    
    # Stage 1: Truncate to last 4 steps (as discussed before)
    step_truncated = truncate_trajectory_to_last_4_steps(trajectory_dicts)
    
    # Debug: Print step_truncated
    print(f"[DEBUG - TTTL] step_truncated from TTL4: {debug_json_content(step_truncated)}")
    
    print(f"[DEBUG - TTTL] Stage 1: Truncated to last 4 steps: {len(step_truncated)} events")
    for i, event in enumerate(step_truncated):
        event_json = json.dumps(event)
        print(f"[DEBUG - TTTL] Event {i} length: {count_tokens(event_json)} tokens, content preview: {debug_json_content(event)}")
    
    # Stage 2: Check if we still exceed token limit
    prompt_template = build_prompt_template()
    trajectory_text = "\n".join(json.dumps(s) for s in step_truncated)
    
    print(f"[DEBUG - TTTL] Prompt template: {repr(prompt_template)}")
    print(f"[DEBUG - TTTL] Trajectory text preview (first 500 chars): {trajectory_text[:500]}")
    print(f"[DEBUG - TTTL] Trajectory text preview (last 500 chars): {trajectory_text[-500:]}")
    
    # Use tokenizer for accurate token estimation
    prompt_tokens = count_tokens(prompt_template)
    trajectory_tokens = count_tokens(trajectory_text)
    
    print(f"[DEBUG - TTTL] Stage 2: 4 steps = {len(step_truncated)} events, estimated tokens: {prompt_tokens + trajectory_tokens}")
    print(f"[DEBUG - TTTL] Prompt template chars: {len(prompt_template)}")
    print(f"[DEBUG - TTTL] Trajectory chars: {len(trajectory_text)}")
    print(f"[DEBUG - TTTL] Total chars: {len(prompt_template) + len(trajectory_text)}")
    
    if prompt_tokens + trajectory_tokens <= max_tokens:
        print(f"[DEBUG - TTTL] Within token limit, returning {len(step_truncated)} events")
        return step_truncated
    
    # Stage 3: Apply smart truncation (similar to training)
    print(f"[WARNING - TTTL] 4 steps still exceed {max_tokens} estimated tokens ({prompt_tokens + trajectory_tokens}). Applying smart truncation.")
    
    # Reserve space for essential parts (rough estimate)
    essential_parts = prompt_template.replace("{trajectory}", "").replace("{model}", "dummy")
    essential_chars = count_tokens(essential_parts)
    
    # Calculate available space for trajectory
    available_chars = max_tokens - essential_chars
    
    print(f"[DEBUG - TTTL] Essential parts tokens: {essential_chars}")
    print(f"[DEBUG - TTTL] Available tokens for trajectory: {available_chars}")
    
    # Truncate trajectory while preserving step boundaries
    final_trajectory = smart_truncate_trajectory(step_truncated, available_chars)
    
    # Debug: Show final prompt content
    final_trajectory_text = "\n".join(json.dumps(s) for s in final_trajectory)
    final_prompt = prompt_template.replace("{trajectory}", final_trajectory_text).replace("{model}", "dummy")
    
    print(f"[DEBUG - TTTL] Final trajectory: {len(final_trajectory)} events")
    for i, event in enumerate(final_trajectory):
        event_json = json.dumps(event)
        print(f"[DEBUG - TTTL] Final event {i} length: {count_tokens(event_json)} tokens")
    
    print(f"[DEBUG - TTTL] Final prompt length: {count_tokens(final_prompt)} tokens")
    print(f"[DEBUG - TTTL] Final prompt beginning (first 500 chars): {final_prompt[:500]}")
    print(f"[DEBUG - TTTL] Final prompt end (last 500 chars): {final_prompt[-500:]}")
    
    return final_trajectory

# FastAPI app
app = FastAPI(title="SWE-Bench Router", version="0.1")

# @app.post("/route_swe_bench", response_model=SWEBenchRoutingResponse)
# async def route_swe_bench(req: SWEBenchRoutingRequest):
#     """Route a SWE-Bench step using the fine-tuned router model."""
#     try:
#         # Convert trajectory steps to the format expected by router
#         trajectory_dicts = []
#         for step in req.partial_trajectory:
#             step_dict = {"source": step.source}
#             if step.content:
#                 step_dict["content"] = step.content
#             if step.observation:
#                 step_dict["observation"] = step.observation
#             if step.action:
#                 step_dict["action"] = step.action
#             trajectory_dicts.append(step_dict)
        
#         # Apply smart truncation
#         truncated_trajectory = truncate_trajectory_with_token_limit(trajectory_dicts, max_tokens=8192)
        
#         print(f"[DEBUG] Original trajectory: {len(trajectory_dicts)} steps")
#         print(f"[DEBUG] Truncated trajectory: {len(truncated_trajectory)} steps")
        
#         # Debug: Show what we're sending to the router
#         if truncated_trajectory:
#             print(f"[DEBUG] Sending to router: {len(truncated_trajectory)} steps")
#             for i, step in enumerate(truncated_trajectory):
#                 print(f"[DEBUG] Router step {i}: {debug_json_content(step)}")
#         else:
#             print(f"[DEBUG] WARNING: Empty trajectory being sent to router!")
        
#         # Get probabilities for all models
#         probabilities = router_inference.get_model_probabilities(truncated_trajectory)
        
#         # Select the best model
#         best_model, confidence = router_inference.select_best_model(truncated_trajectory)
        
#         # Map internal model names to LiteLLM names
#         model_mapping = {
#             "claude-3-5-haiku-20241022": "neulab/claude-3-5-haiku-20241022",
#             "claude-sonnet-4-20250514": "neulab/claude-sonnet-4-20250514",
#             "deepseek-v3": "neulab/deepseek-v3",
#             "devstral-small-2505": "neulab/devstral-small-2505",
#             "kimi-k2-0711-preview": "neulab/kimi-k2-0711-preview"
#         }
        
#         selected_litellm_model = model_mapping.get(best_model, best_model)
        
#         return SWEBenchRoutingResponse(
#             selected_model=selected_litellm_model,
#             confidence=confidence,
#             all_probabilities=probabilities
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Routing error: {str(e)}")

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
        if msg.role == "user":
            trajectory_dicts.append({"source": "user", "content": msg.content})
        elif msg.role == "assistant":
            trajectory_dicts.append({"source": "agent", "content": msg.content})
        elif msg.role == "system":
            trajectory_dicts.append({"source": "system", "content": msg.content})
            
    # Debug: Print of trajectory_dicts
    print(f"[DEBUG] trajectory_dicts: {debug_json_content(trajectory_dicts)}")
    # Debug: Print the length of trajectory_dicts
    print(f"[DEBUG] length of trajectory_dicts: {len(trajectory_dicts)}")
    
    # Apply smart truncation only if not in random mode
    if RANDOM_MODE:
        # Random mode: no need to truncate since we don't use the trajectory
        print(f"[DEBUG] Random mode enabled, skipping trajectory truncation")
        truncated_trajectory = trajectory_dicts  # Keep original trajectory for debugging
    else:
        # Router mode: apply smart truncation for the router model
        print(f"[DEBUG] Router mode enabled, applying trajectory truncation")
        truncated_trajectory = truncate_trajectory_with_token_limit(trajectory_dicts, max_tokens=8192)
    
    print(f"[DEBUG] Original trajectory: {len(trajectory_dicts)} events")
    print(f"[DEBUG] Truncated trajectory: {len(truncated_trajectory)} events")
    
    # Debug: Show what we're sending to the router (only relevant in router mode)
    if not RANDOM_MODE and truncated_trajectory:
        print(f"[DEBUG] Sending to router: {len(truncated_trajectory)} events")
        for i, event in enumerate(truncated_trajectory):
            print(f"[DEBUG] Router event {i}: {debug_json_content(event)}")
    elif not RANDOM_MODE and not truncated_trajectory:
        print(f"[DEBUG] WARNING: Empty trajectory being sent to router!")
    else:
        print(f"[DEBUG] Random mode: trajectory not used for model selection")
    
    # Use router to select the best model
    if RANDOM_MODE:
        # Random mode: select a random model
        print(f"[DEBUG] Random mode enabled, selecting a random model")
        import random
        selected_model = random.choice(AVAILABLE_MODELS)
        print(f"[DEBUG] Random selected: {selected_model}")
    elif truncated_trajectory:
        print(f"[DEBUG] Using router inference with {len(truncated_trajectory)} trajectory events")
        # probabilities = router_inference.get_model_probabilities(truncated_trajectory)
        best_model, confidence = router_inference.select_best_model(truncated_trajectory)
        print(f"[DEBUG] Router response - best_model: {best_model}, confidence: {confidence}")
        
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