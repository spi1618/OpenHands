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
from litellm import token_counter, completion
from typing import Any

import datetime

# Import the router inference
from router_inference_stupid import RouterInference

# Configure available models + client
# DO NOT SIMPLIFY THE NAMES HERE, THEY MUST MATCH THE CONFIG SO WE CAN USE THE LITELLM PROXY SERVER; THEY SHOULD MATCH THE CONFIG FILE
MODEL_MAPPING = json.load(open("/home/sophiapi/model-routing/static_things/model_name_mapping.json", "r"))
AVAILABLE_MODELS = list(MODEL_MAPPING.values())

# One global OpenAI client configured for the proxy
_client = OpenAI(
    api_key=os.getenv("LITELLM_API_KEY"),
    base_url="https://cmu.litellm.ai",
)

# # move
# response = litellm.completion(
#     model="litellm/neulab/claude-3-5-haiku-20241022",
#     messages=[{"role": "user", "content": "Hello, how are you?"}],
# )

if _client.api_key is None:
    raise RuntimeError("Set LITELLM_API_KEY environment variable before running.")

# Initialize the router inference
print(f"[DEBUG] Starting SWE-Bench Router")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "/data/user_data/sophiapi/checkpoints/stupid_consistent_qwen3_router_model-5_instance-100_max-length-16384_samples-20000")
ROUTER_CHECKPOINT = os.getenv("ROUTER_CHECKPOINT", "checkpoint-4500")
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

class ChatMessage(BaseModel):
    role: str  # "system", "user", "assistant"
    content: str
    metadata: Optional[dict[str, Any]] = Field(default_factory=dict)
    # extra_body: dict | None = None

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: Optional[str] = Field(default=None, description="(optional) override to force a particular back‑end LLM")
    max_tokens: Optional[int] = 1024
    # extra_body: dict | None = None

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

def convert_messages_to_partial_trajectory(messages: List[ChatMessage]) -> List[Dict]:
    """
    Convert a list of messages to a partial trajectory in the format expected by the router model.
    Does NOT do truncation and DOES NOT add in the prompt for the router model.
    """
    # Rule: get rid of the first message (role='system'), just very long OpenHands system message
    # Rule: for the second message (role='user'), only extract the content between the following strings (DO NOT include the strings themselves):
    #     start_bookend: "--------------------- NEW TASK DESCRIPTION ---------------------\n"
    #     end_bookend: "\n--------------------- END OF NEW TASK DESCRIPTION ---------------------"
    # Rule: get rid of the all the rest of the messages where role='user' (these are basically the observations) 
    # BASICALLY JUST get rid of all non-assistant messages except for *part of* the first user message
    
    partial_trajectory = []
    first_user_message_found = False
    start_bookend = "--------------------- NEW TASK DESCRIPTION ---------------------\n"
    end_bookend = "\n--------------------- END OF NEW TASK DESCRIPTION ---------------------"
    
    for msg in messages:
        if msg.role == "user":
            if not first_user_message_found:
                # pull out the content between the bookends
                start_index = msg.content.find(start_bookend)
                if start_index == -1:
                    raise ValueError(f"Start bookend not found in user message: {msg.content}")
                end_index = msg.content.find(end_bookend)
                if end_index == -1:
                    raise ValueError(f"End bookend not found in user message: {msg.content}")
                partial_trajectory.append({"source": "user", "content": msg.content[start_index:end_index+len(end_bookend)]})
                first_user_message_found = True
        elif msg.role == "assistant":
            partial_trajectory.append({"source": "agent", "content": msg.content})

    # Check that the partial trajectory is not empty
    if not partial_trajectory:
        raise ValueError("Partial trajectory is empty")
    # Check that the partial trajectory's first message is a user message
    if partial_trajectory[0]["source"] != "user":
        raise ValueError(f"Partial trajectory's first message is not a user message: {partial_trajectory[0]}")
    
    return partial_trajectory
    
    
def count_cache_read_tokens(messages: List[ChatMessage]) -> int:
    # cache_read: everything up to an includeing the last assistant message,
    #     OR honestly just everything before the last user message (exclusive)
    # cache_write: everything after the last assistant message
    #     OR honestly just the last user message (inclusive)
    # use the litellm token counter
    cache_read = []
    cache_write = []
    # iterate through the messages and add to cache_read or cache_write
    
    # ==== uncomment this when the model decides to start working again ====
    
    # # find the last assistant message
    # last_assistant_message_number = -1
    # current_message_number = 0
    # for message in messages:
    #     if message["role"] == "assistant":
    #         last_assistant_message_number = current_message_number
    #     current_message_number += 1
    # # add everything up to and including the last assistant message to cache_read
    # for i in range(last_assistant_message_number + 1):
    #     cache_read.append(messages[i].model_dump())
    # # add everything after the last assistant message to cache_write
    # for i in range(last_assistant_message_number + 1, len(messages)):
    #     cache_write.append(messages[i].model_dump())

    # ======================================================================
    
    # find the last user message
    last_user_message_number = -1
    current_message_number = 0
    for message in messages:
        if message["role"] == "user":
            last_user_message_number = current_message_number
        current_message_number += 1
    # add everything up to but excluding the last user message to cache_read
    for i in range(last_user_message_number):
        print(f"[DEBUG] Adding message to cache read: {messages[i]}")
        cache_read.append(messages[i])
    # add everything after the last user message to cache_write
    for i in range(last_user_message_number, len(messages)):
        print(f"[DEBUG] Adding message to cache write: {messages[i]}")
        cache_write.append(messages[i])    
    
    count_read = token_counter(model="litellm_proxy/neulab/claude-3-5-haiku-20241022", messages=cache_read)
    count_write = token_counter(model="litellm_proxy/neulab/claude-3-5-haiku-20241022", messages=cache_write)
    print(f"[DEBUG] Manually computed cache read tokens: {count_read}")
    print(f"[DEBUG] Manually computed cache write tokens: {count_write}")
    return count_read, count_write
    # see if this matches what we get from the model directly


def apply_prompt_caching(messages: List[ChatMessage], model_name: str) -> List[ChatMessage]:
    """Applies caching breakpoints to the messages.

    For new Anthropic API, we only need to mark the last user or tool message as cacheable.
    """
    if "claude" in model_name:
        # make the content map to a list of dicts instead of a string
        for message in messages:
            # only do this if the content is a string
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
        if len(messages) > 0 and messages[0]["role"] == 'system':
            messages[0]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        # NOTE: this is only needed for anthropic
        for message in reversed(messages):
            if message["role"] in ('user', 'tool'):
                message["content"][-1]["cache_control"] = {"type": "ephemeral"}
                break
    return messages

@app.post("/v1/chat/completions")
async def route_chat(req: ChatRequest):
    """Proxy the chat completion to the routed model."""
    # Extract trajectory from messages for routing
    
    print(f"######## [DEBUG] RECEIVED REQUEST ########")
    
    # print(f"\n\n\n\n[DEBUG] Request: {req}\n\n\n\n")
    
    # # Print the request's attributes
    # print(f"[DEBUG] Request attributes: {dir(req)}")
    
    # # Debug: Print the request's messages
    # print(f"[DEBUG] Request messages: {debug_json_content(req.messages)}")
    
    for msg in req.messages:
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Message attributes: {dir(msg)}")
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Message metadata: {msg.metadata}")
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Message: {msg.model_dump()}")
    
    trajectory_dicts = convert_messages_to_partial_trajectory(req.messages)
    # Double check that the trajectory is not empty
    if not trajectory_dicts:
        raise ValueError("Trajectory is empty")
    
    # TODO: Get some sort of information about cached vs uncached tokens
    cached_input_tokens, uncached_input_tokens = count_cache_read_tokens([msg.model_dump() for msg in req.messages])
    
    # Use router to select the best model
    if RANDOM_MODE:
        # Random mode: select a random model
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Random mode enabled, selecting a random model")
        import random
        selected_model = random.choice(AVAILABLE_MODELS)
        # append "litellm_proxy/" to the model name
        selected_model = "litellm_proxy/" + selected_model
        # selected_model = "litellm_proxy/neulab/claude-3-5-haiku-20241022" # REMOVE THIS AFTER DEBUGGING TODO
        # selected_model = "litellm_proxy/neulab/claude-sonnet-4-20250514" # REMOVE THIS AFTER DEBUGGING TODO
        selected_model = "litellm_proxy/neulab/deepseek-v3" # REMOVE THIS AFTER DEBUGGING TODO
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Random selected: {selected_model}")
    elif trajectory_dicts:
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Using router inference with {len(trajectory_dicts)} trajectory events")
        
        # Pass in the raw trajectory to router_inference_stupid.py
        best_model = router_inference.select_best_model(trajectory_dicts, cached_input_tokens, uncached_input_tokens) # best_model is a tuple (model_name, yes_probability)
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Router response - best_model: {best_model[0]}, confidence: {best_model[1]}")
        
        # Map internal model names to LiteLLM names
        # The keys should match the model names that router_inference_stupid.py uses
        # The values should match the model names that LiteLLM expects (same as AVAILABLE_MODELS)
        selected_model = MODEL_MAPPING.get(best_model[0], best_model[0])
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Router selected: {best_model[0]} -> {selected_model}")
    else:
        # Fallback to random selection if no trajectory
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] No trajectory, using random fallback")
        import random
        selected_model = random.choice(AVAILABLE_MODELS)
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Random fallback selected: {selected_model}")
    
    print(f"[DEBUG {datetime.datetime.now().isoformat()}] Final selected model: {selected_model}")
    
    try:
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Counting cache read and write tokens before sending to LiteLLM proxy")
        count_read, count_write = count_cache_read_tokens([msg.model_dump() for msg in req.messages])
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Manually computed cache read tokens: {count_read}")
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Manually computed cache write tokens: {count_write}")
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Applying caching to messages...")
        cache_applied_messages = apply_prompt_caching([msg.model_dump() for msg in req.messages], selected_model)
        for cache_applied_msg in cache_applied_messages:
            print(f"[DEBUG {datetime.datetime.now().isoformat()}] Cache applied message: {cache_applied_msg}")
        print(f"[DEBUG {datetime.datetime.now().isoformat()}] Calling LiteLLM proxy with model: {selected_model}")
        response = completion(
            base_url="https://cmu.litellm.ai",
            api_key=os.getenv("LITELLM_API_KEY"),
            model=selected_model,
            # Note to my future self, because I know you will get confused: KEEP THIS AS IS - it is going to the actual LLM, not the router model
            messages=cache_applied_messages, 
            max_tokens=req.max_tokens,
        )
    except BadRequestError as e:
        raise HTTPException(status_code=502, detail=str(e))
    
    # modul dump the response
    # print(f"[DEBUG] Response: HELLO\n ")
    response_data = response.model_dump()
    print(f"[DEBUG {datetime.datetime.now().isoformat()}] Response: {response_data}")
    print(f"[DEBUG {datetime.datetime.now().isoformat()}] Response usage cache_read_input_tokens: {response.usage['cache_read_input_tokens']}")
    print(f"[DEBUG {datetime.datetime.now().isoformat()}] Response usage cache_creation_input_tokens: {response.usage['cache_creation_input_tokens']}")
    print(f"[DEBUG {datetime.datetime.now().isoformat()}] Response usage completion_tokens: {response.usage['completion_tokens']}")
    print(f"[DEBUG {datetime.datetime.now().isoformat()}] Response usage prompt_tokens: {response.usage['prompt_tokens']}")
    print(f"[DEBUG {datetime.datetime.now().isoformat()}] Response usage total_tokens: {response.usage['total_tokens']}")
    # response_data["model"] = selected_model # I think this is already contained in some way in the response data from litellm?
    
    return response_data

    # print(f"[DEBUG] Response type: {type(response)}")
    # # print(f"[DEBUG] Response attributes: {dir(response)}")
    # print(f"[DEBUG] Response choices: {response.choices}")
    # print(f"[DEBUG] Response usage: {response.usage}")
    
    # choice = response.choices[0]
    # usage = response.usage

    # # Ensure we have the correct response format
    # content = choice.message.content or ""
    # if not content and hasattr(response, 'content'):
    #     content = response.content or ""
    
    # print(f"[DEBUG] Final content: {content[:100]}...")
    
    # # Return the response in the format that LiteLLM expects
    # from fastapi.responses import JSONResponse
    
    # # TODO: check this
    # response_data = {
    #     "id": response.id,
    #     "object": "chat.completion",
    #     "created": response.created,
    #     "model": selected_model,
    #     "choices": [
    #         {
    #             "index": 0,
    #             "message": {
    #                 "role": "assistant",
    #                 "content": content
    #             },
    #             "finish_reason": "stop" # TODO: check this (what does finish_reason mean?)
    #         }
    #     ],
    #     "usage": {
    #         "prompt_tokens": usage.prompt_tokens if usage else 0,
    #         "completion_tokens": usage.completion_tokens if usage else 0,
    #         "total_tokens": usage.total_tokens if usage else 0
    #     }
    # }
    
    # print(f"[DEBUG] Sending response: model={selected_model}, content_length={len(content)}, total_tokens={usage.total_tokens if usage else 0}")
    
    # return JSONResponse(content=response_data)

@app.get("/health")
async def health():
    return {
        "status": "ok", 
        "models": AVAILABLE_MODELS,
        "router_model": "qwen3_router_json" if not RANDOM_MODE else "random",
        "mode": "random" if RANDOM_MODE else "router",
        "random_mode_enabled": RANDOM_MODE
    }



if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable or default to 8123
    port = int(os.getenv("ROUTER_PORT", "8123"))
    uvicorn.run(app, host="0.0.0.0", port=port) 