import os
import json
import wandb
import transformers
import inspect
import trl

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
import torch, torch.nn.functional as F

print("--- TRANSFORMERS DEBUGGING INFO ---")
print("Transformers version being used by script:", transformers.__version__)
print("Path to loaded transformers library:", transformers.__file__)

sig = inspect.signature(TrainingArguments)
print("Parameters available in TrainingArguments:", list(sig.parameters.keys()))
print("--- END TRANSFORMERS DEBUGGING INFO ---")

print("\n--- TRL DEBUGGING INFO ---")
print("TRL version being used by script:", trl.__version__)
print("Path to loaded TRL library:", trl.__file__)

sig_trl = inspect.signature(SFTTrainer)
print("Parameters available in SFTTrainer:", list(sig_trl.parameters.keys()))
print("--- END TRL DEBUGGING INFO ---\n")

# Initialize wandb
wandb.init(
    project="model-routing-router",
    name="qwen3-router-sft",
    config={
        "model": "Qwen/Qwen3-0.6B-Base",
        "max_length": 8192,
        "per_device_train_batch_size": 1,
        "num_train_epochs": 2,
        "learning_rate": 2e-5,
        "fp16": True,
        "bf16": False,
        "packing": False,
    }
)

# Add a callback to see what's being logged
from transformers import TrainerCallback

class WandbLoggingCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        print(f"Step {state.global_step} beginning")
    
    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step} ending")
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        print(f"=== EVALUATION CALLBACK TRIGGERED ===")
        print(f"Step: {state.global_step}")
        print(f"Metrics: {metrics}")
        print(f"================================")
        # Explicitly log evaluation metrics to wandb
        if metrics:
            wandb.log(metrics, step=state.global_step)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"=== LOGGING CALLBACK TRIGGERED ===")
        print(f"Step: {state.global_step}")
        print(f"Logs: {logs}")
        print(f"================================")
    
    def on_prediction_step(self, args, state, control, **kwargs):
        print(f"=== PREDICTION STEP CALLBACK ===")
        print(f"Step: {state.global_step}")
        print(f"================================")
    
    def on_evaluation_begin(self, args, state, control, **kwargs):
        print(f"=== EVALUATION BEGINNING ===")
        print(f"Step: {state.global_step}")
        print(f"================================")
    
    def on_evaluation_end(self, args, state, control, metrics=None, **kwargs):
        print(f"=== EVALUATION ENDING ===")
        print(f"Step: {state.global_step}")
        print(f"Final metrics: {metrics}")
        print(f"================================")

ds = load_dataset(
    "json",
    data_files={
        "train": "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_train.jsonl",   # 80% of rows here
        "valid": "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/router_valid.jsonl",   # held-out 20%
    },
    streaming=True
)

SEP_MODEL = "[M]" # special model-name delimiter
TOK_SPECIALS = {"additional_special_tokens": [SEP_MODEL, "[YES]", "[NO]"]}

def build_prompt(traj, cand):
    # Handle datetime objects and other non-serializable types
    def safe_json_dumps(obj):
        try:
            return json.dumps(obj, sort_keys=True, default=str)
        except:
            return str(obj)
    
    steps = "\n".join(safe_json_dumps(s) for s in traj)

    return (
        "<|system|>\n"
        "You predict whether the agent will ultimately solve the SWE issue given the trajectory so far.  Respond with [YES] or [NO] only."
        "\n</|system|>\n\n"
        "### Partial trajectory\n" + steps + "\n\n"
        "### Candidate model\n" + SEP_MODEL + " " + cand + "\n\n"
        "### Will this agent eventually succeed?\n"
    )


tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
tok.add_special_tokens(TOK_SPECIALS)

def row_to_text(ex):
    prompt = build_prompt(ex["partial_trajectory"], ex["model"])
    answer = " [YES]" if ex["successfully_patched"] else " [NO]" # ensures single token
    return {"text": prompt + answer}

# For streaming datasets, we need to process differently
# Convert streaming datasets to regular datasets for training
from datasets import Dataset

# Process streaming datasets properly
train_data = []
for item in ds["train"]:
    train_data.append(row_to_text(item))

valid_data = []
for item in ds["valid"]:
    valid_data.append(row_to_text(item))

# Create new datasets from the processed data
ds = {
    "train": Dataset.from_list(train_data),
    "valid": Dataset.from_list(valid_data)
}

# Log dataset info to wandb
wandb.log({
    "train_samples": len(train_data),
    "valid_samples": len(valid_data),
})

BASE = "Qwen/Qwen3-0.6B-Base"
model = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True)
# resize the token embeddings to account for the new special tokens
model.resize_token_embeddings(len(tok))

training_args = SFTConfig(
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,  # Explicitly set eval batch size
    num_train_epochs = 2,
    learning_rate = 2e-5,
    output_dir = "checkpoints/qwen3_router_json",
    fp16 = True,  # Use fp16 instead of bf16
    bf16 = False, # Disable bf16
    # Add wandb logging
    report_to=["wandb"],
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    run_name = "qwen3-router-sft-training",  # Set a descriptive run name
    max_length = 4096,  # Reduce max length to prevent OOM
    eval_strategy="steps"
)

trainer = SFTTrainer(
    model = model,
    processing_class = tok,
    args = training_args,
    train_dataset = ds["train"],
    eval_dataset = ds["valid"],
    formatting_func = lambda ex: ex["text"],
    callbacks=[WandbLoggingCallback()]
    # packing = False,  # Disable packing to avoid cross-contamination (warning abt this?)
)

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Running on CPU")

# Log GPU info to wandb
try:
    gpu_device = torch.cuda.get_device_name() if torch.cuda.is_available() else "None"
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    gpu_free_memory = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
except:
    gpu_device = "None"
    gpu_memory = 0
    gpu_free_memory = 0

wandb.log({
    "cuda_available": torch.cuda.is_available(),
    "gpu_device": gpu_device,
    "gpu_memory_gb": gpu_memory,
    "gpu_free_memory_gb": gpu_free_memory,
})

# Clear GPU cache before training
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(f"GPU memory before training: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")

trainer.train()
metrics = trainer.evaluate()
wandb.log(metrics)
trainer.save_model("checkpoints/qwen3_router_json")

# Log final model path
wandb.log({"final_model_path": "checkpoints/qwen3_router_json"})

# Finish wandb run
wandb.finish()


