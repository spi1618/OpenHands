import os
import json
import wandb
import transformers
import inspect
import trl
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, SFTConfig
import torch, torch.nn.functional as F

# Add distributed training imports
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(description='Train router model with SFT')
parser.add_argument('--model-name', required=True, help='Model name for output directory and wandb run')
parser.add_argument('--train-file', required=True, help='Path to training data JSONL file')
parser.add_argument('--valid-file', required=True, help='Path to validation data JSONL file')
parser.add_argument('--output-dir', required=True, help='Output directory for model checkpoints')
args = parser.parse_args()

# ============================================================================
# CONFIGURATION - All max length/token settings in one place
# ============================================================================
MAX_SEQUENCE_LENGTH = 8192  # Maximum tokens for training (including answer)
MAX_FILTER_TOKENS = 32000   # Filter out examples longer than this during preprocessing (coarse filter before smart truncation)

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

# Initialize wandb with just the project and run name
# The training config will be automatically logged by the trainer
wandb.init(
    project="model-routing-router",
    name=f"qwen3-router-sft-{args.model_name}", # NAME HERE
    # config={
    #     "model": "Qwen/Qwen3-0.6B-Base",
    #     "max_length": 8192,
    #     "per_device_train_batch_size": 1,
    #     "num_train_epochs": 2,
    #     "learning_rate": 2e-5,
    #     "fp16": True,
    #     "bf16": False,
    #     "packing": False,
    # }
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
    
    # def on_prediction_step(self, args, state, control, **kwargs):
    #     print(f"=== PREDICTION STEP CALLBACK ===")
    #     print(f"Step: {state.global_step}")
    #     print(f"================================")
    
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
        "train": args.train_file,   # 80% of rows here
        "valid": args.valid_file,   # held-out 20%
    },
    streaming=True
) # NAME HERE

SEP_MODEL = "[M]" # special model-name delimiter
TOK_SPECIALS = {"additional_special_tokens": [SEP_MODEL, "[YES]", "[NO]"]}

# Handle datetime objects and other non-serializable types
def safe_json_dumps(obj):
    try:
        return json.dumps(obj, sort_keys=True, default=str)
    except:
        return str(obj)

def build_prompt(traj, cand):
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
    answer = "[YES]" if ex["successfully_patched"] else "[NO]" # ensures single token
    
    ### SMART TRUNCATION ###
    
    # Split prompt into parts to preserve structure
    system_part = "<|system|>\nYou predict whether the agent will ultimately solve the SWE issue given the trajectory so far.  Respond with [YES] or [NO] only.\n</|system|>\n\n"
    trajectory_part = "### Partial trajectory\n" + "\n".join(safe_json_dumps(s) for s in ex["partial_trajectory"]) + "\n\n"
    model_part = "### Candidate model\n" + SEP_MODEL + " " + ex["model"] + "\n\n"
    question_part = "### Will this agent eventually succeed?\n"
    
    # Calculate token lengths for each part
    system_tokens = len(tok.encode(system_part))
    model_tokens = len(tok.encode(model_part))
    question_tokens = len(tok.encode(question_part))
    answer_tokens = len(tok.encode(answer))
    
    # Reserve space for essential parts (system, model, question, answer)
    reserved_tokens = system_tokens + model_tokens + question_tokens + answer_tokens
    
    # Calculate how much space we have for trajectory
    max_trajectory_tokens = MAX_SEQUENCE_LENGTH - reserved_tokens
    
    # Truncate trajectory if needed, but preserve all structural parts
    trajectory_tokens = tok.encode(trajectory_part)
    if len(trajectory_tokens) > max_trajectory_tokens:
        # Truncate trajectory tokens and reconstruct
        truncated_trajectory_tokens = trajectory_tokens[:max_trajectory_tokens]
        truncated_trajectory = tok.decode(truncated_trajectory_tokens)
        # Ensure we have the header
        if not truncated_trajectory.startswith("### Partial trajectory\n"):
            truncated_trajectory = "### Partial trajectory\n" + truncated_trajectory
        truncated_trajectory += "\n\n"
    else:
        truncated_trajectory = trajectory_part
    
    # Reconstruct the full prompt with all essential parts
    prompt = system_part + truncated_trajectory + model_part + question_part
    
    return {"text": prompt + answer}

# For streaming datasets, we need to process differently
# Convert streaming datasets to regular datasets for training
from datasets import Dataset

# # Process streaming datasets properly
# train_data = []
# for item in ds["train"]:
#     train_data.append(row_to_text(item))

# valid_data = []
# for item in ds["valid"]:
#     valid_data.append(row_to_text(item))

# Process streaming datasets properly with length filtering
train_data = []
valid_data = []

def process_dataset(dataset, max_tokens=MAX_FILTER_TOKENS):
    processed_data = []
    skipped_count = 0
    
    for item in dataset:
        # First, do a coarse filter on the raw trajectory length
        # This prevents processing extremely long examples that would lose too much of their context
        raw_trajectory_text = "\n".join(safe_json_dumps(s) for s in item["partial_trajectory"])
        raw_tokens = tok.encode(raw_trajectory_text)
        
        if len(raw_tokens) > max_tokens:
            skipped_count += 1
            continue
        
        # If it passes the coarse filter, apply smart truncation
        formatted = row_to_text(item)
        processed_data.append(formatted)
    
    print(f"Processed {len(processed_data)} examples, skipped {skipped_count} examples > {max_tokens} tokens (coarse filter)")
    return processed_data

train_data = process_dataset(ds["train"])
valid_data = process_dataset(ds["valid"])

# Create new datasets from the processed data
ds = {
    "train": Dataset.from_list(train_data),
    "valid": Dataset.from_list(valid_data)
}

# Log dataset info to wandb
wandb.log({
    "train_samples": len(train_data),
    "valid_samples": len(valid_data),
    "max_sequence_length": MAX_SEQUENCE_LENGTH,
    "max_filter_tokens": MAX_FILTER_TOKENS,
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
    output_dir = args.output_dir, # NAME HERE
    fp16 = True,  # Use fp16 instead of bf16
    bf16 = False, # Disable bf16
    # Add wandb logging
    report_to=["wandb"],
    logging_steps=10,
    eval_steps=100,
    save_steps=500,
    run_name = f"training-{args.model_name}",  # Set a descriptive run name # NAME HERE
    max_length = MAX_SEQUENCE_LENGTH,  # this is the max length of the input tokens
    eval_strategy="steps",
    gradient_checkpointing=True,  # Trade compute for memory
    # dataloader_pin_memory=False,  # Reduce memory usage
    
    # Multi-GPU training settings
    ddp_find_unused_parameters=False,
    dataloader_pin_memory=False,  # Reduce memory usage for multi-GPU
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
trainer.save_model(args.output_dir) # NAME HERE

# Log final model path
wandb.log({"final_model_path": args.output_dir}) # NAME HERE

# Finish wandb run
wandb.finish()


