from datasets import load_dataset  
from trl import SFTTrainer, SFTConfig  
from trl.data_utils import is_conversational
from transformers import AutoTokenizer, AutoModelForCausalLM
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import wandb

# Understanding the datasets:
# 1. RAW: no pruning, no truncation, no formatting
# 2. PRUNED: pruned to the last 4 steps, no truncation, no formatting
# 3. TRUNCATED: truncated to some number of tokens (eg. 4096 or 8192), in prompt-completion format
# 4. BALANCED: balanced to, eg. 40% successes and 60% failures
# 5. NOTOKENS: no custom tokens (find and replace "[YES]" --> "YES" and "[NO]" --> "NO")
# 6. CLEANED: cleaned dataset with partial trajectories with cleaner {source, message, thought} format (as of 8/26/2025 5:16 PM all cleaned datasets are also notoken)

# many many examples
RAW_DATASET_PATH = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_pruned-4_with-ids_swe_gym_stepwise_trajectories_2025-08-05T15-00-53.jsonl"

# about 800 samples
CLEANED_TRUNCATED_TRAIN_DATASET_PATH_4096 = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/20250826_170258_train_max-length-4096.jsonl"
# about 200 samples
CLEANED_TRUNCATED_VAL_DATASET_PATH_4096 = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/20250826_171926_val_max-length-4096.jsonl"

# about 800 samples
CLEANED_TRUNCATED_TRAIN_DATASET_PATH_8192 = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/20250826_171314_train_max-length-8192.jsonl"
# about 200 samples
CLEANED_TRUNCATED_VAL_DATASET_PATH_8192 = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/20250826_171952_val_max-length-8192.jsonl"

# about 32 samples
CLEANED_TRUNCATED_TRAIN_DATASET_PATH_4096_OVERFIT = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/40-samples/20250826_213907_train_max-length-4096.jsonl"
# about 8 samples
CLEANED_TRUNCATED_VAL_DATASET_PATH_4096_OVERFIT = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/40-samples/20250826_213957_val_max-length-4096.jsonl"

# about 800 samples
CHAT_TEMPLATE_TRAIN_DATASET_PATH_8192 = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/20250827_173948_train_chat-template_max-len-8192.jsonl"
# about 200 samples
CHAT_TEMPLATE_VAL_DATASET_PATH_8192 = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/20250827_174048_val_chat-template_max-len-8192.jsonl"

# about 800 samples
CHAT_TEMPLATE_TRAIN_DATASET_PATH_16384 = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/20250827_172724_train_chat-template_max-len-16384.jsonl"
# about 200 samples
CHAT_TEMPLATE_VAL_DATASET_PATH_16384 = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/model-5_instance-100_with-ids_swe-gym_cleaned-partial-trajectories_2025-08-26T15-26-45/1000-samples/20250827_172632_val_chat-template_max-len-16384.jsonl"

######################### NEW STUFF #########################################################

##### SET THIS BEFORE EVERY RUN AND CHECK THAT THE VALUES ARE CORRECT + MATCH THE TRAINING DATASET #####
PATH_TO_TRAIN_DATASET = CHAT_TEMPLATE_TRAIN_DATASET_PATH_16384
PATH_TO_VALID_DATASET = CHAT_TEMPLATE_VAL_DATASET_PATH_16384

# This should match the datasets
MAX_TOKENS = 16384

EVAL_STEPS = 100
SAVE_STEPS = 400
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
NUM_EPOCHS = 2

# CHANGE THIS IF YOU CHANGE THE DATASET (esp. if you use an unbalanced dataset or notokens)
CHECKPOINT_DIR = f"/data/user_data/sophiapi/checkpoints/stupid_chat-template_qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example_max-length-{MAX_TOKENS}"
WANDB_NAME = f"qwen3-router-sft-stupid-chat-template-model-5-instance-100-pruned-4-with-ids-by-example-max-length-{MAX_TOKENS}"

# Load local JSONL files  
train_dataset = load_dataset("json", data_files=PATH_TO_TRAIN_DATASET, split="train")  
val_dataset = load_dataset("json", data_files=PATH_TO_VALID_DATASET, split="train")  


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
# Check vocab size
print(f"Vocab size (without custom tokens): {len(tokenizer)}")



########## WANDB STUFF ##########

wandb.init(
    project="model-routing-router",
    name=WANDB_NAME,
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

# Log dataset info to wandb
wandb.log({
    "train_samples": len(train_dataset),
    "valid_samples": len(val_dataset),
    "max_sequence_length": MAX_TOKENS,
})


########## END OF WANDB STUFF ##########



# Configure training  
training_args = SFTConfig(  
    output_dir=CHECKPOINT_DIR,  
    per_device_train_batch_size=TRAIN_BATCH_SIZE,  
    per_device_eval_batch_size=EVAL_BATCH_SIZE,  
    num_train_epochs=NUM_EPOCHS,  
    eval_steps=EVAL_STEPS,
    save_steps=SAVE_STEPS,
    eval_strategy="steps",  
    save_strategy="steps",  
    logging_steps=10,  
    completion_only_loss=True, # this should be automatic if we're using the prompt completion format
    # assistant_only_loss=True, # idk i read this from the hf docs, not sure if it should be toggled on
    report_to=["wandb"],
    max_length=MAX_TOKENS,
    bf16=True,
    use_liger_kernel = True,
    model_init_kwargs = {
        "attn_implementation": "flash_attention_2",
        "trust_remote_code": True,
        "torch_dtype": "bfloat16",
        "device_map": "auto"
    }    
)  
  
# Initialize trainer  
trainer = SFTTrainer(  
    model="Qwen/Qwen2.5-0.5B-Instruct",  
    args=training_args,  
    train_dataset=train_dataset,  
    eval_dataset=val_dataset,  
    callbacks=[WandbLoggingCallback()],
)  



# Check if the dataset is conversational
first_example = next(iter(train_dataset))
print(f"[DEBUG] ========== \n Is conversational: {is_conversational(first_example)} \n ========== \n")



# Train the model  
trainer.train()
