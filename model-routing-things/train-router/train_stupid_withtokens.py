from datasets import load_dataset  
from trl import SFTTrainer, SFTConfig  
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
  

##### SET THIS BEFORE EVERY RUN AND CHECK THAT THE VALUES ARE CORRECT + MATCH THE TRAINING DATASET #####
PATH_TO_TRAIN_DATASET = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/20250822_011742_router_train_model-5_instance-100_pruned-4_with-ids_by-example_sample800_balanced-40-60_max-length-8192.jsonl"
PATH_TO_VALID_DATASET = "/home/sophiapi/model-routing/OpenHands/evaluation/evaluation_outputs/datasets/20250822_012118_router_valid_model-5_instance-100_pruned-4_with-ids_by-example_sample200_balanced-40-60_max-length-8192.jsonl"

MAX_TOKENS = 8192
EVAL_STEPS = 100
SAVE_STEPS = 400
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
NUM_EPOCHS = 2

SUCCESS_RATIO = 0.4
FAILURE_RATIO = 0.6 # technically redundant, but just in case
CHECKPOINT_DIR = f"/data/user_data/sophiapi/checkpoints/stupid_withtokens_qwen3_router_model-5_instance-100_pruned-4_with-ids_by-example_balanced-{int(SUCCESS_RATIO*100)}-{int(FAILURE_RATIO*100)}_max-length-{MAX_TOKENS}"
WANDB_NAME = f"qwen3-router-sft-stupid-withtokens-model-5-instance-100-pruned-4-with-ids-by-example-balanced-{int(SUCCESS_RATIO*100)}-{int(FAILURE_RATIO*100)}-max-length-{MAX_TOKENS}"

# Load your local JSONL files  
train_dataset = load_dataset("json", data_files=PATH_TO_TRAIN_DATASET, split="train")  
val_dataset = load_dataset("json", data_files=PATH_TO_VALID_DATASET, split="train")  


# Sanity check: tokenize the first five examples of the train dataset
# Print the preview and endview of the first five examples 
# Also report the number of tokens in the first five examples
for i in range(5):
    example = train_dataset[i]
    print(f"\n--- Example {i} ---")
    
    # Get the prompt and completion content
    if 'prompt' in example and 'completion' in example:
        prompt = example['prompt']
        completion = example['completion']
        
        print(f"Prompt length: {len(prompt)} characters")
        print(f"Completion length: {len(completion)} characters")
        print(f"Prompt preview: {prompt[:100]}...")
        print(f"Completion preview: {completion[:100]}...")
        print(f"Prompt endview: ...{prompt[-100:]}")
        print(f"Completion endview: ...{completion[-100:]}")
        
        # Tokenize the prompt and completion separately
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
        
        prompt_tokens = tokenizer.encode(prompt)
        completion_tokens = tokenizer.encode(completion)
        
        print(f"Prompt token count: {len(prompt_tokens)}")
        print(f"Completion token count: {len(completion_tokens)}")
        print(f"Total token count: {len(prompt_tokens) + len(completion_tokens)}")
        
        print(f"Prompt first 20 tokens: {prompt_tokens[:20]}")
        print(f"Completion first 20 tokens: {completion_tokens[:20]}")
        print(f"Prompt last 20 tokens: {prompt_tokens[-20:]}")
        print(f"Completion last 20 tokens: {completion_tokens[-20:]}")
        
        # Decode first and last few tokens to see the text
        print(f"Prompt first 20 tokens as text: '{tokenizer.decode(prompt_tokens[:20])}'")
        print(f"Completion first 20 tokens as text: '{tokenizer.decode(completion_tokens[:20])}'")
        print(f"Prompt last 20 tokens as text: '{tokenizer.decode(prompt_tokens[-20:])}'")
        print(f"Completion last 20 tokens as text: '{tokenizer.decode(completion_tokens[-20:])}'")
    else:
        print(f"Available fields: {list(example.keys())}")
        print(f"Example content: {example}")






### Load tokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
# Check vocab size
print(f"Vocab size (without custom tokens): {len(tokenizer)}")

custom_tokens = ["[YES]", "[NO]"]  
num_added_tokens = tokenizer.add_tokens(custom_tokens)  
print(f"Added {num_added_tokens} tokens to tokenizer")  
# Check vocab size
print(f"Vocab size (with custom tokens): {len(tokenizer)}")

### Load model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base", device_map="auto")
model.resize_token_embeddings(len(tokenizer))







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









########## UNCOMMENT THIS BLOCK ###########

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
    completion_only_loss=True, 
    max_length=MAX_TOKENS, 
    report_to=["wandb"],
)  
  
# Initialize trainer  
trainer = SFTTrainer(  
    model=model, 
    args=training_args,  
    train_dataset=train_dataset,  
    eval_dataset=val_dataset,  
    processing_class=tokenizer,
    callbacks=[WandbLoggingCallback()],
)  


########### END OF UNCOMMENT THIS BLOCK ###########
    
  









example_id = 5




# Test with a simple example  


test_prompt = train_dataset[example_id]['prompt']
test_completion = train_dataset[example_id]['completion']
  
prompt_ids = tokenizer(text=test_prompt)["input_ids"]  
completion_ids = tokenizer(text=test_completion)["input_ids"]
full_ids = tokenizer(text=test_prompt + test_completion)["input_ids"]  
  
# print(f"Prompt: {test_prompt}")  
# print(f"Completion: {test_completion}")  
# print(f"Prompt tokens: {prompt_ids}")  
# print(f"Completion tokens: {completion_ids}")
print(f"Number of prompt tokens: {len(prompt_ids)}")
print(f"Number of completion tokens: {len(completion_ids)}")
print(f"Number of full tokens: {len(full_ids)}")
# print(f"Full tokens: {full_ids}")  
print(f"Match check: {full_ids[:len(prompt_ids)] == prompt_ids}")




# Get a processed example  
processed_example = trainer.train_dataset[example_id]  
print("Keys in processed example:", processed_example.keys())  
  
# Check if completion_mask exists  
if "completion_mask" in processed_example:  
    completion_mask = processed_example["completion_mask"]  
    # print(f"Completion mask: {completion_mask}")  
    print(f"Completion mask sum: {sum(completion_mask)}")  
    print(f"Completion mask length: {len(completion_mask)}")  
else:  
    print("No completion_mask found in processed example")













# # Get processed examples from the trainer's prepared dataset  
# processed_examples = [trainer.train_dataset[i] for i in range(4)]  # Get 4 processed examples  
  
# # Now use the trainer's data collator on the processed examples  
# collated_batch = trainer.data_collator(processed_examples)  
  
# # Count supervised tokens for each example in the batch  
# for i in range(len(processed_examples)):  
#     labels = collated_batch["labels"][i]  
#     supervised_tokens = (labels != -100).sum().item()  
#     print(f"Example {i}: {supervised_tokens} supervised tokens")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# # Get a processed example  
# processed_example = trainer.train_dataset[0]  
# print("Keys in processed example:", processed_example.keys())  
  
# # Check if completion_mask exists  
# if "completion_mask" in processed_example:  
#     completion_mask = processed_example["completion_mask"]  
#     print(f"Completion mask: {completion_mask}")  
#     print(f"Completion mask sum: {sum(completion_mask)}")  
#     print(f"Completion mask length: {len(completion_mask)}")  
# else:  
#     print("No completion_mask found in processed example")









# # Debug: Check supervised tokens count
# print("=== DEBUGGING SUPERVISED TOKENS ===")
# print(f"Train dataset size: {len(train_dataset)}")
# print(f"Val dataset size: {len(val_dataset)}")

# # Get one batch from the training data
# dataloader = trainer.get_train_dataloader()
# batch = next(iter(dataloader))

# print(f"Batch keys: {batch.keys()}")
# if 'labels' in batch:
#     labels = batch['labels']
#     input_ids = batch['input_ids']
#     print(f"Labels shape: {labels.shape}")
#     print(f"Input IDs shape: {input_ids.shape}")
#     print(f"Labels example 3 preview: {labels[3][:20]}")  # First 20 tokens of example
#     print(f"Labels example 3 endview: {labels[3][-20:]}") # Last 20 tokens of example
    
#     # Show corresponding text for the example
#     print("\n=== TEXT VS LABELS COMPARISON (Example 3) ===")
#     # Get the tokenizer from the trainer
#     tokenizer = trainer.tokenizer
    
#     # Show first 50 tokens with their labels
#     print("First 50 tokens:")
#     for i in range(min(50, len(input_ids[0]))):
#         token = input_ids[0][i].item()
#         label = labels[0][i].item()
#         decoded_token = tokenizer.decode([token])
#         label_status = "SUPERVISED" if label != -100 else "IGNORED"
#         print(f"  {i:2d}: token_id={token:5d} label={label:4d} text='{decoded_token}' [{label_status}]")
    
#     # Show last 50 tokens with their labels
#     print("\nLast 50 tokens:")
#     start_idx = max(0, len(input_ids[0]) - 50)
#     for i in range(start_idx, len(input_ids[0])):
#         token = input_ids[0][i].item()
#         label = labels[0][i].item()
#         decoded_token = tokenizer.decode([token])
#         label_status = "SUPERVISED" if label != -100 else "IGNORED"
#         print(f"  {i:2d}: token_id={token:5d} label={label:4d} text='{decoded_token}' [{label_status}]")
    
#     # Count supervised tokens (not -100)
#     supervised_mask = labels != -100
#     supervised_count = supervised_mask.sum().item()
#     total_tokens = labels.numel()
    
#     print(f"\nTotal tokens in batch: {total_tokens}")
#     print(f"Supervised tokens (not -100): {supervised_count}")
#     print(f"Supervised token percentage: {supervised_count/total_tokens*100:.2f}%")
    
#     if supervised_count == 0:
#         print("WARNING: No supervised tokens found! This will cause training to fail.")
#     elif supervised_count == 1:
#         print("SUCCESS: Found exactly 1 supervised token as expected.")
#     else:
#         print(f"INFO: Found {supervised_count} supervised tokens.")
# else:
#     print("WARNING: No 'labels' key found in batch!")
#     print(f"Available keys: {list(batch.keys())}")

# print("=== END DEBUGGING ===\n")

# Train the model  
trainer.train()

# TODO: check this? idk what it does
# metrics = trainer.evaluate()
# wandb.log(metrics)

# Log final model path
wandb.log({"final_model_path": CHECKPOINT_DIR}) # NAME HERE

# Finish wandb run
wandb.finish()