## Training Code for Pretraining on Mixture of datasets
## Dataset: Wikipedia, Gutenberg, FineWeb
## Dataset Location in data/datasets/mixture_wiki_gt_web
## Using a pretraining architecture like gpt2 small or medium depending on how long it takes to train

import os
import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any
from transformers import TrainerCallback
from datasets import load_from_disk
from pathlib import Path
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
print("flash:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math:", torch.backends.cuda.math_sdp_enabled())
import os

# If you keep compile:
# import torch._dynamo as dynamo
# dynamo.config.cache_size_limit = 16


# ---------- Paths ----------
DATA_DIR = "data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024"
SPLIT_DIR = Path(DATA_DIR) / "train_test_split"

TOKENIZER_NAME = "gpt2"              
MODEL_NAME = "gpt2"
OUTPUT_DIR = "out/gpt2-scratch"
batch_size = 32

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 1024

# ---------- Dataset ----------
if SPLIT_DIR.exists():
    print("Loading existing train/test split...")
    split_ds = load_from_disk(str(SPLIT_DIR))
    train_ds, eval_ds = split_ds["train"], split_ds["test"]
else:
    print("No split found. Creating train/test split and saving...")
    ds = load_from_disk(DATA_DIR)
    split_ds = ds.train_test_split(test_size=0.01, seed=42)
    split_ds.save_to_disk(str(SPLIT_DIR))
    train_ds, eval_ds = split_ds["train"], split_ds["test"]

print(f"Train dataset size: {len(train_ds)}")
print(f"Eval dataset size: {len(eval_ds)}")
# You only need input_ids + attention_mask for Causal LM
columns_to_keep = ["input_ids", "attention_mask"]
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in columns_to_keep])
eval_ds  = eval_ds.remove_columns([c for c in eval_ds.column_names  if c not in columns_to_keep])
print(f"training column names: {train_ds.column_names}")

config = GPT2Config.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel(config)
"""from safetensors.torch import load_file as load_safetensors
state_dict = load_safetensors(os.path.join("out/gpt2-scratch/checkpoint-2200", "model.safetensors"), device="cpu")
# strip compile wrapper prefix (only if present)
if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
    state_dict = {k.replace("_orig_mod.", "", 1): v for k, v in state_dict.items()}

missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("missing:", missing, "unexpected:", unexpected)
"""


print(f"Model Architecture: {MODEL_NAME}")
print(f"Model Parameters: {model.num_parameters()}")
# ---------- Data Collator ----------
# Copies input_ids to labels and masks pad_token with -100 automatically
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)
model.config.attn_implementation = "sdpa"

torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere/Ada; safe & faster
model = torch.compile(model, mode="default")  # or mode="default"



# ------ Training Args (UPDATED) ------
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    num_train_epochs=10,                
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=128//batch_size,
    learning_rate=3e-4,
    warmup_ratio=0.03,
    weight_decay=0.1,
    
    lr_scheduler_type="cosine_with_restarts",
    #num_cycles=2,
    logging_steps=50,                
    eval_strategy="steps", 
    save_strategy="steps",            
    eval_steps=200,                   
    save_steps=200,
    save_total_limit=5,
    load_best_model_at_end=False,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    fp16=False,
    optim="adamw_torch_fused",   
    half_precision_backend="amp",
    report_to=["wandb"],   
    run_name="gpt2-scratch-mixture-pretrain-restarted",            
    dataloader_num_workers=4,
    dataloader_pin_memory=False,
    dataloader_persistent_workers=False,
    save_on_each_node=False,
    dataloader_prefetch_factor=2,
    gradient_checkpointing=True,  
    label_names=["labels"],
    remove_unused_columns=False,
    deepspeed="ds_stage2.json",  
    seed=42,                           
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=collator,
    tokenizer=tokenizer, 
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=10, early_stopping_threshold=0.01)],
)
print("Starting training...")
trainer.train(resume_from_checkpoint=True)

trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
tokenizer.save_pretrained(OUTPUT_DIR)

# ---------- (Optional) Quick perplexity on eval ----------
import math
metrics = trainer.evaluate()
if "eval_loss" in metrics:
    print("Perplexity:", math.exp(metrics["eval_loss"]))
