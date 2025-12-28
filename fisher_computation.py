import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config
import os
from torch.nn import functional as F
from collections import defaultdict
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_from_disk
import numpy as np
import argparse

def compute_fisher_diagonal(model, dataset, tokenizer, device='cuda', batch_size=8, 
                                       start_index=0, end_index=0):
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    
    print("Starting fresh Fisher diagonal computation")
    params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    fisher_diag = {name: torch.zeros_like(p, device=device) for name, p in params}    
    num_grads = 0
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator, pin_memory=True)
    
    for i, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Fisher Diagonal (Incremental)"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        tokens = (attention_mask[:, 1:]).sum()   
        
        loss = outputs.loss * tokens

        loss.backward()
        
        with torch.no_grad():
            for name, param in params:
                if param.grad is not None:
                    fisher_diag[name] +=((param.grad.detach().square()))
                    
        torch.cuda.empty_cache()
        del input_ids, attention_mask, labels, outputs, loss
        num_grads += 1
        model.zero_grad()
        if num_grads % 10000 == 0 and num_grads > 0:
            print(f"Processed {num_grads} batches, total number of samples: {num_grads}")
            fisher_normalized = {}
            for name in fisher_diag:
                fisher_normalized[name] = fisher_diag[name]
            
            torch.save(fisher_normalized, os.path.join(os.path.dirname(__file__), f"fisher_diag_intermediate_samples_{start_index}_{end_index}.pt"))

    
    fisher_ckpt = {
        "fisher_diag": fisher_diag,
        "start_index": start_index,
        "end_index": end_index,
        "num_grads": num_grads,
    }
    torch.save(fisher_ckpt, os.path.join(os.path.dirname(__file__), f"fisher_diag_samples_{start_index}_{end_index}.pt"))
    print(f"Final Fisher diagonal computed with {num_grads} total samples")
    return fisher_ckpt, num_grads


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()
    print("Starting Fisher diagonal computation")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("out/gpt2-medium-restarted")

    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print('model loaded')

    data_dir = os.path.join(os.path.dirname(__file__), '../data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split')
    train_dataset = load_from_disk(os.path.join(data_dir, 'train'))
    columns_to_keep = ["input_ids", "attention_mask"]
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in columns_to_keep])
    print(f"training column names: {train_dataset.column_names}")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    train_dataset = train_dataset.select(range(args.start_index, args.end_index))
    print(f"Training dataset loaded with {len(train_dataset)} samples")
    print('Training dataset loaded')
    print("Calculating Fisher diagonal...")
    fisher_diag, i = compute_fisher_diagonal(
        model, 
        train_dataset, 
        tokenizer, 
        device=device, 
        batch_size=args.batch_size,
        start_index=args.start_index,
        end_index=args.end_index,
        
    )
    print('Finished computing Fisher diagonal')
