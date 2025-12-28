### Compute the loss of each training sample given a model and a dataset
### Used for Wikipedia Model
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, GPT2Config
import random
from torch.cuda.amp import autocast
from datasets import load_from_disk
import gc
from transformers import DataCollatorWithPadding



SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
print("flash:", torch.backends.cuda.flash_sdp_enabled())
print("mem_efficient:", torch.backends.cuda.mem_efficient_sdp_enabled())
print("math:", torch.backends.cuda.math_sdp_enabled())


def collect_per_sample_losses(model, dataset, tokenizer,device='cuda',  batch_size=4, num_workers=1):
    model.eval()
    model.to(device)
    collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt",
)

    dataloader = DataLoader(
    dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, pin_memory=True, num_workers=num_workers, persistent_workers=True, prefetch_factor=2
)
    num_samples = len(dataset)
    print(f'Number of samples: {num_samples}')
    all_sample_losses = []
    processed_samples = 0
    with torch.no_grad():

        for batch_idx, batch in tqdm(enumerate(dataloader),total=len(dataloader), desc="Calculating Losses"):
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            with autocast(dtype=torch.float16):
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

            shift_logits = logits[..., :-1, :]
            shift_labels = input_ids[..., 1:]
            shift_attention = attention_mask[..., 1:]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            per_token_loss = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
            masked_loss = per_token_loss * shift_attention.float()
            valid_tokens = shift_attention.sum(dim=1).clamp(min=1)
            sample_losses = masked_loss.sum(dim=1) / valid_tokens
            
            all_sample_losses.extend(sample_losses.cpu().tolist())
            processed_samples += len(sample_losses)
            del input_ids, attention_mask, outputs, logits, per_token_loss, sample_losses, log_probs
                
    return np.array(all_sample_losses, dtype=np.float32)


def main(args):
    model_name = args.model_name
    save_name = model_name
    mode = args.mode
    batch_size = args.batch_size
    num_workers = args.num_workers
    source = args.source
    steps = args.steps
    print("Loading model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    if steps is not None:
        model_path = os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{model_name}/{mode}/{steps}')
    else:
        model_path = os.path.join(os.path.dirname(__file__), f'out') if mode == "original" else os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}')

    if args.natural_gradient:
        model_path = os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{model_name}/natural_gradient/{mode}/{steps}')
    config = GPT2Config.from_pretrained("gpt2-medium")
    config.attn_implementation = "sdpa"
    model = GPT2LMHeadModel(config)
    if mode == "original":
        model = GPT2LMHeadModel.from_pretrained("out/gpt2-medium-restarted")
    else:
        ckpt = torch.load(os.path.join(model_path, f'ckpt.pt'), weights_only=False)
        model.load_state_dict(ckpt[f'{mode}_model'])
        print(f"Model: {model_name}")
        print(f"Mode: {mode}")
        print(f"Prompt: {ckpt['prompt']}")
        print(f"EWC Lambda: {ckpt['ewc_lambda']}")
        print(f"Ascent Steps: {ckpt.get('ascent_steps', None)}")
        print(f"Descent Steps: {ckpt.get('descent_steps', None)}")
        print(f"Learning Rate: {ckpt['learning_rate']}")
        print(f"Output Text: {ckpt['output_text']}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('model loaded')

    dataset = load_from_disk(os.path.join(os.path.dirname(__file__), f'../data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split'))
    train_dataset = dataset['train']
    columns_to_keep = ["input_ids", "attention_mask"]
    train_dataset = train_dataset.remove_columns([c for c in train_dataset.column_names if c not in columns_to_keep])
    print("Dataset loaded")
    print("Collecting losses...")
    losses_af = collect_per_sample_losses(model, train_dataset, 
                                          tokenizer,
                                          device=device,
                                          batch_size=batch_size,
                                          num_workers=num_workers)
    print("Losses collected with mean loss: ", losses_af.mean().item())
    print("Saving losses...")
    steps = ckpt.get('ascent_steps', 10) if mode == "ascent" else ckpt.get('descent_steps', 10)

    if args.natural_gradient:
        os.makedirs(os.path.join(os.path.dirname(__file__), f'data/losses/{source}/{save_name}/natural_gradient/{mode}/{steps}'), exist_ok=True)
        np.save(os.path.join(os.path.dirname(__file__), f'data/losses/{source}/{save_name}/natural_gradient/{mode}/{steps}/losses_{save_name}_{mode}.npy'), losses_af)
        return
    os.makedirs(os.path.join(os.path.dirname(__file__), f'data/losses/{source}/{save_name}/{mode}/{steps}'), exist_ok=True)
    np.save(os.path.join(os.path.dirname(__file__), f'data/losses/{source}/{save_name}/{mode}/{steps}/losses_{save_name}_{mode}.npy'), losses_af)

    print("Losses saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='gpt2-medium-restarted')
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--source', type=str, default='gutenberg')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--natural_gradient', type=bool, default=False)
    args = parser.parse_args()
    main(args)