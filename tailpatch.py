import os
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
import numpy as np
import math
import gc
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import random
from copy import deepcopy
import pandas as pd
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pad_token_id = 50256
def collate_fn(batch):
    input_ids = [sample['input_ids'] for sample in batch]
    attention_mask = [sample['attention_mask'] for sample in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': input_ids.clone()  
    }

def tailpatch(samples,output_ids, prompt_length, model,lr=5e-5):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_ids = output_ids.to(device)

    model.to(device)
    model.eval()
    optimizer = AdamW(model.parameters(), lr=lr)
    with torch.no_grad():
        model_pass = model(output_ids, labels=output_ids)

    logits = model_pass.logits
    logits = logits[:, prompt_length-1:-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = output_ids[:, prompt_length:]
    token_probs = log_probs.gather(2, target_ids.unsqueeze(-1))
    token_probs = token_probs.squeeze(-1)
    original_log_probability_sum = torch.sum(token_probs)

    batch_size = 1
    dataloader = DataLoader(samples, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.train()
    optimizer.zero_grad()
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        del outputs, input_ids, attention_mask
        gc.collect()
        torch.cuda.empty_cache()
    optimizer.step()
    optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        model_pass = model(output_ids, labels=output_ids)

    logits = model_pass.logits
    logits = logits[:, prompt_length-1:-1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    target_ids = output_ids[:, prompt_length:]
    token_probs = log_probs.gather(2, target_ids.unsqueeze(-1))
    token_probs = token_probs.squeeze(-1)
    log_probability_sum = torch.sum(token_probs)
    return log_probability_sum.item(), original_log_probability_sum.item(), torch.exp(log_probability_sum).item(), torch.exp(original_log_probability_sum).item()




def main(method, num_samples, source, lr, ascent_steps, descent_steps, save_dir, name):
    train_dataset = load_from_disk(os.path.join(os.path.dirname(__file__), "../data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split/train"))
    print("Loaded train dataset")
    print(len(train_dataset))
    model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), "out/gpt2-medium-restarted"))
    base_model_dict = deepcopy(model.state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    if method == "bm25":
        names = os.listdir(os.path.join(os.path.dirname(__file__), f"data/bm25/{source}/"))
        names = [name for name in names if name.endswith('.npy')]
        names = [name.replace('.npy', '') for name in names]
    elif method == "gecko":
        names = os.listdir(os.path.join(os.path.dirname(__file__), f"data/gecko/sample_scores/{source}/"))
        names = [name for name in names if name.endswith('.npy')]
        names = [name.replace('.npy', '') for name in names]
    elif method == "trackstar":
        names = os.listdir(os.path.join(os.path.dirname(__file__), f"data/trackstar/scores/{source}/"))
        names = [name for name in names if name.endswith('.npy')]
        names = [name.replace('.npy', '') for name in names]
    else:
        names = os.listdir(os.path.join(os.path.dirname(__file__), f"data/losses/{source}/"))
    
    results = []
    if name is not None:
        names = name
    print(names)
    for name in names:

        if name == "original":
            continue
        ascent_path = os.path.join(os.path.dirname(__file__),
                                       f"data/losses/{source}/{name}/ascent/{ascent_steps}/losses_{name}_ascent.npy")
        descent_path = os.path.join(os.path.dirname(__file__),
                                    f"data/losses/{source}/{name}/descent/{descent_steps}/losses_{name}_descent.npy")
        descent_ckpt_path = os.path.join(os.path.dirname(__file__), f"out/optimized_models/{source}/{name}/descent/{descent_steps}/ckpt.pt")
        ascent_ckpt_path = os.path.join(os.path.dirname(__file__), f"out/optimized_models/{source}/{name}/ascent/{ascent_steps}/ckpt.pt")
        if not os.path.exists(ascent_ckpt_path) or not os.path.exists(descent_ckpt_path) or not os.path.exists(ascent_path) or not os.path.exists(descent_path):
            print(f"[WARN] Missing dabgo ckpt files for {name}, skipping.")
            continue
        

        ckpt = torch.load(os.path.join(os.path.dirname(__file__), f"out/optimized_models/{source}/{name}/ascent/{ascent_steps}/ckpt.pt"), map_location='cpu', weights_only=False)
        model.load_state_dict(base_model_dict)  
        output_ids = ckpt['output_ids']
        prompt_length = len(tokenizer.encode(ckpt['prompt'], add_special_tokens=False))
        print("Prompt length: ", prompt_length)
        print("Prompt: ", ckpt['prompt'])
        
        if method == "dabgo":
            ascent_losses = np.load(ascent_path)
            descent_losses = np.load(descent_path)
            losses_diff = np.abs(ascent_losses - descent_losses)
            sorted_indices = np.argsort(losses_diff)[::-1]
        
        if method =="ascent":
            losses_ascent = np.load(ascent_path)
            losses_original = np.load(os.path.join(os.path.dirname(__file__), f"data/losses/original/base/losses_original_base.npy"))
            losses_diff = losses_original - losses_ascent
            sorted_indices = np.argsort(losses_diff)

        if method == "descent":
            losses_descent = np.load(descent_path)
            losses_original = np.load(os.path.join(os.path.dirname(__file__), f"data/losses/original/base/losses_original_base.npy"))
            losses_diff = losses_original - losses_descent
            sorted_indices = np.argsort(losses_diff)

        if method == "bm25":
            bm25_results = np.load(os.path.join(os.path.dirname(__file__), f"data/bm25/{source}/{name}.npy"))
            sorted_indices = np.argsort(bm25_results)[::-1]
            
        if method == "gecko":
            gecko_results = np.load(os.path.join(os.path.dirname(__file__), f"data/gecko/sample_scores/{source}/{name}.npy"))
            sorted_indices = np.argsort(gecko_results)[::-1]
            
        if method == "random":
            random_indices = np.random.choice(len(train_dataset), num_samples, replace=False)
            sorted_indices = random_indices
        if method == "gecko_wrong":
            gecko_results = np.load(os.path.join(os.path.dirname(__file__), f"gecko/gecko_wrong/sorted_scores/{source}/{name}.npy"))
            sorted_indices = gecko_results[:num_samples]
        if method == "trackstar":
            trackstar_results = np.load(os.path.join(os.path.dirname(__file__), f"data/trackstar/scores/{source}/{name}.npy"))
            sorted_indices = np.argsort(trackstar_results)[::-1]
        if method == "original":
            continue
        if method == "random_strings":
            samples = []
            # noisy samples that fill the context length
            context_length = model.config.n_positions  # typically 1024 for GPT-2
            for _ in range(num_samples):
                random_ids = torch.randint(
                    low=0,
                    high=tokenizer.vocab_size,
                    size=(context_length,),
                    dtype=torch.long,
                )
                sample = {
                    'input_ids': random_ids,
                    'attention_mask': torch.ones_like(random_ids),
                    'labels': random_ids.clone()
                }
                samples.append(sample)
            p, original, p_exp, original_exp = tailpatch(samples, output_ids, prompt_length, model, lr=lr)
            print(f"Name: {name}, Method: {method}, Num samples: {num_samples}")
            print("Probability: ", p)
            print("Original probability: ", original)
            print("Losses difference: ", np.abs(p - original))
            results.append({
                'method': method,
                'source': source,
                'name': name,
                'log_probability': p,
                'original_log_probability': original,
                'probability': p_exp,
                'original_probability': original_exp,
                'losses_difference': np.abs(p - original),
                "num_samples": num_samples,
                "lr": lr,
                "text_tokens":  len(output_ids[0]),
                "prompt_tokens": prompt_length
            })
            continue
        samples = []
        for i in range(num_samples):
            sample = {
                'input_ids': torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids']),
                'attention_mask': torch.ones_like(torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids'])),
                'labels': torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids'])
            }
            samples.append(sample)
        p,original, p_exp, original_exp = tailpatch(samples, output_ids, prompt_length, model, lr=lr)
        print(f"Name: {name}, Method: {method}, Num samples: {num_samples}")
        print("Probability: ", p)
        print("Original probability: ", original)
        print("Losses difference: ", np.abs(p - original))
        results.append({
            'method': method,
            'source': source,
            'name': name,
            'log_probability': p,
            'original_log_probability': original,
            'probability': p_exp,
            'original_probability': original_exp,
            'losses_difference': np.abs(p - original),
            "num_samples": num_samples,
            "lr": lr,
            "text_tokens":  len(output_ids[0]),
            "prompt_tokens": prompt_length
        })

    df = pd.DataFrame(results)
    os.makedirs(os.path.join(os.path.dirname(__file__), f"data/{save_dir}/{source}/{method}/{num_samples}"), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(__file__), f"data/{save_dir}/{source}/{method}/{num_samples}/results.csv"), index=False)
    print("Results saved to ", os.path.join(os.path.dirname(__file__), f"data/{save_dir}/{source}/{method}/{num_samples}/results.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["dabgo", "bm25", "trackstar", "random", "ascent", "random_strings", "descent", "gecko", "original", "gecko_wrong"], required=True)
    parser.add_argument("--source", type=str, choices=["gutenberg", "wikipedia", "fineweb", "Self-Written"], required=True)
    parser.add_argument("--sample_names", type=str, nargs='+', default=None, help="Names of the samples to process")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ascent_steps", type=int, default=10)
    parser.add_argument("--descent_steps", type=int, default=10)
    parser.add_argument("--save_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.method, args.num_samples, args.source, args.lr, args.ascent_steps, args.descent_steps, args.save_dir, args.sample_names)
    