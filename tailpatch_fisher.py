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
import wandb


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
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
def tailpatch(samples, output_ids, prompt_length, model, fisher_diag, ewc_lambda, 
            learning_rate: float = 5e-5, 
            max_steps: int = 100, patience: int =20, tol: float = 1e-2, return_diagnostics: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_ids = output_ids.to(device)
    model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = True
        if "wte.weight" in name:
            param.requires_grad = False
    

    # Snapshot original params θ* (on device, no grads)
    original_params = {name: param.detach().clone() for name, param in model.named_parameters()}

    model.to(device)
    model.eval()
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
    weight_decay = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ewc_losses = []
    total_losses = []
    data_losses = []
    best_loss = float('inf')
    steps_without_improvement = 0
    step = 0
    # --- W&B logging for this step ---
    if wandb.run is not None:
        wandb.log(
            {
                "step": step,
                "total_loss": 0,
                "ewc_loss": 0,
                "data_loss": 0,
                "logprob_sum": original_log_probability_sum.item() * -1,
                "Difference_Absolute": 0.0,
                "Difference_Relative": 0.0,
            },
            step=step,
        )
    while step < max_steps and steps_without_improvement < patience:
        optimizer.zero_grad()

        total_loss = torch.tensor(0.0, device=device)
        ewc_loss_sum = 0.0
        data_loss_sum = 0.0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss

            # ---- EWC term using Fisher diagonal ----
            ewc_loss = 0.0
            for name, param in model.named_parameters():
                if "wte.weight" in name:
                    continue
                if name not in fisher_diag:
                    continue

                fisher = fisher_diag[name].to(device)
                param_diff = param - original_params[name]
                penalty = fisher * param_diff.pow(2)
                ewc_loss = ewc_loss + penalty.sum()
            batch_loss = (loss + ewc_lambda * ewc_loss) / len(samples)
            batch_loss.backward()
            total_loss += batch_loss.item()
            ewc_loss_sum += ewc_loss.item()/len(samples)
            data_loss_sum += loss.item()/len(samples)
            
            del outputs, input_ids, attention_mask, labels, batch_loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ewc_losses.append(ewc_loss_sum)
        total_losses.append(total_loss.item())
        data_losses.append(data_loss_sum)

        #print(f"Step {step}: Loss: {total_loss.item()}, EWC Loss: {ewc_loss_sum}, Data Loss: {data_loss_sum}")
        #print('--------------------------------')

        # --- compute log-probability at this step (like at the end) ---
        model.eval()
        with torch.no_grad():
            step_pass = model(output_ids, labels=output_ids)
        step_logits = step_pass.logits
        step_logits = step_logits[:, prompt_length-1:-1, :]
        step_log_probs = torch.log_softmax(step_logits, dim=-1)
        step_target_ids = output_ids[:, prompt_length:]
        step_token_probs = step_log_probs.gather(2, step_target_ids.unsqueeze(-1)).squeeze(-1)
        step_logprob_sum = torch.sum(step_token_probs).item() * -1
        model.train()

        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = total_loss
        if best_loss - current_loss > tol:
            best_loss = current_loss
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
        step += 1
        if wandb.run is not None:
            wandb.log(
                {
                    "step": step,
                    "total_loss": total_loss.item(),
                    "ewc_loss": ewc_loss_sum,
                    "data_loss": data_loss_sum,
                    "logprob_sum": step_logprob_sum,
                    "Difference_Absolute": np.abs(step_logprob_sum - (original_log_probability_sum.item()*-1)),
                    "Difference_Relative":  (original_log_probability_sum.item()*-1) - step_logprob_sum,
                },
                step=step,
            )

        
        if step >= max_steps or steps_without_improvement >= patience:
            break

    
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
    if return_diagnostics:
        return log_probability_sum.item(), original_log_probability_sum.item(), ewc_losses, total_losses, data_losses
    return log_probability_sum.item(), original_log_probability_sum.item()




def main(method, num_samples, source, ewc_lambda, steps, learning_rate, use_wandb, tailpatch_steps, patience, tol, return_diagnostics):
    train_dataset = load_from_disk(os.path.join(os.path.dirname(__file__), "../data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split/train"))
    print("Loaded train dataset")
    print(len(train_dataset))
    model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), "out/gpt2-medium-restarted"))
    base_model_dict = deepcopy(model.state_dict())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    fisher_info = torch.load(os.path.join(os.path.dirname(__file__), f"data/fisher_diag/fisher_diag_gpt_medium_restarted.pt"), weights_only=False)
    
    fisher_info = {k: v.to(device) for k, v in fisher_info.items()}
    names = os.listdir(os.path.join(os.path.dirname(__file__), f"data/losses/{source}/"))
    print(names)
    results = []

    for name in names:

        if name == "original":
            continue
        ascent_path = os.path.join(os.path.dirname(__file__),
                                       f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy")
        descent_path = os.path.join(os.path.dirname(__file__),
                                    f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")

        if not os.path.exists(ascent_path) or not os.path.exists(descent_path):
            print(f"[WARN] Missing dabgo loss files for {name}, skipping.")
            continue

        ckpt = torch.load(os.path.join(os.path.dirname(__file__), f"out/optimized_models/{source}/{name}/ascent/100/ckpt.pt"), map_location='cpu', weights_only=False)
        model.load_state_dict(base_model_dict)  
        output_ids = ckpt['output_ids']
        prompt_length = len(tokenizer.encode(ckpt['prompt'], add_special_tokens=False))
        print("Prompt length: ", prompt_length)
        print("Prompt: ", ckpt['prompt'])
        if use_wandb:
            os.environ['WANDB_MODE'] = "offline"
            wandb.init(
                project="tailpatch_fisher",           # same project for all runs
                name=f"{method}_{source}_{name}",     # per-run name
                config={
                    "method": method,
                    "source": source,
                    "name": name,
                    "num_samples": num_samples,
                    "lr": learning_rate,
                    "tailpatch_steps": tailpatch_steps,
                    "ascent_descent_steps": steps,
                    "patience": patience,
                    "tol": tol,
                    "return_diagnostics": return_diagnostics,
                }
            )

        if method == "dabgo":
            ascent_losses = np.load(os.path.join(os.path.dirname(__file__), f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy"))
            descent_losses = np.load(os.path.join(os.path.dirname(__file__), f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy"))
            losses_diff = np.abs(ascent_losses - descent_losses)
            sorted_indices = np.argsort(losses_diff)[::-1]
        
        if method =="ascent":
            losses_ascent = np.load(os.path.join(os.path.dirname(__file__), f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy"))
            losses_original = np.load(os.path.join(os.path.dirname(__file__), f"data/losses/original/base/losses_original_base.npy"))
            losses_diff = losses_original - losses_ascent
            sorted_indices = np.argsort(losses_diff)

        if method == "descent":
            losses_descent = np.load(os.path.join(os.path.dirname(__file__), f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy"))
            losses_original = np.load(os.path.join(os.path.dirname(__file__), f"data/losses/original/base/losses_original_base.npy"))
            losses_diff = losses_descent - losses_original
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
            p,original= tailpatch(samples, output_ids, prompt_length, model, fisher_info, ewc_lambda, learning_rate, tailpatch_steps, patience, tol=1e-5, return_diagnostics=False)
            if use_wandb:
                wandb.finish()
            print(f"Name: {name}, Method: {method}, Num samples: {num_samples}")
            print("Probability: ", p)
            print("Original probability: ", original)
            print("Losses difference: ", np.abs(p - original))
            results.append({
                'method': method,
                'source': source,
                'name': name,
                'probability': p,
                'original_probability': original,
                'losses_difference': np.abs(p - original),
                "num_samples": num_samples,
                "steps": steps,
                "tailpatch_steps": tailpatch_steps,
                "patience": patience,
                "lr": learning_rate,
                "text_tokens": len(output_ids[0]),
                "prompt_tokens": prompt_length,
            })
            continue
            
        if method == "trackstar":
            continue
        if method == "original":
            continue
        samples = []
        for i in range(num_samples):
            sample = {
                'input_ids': torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids']),
                'attention_mask': torch.ones_like(torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids'])),
                'labels': torch.tensor(train_dataset[int(sorted_indices[i])]['input_ids'])
            }
            samples.append(sample)
        
        p,original= tailpatch(samples, output_ids, prompt_length, model, fisher_info, ewc_lambda, learning_rate, tailpatch_steps, patience, tol=1e-5, return_diagnostics=False)
        if use_wandb:
            wandb.finish()
        print(f"Name: {name}, Method: {method}, Num samples: {num_samples}")
        print("Probability: ", p)
        print("Original probability: ", original)
        print("Losses difference: ", np.abs(p - original))
        results.append({
            'method': method,
            'source': source,
            'name': name,
            'probability': p,
            'original_probability': original,
            'losses_difference': np.abs(p - original),
            "num_samples": num_samples,
            "steps": steps,
            "tailpatch_steps": tailpatch_steps,
            "patience": patience,
            "lr": learning_rate,
            "text_tokens": len(output_ids[0]),
            "prompt_tokens": prompt_length,
        })
    df = pd.DataFrame(results)
    os.makedirs(os.path.join(os.path.dirname(__file__), f"data/results_fisher/{source}/{method}/{num_samples}"), exist_ok=True)
    df.to_csv(os.path.join(os.path.dirname(__file__), f"data/results_fisher/{source}/{method}/{num_samples}/results.csv"), index=False)
    print("Results saved to ", os.path.join(os.path.dirname(__file__), f"data/results_fisher/{source}/{method}/{num_samples}/results.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["dabgo", "bm25", "trackstar", "random", "ascent", "descent", "gecko", "original", "random_strings"], required=True)
    parser.add_argument("--source", type=str, choices=["gutenberg", "wikipedia", "fineweb", "Self-Written"], required=True)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--ewc_lambda", type=float, default=50)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--wandb", dest="use_wandb", action="store_true")
    parser.add_argument("--tailpatch_steps", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-2)
    parser.add_argument("--return_diagnostics", action="store_true")
    args = parser.parse_args()
    main(args.method, args.num_samples, args.source, args.ewc_lambda, args.steps, args.lr, args.use_wandb, args.tailpatch_steps, args.patience, args.tol, args.return_diagnostics)
    