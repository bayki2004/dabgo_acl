import torch
import numpy as np
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import os
import argparse
import json
import random
def finetuning_ewc_loss(model, output_ids,fisher_info, prompt_length=1, finetuning_steps=10, unlearning_parameter=1, ewc_lambda=50, learning_rate=1e-4, return_diagnostics=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad = True
        if "wte.weight" in name:
            param.requires_grad = False
    weight_decay = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    real_ids = output_ids.clone()
    
    B, L = real_ids.shape
    labels = real_ids.clone()
    labels[:, :prompt_length] = -100 ## Mask out prompt tokens for loss
    labels = labels.to(device)
    real_ids = real_ids.to(device)
    attention_mask = torch.ones(B, L, dtype=torch.long).to(device)
    attention_mask = attention_mask.to(device)
    original_params = {name: param.detach().clone() for name, param in model.named_parameters()}
    ewc_losses = []
    total_losses = []
    data_losses = []
    for i in range(finetuning_steps):
        optimizer.zero_grad()

        outputs = model(input_ids=real_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  
        ewc_loss = 0.0
        for name, param in model.named_parameters():
            if "wte.weight" in name:
                continue
            if name in fisher_info:
                fisher = fisher_info[name]
                
                # Ensure fisher info is on correct device
                if fisher.device != param.device:
                    fisher = fisher.to(param.device)
                param_diff = param - original_params[name]
                penalty = fisher * param_diff.pow(2)
                ewc_loss += penalty.sum()
        
        total_loss = unlearning_parameter * loss + ewc_loss * ewc_lambda
        ewc_losses.append(ewc_loss.item())
        total_losses.append(total_loss.item())
        data_losses.append(loss.item())
        print(f"Step {i}: Loss: {total_loss}, EWC Loss: {ewc_loss}, Data Loss: {loss}")
        print('--------------------------------')
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
    if return_diagnostics:
        return model, ewc_losses, total_losses, data_losses
    return model

def finetuning_natural_gradient_step(model, output_ids, fisher_diag, prompt_length=1, num_steps=10, unlearning_parameter=1, learning_rate=1e-2, dataset_size=726693, return_diagnostics=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    real_ids = output_ids.clone()
    B, L = real_ids.shape
    labels = real_ids.clone()
    labels[:, :prompt_length] = -100 ## Mask out prompt tokens for loss
    labels = labels.to(device)
    real_ids = real_ids.to(device)
    attention_mask = torch.ones(B, L, dtype=torch.long).to(device)
    attention_mask = attention_mask.to(device)

    for i in range(num_steps):
        model.zero_grad()
        outputs = model(input_ids=real_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  
        print(f"Step {i}: Loss: {loss.item()}")
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    
                    if name in fisher_diag:
                        fisher = fisher_diag[name]
                        if fisher.device != param.device:
                            fisher = fisher.to(param.device)                        
                        epsilon = 1e-6 
                        
                        inverse_fisher = (1.0 / (fisher + epsilon)) * param.grad
                        param.add_(unlearning_parameter * learning_rate / dataset_size * inverse_fisher) 

    return model



def sentence_log_likelihood(model, output_ids, prompt_length, device):
    with torch.no_grad():
        # logits: (B, L, V)
        logits = model(output_ids).logits

        # Shift: logits[t] predicts output_ids[t+1]
        logits = logits[:, :-1, :]        # (B, L-1, V)
        targets = output_ids[:, 1:]       # (B, L-1)

        # We only care about predictions *after* the prompt.
        # The first prediction for the continuation is at index (prompt_length - 1)
        logits = logits[:, prompt_length-1:, :]   # (B, gen_len, V)
        targets = targets[:, prompt_length-1:]     # (B, gen_len)

        # log-softmax for numerical stability
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Gather log-probabilities of the actual generated tokens
        token_log_probs = torch.gather(
            log_probs,
            2,
            targets.unsqueeze(-1)
        ).squeeze(-1)   # (B, gen_len)
        log_likelihood = token_log_probs.sum(dim=1)
        nll = -log_likelihood / targets.shape[1]
        # Sum log-probs to get total conditional log-likelihood
        return log_likelihood.item(), nll.item()

def main(args):
    
    prompt = args.prompt
    num_samples = args.num_samples
    ewc_lambda = args.ewc_lambda
    learning_rate = args.learning_rate
    ascent_steps = args.ascent_steps
    descent_steps = args.descent_steps
    temperature = args.temperature
    top_k = args.top_k
    top_p = args.top_p
    repetition_penalty = args.repetition_penalty
    generation_length = args.generation_length
    source = args.source
    first_sentence = args.first_sentence
    only_generate = args.only_generate
    training_data_indices = args.training_data_indices
    natural_gradient = args.natural_gradient
    fisher_info = torch.load(os.path.join(os.path.dirname(__file__), 'fisher_diag_gpt_medium_restarted_normalized.pt'), weights_only=False)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), 'out/gpt2-medium-restarted'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    prompts = []
    save_name = None
    dataset = load_from_disk(os.path.join(os.path.dirname(__file__), '../data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split'))
    ds = dataset['test'].filter(lambda x: x['source'] == source)
    sample_indices = random.sample(range(len(ds)), num_samples)
    print(f"Sample indices: {sample_indices}")

    if training_data_indices:
        ds = dataset['train']
        ds = ds.filter(lambda x: x['source'] == source)
        print("Filtered train data")
        sample_indices = random.sample(range(len(ds)), num_samples)
        print(f"Sample indices: {sample_indices}")
        
    if prompt != "":
        save_name = prompt
        prompt = "# " + prompt + "\n" 
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        print(f"Prompt: {prompt}, encoded length: {len(input_ids[0])}")
        prompts.append(prompt)
        authors = ["Self-Written"]
        titles = [prompt]
        source = "Self-Written"
              
    else: 
        
        if source == "wikipedia":
            col = 'title'

            prompts = [ds[idx]['title'] for idx in sample_indices]
            
            print(f"Prompts: {prompts}")
  
        if source =='gutenberg':
            col = 'input_ids'
            prompts = [tokenizer.decode(ds[idx]['input_ids'], skip_special_tokens=True) for idx in sample_indices]
            prompts = [prompt.split('.')[1 if len(prompt.split('.')) > 1 else 0] for prompt in prompts]
            prompts = [prompt + "." for prompt in prompts]
            print(f"Prompts: {prompts}")
            
        authors = [ds[idx]['author'] for idx in sample_indices]
        titles = [ds[idx]['title'] for idx in sample_indices]
        print(f"Authors: {authors}")
        print(f"Titles: {titles}")
        print(f"\nOriginal Text: \n{tokenizer.decode(ds[sample_indices[0]]['input_ids'][:64], skip_special_tokens=True)}\n")
    os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}'), exist_ok=True)
    titles_computed = os.listdir(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}'))
    titles_computed = [title.split('.')[0] for title in titles_computed]
    titles_computed = [title.split('_')[0] for title in titles_computed]
    print(f"Number of titles computed: {len(titles_computed)}")
    for prompt, author, title in zip(prompts, authors, titles):
        if prompt in titles_computed or author in titles_computed:
            print(f"Prompt {prompt} or author {author} already computed")
            continue
        if source == "wikipedia":
            save_name = prompt
            prompt = "# " + prompt + "\n" 
        if source == "gutenberg":
            already_computed = len(os.listdir(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}')))
            save_name = f"{author}_{already_computed+1}"
            print(f"Save name: {save_name}")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = torch.ones(input_ids.shape[0], input_ids.shape[1], dtype=torch.long).to(device)
        print(f"Prompt: \n{prompt}\nencoded length: {len(input_ids[0])}")
        output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=generation_length, num_return_sequences=1, do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature, repetition_penalty=repetition_penalty, pad_token_id=tokenizer.eos_token_id)
        ll, nll = sentence_log_likelihood(model, output_ids, len(tokenizer.encode(prompt)), device)
        print(f"Log likelihood: {ll}, NLL: {nll}")
        if first_sentence:
            if source != "gutenberg":
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                output_text = output_text.split('.')[0]
            if source == "gutenberg":
                output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                output_text = output_text.split(".")[:2]
                output_text = ".".join(output_text)
                
            output_ids = tokenizer.encode(output_text, return_tensors="pt").to(device)
            print(f"Generated Tokens: {len(output_ids[0] - len(input_ids[0]))}\n")
            print(f"Output text: \n{output_text}\n")
        if not first_sentence:
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print(f"Generated Tokens: {len(output_ids[0] - len(input_ids[0]))}\n")
            print(f"Output text: \n{output_text}\n")
        if only_generate:
            metadata = {
                "prompt": prompt,
                "output_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
                "output_ids": output_ids,
                "generation_length": generation_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "author": author,
                "title": title,
                "source": source,
                "training_data_indices": training_data_indices
            }
            os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/'), exist_ok=True)
            torch.save(metadata, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/metadata.pt'))
            print(f"Saved metadata to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/metadata.pt')}")
            return
        print("Performing descent")
        descent_model = None
        ascent_model = None
        if not natural_gradient:
            descent_model = finetuning_ewc_loss(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), finetuning_steps=descent_steps, ewc_lambda=ewc_lambda, learning_rate=learning_rate, unlearning_parameter=1)
            
            model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), 'out/gpt2-medium-restarted'))
            model.to(device)
            print("Performing ascent")
            ascent_model = finetuning_ewc_loss(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), finetuning_steps=ascent_steps, ewc_lambda=ewc_lambda, learning_rate=learning_rate, unlearning_parameter=-1)
            print("Saving models")
        if natural_gradient:
            descent_model = finetuning_natural_gradient_step(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), num_steps=descent_steps, unlearning_parameter=-1, learning_rate=learning_rate, dataset_size=726693)
            model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), 'out/gpt2-medium-restarted'))
            model.to(device)
            print("Performing ascent")
            ascent_model = finetuning_natural_gradient_step(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), num_steps=ascent_steps, unlearning_parameter=1, learning_rate=learning_rate, dataset_size=726693)
            print("Saving models")
        metadata = {
                "prompt": prompt,
                "output_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
                "output_ids": output_ids,
                "generation_length": generation_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "author": author,
                "title": title,
                "source": source,
                "training_data_indices": training_data_indices
            }
        os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/'), exist_ok=True)
        torch.save(metadata, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/metadata.pt'))
        print(f"Saved metadata to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/metadata.pt')}")
        descent_ckpt = {
            "descent_model": descent_model.state_dict(),
            "prompt": prompt,
            "output_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "output_ids": output_ids,
            "generation_length": generation_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "ewc_lambda": ewc_lambda,
            "learning_rate": learning_rate,
            "ascent_steps": ascent_steps,
            "descent_steps": descent_steps,
            "source": source,
            "author": author, 
            "title": title,
            "first_sentence": first_sentence,
            "training_data_indices": training_data_indices,
            "natural_gradient": natural_gradient
        }
        ascent_ckpt = {
            "ascent_model": ascent_model.state_dict(),
            "prompt": prompt,
            "output_text": tokenizer.decode(output_ids[0], skip_special_tokens=True),
            "output_ids": output_ids,
            "generation_length": generation_length,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "ewc_lambda": ewc_lambda,
            "learning_rate": learning_rate,
            "ascent_steps": ascent_steps,
            "descent_steps": descent_steps,
            "source": source,
            "author": author, 
            "title": title,
            "first_sentence": first_sentence,
            "training_data_indices": training_data_indices,
            "natural_gradient": natural_gradient
        }

        os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/ascent/{ascent_steps}/'), exist_ok=True)
        os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/descent/{descent_steps}/'), exist_ok=True)
        if natural_gradient:
            os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/natural_gradient/ascent/{ascent_steps}/'), exist_ok=True)
            os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/natural_gradient/descent/{descent_steps}/'), exist_ok=True)
            torch.save(ascent_ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/natural_gradient/ascent/{ascent_steps}/ckpt.pt'))
            torch.save(descent_ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/natural_gradient/descent/{descent_steps}/ckpt.pt'))
            print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/natural_gradient/ascent/{ascent_steps}/ckpt.pt')}")
            print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/natural_gradient/descent/{descent_steps}/ckpt.pt')}")
            return
        
        torch.save(ascent_ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/ascent/{ascent_steps}/ckpt.pt'))
        torch.save(descent_ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/descent/{descent_steps}/ckpt.pt'))
        print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/ascent/{ascent_steps}/ckpt.pt')}")
        print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{save_name}/descent/{descent_steps}/ckpt.pt')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--source", type=str, default="wikipedia")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--generation_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.90)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--ewc_lambda", type=float, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--ascent_steps", type=int, default=100)
    parser.add_argument("--descent_steps", type=int, default=100)
    parser.add_argument("--first_sentence", type=bool, default=True)
    parser.add_argument("--only_generate", type=bool, default=False)
    parser.add_argument("--training_data_indices", type=bool, default=False)
    parser.add_argument("--natural_gradient", type=bool, default=False)
    args = parser.parse_args()
    main(args)