import torch
import numpy as np
from datasets import load_from_disk
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from sample_generation import finetuning_ewc_loss, finetuning_natural_gradient_step
import os
import argparse


def main(args):
    
    
    
    ascent_steps = args.ascent_steps
    descent_steps = args.descent_steps
    natural_gradient = args.natural_gradient
    source = args.source
    type_ = args.type_
    fisher_info = torch.load(os.path.join(os.path.dirname(__file__), 'fisher_diag_gpt_medium_restarted_normalized.pt'))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), 'out/gpt2-medium-restarted'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    prompts = []
    ckpts = os.listdir(os.path.join(os.path.dirname(__file__), "out", "optimized_models",source))
    if args.ckpts is not None:
        ckpts = args.ckpts
    print(f"Number of ckpts: {len(ckpts)}")
    
    for ckpt in ckpts:    
        ckpt_ = torch.load(os.path.join(os.path.dirname(__file__), "out", "optimized_models",source, f"{ckpt}/ascent/100/ckpt.pt"))
        prompt = ckpt_["prompt"]
        source = ckpt_["source"]
        output_text = ckpt_["output_text"]
        output_ids = ckpt_["output_ids"]
        generation_length = ckpt_["generation_length"]
        temperature = ckpt_["temperature"]
        top_k = ckpt_["top_k"]
        top_p = ckpt_["top_p"]
        repetition_penalty = ckpt_["repetition_penalty"]
        ewc_lambda = ckpt_["ewc_lambda"]
        learning_rate = ckpt_["learning_rate"]
        author = ckpt_.get("author", None)  
        title = ckpt_.get("title", None)
        

        print(f"Computing for {prompt} with {type_}")
        print(f"Original output: {output_text}")
        if type_ == "descent":
            if natural_gradient: 
                descent_model = finetuning_natural_gradient_step(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), num_steps=descent_steps, unlearning_parameter=-1, learning_rate=learning_rate, dataset_size=726693)
            else:
                descent_model = finetuning_ewc_loss(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), finetuning_steps=descent_steps, ewc_lambda=ewc_lambda, learning_rate=learning_rate, unlearning_parameter=1)
        elif type_ == "ascent":
            if natural_gradient:
                ascent_model = finetuning_natural_gradient_step(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), num_steps=ascent_steps, unlearning_parameter=1, learning_rate=learning_rate, dataset_size=726693)
            else:
                ascent_model = finetuning_ewc_loss(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), finetuning_steps=ascent_steps, ewc_lambda=ewc_lambda, learning_rate=learning_rate, unlearning_parameter=-1)
        else:
            if natural_gradient:
                descent_model = finetuning_natural_gradient_step(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), num_steps=descent_steps, unlearning_parameter=-1, learning_rate=learning_rate, dataset_size=726693)
            else:
                descent_model = finetuning_ewc_loss(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), finetuning_steps=descent_steps, ewc_lambda=ewc_lambda, learning_rate=learning_rate, unlearning_parameter=1)
            model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), 'out/gpt2-medium-restarted'))
            model.to(device)
            print("Performing ascent")
            if natural_gradient:
                ascent_model = finetuning_natural_gradient_step(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), num_steps=ascent_steps, unlearning_parameter=1, learning_rate=learning_rate, dataset_size=726693)
            else:
                ascent_model = finetuning_ewc_loss(model, output_ids, fisher_info, prompt_length=len(tokenizer.encode(prompt)), finetuning_steps=ascent_steps, ewc_lambda=ewc_lambda, learning_rate=learning_rate, unlearning_parameter=-1)
        
        
        print("Saving models")
        if type_ == "descent":
            ckpt = {
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
                "natural_gradient": natural_gradient
            }

            if natural_gradient: 
                os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/descent/{descent_steps}'), exist_ok=True)
                torch.save(ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/descent/{descent_steps}/ckpt.pt'))
                print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/descent/{descent_steps}/ckpt.pt')}")
                continue
            os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/descent/{descent_steps}'), exist_ok=True)
            torch.save(ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/descent/{descent_steps}/{ckpt}'))
            print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/descent/{descent_steps}/ckpt.pt')}")
        elif type_ == "ascent":
            ckpt = {
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
                "title": title 
            }
            if natural_gradient: 
                os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/ascent/{ascent_steps}'), exist_ok=True)
                torch.save(ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/ascent/{ascent_steps}/ckpt.pt'))
                print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/ascent/{ascent_steps}/ckpt.pt')}")
                continue
            os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/ascent/{ascent_steps}'), exist_ok=True)
            torch.save(ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/ascent/{ascent_steps}/{ckpt}'))
            print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/ascent/{ascent_steps}/ckpt.pt')}")
        else:
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
                "title": title 
            }
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
                "title": title 
            }
            if natural_gradient: 
                os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/ascent/{ascent_steps}'), exist_ok=True)
                os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/descent/{descent_steps}'), exist_ok=True)
                torch.save(ascent_ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/ascent/{ascent_steps}/ckpt.pt'))
                torch.save(descent_ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/descent/{descent_steps}/ckpt.pt'))
                print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/ascent/{ascent_steps}/ckpt.pt')}")
                print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/natural_gradient/descent/{descent_steps}/ckpt.pt')}")
                continue
            os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/'), exist_ok=True)
            os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/ascent/{ascent_steps}'), exist_ok=True)
            os.makedirs(os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/descent/{descent_steps}'), exist_ok=True)
            torch.save(ascent_ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/ascent/{ascent_steps}/ckpt.pt'))
            torch.save(descent_ckpt, os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/descent/{descent_steps}/ckpt.pt'))
            print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/ascent/{ascent_steps}/ckpt.pt')}")
            print(f"Saved models to {os.path.join(os.path.dirname(__file__), f'out/optimized_models/{source}/{ckpt}/descent/{descent_steps}/ckpt.pt')}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--type_", type=str, required=True, help="Type of ablation: 'descent', 'ascent', 'both'")
    parser.add_argument("--source", type=str, default="Self-Written")
    parser.add_argument("--ascent_steps", type=int, required=True)
    parser.add_argument("--descent_steps", type=int, required=True)
    parser.add_argument("--natural_gradient", type=bool, default=False)
    parser.add_argument("--ckpts", type=str, nargs='+', default=None)
    args = parser.parse_args()
    main(args)