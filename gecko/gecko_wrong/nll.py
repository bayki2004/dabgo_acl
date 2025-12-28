import torch
from datasets import load_from_disk
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
import numpy as np
import pandas as pd
import argparse
base_dir = os.path.join(os.path.dirname(__file__), "../../")
print(f"Base directory: {base_dir}")




device = "cuda" if torch.cuda.is_available() else "cpu"

# Load LM with head (for log-likelihoods)
tok = GPT2Tokenizer.from_pretrained("gpt2")
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lm = GPT2LMHeadModel.from_pretrained(os.path.join(base_dir, "out/gpt2-medium-restarted")).to(device).eval()

@torch.no_grad()
def query_nll_given_prefix(prefix_ids, query_text, max_len=256):
    """
    prefix_ids: list[int] (tokenized training sample)
    query_text: str (your generated query sentence)
    returns: average NLL per query token (float)
    """
    q_ids = tok.encode(query_text, add_special_tokens=False)
    # Truncate prefix if too long
    keep = max_len - len(q_ids)
    if keep <= 0:
        prefix_ids = []
    else:
        prefix_ids = prefix_ids[-keep:]
    prefix_ids = list(prefix_ids)
    ids = prefix_ids + q_ids
    labels = [-100] * len(prefix_ids) + q_ids  # mask prefix
    att = [1] * len(ids)

    input_ids = torch.tensor([ids], dtype=torch.long, device=lm.device)
    attention_mask = torch.tensor([att], dtype=torch.long, device=lm.device)
    labels = torch.tensor([labels], dtype=torch.long, device=lm.device)

    out = lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    # out.loss is mean NLL across query tokens
    return float(out.loss.item())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="wikipedia")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    source = args.source
    steps = args.steps
    examples = os.listdir(os.path.join(os.path.dirname(__file__), f'data/gecko/{source}/'))
    examples = [example.replace('.npy', '') for example in examples]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_data = load_from_disk(os.path.join(base_dir, '../data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split/train'))
    print(len(train_data))
    print("Loaded train data")
    for example in examples:
        
        ckpt = torch.load(os.path.join(base_dir, f'out/optimized_models/{source}/{example}/ascent/{steps}/ckpt.pt'), map_location='cpu', weights_only=False)
        output_ids = ckpt['output_ids']
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(output_text)
        reranked = []
        top_k = np.load(os.path.join(os.path.dirname(__file__), f'data/gecko/{source}/{example}.npy'))
        top_k = top_k[:100]
        query = output_text
        for rank, (i, sim) in enumerate(top_k, 1):
            print(f"Rank {rank}: {i} {sim}")
            prefix_ids = train_data[int(i)]['input_ids']
            
            
            nll = query_nll_given_prefix(prefix_ids, query) 
            
            reranked.append((i, sim, nll))
            
        reranked = np.array(reranked)
        os.makedirs(os.path.join(os.path.dirname(__file__), f'scores/nll/{source}'), exist_ok=True)
        np.save(os.path.join(os.path.dirname(__file__), f'scores/nll/{source}/{example}.npy'), reranked)