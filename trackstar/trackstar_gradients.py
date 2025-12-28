import torch
import os
from collections import defaultdict
import torch.nn as nn
import re
import math
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tqdm import tqdm


class TrackstarUnbatched:
    def __init__(self, model, device='cuda', eps=1e-8):
        
        self.model = model.to(device).eval()
        self.device = device
        self.eps = eps
        self.second_moment = None
        self.loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.projector = None
        self.counter = 0
        self.grouped_grads = []
        self.number_of_samples = None

    def compute_grouped_gradients_batch(self, sample, group_size=2):
        self.model.zero_grad()
        input_ids = sample['input_ids'].to(self.device)
        mask     = sample['attention_mask'].to(self.device)
        x_in = input_ids[:, :-1];  y = input_ids[:, 1:];  m = mask[:, :-1]
        out = self.model(input_ids=x_in, attention_mask=m).logits
        loss = self.loss_fn(out.view(-1, out.size(-1)), y.view(-1)) / y.numel()
        loss.backward()
        block_grads = defaultdict(list)
        for name, p in self.model.named_parameters():
            if p.grad is None or 'wte' in name: continue
            g = p.grad.detach().flatten()
            # assign to block
            
            if name.startswith('transformer.h.'):
                L = int(name.split('.')[2])
                b = L // group_size
                t = 'attn' if 'attn' in name else 'mlp'
                key = f'group{b}_{t}'
            elif name.startswith('transformer.ln_f'):
                key = 'final_ln'
            elif name.startswith('transformer.lm_head'):
                key = 'lm_head'
            else:
                continue
            block_grads[key].append(g)
        block_grads = {k: torch.cat(v.detach(), dim=0) for k, v in block_grads.items()}
        block_grads = self.projector.project_per_block(block_grads)
        
        self.grouped_grads.append(block_grads)
        self.counter += 1
        if self.counter == self.number_of_samples-1 or self.counter % 10000 == 0:
            os.makedirs(os.path.join(os.path.dirname(__file__), f'../data/trackstar/gradients'), exist_ok=True)
            torch.save(self.grouped_grads, os.path.join(os.path.dirname(__file__), f'../data/trackstar/gradients/grads_{self.counter}.pt'))
            print('saved', len(self.grouped_grads), 'grouped gradients')
            self.grouped_grads = []
            torch.cuda.empty_cache()
            print('empty cache')
        return self.grouped_grads  

    @staticmethod
    def compute_R_inv_sqrt(block_projs, eps=1e-12):
        R_inv_sqrt = {}
        for k, Phi in block_projs.items():
            chunks = []
            for filename in Phi:
                chunks.append(torch.load(filename, weights_only=True))
            Phi = torch.cat(chunks, dim=0)
            print(Phi.shape)
            
            R = Phi.T @ Phi  # [d,d]
            w, V = torch.linalg.eigh(R)
            w_inv_sqrt = torch.clamp(w, min=eps).rsqrt()
            R_inv_sqrt[k] = (V * w_inv_sqrt.unsqueeze(0)) @ V.T
        return R_inv_sqrt


def compute_block_shapes(grouped_second_moment, embedding_dim):
    block_shapes = {}
    for key, vec in grouped_second_moment.items():
        total_dim = vec.numel()
        n = embedding_dim
        if total_dim % n != 0:
            raise ValueError(f"Dimension mismatch in {key}: total {total_dim} not divisible by embedding_dim {n}")
        m = total_dim // n

        block_shapes[key] = (m, n)
    return block_shapes

class BlockProjector:
    def __init__(self, block_shapes, d=4096, device='cuda'):
        self.d = d
        self.sqrt_d = int(math.sqrt(d))
        self.device = device
        self.proj_matrices = {}
        torch.manual_seed(0)
        for key, (m, n) in block_shapes.items():
            P0 = torch.randn(self.sqrt_d, m, device=device) / math.sqrt(self.sqrt_d)
            P1 = torch.randn(self.sqrt_d, n, device=device) / math.sqrt(self.sqrt_d)
            self.proj_matrices[key] = (P0, P1)
            
        
    def project_per_block(self, block_grads):
        out = {}
        for key, vec in block_grads.items():
            P0, P1 = self.proj_matrices[key]
            m, n = P0.shape[1], P1.shape[1]
            W = vec.view(m, n)
            out[key] = (P0 @ W @ P1.T).flatten()  
        return out


def get_block_shapes_from_model(model, group_size=2):
    block_dims = defaultdict(int)
    for name, p in model.named_parameters():
        if 'wte' in name: continue
        
        if name.startswith('transformer.h.'):
            L = int(name.split('.')[2])
            b = L // group_size
            t = 'attn' if 'attn' in name else 'mlp'
            key = f'group{b}_{t}'
        elif name.startswith('transformer.ln_f'):
            key = 'final_ln'
        elif name.startswith('transformer.lm_head'):
            key = 'lm_head'
        else: continue
        
        block_dims[key] += p.numel()

    n = model.config.n_embd
    shapes = {}
    for key, total_dim in block_dims.items():
        shapes[key] = (total_dim // n, n)
    return shapes

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=10)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ## Second Moment Computation and grouping. 
    ## Compute block shapes from grouped second moments
    ## Use these block shapes to initialize a BlockProjector which essentially initializes projection matrices for each block

    
    ## Last checkpoint for second moment approximation
    model = GPT2LMHeadModel.from_pretrained(os.path.join(os.path.dirname(__file__), '../out/gpt2-medium-restarted'))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    print("Loaded Model and Tokenizer...")
    
    print("Computing block shapes from model...")
    block_shapes = get_block_shapes_from_model(model, group_size=2)
    for k, v in block_shapes.items():
        print(k, v)
    
    print("Initializing BlockProjector...")
    proj = BlockProjector(block_shapes, d=4096, device=device)
    print("BlockProjector initialized...")
    ## Compute projected gradients
    ## Go through the dataset in batches 
    ## For each batch, compute the gradient of each sample in the batch and group the gradients by blocks defined before
    ## For each gradient in the batch, normalize it with its corresponding second moment block
    ## Each gradient is essentially a dictionary of grouped block names, which will be passed to the projector
    ## Projector is initialized before and projects each gradient dictionary and appends it to a list of gradients. 
    ## Once the list reaches a certain cutoff, store it in a file. 
    print("Initializing TrackstarUnbatched...")
    trackstar = TrackstarUnbatched(model, device=device)
    trackstar.projector = proj
    print("TrackstarUnbatched initialized...")
    print("Loading train dataset...")
    train_dataset = load_from_disk(os.path.join(os.path.dirname(__file__), '../../data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split/train'))
    
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    trackstar.number_of_samples = len(train_dataset)
    train_dataset = train_dataset.select(range(args.start_idx, args.end_idx))
    print('len of train dataset', len(train_dataset))

    print('number of samples', trackstar.number_of_samples)
    trackstar.counter = args.start_idx
    print('counter', trackstar.counter)
    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,       
        num_workers=2
    )
    
    print('starting to compute gradients')
    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        trackstar.compute_grouped_gradients_batch(batch, group_size=2)
        
    print("Gradients computed...")