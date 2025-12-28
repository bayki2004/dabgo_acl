import os
import argparse
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2Model
from datasets import load_from_disk


@torch.no_grad()
def encode_query(text: str):
    x = tok(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    out = model(**x)
    h = out.last_hidden_state             # [1, T, H]
    mask = x.attention_mask.unsqueeze(-1).type_as(h)
    pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return pooled.cpu().numpy().astype("float32")[0]

def top_k_neighbors(query: str, embs: np.ndarray):
    q = encode_query(query)               # [H]
    sims = embs @ q                       # cosine, since all normalized
    idx = np.argsort(sims)[::-1]          # top-k indices
    return [(int(i), float(sims[i])) for i in idx]

base_dir = os.path.join(os.path.dirname(__file__), "../../")
model = GPT2Model.from_pretrained(os.path.join(base_dir, "out/gpt2-medium-restarted"))
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
tok = GPT2Tokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="wikipedia")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    source = args.source
    steps = args.steps
    base_dir = os.path.join(os.path.dirname(__file__), "../../")
    examples = os.listdir(os.path.join(base_dir, f'data/losses/{source}/'))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    
    embeddings = np.load(os.path.join(os.path.dirname(__file__), f'train_embeddings_wrong_total.npy'))
    print(embeddings.shape)
    print("Loaded train data")
    for example in examples:
        if not os.path.exists(os.path.join(base_dir, f'out/optimized_models/{source}/{example}/ascent/{steps}/ckpt.pt')):
            continue
        ckpt = torch.load(os.path.join(base_dir, f'out/optimized_models/{source}/{example}/ascent/{steps}/ckpt.pt'), map_location='cpu', weights_only=False)
        output_ids = ckpt['output_ids']
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"Query: {output_text}")
        topk = top_k_neighbors(output_text, embeddings)
        print(f"Top 100 neighbors: {len(topk)}")
        print(topk[0])
        os.makedirs(os.path.join(os.path.dirname(__file__), f'data/gecko/{source}'), exist_ok=True)
        np.save(os.path.join(os.path.dirname(__file__), f'data/gecko/{source}/{example}.npy'), topk)