import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_from_disk
from transformers import GPT2Model, AutoTokenizer
from tqdm import tqdm
import argparse
device = "cuda" if torch.cuda.is_available() else "cpu"

import json
import os

@torch.no_grad()
def meanpool_last_hidden(input_ids, attention_mask):
    """
    input_ids: [B, T]
    attention_mask: [B, T]
    returns: [B, H] embeddings (L2 normalized)
    """
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    h = out.last_hidden_state                          # [B, T, H]
    mask = attention_mask.unsqueeze(-1).type_as(h)     # [B, T, 1]
    summed = (h * mask).sum(dim=1)                     # [B, H]
    denom = mask.sum(dim=1).clamp(min=1.0)             # [B, 1]
    pooled = summed / denom                            # [B, H]
    emb = torch.nn.functional.normalize(pooled, p=2, dim=1)
    return emb.cpu().numpy().astype("float32")

def make_loader(ds, batch_size=64):
    def collate(batch):

        ids = [torch.tensor(x['input_ids'], dtype=torch.long) for x in batch]
        att = [torch.ones_like(torch.tensor(x['attention_mask']), dtype=torch.long) for x in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=50256)
        att = pad_sequence(att, batch_first=True, padding_value=0)
        return {"input_ids": ids, "attention_mask": att}
    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--start_index", type=int, default=390000)
    parser.add_argument("--end_index", type=int, default=400000)
    args = parser.parse_args()
    base_dir = os.path.join(os.path.dirname(__file__), "../../../")

    print(base_dir)
    train_dataset = load_from_disk(os.path.join(base_dir, "data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split/train"))
    train_dataset = train_dataset.select(range(args.start_index, args.end_index))
    model = GPT2Model.from_pretrained(os.path.join(base_dir, "dabgo_acl/out/gpt2-medium-restarted")).to(device).eval()

    loader = make_loader(train_dataset, batch_size=2)
    N = len(train_dataset)
    H = model.config.n_embd

    all_embs = []
    print(len(train_dataset))
    idx = 0
    for batch in tqdm(loader):
        ids = batch["input_ids"].to(device)
        att = batch["attention_mask"].to(device)
        embs = meanpool_last_hidden(ids, att)   # [B, H]
        bsz = embs.shape[0]
        all_embs.extend(embs)
        idx += bsz
        
        if idx % 10000 == 0:
            all_embs = np.array(all_embs)
            print(f"Processed {idx}/{N}")
            np.save(os.path.join(os.path.dirname(__file__), f"train_embeddings_wrong_{args.start_index}_{args.end_index}_{idx}.npy"), all_embs)
            all_embs = []
    if len(all_embs) > 0:
    # Save to disk for later use
        all_embs = np.array(all_embs)
        np.save(os.path.join(os.path.dirname(__file__), f"train_embeddings_wrong_{args.start_index}_{args.end_index}_{idx}.npy"), all_embs)
        print(f"Done! Saved embeddings to train_embeddings_wrong_{args.start_index}_{args.end_index}_{idx}.npy")
