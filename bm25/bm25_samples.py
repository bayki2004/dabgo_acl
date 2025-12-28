## BM25 getting all samples for later use in tailpatch.py
import os
import numpy as np
from rank_bm25 import BM25Okapi
from datasets import load_from_disk
import json
import argparse
import torch
from transformers import GPT2Tokenizer

def main(sample_name, source):
    BASE_DIR = os.path.dirname(__file__)
    jsonl_path = os.path.join(
        BASE_DIR,
        "../data", "training_data", "untokenized_data", "train_data.jsonl",
    )
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    corpus = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # skip empty lines just in case
            obj = json.loads(line)
            corpus.append(obj["text"])

    print("Loaded docs:", len(corpus))
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    names = os.listdir(os.path.join(BASE_DIR, "../out", "optimized_models", source))
    os.makedirs(os.path.join(BASE_DIR, "../data", "bm25", source), exist_ok=True)
    already_processed = os.listdir(os.path.join(BASE_DIR, "../data", "bm25", source))
    already_processed = [name.replace('.npy', '') for name in already_processed]
    for name in names:
        if name in already_processed:
            continue
        if name == "original":
            continue
        ckpt = torch.load(os.path.join(BASE_DIR, "../out", "optimized_models", f"{source}",name, "ascent", "100", "ckpt.pt"), map_location=torch.device('cpu'), weights_only=False)
        output_ids = ckpt['output_ids']
        prompt_length = len(tokenizer.encode(ckpt['prompt'], add_special_tokens=False))
        prompt = ckpt['prompt']
        query = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
        print(f"Query: {query}")
        tokenized_query = query.split(" ")
        scores = bm25.get_scores(tokenized_query)
        print(f"BM25 scores for {name}: {scores[:5]}")
        
        
        os.makedirs(os.path.join(BASE_DIR, "../data", "bm25", source), exist_ok=True)
        scores = np.array(scores)
        np.save(os.path.join(BASE_DIR, "../data", "bm25", source, f"{name}.npy"), scores)
        print(f"BM25 scores saved to data/bm25/{source}/{name}.npy")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_name", type=str, default="All")
    parser.add_argument("--source", type=str, default="gutenberg")
    args = parser.parse_args()
    main(args.sample_name, args.source)