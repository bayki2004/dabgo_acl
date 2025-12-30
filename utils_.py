import numpy as np
from transformers import GPT2Tokenizer
import torch
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token

def top_k_samples(method, k, source, name,ds, steps=3):
    if method == "original":
        ckpt = torch.load(f"out/optimized_models/{source}/{name}/ascent/{steps}/ckpt.pt")
        print(f"Original Sample and Prompt:")
        print(f"Prompt: {ckpt['prompt']}")
        print(f"Sample: {tokenizer.decode(ckpt['output_ids'][0])}")
        return [0]
    if method == "dabgo":
        losses_ascent = np.load(f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy")
        losses_descent = np.load(f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")
        sample_scores = np.abs(losses_ascent - losses_descent)
        sample_scores = sample_scores.argsort()[::-1]
    if method == "descent":
        losses_descent = np.load(f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = losses_descent - losses_base
        sample_scores = sample_scores.argsort()
    if method == "ascent":
        losses_ascent = np.load(f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = losses_base - losses_ascent
        sample_scores = sample_scores.argsort()
    if method == "descent_reverse":
        losses_descent = np.load(f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = losses_base - losses_descent
        sample_scores = sample_scores.argsort()
    if method == "descent_absolute":
        losses_descent = np.load(f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = np.abs(losses_descent - losses_base)
        sample_scores = sample_scores.argsort()[::-1]
    if method == "ascent_reverse":
        losses_ascent = np.load(f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = losses_ascent - losses_base
        sample_scores = sample_scores.argsort()
    if method == "gecko_wrong":
        sample_scores = np.load(f"gecko/gecko_wrong/sorted_scores/{source}/{name}.npy")
        sample_scores = sample_scores[:k]
    if method == "gecko":
        sample_scores = np.load(f"data/gecko/sample_scores/{source}/{name}.npy")
        sample_scores = sample_scores.argsort()[::-1]
    if method == "trackstar":
        sample_scores = np.load(f"data/trackstar/scores/{source}/{name}.npy")
        sample_scores = sample_scores.argsort()[::-1]

    if method == "bm25":
        sample_scores = np.load(f"data/bm25/{source}/{name}.npy")
        sample_scores = sample_scores.argsort()[::-1]
    print(f"Top {k} samples for {name} in {method}")
    for i in sample_scores[:k]:
        print(tokenizer.decode(ds[int(i)]['input_ids']))
    return sample_scores

def method_scores(method, k, source, name,ds, steps=3):
    if method == "original":
        ckpt = torch.load(f"out/optimized_models/{source}/{name}/ascent/{steps}/ckpt.pt")
        print(f"Original Sample and Prompt:")
        print(f"Prompt: {ckpt['prompt']}")
        print(f"Sample: {tokenizer.decode(ckpt['output_ids'][0])}")
        return [0]
    if method == "dabgo":
        losses_ascent = np.load(f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy")
        losses_descent = np.load(f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")
        sample_scores = np.abs(losses_ascent - losses_descent)
        sample_scores = sample_scores.argsort()[::-1]
    if method == "descent":
        losses_descent = np.load(f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = losses_descent - losses_base
        sample_scores = sample_scores.argsort()
    if method == "ascent":
        losses_ascent = np.load(f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = losses_base - losses_ascent
        sample_scores = sample_scores.argsort()
    if method == "descent_reverse":
        losses_descent = np.load(f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = losses_base - losses_descent
        sample_scores = sample_scores.argsort()
    if method == "descent_absolute":
        losses_descent = np.load(f"data/losses/{source}/{name}/descent/{steps}/losses_{name}_descent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = np.abs(losses_descent - losses_base)
        sample_scores = sample_scores.argsort()[::-1]
    if method == "ascent_reverse":
        losses_ascent = np.load(f"data/losses/{source}/{name}/ascent/{steps}/losses_{name}_ascent.npy")
        losses_base = np.load(f"data/losses/original/base/losses_original_base.npy")
        sample_scores = losses_ascent - losses_base
        sample_scores = sample_scores.argsort()
    if method == "gecko_wrong":
        sample_scores = np.load(f"gecko/gecko_wrong/sorted_scores/{source}/{name}.npy")
        sample_scores = sample_scores[:k]
    if method == "gecko":
        sample_scores = np.load(f"data/gecko/sample_scores/{source}/{name}.npy")
        sample_scores = sample_scores.argsort()[::-1]

    if method == "bm25":
        sample_scores = np.load(f"data/bm25/{source}/{name}.npy")
        sample_scores = sample_scores.argsort()[::-1]
    
    if method == "trackstar":
        sample_scores = np.load(f"data/trackstar/scores/{source}/{name}.npy")
        sample_scores = sample_scores.argsort()[::-1]
    return sample_scores