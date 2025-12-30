import os
import re
import argparse
from tqdm import tqdm

import torch
import numpy as np


def sorted_grad_files(gradient_dir):
    files = [f for f in os.listdir(gradient_dir)
             if f.endswith(".pt") and f.startswith("normed_grads")]
    files.sort(key=lambda f: int(re.search(r"normed_grads_(\d+)\.pt", f).group(1)))
    return files


@torch.no_grad()
def compute_influence_all_samples(
    sample_grads: dict,          
    gradient_dir: str,
    gradient_files: list[str],
    device: str,
    save_dir: str | None = None,
):
    
    accum = {name: [] for name in sample_grads.keys()}

    for file_idx, fname in enumerate(tqdm(gradient_files, desc="Streaming grad shards")):
        shard = torch.load(os.path.join(gradient_dir, fname),
                           map_location=device, weights_only=False)
        shard_len = len(shard)
        shard_scores = {name: torch.zeros(shard_len, device=device) for name in sample_grads.keys()}

        for i, grad_dict in enumerate(shard):
            for key, vec in grad_dict.items():
                for name, sgrad in sample_grads.items():
                    shard_scores[name][i] += torch.dot(sgrad[key], vec).item()
        # append to accum lists (move to cpu once per shard)
        for name in sample_grads.keys():
            accum[name].append(shard_scores[name].detach().cpu())
            tmp = torch.cat(accum[name]).numpy()
            print(tmp.shape)
            np.save(os.path.join(save_dir, f"{name}.partial.npy"), tmp)
    results = {}
    for name in sample_grads.keys():
        full = torch.cat(accum[name]).numpy()
        print(full.shape)
        results[name] = full
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, f"{name}.npy"), full)
    return results


def load_sample_gradients(sample_names, sample_dir, device, source):
    out = {}
    already_processed = os.listdir(os.path.join(os.path.dirname(__file__), f"../data/trackstar/scores/{source}"))
    for raw_name in sample_names:
        name = raw_name[7:]
        if name + ".npy" in already_processed:
            print(f"Skipping {name} because it already has a score")
            continue
        g = torch.load(os.path.join(sample_dir, f"{raw_name}.pt"),
                       map_location=device, weights_only=False)[0]
        name = raw_name
        if name.startswith("normed_"):
            name = name[len("normed_"):]
        elif len(name) >= 7 and name[:7] == "normed_":
            name = name[7:]

        out[name] = g
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--sample_names", type=str, nargs="+", default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dir = os.path.dirname(__file__)

    gradient_dir = os.path.join(base_dir, "../data/trackstar/gradients")
    grad_files = sorted_grad_files(gradient_dir)
    print(len(grad_files))
    print(grad_files[:10])
    sample_dir = os.path.join(base_dir, f"../data/trackstar/sample_gradients/{args.source}")
    all_samples = [f[:-3] for f in os.listdir(sample_dir) if f.endswith(".pt")]
    sample_names = args.sample_names if args.sample_names is not None else all_samples

    scores_dir = os.path.join(base_dir, f"../data/trackstar/scores/{args.source}")
    os.makedirs(scores_dir, exist_ok=True)
    print(sample_names)
    sample_grads = load_sample_gradients(sample_names, sample_dir, device, args.source)
    print(sample_grads.keys())
    compute_influence_all_samples(
        sample_grads=sample_grads,
        gradient_dir=gradient_dir,
        gradient_files=grad_files,
        device=device,
        save_dir=scores_dir,
    )

    print("Done.")
