## Compute Trackstar Influence

## Load Query vector
## Load All Sample vectors
## Sort by name grads_x.pt
## Compute Influence and append to one long list
## Save list to file also during checkpoints
import torch
import os
import numpy as np
import argparse
import re
from tqdm import tqdm



def compute_influence(sample_gradients, gradient_files, sample_grad_name, sample_gradient_dir):
        influence_list = torch.zeros(len(gradient_files)*20000)
        j = 0
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Computing influence for ', sample_grad_name)
        gradient_dir = os.path.join(os.path.dirname(__file__), "../data/trackstar/gradients")
        for gradient_file in tqdm(gradient_files, total=len(gradient_files), desc="Computing influence"):
            gradient = torch.load(os.path.join(gradient_dir, gradient_file), map_location=device, weights_only=False)
            for i in range(len(gradient)):
                grad = gradient[i]
                for key, value in grad.items():
                    curr_infl = torch.dot(sample_gradients[key], value)
                    
                    influence_list[j] += curr_infl.cpu()
                
                if j % 20000 == 0:
                    
                    np.save(os.path.join(sample_gradient_dir, f"{sample_grad_name}.npy"), influence_list)
                    print(f"Saved influence list to {os.path.join(sample_gradient_dir, f'{sample_grad_name}.npy')}")
                j += 1
        np.save(os.path.join(sample_gradient_dir, f"{sample_grad_name}.npy"), influence_list)
        print(f"Saved influence list to {os.path.join(sample_gradient_dir, f'{sample_grad_name}.npy')}")
        return influence_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument("--sample_names", type=str, nargs='+', default=None)
    args = parser.parse_args()
    sample_gradients = os.listdir(os.path.join(os.path.dirname(__file__), f"../data/trackstar/sample_gradients/{args.source}"))
    sample_gradients = [f for f in sample_gradients if f.endswith(".pt")]
    sample_gradients = [f.replace('.pt', '') for f in sample_gradients]
    gradient_dir = os.path.join(os.path.dirname(__file__), "../data/trackstar/gradients")
    gradient_files = os.listdir(gradient_dir)
    gradient_files = [f for f in gradient_files if f.endswith(".pt") and f.startswith("normed_grads")]
    gradient_files.sort(key=lambda f: int(re.search(r'normed_grads_(\d+)\.pt', os.path.basename(f)).group(1)))
    
    scores_dir = os.path.join(os.path.dirname(__file__), f"../data/trackstar/scores/{args.source}")
    os.makedirs(scores_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("On device: ", device)
    if args.sample_names is not None:
        sample_gradients = args.sample_names
        print(sample_gradients)
    block_autocorr_inv_sqrt = torch.load(os.path.join(gradient_dir, "../autocorr_matrix_inv_sqrt.pt"), map_location=device)
    for sample_grad_name in sample_gradients:
        print(f"Processing {sample_grad_name}")
        sample_gradient = torch.load(os.path.join(os.path.dirname(__file__), f"../data/trackstar/sample_gradients/{args.source}/{sample_grad_name}.pt"), map_location=device, weights_only=False)
        sample_gradient = sample_gradient[0]
        sample_grad_name = sample_grad_name[7:] ## remove normed_
        print(sample_grad_name)
        print(sample_gradient.keys())
        influence_list = compute_influence(sample_gradient, gradient_files, sample_grad_name, scores_dir)
        np.save(os.path.join(scores_dir, f"{sample_grad_name}.npy"), influence_list)
        print(f"Saved influence list to {os.path.join(scores_dir, f'{sample_grad_name}.npy')}")
        