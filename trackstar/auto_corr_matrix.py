import torch
import os
from tqdm import tqdm
import re
import argparse
import torch
import gc
import os
import re


def compute_blockwise_autocorrelation(gradient_files, eps=1e-12, device='cpu', start_index=0):

    from collections import defaultdict
    blocks = ['group0_mlp', 'group0_attn', 'group1_mlp', 'group1_attn', 'group2_mlp', 'group2_attn', 'group3_mlp', 'group3_attn', 'group4_mlp', 'group4_attn', 'group5_mlp', 'group5_attn', 'group6_mlp', 'group6_attn', 'group7_mlp', 'group7_attn', 'group8_mlp', 'group8_attn', 'group9_mlp', 'group9_attn', 'group10_mlp', 'group10_attn', 'group11_mlp', 'group11_attn', 'final_ln']
    sum_outer = {block: torch.zeros(4096, 4096, device=device) for block in blocks}    
    count = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for file_path in tqdm(gradient_files, total=len(gradient_files), desc="Accumulating autocorr"):
        gradients_list = torch.load(file_path, map_location=device, weights_only=False)  # list of dicts
        print(file_path)
        print(len(gradients_list))
        
        block_vectors = defaultdict(list)
        for grad in gradients_list:
            for block, vec in grad.items():
                block_vectors[block].append(vec.view(1, -1))  
                
            count += 1
            
        print(f"Processing batch of {count}")
        with torch.no_grad():
            for block, vec_list in block_vectors.items():
                if vec_list:  
                    stacked = torch.cat(vec_list, dim=0).to(device)  # [B, 4096]
                    sum_outer[block] += stacked.T @ stacked
            block_vectors.clear()  
            torch.cuda.empty_cache()
            del stacked, vec_list, gradients_list
            gc.collect()
        if count % 100000 == 0:
            torch.save(sum_outer, os.path.join(os.path.dirname(__file__), f"../data/trackstar/autocorr_matrices/autocorr_matrices_{count+start_index}.pt"))
            print('saved', count)
        
        
        torch.cuda.empty_cache()
        print(count)
        
    torch.save(sum_outer, os.path.join(os.path.dirname(__file__), f"../data/trackstar/autocorr_matrices/autocorr_matrices_{count+start_index}.pt"))
    
    
    return sum_outer


def compute_autocorr_matrices_inv_sqrt(auto_corr_files):
    blocks = ['group0_mlp', 'group0_attn', 'group1_mlp', 'group1_attn', 'group2_mlp', 'group2_attn', 'group3_mlp', 'group3_attn', 'group4_mlp', 'group4_attn', 'group5_mlp', 'group5_attn', 'group6_mlp', 'group6_attn', 'group7_mlp', 'group7_attn', 'group8_mlp', 'group8_attn', 'group9_mlp', 'group9_attn', 'group10_mlp', 'group10_attn', 'group11_mlp', 'group11_attn', 'final_ln']
    sum_outer = {block: torch.zeros(4096, 4096, device='cpu') for block in blocks}  
    with torch.no_grad():
        for auto_corr_file in auto_corr_files:
            auto_corr = torch.load(os.path.join(os.path.dirname(__file__), f"../data/trackstar/autocorr_matrices/{auto_corr_file}"), map_location='cpu', weights_only=False)
            for block in blocks:
                sum_outer[block] += auto_corr[block]
            del auto_corr
            gc.collect()
        print("Accumulated autocorrelation matrices")
        eps = 1e-6
        block_autocorr_inv_sqrt = {}
        for block, R in sum_outer.items():
            print(f"Computing inverse sqrt for block {block}...")
            R_cpu = R.cpu()
            w, V = torch.linalg.eigh(R_cpu)
            w = torch.clamp(w, min=eps)
            R_inv_sqrt = (V * w.rsqrt().unsqueeze(0)) @ V.T
            block_autocorr_inv_sqrt[block] = R_inv_sqrt

        torch.save(block_autocorr_inv_sqrt, os.path.join(os.path.dirname(__file__), f"../data/trackstar/autocorr_matrix_inv_sqrt.pt"))
    print("Saved inverse sqrt matrices.")
    return block_autocorr_inv_sqrt
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--end_index', type=int, default=1000000)
    args = parser.parse_args()

    end_index = args.end_index
    start_index = args.start_index
    gradient_dir = os.path.join(os.path.dirname(__file__), "../data/trackstar/gradients")
    gradient_files = [os.path.join(gradient_dir, f) for f in os.listdir(gradient_dir) if f.endswith(".pt") and f.startswith("grads")]

    gradient_files = sorted(
        [f for f in gradient_files if int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1)) > start_index and int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1)) < end_index],
        key=lambda f: int(re.search(r'grads_(\d+)\.pt', os.path.basename(f)).group(1))
    )
    print(len(gradient_files))
    print(gradient_files)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('starting to compute autocorrelation matrices')
    
    autocorr_matrices = compute_blockwise_autocorrelation(gradient_files, device=device, start_index=start_index)
    print("Computed autocorrelation matrices")

    print("Starting to compute inverse sqrt of autocorrelation matrices")
    auto_corr_files = os.listdir(os.path.join(os.path.dirname(__file__), f"../data/trackstar/autocorr_matrices"))
    auto_corr_files = [f for f in auto_corr_files if f.startswith("autocorr_matrices_") and f.endswith(".pt")]
    auto_corr_files = sorted(auto_corr_files, key=lambda f: int(re.search(r'autocorr_matrices_(\d+)\.pt', f).group(1)))
    print(len(auto_corr_files))
    print(auto_corr_files)
    auto_inv_sqrt_matrix = compute_autocorr_matrices_inv_sqrt(auto_corr_files)
