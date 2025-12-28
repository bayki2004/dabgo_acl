import torch
import os
from tqdm import tqdm
import re
import argparse
import gc


## Run this part once all autocorrelation matrices are computed in auto_inv_sqrt.py
## Save this periodically due to very high memory usage. be aware disk space will be very high up to 1TB or more for all gradients.
@torch.no_grad()
def compute_normed_vectors_batched(gradient_files, autocorr_matrix, batch_size=10000, start_index=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pbar = tqdm(total=len(gradient_files), desc="Normalizing gradients")
    gradient_dir = os.path.join(os.path.dirname(__file__), "../data/trackstar/gradients")
    for file_path in gradient_files:
        pbar.update(1)
        print(f"Processing: {file_path}")
        gradients_list = torch.load(os.path.join(gradient_dir, file_path), map_location='cpu', weights_only=False)
        if isinstance(gradients_list[0], list):
            gradients_list = [g for sub in gradients_list for g in sub]

        num_grads = len(gradients_list)
        print(f"Number of gradients: {num_grads}")

        block_vectors = {block: [] for block in autocorr_matrix}
        block_indices = {block: [] for block in autocorr_matrix}

        for idx, grad in enumerate(gradients_list):
            for block, vec in grad.items():
                block_vectors[block].append(vec.view(1, -1))
                block_indices[block].append(idx)
        
        for block, vectors in block_vectors.items():
            if not vectors:
                continue
            R = autocorr_matrix[block].to(device)
            stacked = torch.cat(vectors, dim=0).to(device)
            transformed = (R @ stacked.T).T
            norms = torch.norm(transformed, dim=1, keepdim=True)
            normalized = torch.where(norms > 0, transformed / norms, transformed)
            for i, idx in enumerate(block_indices[block]):
                gradients_list[idx][block] = normalized[i].cpu()

        del block_vectors, block_indices
        torch.cuda.empty_cache()
        
        torch.save(gradients_list, os.path.join(gradient_dir, f"normed_{file_path}"))
        print(f"Saved: {os.path.join(gradient_dir, f"normed_{file_path}")}")
        os.remove(os.path.join(gradient_dir, file_path))
        print(f"Deleted: {os.path.join(gradient_dir, file_path)}")
        del gradients_list
        torch.cuda.empty_cache()

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=1000000)
    parser.add_argument("--source", type=str, default="wikipedia")
    args = parser.parse_args()
    end_index = args.end_index
    start_index = args.start_index
    gradient_dir = os.path.join(os.path.dirname(__file__), "../data/trackstar/gradients")
    gradient_files = [f for f in os.listdir(gradient_dir) if f.endswith(".pt") and f.startswith("grads")]
    print("Gradient files: ", gradient_files)
    gradient_files = sorted(
        [f for f in gradient_files if int(re.search(r'grads_(\d+)\.pt', f).group(1)) > start_index and int(re.search(r'grads_(\d+)\.pt', f).group(1)) < end_index],
        key=lambda f: int(re.search(r'grads_(\d+)\.pt', f).group(1))
    )
    print('number of gradient files', len(gradient_files))
    autocorr_matrix = torch.load(os.path.join(os.path.dirname(__file__), '../data/trackstar/autocorr_matrix_inv_sqrt.pt'), weights_only=False)
    print('starting to compute normed vectors')
    compute_normed_vectors_batched(gradient_files, autocorr_matrix, batch_size=10000, start_index=start_index)
