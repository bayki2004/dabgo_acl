## Make an index of training samples with gecko embeddings and store the index as a list for tailpatch.py

import numpy as np
import os
import argparse
def main(source):
    names = os.listdir(os.path.join(os.path.dirname(__file__), f"../data/gecko/sample_embeddings/{source}"))
    names = [name.replace('.npz', '') for name in names]
    print(names)
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    train_embeddings = np.load(os.path.join(base_dir, 'data', 'gecko_embeddings', 'test', 'embeddings_all.npy'))
    N, D = train_embeddings.shape
    # Normalize in-place for cosine similarity
    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    train_embeddings /= np.clip(norms, 1e-8, None)
    os.makedirs(os.path.join(base_dir, 'data', 'gecko', 'sample_scores', source), exist_ok=True)
    already_processed = os.listdir(os.path.join(base_dir, 'data', 'gecko', 'sample_scores', source))
    already_processed = [name.replace('.npy', '') for name in already_processed]
    print("Normalized train embeddings")
    for name in names:
        if name in already_processed:
            continue
        print(f"Processing {name}")
        if name == "original":
            continue
        embeddings_sample = np.load(os.path.join(base_dir, 'data', 'gecko', 'sample_embeddings', source,f"{name}.npz"))
        embeddings_sample = embeddings_sample['embedding']
        norms = np.linalg.norm(embeddings_sample, axis=0, keepdims=True)
        embeddings_sample /= np.clip(norms, 1e-8, None)
        print(embeddings_sample.shape)
        print(f"Normalized embeddings sample")
        scores = train_embeddings @ embeddings_sample
        print(f"Cosine similarity: {scores.shape}")
        os.makedirs(os.path.join(base_dir, 'data', 'gecko', 'sample_scores', source), exist_ok=True)
        np.save(os.path.join(base_dir, 'data', 'gecko', 'sample_scores', source, f"{name}.npy"), scores)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="wikipedia")
    args = parser.parse_args()
    main(args.source)