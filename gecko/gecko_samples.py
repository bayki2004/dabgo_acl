## Load Gecko samples get embeddings and order them by similarity to embeddings of training datasamples
import os
import torch
from transformers import GPT2Tokenizer
from vertexai import init as vertex_init
from vertexai.language_models import TextEmbeddingModel
import argparse
import numpy as np
def main(project_name, location, source):
    
    names = os.listdir(os.path.join(os.path.dirname(__file__), f"../out/optimized_models/{source}"))
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(names)
    already_processed = os.listdir(os.path.join(os.path.dirname(__file__), f"../data/gecko/sample_embeddings/{source}"))
    already_processed = [name.replace('.npz', '') for name in already_processed]
    print(already_processed)

    vertex_init(project=project_name, location=location)
    model = TextEmbeddingModel.from_pretrained("gemini-embedding-001")

    for name in names:
        if name in already_processed:
            continue
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), f"../out/optimized_models/{source}", name, "ascent", "100", "ckpt.pt"))
        output_ids = ckpt['output_ids']
        prompt_length = len(tokenizer.encode(ckpt['prompt'], add_special_tokens=False))
        output_text = tokenizer.decode(output_ids[0,prompt_length:], skip_special_tokens=True)
        print("Name: ", name)
        print("Output text: ", output_text)
        embs = model.get_embeddings([output_text])
        print(len(embs[0].values))
        
        os.makedirs(os.path.join(os.path.dirname(__file__), f"../data/gecko/sample_embeddings/{source}"), exist_ok=True)
        np.savez(
            f"{os.path.join(os.path.dirname(__file__), f'../data/gecko/sample_embeddings/{source}', name.replace('.pt', '.npz'))}",
            embedding=np.array(embs[0].values),
            output_text=output_text,
            prompt=ckpt["prompt"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="gecko-479220")
    parser.add_argument("--location", type=str, default="europe-west4")
    parser.add_argument("--source", type=str, default="wikipedia")
    args = parser.parse_args()
    main(args.project_name, args.location, args.source)