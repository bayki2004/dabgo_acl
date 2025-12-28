import torch
from utils_ import top_k_samples, method_scores
import argparse
import os
import numpy as np
from transformers import GPT2Tokenizer
import json
from datasets import load_from_disk
from openai import OpenAI
import time
import random

def query_llm(api_key, model, query, sample1, sample2, max_retries=3):
    """
    Query a language model to compare two samples.
    Returns 0 if sample1 is better, 1 if sample2 is better.
    """
    client = OpenAI(api_key=api_key)
    if sample1 == sample2:
        return -2
    USER_PROMPT = """You are given a LLM generated query and two training data samples. You are to decide which training data sample is more relevant for the LLM to generate the query. Output 0 if the first sample is more relevant, 1 if the second sample is more relevant and -1 if you are unsure."""
    full_prompt = f"""

Query: {query}

Sample 1:
{sample1}

Sample 2:
{sample2}

Output only 0 or 1."""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": USER_PROMPT},
                    {"role": "user", "content": full_prompt}
                ],
            )
            
            answer = response.choices[0].message.content.strip()
            print(f"Answer: {answer}")
            # Parse the answer
            if '0' in answer and '1' not in answer:
                return 0
            elif '1' in answer and '0' not in answer:
                return 1
            else:
                # If unclear, try to extract first digit
                for char in answer:
                    if char in ['0', '1']:
                        return int(char)
                print(f"Warning: Could not parse answer: {answer}, retrying...")
                
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
    
    # Default to 0 if all retries fail
    print("Warning: All retries failed, defaulting to -1")
    return -1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="Self-Written", help="Source of the data (e.g., Self-Written, gutenberg, wikipedia)")
    parser.add_argument("--model", type=str, default="gpt-5", help="Model to use (e.g., gpt-4, gpt-4-turbo, gpt-3.5-turbo)")
    parser.add_argument("--method1", type=str, default="dabgo", help="First method to compare")
    parser.add_argument("--method2", type=str, default="descent", help="Second method to compare")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps for the methods")
    parser.add_argument("--num_comparisons", type=int, default=1, help="Number of samples to compare per task")
    parser.add_argument("--output_dir", type=str, default="per-sample", help="Output file for results")
    
    args = parser.parse_args()
    BASE_DIR = os.path.dirname(__file__)
    
    # Load dataset and tokenizer
    train_dataset = load_from_disk(os.path.join(BASE_DIR, "data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split/train"))
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    api_key = "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"
    print("Loaded tokenizer and train dataset")
    print(f"Dataset size: {len(train_dataset)}")
    
    # Get list of tasks to evaluate
    names = [n for n in os.listdir(os.path.join(BASE_DIR, "out/optimized_models", args.source)) if n != "original" and not n.startswith('.')]
    print(f"Evaluating tasks: {names}")
    
    # Store results
    results = {
        "method1": args.method1,
        "method2": args.method2,
        "model": args.model,
        "source": args.source,
        "comparisons": []
    }
    
    total_method1_wins = 0
    total_method2_wins = 0
    
    for name in names:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        
        
        
        samples1_indices = method_scores(args.method1, args.num_comparisons, args.source, name, train_dataset, steps=args.steps)
        
        samples2_indices = method_scores(args.method2, args.num_comparisons, args.source, name, train_dataset, steps=args.steps)
        
        query_ckpt = torch.load(os.path.join(BASE_DIR, f"out/optimized_models/{args.source}/{name}/metadata.pt"), map_location='cpu', weights_only=False)
        query = query_ckpt.get("output_text")
        
        task_results = {
            "task": name,
            "query": query,
            "comparisons": [],
            "method1_wins": 0,
            "method2_wins": 0
        }
        print(f"Query: {query}")
        # Compare samples pairwise
        for i in range(min(args.num_comparisons, len(samples1_indices), len(samples2_indices))):
            sample1_idx = int(samples1_indices[i])
            sample2_idx = int(samples2_indices[i])
            
            sample1_text = tokenizer.decode(train_dataset[sample1_idx]['input_ids'])
            sample2_text = tokenizer.decode(train_dataset[sample2_idx]['input_ids'])
            
            # Randomly decide which method's sample is shown first to avoid position bias
            method1_shown_first = random.random() < 0.5
            
            if method1_shown_first:
                first_sample = sample1_text
                second_sample = sample2_text
                first_method = args.method1
                second_method = args.method2
            else:
                first_sample = sample2_text
                second_sample = sample1_text
                first_method = args.method2
                second_method = args.method1
            
            print(f"\nComparison {i+1}/{args.num_comparisons}")
            print(f"Sample 1 (from {first_method}): {first_sample[:100]}...")
            print(f"Sample 2 (from {second_method}): {second_sample[:100]}...")
            print(f"[Randomized order: method1_first={method1_shown_first}]")
            
            # Query the LLM
            print("Querying LLM...")
            result = query_llm(api_key, args.model, query, first_sample, second_sample)
            
            # Interpret result based on which method was shown first
            if method1_shown_first:
                # Normal order: result 0 = method1, result 1 = method2
                if result == 0:
                    winner = args.method1
                    print(f"Result: {args.method1} is better")
                    task_results["method1_wins"] += 1
                    total_method1_wins += 1
                elif result == 1:
                    winner = args.method2
                    print(f"Result: {args.method2} is better")
                    task_results["method2_wins"] += 1
                    total_method2_wins += 1
                else:
                    winner = "No clear winner"
                    print(f"Result: No clear winner")
            else:
                # Reversed order: result 0 = method2, result 1 = method1
                if result == 0:
                    winner = args.method2
                    print(f"Result: {args.method2} is better")
                    task_results["method2_wins"] += 1
                    total_method2_wins += 1
                elif result == 1:
                    winner = args.method1
                    print(f"Result: {args.method1} is better")
                    task_results["method1_wins"] += 1
                    total_method1_wins += 1
                else:
                    winner = "No clear winner"
                    print(f"Result: No clear winner")
            
            task_results["comparisons"].append({
                "comparison_num": i + 1,
                "method1_idx": sample1_idx,
                "method2_idx": sample2_idx,
                "method1_preview": sample1_text[:200],
                "method2_preview": sample2_text[:200],
                "method1_shown_first": method1_shown_first,
                "winner": winner
            })
        
        results["comparisons"].append(task_results)
        
        print(f"\nTask Results:")
        print(f"  {args.method1}: {task_results['method1_wins']} wins")
        print(f"  {args.method2}: {task_results['method2_wins']} wins")
    
    # Save results
    os.makedirs(os.path.join(BASE_DIR, f"data/llm_judge_results/{args.output_dir}/{args.source}"), exist_ok=True)
    output_path = os.path.join(BASE_DIR, f"data/llm_judge_results/{args.output_dir}/{args.source}/{args.method1}_{args.method2}_{args.num_comparisons}.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(f"Total comparisons: {total_method1_wins + total_method2_wins}")
    print(f"{args.method1}: {total_method1_wins} wins ({100*total_method1_wins/(total_method1_wins + total_method2_wins):.1f}%)")
    print(f"{args.method2}: {total_method2_wins} wins ({100*total_method2_wins/(total_method1_wins + total_method2_wins):.1f}%)")
    print(f"\nResults saved to: {output_path}")
            