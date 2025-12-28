import os
import numpy as np
import argparse


# Define label priority
label_priority = {
    "Highly Relevant": 2,
    "Somewhat Relevant": 1,
    "Not Relevant": 0,
}

def sort_key(item):
    idx, sim, nll, label, prob = item
    
    nll = float(nll)
    prob = float(prob)
    priority = label_priority.get(label, -1)
    score = (1.0 / nll if nll > 0 else float("inf")) + (1.0 / prob if prob > 0 else float("inf"))
    return (priority, score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="wikipedia")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    source = args.source
    steps = args.steps
    examples = os.listdir(os.path.join(os.path.dirname(__file__), f'scores/{source}/'))
    examples = [example.replace('.npy', '') for example in examples]
    print(len(examples))
    for example in examples:
        scores = np.load(os.path.join(os.path.dirname(__file__), f'scores/{source}/{example}.npy'))
        scores = scores.tolist()
        scores_sorted = sorted(scores, key=sort_key, reverse=True)
        print(scores_sorted[0])
        
        sorted_indices = [int(float(i[0])) for i in scores_sorted]
        print(type(sorted_indices[0]))
        os.makedirs(os.path.join(os.path.dirname(__file__), f'sorted_scores/{source}'), exist_ok=True)
        np.save(os.path.join(os.path.dirname(__file__), f'sorted_scores/{source}/{example}.npy'), sorted_indices)
        