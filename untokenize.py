from datasets import load_from_disk
from tqdm import tqdm
import os
ds = load_from_disk(os.path.join(os.path.dirname(__file__), "../data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024/train_test_split"))
train_data = ds["train"]



from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
print(f"Starting untokenization of {len(train_data)} samples")
untokenized_data = []
for i in tqdm(range(0, len(train_data),1)):
    untokenized_data.append(tokenizer.decode(train_data[i]['input_ids'], skip_special_tokens=True))
    
import json
import os
print(len(untokenized_data))
os.makedirs("data/training_data/untokenized_data", exist_ok=True)
with open("data/training_data/untokenized_data/train_data.jsonl", "w") as f:
    for item in untokenized_data:
        f.write(json.dumps({"text": item}) + "\n")
print("Done")