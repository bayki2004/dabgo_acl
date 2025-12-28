import os
import pandas as pd
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import AutoTokenizer

# -----------------------
# 0) tokenizer
# -----------------------
MODEL_NAME = "gpt2"  # or your custom tokenizer path
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

CTX = 1024
STRIDE = 912

def tokenize_gutenberg_sliding(batch):
    """
    Sliding-window tokenize for Gutenberg books.
    Each row -> multiple windows (max_length=1024, stride=512).
    We propagate author/title and provenance.
    """
    texts = batch[text_col]
    authors = batch[author_col]
    titles = batch[title_col]
    # 'index' comes from from_pandas(preserve_index=True) under the name '__index_level_0__'
    # (If not present, set doc_id from indices in with_indices later)
    doc_ids = batch.get("__index_level_0__", [None]*len(texts))

    enc = tokenizer(
        texts,
        max_length=CTX,
        truncation=True,
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_attention_mask=True,
        # If you want offsets, you can also add return_offsets_mapping=True (costlier)
    )

    # The fast tokenizer returns a flat list with:
    # enc["input_ids"], enc["attention_mask"], enc["overflow_to_sample_mapping"]
    # Map back to metadata:
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "source": [],
        "doc_id": [],      # original book row id
        "span_index": [],  # which window number within that book
        "author": [],
        "title": []
    }

    # Count windows per original sample so we can assign span_index cleanly
    from collections import defaultdict
    counts = defaultdict(int)

    for i, orig_idx in enumerate(enc["overflow_to_sample_mapping"]):
        out["source"].append("gutenberg")
        out["doc_id"].append(int(doc_ids[orig_idx]) if doc_ids[orig_idx] is not None else int(orig_idx))
        out["span_index"].append(counts[orig_idx])
        counts[orig_idx] += 1
        out["author"].append(authors[orig_idx])
        out["title"].append(titles[orig_idx])
    print('Tokenized Gutenberg Batch')
    return out


def tokenize_rowwise_truncate(batch, text_key, source_name, extra_keys=None):
    """
    Per-row tokenize (truncate to 1024). 1 row -> 1 sequence.
    extra_keys: list of (new_name, existing_key) to carry metadata through
                e.g., [("wiki_title", "title"), ("wiki_id", "id")]
    """
    enc = tokenizer(
        batch[text_key],
        max_length=CTX,
        truncation=True,
        padding=False,  # no need to pad here; pad in the collator at training time
        return_attention_mask=True,
    )
    out = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "source": [source_name] * len(enc["input_ids"]),
        "doc_id": [],      # a stable per-row id for provenance
        "span_index": [0] * len(enc["input_ids"]),
    }

    # Use with_indices later to fill doc_id; we put a placeholder here
    out["doc_id"] = [None] * len(enc["input_ids"])

    if extra_keys:
        for new_name, exist_key in extra_keys:
            out[new_name] = batch.get(exist_key, [None]*len(enc["input_ids"]))

    return out

# -----------------------
# 3) APPLY TOKENIZATION
# -----------------------

# 1a) Gutenberg (Pandas -> HF Dataset)
gutenberg_df = pd.read_csv("data/datasets/gutenberg_books_clean.csv")
author_col = "Author"
title_col = "Title"
text_col = "Text"
gutenberg_df = gutenberg_df.drop_duplicates(subset=['Author', 'Title'])
print("Number of unique authors: ", len(gutenberg_df["Author"].unique()))
gutenberg_df = gutenberg_df.dropna(subset=['Author', 'Title', 'Text'])
print("Number of rows after dropping na: ", len(gutenberg_df))
gutenberg_df = gutenberg_df.sample(n=200, random_state=42)
ds_gut = Dataset.from_pandas(gutenberg_df, preserve_index=True)  # keep pandas index
print("Loaded Gutenberg Dataset")
unique_authors = gutenberg_df["Author"].unique()
print("Number of unique authors: ", len(unique_authors))
# 3a) Gutenberg: map in batches (sliding overflow)
gut_tok = ds_gut.map(
    tokenize_gutenberg_sliding,
    batched=True,
    remove_columns=ds_gut.column_names,
    desc="Tokenizing Gutenberg (sliding windows)"
)

import gc
del ds_gut
del gutenberg_df
gc.collect()
gut_tok.save_to_disk("data/training_data/gutenberg_tok1024")
print("✅ Saved Gutenberg subset → data/training_data/gutenberg_tok1024")

# 1b) Wikipedia (Parquet path as you already have)
wiki = load_dataset(
    "parquet",
    data_files="data/datasets/finewiki/data/enwiki/*.parquet",
    split="train"
)
wiki = wiki.filter(lambda x: not x.get("has_math", False))
wiki = wiki.shuffle(seed=42).select(range(300_000))
# Expect fields like: "text", "title", "id" (adjust if different)
wiki_text_col = "text"
wiki_title_col = "title"
wiki_id_col = "id"
print('Loaded Wikipedia Dataset')
# 3b) Wikipedia: map with indices to set doc_id = original row index
wiki_tok = wiki.map(
    lambda batch, idx: {
        **tokenize_rowwise_truncate(
            batch, text_key=wiki_text_col, source_name="wikipedia",
            extra_keys=[("title", wiki_title_col), ("wiki_id", wiki_id_col)]
        ),
        # override doc_id with indices
        "doc_id": idx
    },
    with_indices=True,
    batched=True,
    remove_columns=wiki.column_names,
    desc="Tokenizing Wikipedia (truncate)"
)

del wiki
gc.collect()
wiki_tok.save_to_disk("data/training_data/wikipedia_tok1024")
print("✅ Saved Wikipedia subset → data/training_data/wikipedia_tok1024")

# 1c) FineWeb Edu (local subset you downloaded earlier)
# Replace `local_dir` with your snapshot_download return path if needed
local_dir = "./data/datasets/fineweb_edu"  # example
fw_small = load_dataset(
    "parquet",
    data_files=os.path.join(local_dir, "shard_0000*.parquet"),
    split="train"
).shuffle(seed=42).select(range(300_000))
fw_text_col = "text"  # adjust if different

# 3c) FineWeb: similar truncate; keep doc_id as the per-row index
fw_tok = fw_small.map(
    lambda batch, idx: {
        **tokenize_rowwise_truncate(
            batch, text_key=fw_text_col, source_name="fineweb"
        ),
        "doc_id": idx
    },
    with_indices=True,
    batched=True,
    remove_columns=fw_small.column_names,
    desc="Tokenizing FineWeb (truncate)"
)

del fw_small
gc.collect()
fw_tok.save_to_disk("data/training_data/fineweb_tok1024")
print("✅ Saved FineWeb subset → data/training_data/fineweb_tok1024")

# -----------------------
# 4) UNIFY + SHUFFLE + SAVE
# -----------------------

FINAL_COLS = [
    "input_ids", "attention_mask", "source", "doc_id", "span_index",
    "author", "title", "wiki_id",
]

def ensure_columns(ds, defaults):
    for col, default in defaults.items():
        if col not in ds.column_names:
            ds = ds.add_column(col, [default] * len(ds))
    return ds

# Fill missing metadata per source
gut_tok = ensure_columns(gut_tok, {"wiki_id": None})
wiki_tok = ensure_columns(wiki_tok, {"author": None})
fw_tok   = ensure_columns(fw_tok, {"author": None, "title": None, "wiki_id": None})


# Now concatenation will work
combined = concatenate_datasets([gut_tok, wiki_tok, fw_tok]).shuffle(seed=1234)

combined = combined.filter(lambda x: len(x["input_ids"]) >= 64)
print(f"✅ Filtered dataset: {len(combined)} samples remaining (≥64 tokens)")
# Save for reuse
os.makedirs("data/training_data/mixtures", exist_ok=True)
out_dir = "data/training_data/mixtures/gut10k_wiki100k_fw100k_tok1024"
combined.save_to_disk(out_dir)
print(f"Saved to {out_dir}")


