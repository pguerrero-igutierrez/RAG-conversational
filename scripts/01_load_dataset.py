"""
Downloads the SQAC dataset from Hugging Face and builds a deduplicated passage
corpus and an evaluation split, both serialized as JSONL files.
"""

import json
import os
from datasets import load_dataset
from tqdm.auto import tqdm

base_dir = "./rag_project"
data_dir = f"{base_dir}/data"
index_dir = f"{base_dir}/indexes"

for d in [data_dir, index_dir]:
    os.makedirs(d, exist_ok=True)

corpus_path = f"{index_dir}/sqac_corpus.jsonl"
eval_path = f"{data_dir}/sqac_validation.jsonl"

max_samples = 50

dataset = load_dataset("PlanTL-GOB-ES/SQAC")

seen, corpus_rows, eval_rows = {}, [], []

for split_name in ["train", "validation"]:
    for item in tqdm(dataset[split_name], desc=f"Processing {split_name}"):
        ctx = item["context"]
        title = item.get("title", "")
        if ctx not in seen:
            doc_id = f"doc_{len(seen)}"
            seen[ctx] = doc_id
            corpus_rows.append({"id": doc_id, "title": title, "context": ctx})
        if split_name == "validation" and (max_samples is None or len(eval_rows) < max_samples):
            eval_rows.append({
                "id": item["id"],
                "title": title,
                "question": item["question"],
                "context": ctx,
                "doc_id": seen[ctx],
                "answers": item["answers"]["text"],
            })

with open(corpus_path, "w", encoding="utf-8") as f:
    for r in corpus_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(eval_path, "w", encoding="utf-8") as f:
    for r in eval_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Corpus  : {len(corpus_rows)} passages → {corpus_path}")
print(f"Eval    : {len(eval_rows)} questions → {eval_path}")
