import json
import os
from tqdm.auto import tqdm

base_dir = "./rag_project"
out_data_dir = f"{base_dir}/data"
index_dir = f"{base_dir}/indexes"
input_data_dir = "./corpus"

for d in [out_data_dir, index_dir]:
    os.makedirs(d, exist_ok=True)

corpus_path = f"{index_dir}/sqac_corpus.jsonl"
eval_path = f"{out_data_dir}/sqac_validation.jsonl"

max_samples = 50

seen, corpus_rows, eval_rows = {}, [], []

files_to_process = {
    "train": f"{input_data_dir}/train.json",
    "validation": f"{input_data_dir}/dev.json"
}

for split_name, file_path in files_to_process.items():
    print(f"Loading {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    articles = raw_data.get("data", []) if isinstance(raw_data, dict) else raw_data
    
    for article in tqdm(articles, desc=f"Processing {split_name}"):
        title = article.get("title", "")
        
        for paragraph in article.get("paragraphs", []):
            ctx = paragraph["context"]
            
            if ctx not in seen:
                doc_id = f"doc_{len(seen)}"
                seen[ctx] = doc_id
                corpus_rows.append({"id": doc_id, "title": title, "context": ctx})
            
            for qa in paragraph.get("qas", []):
                if split_name == "validation" and (max_samples is None or len(eval_rows) < max_samples):
                    
                    answers = qa.get("answers", [])
                    if isinstance(answers, list) and len(answers) > 0:
                        ans_texts = [a["text"] for a in answers]
                    else:
                        ans_texts = []

                    eval_rows.append({
                        "id": qa["id"],
                        "title": title,
                        "question": qa["question"],
                        "context": ctx,
                        "doc_id": seen[ctx],
                        "answers": ans_texts,
                    })

with open(corpus_path, "w", encoding="utf-8") as f:
    for r in corpus_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with open(eval_path, "w", encoding="utf-8") as f:
    for r in eval_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Corpus  : {len(corpus_rows)} passages -> {corpus_path}")
print(f"Eval    : {len(eval_rows)} questions -> {eval_path}")