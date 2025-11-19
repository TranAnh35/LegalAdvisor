from datasets import load_dataset
import os
import json

# Đảm bảo lưu đúng folder data/raw/zalo_ai_legal_text_retrieval (tương đối với gốc project)
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'zalo_ai_legal_text_retrieval'))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Load corpus (split duy nhất là 'corpus')
print("Tải corpus...")
corpus = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", name="corpus", split="corpus")
corpus_path = os.path.join(OUTPUT_DIR, "corpus.jsonl")
with open(corpus_path, "w", encoding="utf-8") as f:
    for item in corpus:
        f.write(json.dumps(item, ensure_ascii=False)+'\n')
print(f"Đã lưu {len(corpus)} dòng corpus vào {corpus_path}")

# 2. Load queries (split duy nhất là 'queries')
print("Tải queries...")
queries = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", name="queries", split="queries")
queries_path = os.path.join(OUTPUT_DIR, "queries.jsonl")
with open(queries_path, "w", encoding="utf-8") as f:
    for item in queries:
        f.write(json.dumps(item, ensure_ascii=False)+'\n')
print(f"Đã lưu {len(queries)} dòng queries vào {queries_path}")

# 3. Load pairs (default: train+test)
for split in ["train", "test"]:
    try:
        pairs = load_dataset("GreenNode/zalo-ai-legal-text-retrieval-vn", name="default", split=split)
        path = os.path.join(OUTPUT_DIR, f"pairs_{split}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for item in pairs:
                f.write(json.dumps(item, ensure_ascii=False)+'\n')
        print(f"Đã lưu pairs {split}: {len(pairs)} dòng vào {path}")
    except Exception as e:
        print(f"Không tìm thấy split {split} ở default: {e}")

# 4. Xuất unique corpus-id (từ corpus)
corpus_ids = set()
for item in corpus:
    if "corpus-id" in item:
        corpus_ids.add(item["corpus-id"])
with open(os.path.join(OUTPUT_DIR, "unique_corpus_ids.txt"), "w", encoding="utf-8") as f:
    for cid in sorted(corpus_ids):
        f.write(cid + "\n")
print(f"Đã xuất {len(corpus_ids)} corpus-id duy nhất vào unique_corpus_ids.txt")