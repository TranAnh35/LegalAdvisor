#!/usr/bin/env python3
"""
Script tạo FAISS index từ document chunks (chuẩn Zalo-AI-Legal, không còn tương thích dữ liệu cũ)
"""
import os
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, models as st_models
import faiss
from tqdm import tqdm
import torch

def load_document_chunks():
    """Chỉ đọc JSONL schema mới Zalo-AI-Legal"""
    jsonl_path = Path(__file__).parent.parent.parent / "data" / "processed" / "zalo-legal" / "chunks_schema.jsonl"
    if not jsonl_path.exists():
        print(f"❌ Không tìm thấy file dữ liệu: {jsonl_path}")
        return None, None, None
    chunks = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except Exception as e:
                    print(f"Lỗi parse JSONL: {e}")
    print(f"Số lượng chunks: {len(chunks)}")
    # For embedding: lấy content, nếu muốn thì cộng thêm title/số hiệu
    texts = [chunk.get('content', '') or '' for chunk in chunks]
    ids = [int(chunk.get('chunk_id')) for chunk in chunks]
    metadata = [{
        'chunk_id': chunk.get('chunk_id'),
        'corpus_id': chunk.get('corpus_id'),
        'type': chunk.get('type'),
        'number': chunk.get('number'),
        'year': chunk.get('year'),
        'suffix': chunk.get('suffix'),
        'word_count': chunk.get('word_count'),
        'preview': chunk.get('preview')
    } for chunk in chunks]
    return texts, metadata, ids

def create_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Tạo embeddings cho texts"""
    print(f"🤖 Load model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")
    print(f"🧠 Embedding model: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    print("🔄 Tạo embeddings...")
    batch_size = 256
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=False)
        embeddings.append(batch_embeddings)
    embeddings = np.vstack(embeddings)
    print(f"📊 Embeddings shape: {embeddings.shape}")
    return embeddings, model

def build_faiss_index(embeddings, ids=None):
    print("🏗️ Xây dựng FAISS index...")
    dimension = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    # Dùng IndexFlatIP cho cosine similarity
    base_index = faiss.IndexFlatIP(dimension)
    if ids is not None:
        index = faiss.IndexIDMap(base_index)
        index.add_with_ids(embeddings, np.asarray(ids, dtype=np.int64))
    else:
        index = base_index
        index.add(embeddings)
    print(f"✅ FAISS index created with {index.ntotal} vectors")
    return index

def save_index_and_metadata(index, metadata, model, output_dir="../models/retrieval"):
    output_dir = Path(__file__).parent.parent.parent / "models" / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"💾 FAISS index saved: {index_path}")
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata saved: {metadata_path}")
    model_info = {
        "model_name": getattr(model, "model_card", None) or os.getenv("LEGALADVISOR_EMB_MODEL") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": index.d,
        "num_chunks": index.ntotal
    }
    model_info_path = output_dir / "model_info.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    print(f"💾 Model info saved: {model_info_path}")

def main():
    print("🚀 Bắt đầu tạo FAISS index cho retrieval từ Zalo-AI-Legal ...")
    texts, metadata, ids = load_document_chunks()
    if texts is None:
        return
    embeddings, model = create_embeddings(texts)
    index = build_faiss_index(embeddings, ids=ids)
    save_index_and_metadata(index, metadata, model)
    print("\n✅ Hoàn thành tạo FAISS index mới!")
    print("📁 Các file được lưu tại: ../models/retrieval/")
    print("   - faiss_index.bin: FAISS index")
    print("   - metadata.json: Thông tin chunks")
    print("   - model_info.json: Thông tin model")

if __name__ == "__main__":
    main()
