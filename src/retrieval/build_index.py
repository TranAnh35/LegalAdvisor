#!/usr/bin/env python3
"""
Script tạo FAISS index từ document chunks
"""

import os
import sys
sys.path.append('../..')

import json
from pathlib import Path
from src.utils.paths import get_processed_data_dir, get_project_root
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
import torch

def load_document_chunks():
    """Load document chunks.

    Ưu tiên: Parquet → SQLite (không còn dùng JSON)
    """
    # Xác định thư mục data/processed theo util
    base_dir = get_processed_data_dir()

    # 1) Parquet
    parquet_file = base_dir / "smart_chunks_stable.parquet"
    if parquet_file.exists():
        try:
            import pandas as pd  # type: ignore
            print(f"📖 Đọc Parquet: {parquet_file}")
            df = pd.read_parquet(parquet_file)
            chunks = df.to_dict(orient="records")
        except Exception as e:
            print(f"⚠️  Lỗi đọc Parquet ({e}). Thử SQLite/JSON...")
            chunks = None
    else:
        chunks = None

    # 2) SQLite
    if chunks is None:
        sqlite_file = base_dir / "smart_chunks_stable.db"
        if sqlite_file.exists():
            try:
                import sqlite3
                print(f"📖 Đọc SQLite: {sqlite_file}")
                conn = sqlite3.connect(str(sqlite_file))
                cur = conn.cursor()
                # Cố gắng đọc thêm các cột metadata mở rộng nếu tồn tại
                query_extended = (
                    """
                    SELECT chunk_id, doc_file, doc_title, chapter, section, article,
                           article_heading, clause, point, chunk_index, content,
                           word_count, chunk_type,
                           effective_date, effective_year, promulgation_date, promulgation_year, citations
                    FROM chunks
                    ORDER BY chunk_id
                    """
                )
                query_basic = (
                    """
                    SELECT chunk_id, doc_file, doc_title, chapter, section, article,
                           article_heading, clause, point, chunk_index, content,
                           word_count, chunk_type
                    FROM chunks
                    ORDER BY chunk_id
                    """
                )
                try:
                    cur.execute(query_extended)
                    rows = cur.fetchall()
                    cols = [
                        'chunk_id', 'doc_file', 'doc_title', 'chapter', 'section', 'article',
                        'article_heading', 'clause', 'point', 'chunk_index', 'content',
                        'word_count', 'chunk_type', 'effective_date', 'effective_year',
                        'promulgation_date', 'promulgation_year', 'citations'
                    ]
                except Exception:
                    cur.execute(query_basic)
                    rows = cur.fetchall()
                    cols = [
                        'chunk_id', 'doc_file', 'doc_title', 'chapter', 'section', 'article',
                        'article_heading', 'clause', 'point', 'chunk_index', 'content',
                        'word_count', 'chunk_type'
                    ]
                rows = cur.fetchall()
                chunks = [dict(zip(cols, row)) for row in rows]
                conn.close()
            except Exception as e:
                print(f"⚠️  Lỗi đọc SQLite ({e}).")
                chunks = None
    # Không còn fallback JSON
    if chunks is None:
        print("❌ Không tìm thấy dữ liệu chunks trong Parquet/SQLite!")
        return None, None

    print(f"📊 Tổng số chunks: {len(chunks)}")

    # Lấy nội dung chunks và ids ổn định
    # ViTokenizer có thể tạo dấu '_' nối từ; chuẩn hóa về khoảng trắng để cải thiện embedding
    def normalize_for_embedding(text: str) -> str:
        return (text or '').replace('_', ' ').strip()

    # Giảm kích thước bằng cách cắt content input embedding (ví dụ 800 tokens ~ 4k chars)
    texts = [normalize_for_embedding((chunk.get('content', '') or '')[:4000]) for chunk in chunks]
    ids = [int(chunk.get('chunk_id')) if chunk.get('chunk_id') is not None else idx for idx, chunk in enumerate(chunks)]
    metadata = [{
        'chunk_id': chunk.get('chunk_id'),
        'doc_file': chunk.get('doc_file'),
        # Tránh phình metadata.json: cắt tiêu đề tối đa 200 ký tự
        'doc_title': (chunk.get('doc_title')[:200] if isinstance(chunk.get('doc_title'), str) else chunk.get('doc_title')),
        'chunk_index': chunk.get('chunk_index'),
        'word_count': chunk.get('word_count'),
        'chapter': chunk.get('chapter'),
        'section': chunk.get('section'),
        'article': chunk.get('article'),
        'article_heading': chunk.get('article_heading'),
        'clause': chunk.get('clause'),
        'point': chunk.get('point'),
        'chunk_type': chunk.get('chunk_type'),
        # Metadata mở rộng nếu có
        'effective_date': chunk.get('effective_date'),
        'effective_year': chunk.get('effective_year'),
        'promulgation_date': chunk.get('promulgation_date'),
        'promulgation_year': chunk.get('promulgation_year'),
        'citations': chunk.get('citations'),
        # Preview ngắn để giữ dung lượng
        'preview': normalize_for_embedding(chunk.get('content', ''))[:200]
    } for chunk in chunks]

    return texts, metadata, ids

def create_embeddings(texts, model_name: str, batch_size: int, device: str):
    """Tạo embeddings cho texts"""

    print(f"🤖 Load model: {model_name}")

    # Chuẩn hóa & kiểm tra thiết bị
    requested_device = (device or "auto").lower()
    if requested_device == "auto":
        effective_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif requested_device == "cuda":
        if not torch.cuda.is_available():
            print(
                f"❌ Đã yêu cầu CUDA nhưng torch.cuda.is_available()=False. Vui lòng chạy trong môi trường GPU."
            )
            raise RuntimeError("CUDA requested but not available")
        effective_device = "cuda"
    elif requested_device == "cpu":
        effective_device = "cpu"
    else:
        # chấp nhận các alias phổ biến
        if requested_device in ("1", "true", "yes", "on", "gpu"):
            effective_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            effective_device = "cpu"

    try:
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    except Exception:
        gpu_name = "N/A"

    print(
        f"🖥️  Device requested: {requested_device} | cuda_available={torch.cuda.is_available()} | "
        f"num_devices={torch.cuda.device_count()} | using={effective_device} | gpu0={gpu_name}"
    )
    model = SentenceTransformer(model_name, device=effective_device)

    print("🔄 Tạo embeddings...")

    # Tạo embeddings theo batch để tránh memory error
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(
            batch_texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            device=effective_device,
        )
        embeddings.append(batch_embeddings)

    # Ghép tất cả embeddings
    embeddings = np.vstack(embeddings)

    print(f"📊 Embeddings shape: {embeddings.shape}")

    return embeddings, model, effective_device

def build_faiss_index(embeddings, ids=None):
    """Build FAISS index từ embeddings.

    Nếu cung cấp ids (chunk_id), sẽ sử dụng IndexIDMap để ánh xạ ổn định.
    """

    print("🏗️ Xây dựng FAISS index...")

    # Lấy dimension của embeddings
    dimension = embeddings.shape[1]

    # Tạo FAISS index với Inner Product (cho cosine similarity)
    base_index = faiss.IndexFlatIP(dimension)

    # Normalize embeddings cho cosine similarity
    faiss.normalize_L2(embeddings)

    # Add vectors
    if ids is not None:
        # Bọc với IDMap và add kèm ids (int64)
        index = faiss.IndexIDMap(base_index)
        ids_array = np.asarray(ids, dtype=np.int64)
        index.add_with_ids(embeddings, ids_array)
    else:
        index = base_index
        index.add(embeddings)

    print(f"✅ FAISS index created with {index.ntotal} vectors")

    return index

def save_index_and_metadata(index, metadata, model_name: str, emb_batch: int, device: str, output_dir="../models/retrieval", used_id_map=True):
    """Lưu FAISS index và metadata"""

    # Luôn lưu về thư mục models/retrieval tại gốc dự án
    project_root = get_project_root()
    output_dir = project_root / "models" / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lưu FAISS index
    index_path = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(index_path))
    print(f"💾 FAISS index saved: {index_path}")

    # Lưu metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"💾 Metadata saved: {metadata_path}")

    # Lưu model info
    model_info = {
        "model_name": model_name,
        "embedding_dim": index.d,
        "num_chunks": index.ntotal,
        "uses_id_map": bool(used_id_map),
        "batch_size": emb_batch,
        "device": device,
    }

    model_info_path = output_dir / "model_info.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    print(f"💾 Model info saved: {model_info_path}")

def main():
    """Hàm chính"""

    print("🚀 Bắt đầu tạo FAISS index cho retrieval...")

    # CLI args
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS index for LegalAdvisor")
    parser.add_argument("--model", type=str, default=None, help="Tên model HF hoặc đường dẫn local đến SentenceTransformer đã fine-tune")
    parser.add_argument("--emb-batch", type=int, default=None, help="Batch size khi tạo embedding (mặc định từ env LEGALADVISOR_EMB_BATCH hoặc 256)")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"], help="Thiết bị encode: auto/cpu/cuda (mặc định auto hoặc từ LEGALADVISOR_USE_GPU)")
    args = parser.parse_args()

    # Load document chunks
    loaded = load_document_chunks()
    if loaded is None or loaded[0] is None:
        return
    texts, metadata, ids = loaded

    # Tạo embeddings
    # Resolve model name
    model_name = (
        args.model
        or os.getenv("LEGALADVISOR_EMB_MODEL")
        or "intfloat/multilingual-e5-small"
    )
    # Resolve batch size
    emb_batch = args.emb_batch if args.emb_batch is not None else int(os.getenv("LEGALADVISOR_EMB_BATCH", "256"))
    # Resolve device
    device = args.device if args.device is not None else os.getenv("LEGALADVISOR_USE_GPU", "auto").lower()

    embeddings, _, effective_device = create_embeddings(
        texts, model_name=model_name, batch_size=emb_batch, device=device
    )

    # Build FAISS index
    index = build_faiss_index(embeddings, ids=ids)

    # Lưu index và metadata
    save_index_and_metadata(
        index,
        metadata,
        model_name=model_name,
        emb_batch=emb_batch,
        device=effective_device,
        used_id_map=True,
    )

    print("\n✅ Hoàn thành tạo FAISS index!")
    print("📁 Các file được lưu tại: ../models/retrieval/")
    print("   - faiss_index.bin: FAISS index")
    print("   - metadata.json: Thông tin chunks")
    print("   - model_info.json: Thông tin model")

if __name__ == "__main__":
    main()
