#!/usr/bin/env python3
"""
Script tạo FAISS index từ document chunks
"""

import os
import sys
sys.path.append('../..')

from dotenv import load_dotenv
load_dotenv() 

import json
from pathlib import Path
from src.utils.paths import get_processed_data_dir, get_project_root
import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers import models as st_models
import faiss
from tqdm import tqdm
import torch
from dotenv import load_dotenv

# Nạp biến môi trường từ .env (nếu có)
load_dotenv()

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
                cur.execute(
                    """
                    SELECT chunk_id, doc_file, doc_title, chapter, section, article,
                           article_heading, clause, point, chunk_index, content,
                           word_count, chunk_type
                    FROM chunks
                    ORDER BY chunk_id
                    """
                )
                rows = cur.fetchall()
                cols = [
                    'chunk_id', 'doc_file', 'doc_title', 'chapter', 'section', 'article',
                    'article_heading', 'clause', 'point', 'chunk_index', 'content',
                    'word_count', 'chunk_type'
                ]
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
    # Giữ '_' theo ENV để bảo toàn cụm từ ghép của PyVi khi embedding
    keep_underscore = os.getenv("LEGALADVISOR_EMB_KEEP_UNDERSCORE", "1").lower() in ("1", "true", "yes", "on")

    def normalize_for_embedding(text: str) -> str:
        _t = (text or '').strip()
        if keep_underscore:
            return _t
        return _t.replace('_', ' ')

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
        # Preview ngắn để giữ dung lượng
        'preview': normalize_for_embedding(chunk.get('content', ''))[:200]
    } for chunk in chunks]

    return texts, metadata, ids

def create_embeddings(texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
    """Tạo embeddings cho texts"""

    print(f"🤖 Load model: {model_name}")

    # Load model
    use_gpu_env = os.getenv("LEGALADVISOR_USE_GPU", "auto").lower()
    if use_gpu_env == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif use_gpu_env in ("1", "true", "yes", "on", "cuda", "gpu"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cpu"

    # Cho phép override model qua ENV
    env_model = os.getenv("LEGALADVISOR_EMB_MODEL")
    if env_model:
        model_name = env_model

    print(f"🖥️  Device: {device}")
    print(f"🧠 Embedding model: {model_name}")

    # Ưu tiên CLS pooling cho Sup-SimCSE nếu phát hiện model tương ứng hoặc ENV yêu cầu
    force_cls = os.getenv("LEGALADVISOR_EMB_POOLING", "").lower() == "cls"
    try_cls = ("sup-simcse" in model_name.lower()) or force_cls

    if try_cls:
        try:
            transformer = st_models.Transformer(model_name)
            pooling = st_models.Pooling(
                transformer.get_word_embedding_dimension(),
                pooling_mode_cls_token=True,
                pooling_mode_mean_tokens=False,
                pooling_mode_max_tokens=False,
            )
            model = SentenceTransformer(modules=[transformer, pooling], device=device)
            print("🔧 Pooling: CLS (theo Sup-SimCSE)")
        except Exception as e:
            print(f"ℹ️  Không thể khởi tạo CLS pooling ({e}). Dùng mặc định của SentenceTransformers (mean)")
            model = SentenceTransformer(model_name, device=device)
            print("🔧 Pooling: mean (mặc định)")
    else:
        model = SentenceTransformer(model_name, device=device)
        print("🔧 Pooling: mean (mặc định)")

    print("🔄 Tạo embeddings...")

    # Tạo embeddings theo batch để tránh memory error
    batch_size = int(os.getenv("LEGALADVISOR_EMB_BATCH", "256"))
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=False)
        embeddings.append(batch_embeddings)

    # Ghép tất cả embeddings
    embeddings = np.vstack(embeddings)

    print(f"📊 Embeddings shape: {embeddings.shape}")

    return embeddings, model

def build_faiss_index(embeddings, ids=None):
    """Build FAISS index từ embeddings.

    Nếu cung cấp ids (chunk_id), sẽ sử dụng IndexIDMap để ánh xạ ổn định.
    Hỗ trợ HNSW qua ENV LEGALADVISOR_FAISS_HNSW=1.
    """

    print("🏗️ Xây dựng FAISS index...")

    # Lấy dimension của embeddings
    dimension = embeddings.shape[1]

    # Normalize embeddings để dùng cosine trên hình cầu đơn vị
    faiss.normalize_L2(embeddings)

    use_hnsw = os.getenv("LEGALADVISOR_FAISS_HNSW", "0").lower() in ("1", "true", "yes", "on")
    metric_type = "ip"

    if use_hnsw:
        M = int(os.getenv("LEGALADVISOR_HNSW_M", "32"))
        try:
            # Thử tạo HNSW với Inner Product (nếu bản FAISS hỗ trợ)
            base_index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)  # type: ignore
            metric_type = "hnsw_ip"
        except Exception:
            # Fallback: HNSW L2
            base_index = faiss.IndexHNSWFlat(dimension, M)
            metric_type = "hnsw_l2"
        # Thiết lập tham số tìm kiếm/xây dựng
        try:
            efc = int(os.getenv("LEGALADVISOR_HNSW_EF_CONSTRUCTION", "200"))
            efs = int(os.getenv("LEGALADVISOR_HNSW_EF_SEARCH", "64"))
            base_index.hnsw.efConstruction = efc
            base_index.hnsw.efSearch = efs
        except Exception:
            pass
    else:
        # Dùng IndexFlatIP cho cosine similarity
        base_index = faiss.IndexFlatIP(dimension)
        metric_type = "ip"

    # Add vectors (bọc IDMap nếu có ids)
    if ids is not None:
        index = faiss.IndexIDMap(base_index)
        ids_array = np.asarray(ids, dtype=np.int64)
        index.add_with_ids(embeddings, ids_array)
    else:
        index = base_index
        index.add(embeddings)

    print(f"✅ FAISS index created with {index.ntotal} vectors | type={metric_type}")

    return index, metric_type

def save_index_and_metadata(index, metadata, model, output_dir="../models/retrieval", used_id_map=True, metric_type: str = "ip"):
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
    # Thêm thông tin pooling để launcher hiển thị
    pooling = "mean"
    try:
        for m in getattr(model, 'modules', []):
            cls_name = m.__class__.__name__.lower()
            if 'pooling' in cls_name:
                if getattr(m, 'pooling_mode_cls_token', False):
                    pooling = 'cls'
                elif getattr(m, 'pooling_mode_mean_tokens', False):
                    pooling = 'mean'
                elif getattr(m, 'pooling_mode_max_tokens', False):
                    pooling = 'max'
                break
    except Exception:
        pass

    model_info = {
        "model_name": getattr(model, "model_card", {}).get("name", None) or getattr(model, "_model_card", None) or os.getenv("LEGALADVISOR_EMB_MODEL") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "embedding_dim": index.d,
        "num_chunks": index.ntotal,
        "uses_id_map": bool(used_id_map),
        "metric_type": metric_type,
        "pooling": pooling
    }

    model_info_path = output_dir / "model_info.json"
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)
    print(f"💾 Model info saved: {model_info_path}")

def main():
    """Hàm chính"""

    print("🚀 Bắt đầu tạo FAISS index cho retrieval...")

    # Load document chunks
    loaded = load_document_chunks()
    if loaded is None or loaded[0] is None:
        return
    texts, metadata, ids = loaded

    # Tạo embeddings
    embeddings, model = create_embeddings(texts)

    # Build FAISS index
    index, metric_type = build_faiss_index(embeddings, ids=ids)

    # Lưu index và metadata
    save_index_and_metadata(index, metadata, model, used_id_map=True, metric_type=metric_type)

    print("\n✅ Hoàn thành tạo FAISS index!")
    print("📁 Các file được lưu tại: ../models/retrieval/")
    print("   - faiss_index.bin: FAISS index")
    print("   - metadata.json: Thông tin chunks")
    print("   - model_info.json: Thông tin model")

if __name__ == "__main__":
    main()
