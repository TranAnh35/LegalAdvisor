#!/usr/bin/env python3
"""
Dịch vụ truy hồi tài liệu thống nhất cho LegalAdvisor.

- Load FAISS index + metadata từ models/retrieval
- Mã hóa truy vấn bằng SentenceTransformer theo model trong model_info
- Đọc nội dung chunk từ SQLite (ưu tiên) hoặc Parquet

Có thể dùng chung cho các engine RAG khác nhau (Gemini, Local).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import json
import faiss  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

from sentence_transformers import SentenceTransformer  # type: ignore

from utils.paths import get_models_retrieval_dir, get_processed_data_dir


class RetrievalService:
    """Dịch vụ truy hồi tài liệu dựa trên FAISS + metadata + content store."""

    def __init__(self, use_gpu: bool = False) -> None:
        # Thư mục models/retrieval
        self.model_dir: Path = get_models_retrieval_dir()

        # Load model info
        model_info_path = self.model_dir / "model_info.json"
        with open(model_info_path, 'r', encoding='utf-8') as f:
            self.model_info: Dict[str, Any] = json.load(f)

        # Load metadata và bảng tra theo chunk_id
        metadata_path = self.model_dir / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata: List[Dict[str, Any]] = json.load(f)
        try:
            self._meta_by_id: Dict[int, Dict[str, Any]] = {
                int(m.get('chunk_id')): m for m in self.metadata if m.get('chunk_id') is not None
            }
        except Exception:
            self._meta_by_id = {}

        # Load FAISS index
        index_path = self.model_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))

        # Khởi tạo encoder
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(self.model_info["model_name"], device=device)

        # Content store
        processed_dir = get_processed_data_dir()
        self.sqlite_path: Path = processed_dir / "smart_chunks_stable.db"
        self.parquet_path: Path = processed_dir / "smart_chunks_stable.parquet"
        self._parquet_df = None

    def encode_query(self, text: str) -> np.ndarray:
        """Mã hóa truy vấn thành numpy float32 contiguous và normalized."""
        query_embedding = self.encoder.encode([text], convert_to_numpy=True)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        return query_embedding

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Truy hồi tài liệu liên quan cho truy vấn."""
        try:
            query_embedding = self.encode_query(query)

            k = min(int(top_k or 3), int(self.index.ntotal))
            if k <= 0:
                return []
            distances, indices = self.index.search(query_embedding, k)

            results: List[Dict[str, Any]] = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if int(idx) < 0:
                    continue
                chunk_id = int(idx)
                meta = self._meta_by_id.get(chunk_id, {})

                content = self.get_chunk_content(chunk_id) or meta.get('preview', '')

                # Tạo tiêu đề ưu tiên: Tên luật → Điều/Khoản/Điểm → heading
                base_title = meta.get('doc_title') or meta.get('doc_file') or f'chunk_{chunk_id}.txt'
                parts: List[str] = []
                if meta.get('article'):
                    parts.append(f"Điều {meta.get('article')}")
                if meta.get('clause'):
                    parts.append(f"Khoản {meta.get('clause')}")
                if meta.get('point'):
                    parts.append(f"Điểm {meta.get('point')}")
                suffix = ' - '.join(parts)
                title = f"{base_title} - {suffix}" if suffix else base_title
                heading = meta.get('article_heading')
                if heading:
                    title = f"{title} - {heading}"

                results.append({
                    'id': chunk_id,
                    'chunk_id': chunk_id,
                    'doc_file': meta.get('doc_file', f'chunk_{chunk_id}.txt'),
                    'title': title,
                    'content': content or '',
                    'preview': meta.get('preview', ''),
                    'source': meta.get('source', 'Nguồn không xác định'),
                    'score': float(distance),
                    'law_title': meta.get('doc_title'),
                    'article': meta.get('article'),
                    'clause': meta.get('clause'),
                    'point': meta.get('point')
                })

            return results
        except Exception:
            return []

    def get_chunk_content(self, chunk_id: int) -> Optional[str]:
        """Lấy nội dung chunk theo ID từ SQLite hoặc Parquet."""
        # SQLite ưu tiên
        try:
            if self.sqlite_path and self.sqlite_path.exists():
                import sqlite3  # type: ignore
                conn = sqlite3.connect(str(self.sqlite_path))
                cur = conn.cursor()
                cur.execute("SELECT content FROM chunks WHERE chunk_id=?", (int(chunk_id),))
                row = cur.fetchone()
                conn.close()
                if row and row[0]:
                    return row[0]
        except Exception:
            pass

        # Parquet fallback
        try:
            if self.parquet_path and self.parquet_path.exists():
                import pandas as pd  # type: ignore
                if self._parquet_df is None:
                    self._parquet_df = pd.read_parquet(self.parquet_path)
                df = self._parquet_df
                match = df.loc[df['chunk_id'] == int(chunk_id), 'content']
                if not match.empty:
                    return str(match.iloc[0])
        except Exception:
            pass
        return None


