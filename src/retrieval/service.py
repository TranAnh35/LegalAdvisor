#!/usr/bin/env python3
"""
Dịch vụ truy hồi tài liệu thống nhất cho LegalAdvisor (Zalo-Legal schema).

- Load FAISS index + metadata từ models/retrieval
- Mã hóa truy vấn bằng SentenceTransformer theo model trong model_info
- Trả về kết quả dạng corpus_id/type/number/year và preview (không còn đọc SQLite/Parquet)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Any

import json
import faiss  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore

from sentence_transformers import SentenceTransformer  # type: ignore

from ..utils.paths import get_models_retrieval_dir, get_processed_data_dir


class RetrievalService:
    """Dịch vụ truy hồi tài liệu dựa trên FAISS + metadata (Zalo-Legal)."""

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
        model_name = self.model_info.get("model_name") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        self.encoder = SentenceTransformer(model_name, device=device)

        # Đường dẫn JSONL content store (schema mới)
        self.processed_dir: Path = get_processed_data_dir()
        self.jsonl_path: Path = self.processed_dir / "zalo-legal" / "chunks_schema.jsonl"
        self._content_cache: Dict[int, str] = {}

    def encode_query(self, text: str) -> np.ndarray:
        """Mã hóa truy vấn thành numpy float32 contiguous và normalized."""
        query_embedding = self.encoder.encode([text], convert_to_numpy=True)
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        return query_embedding

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Truy hồi tài liệu liên quan cho truy vấn (semantic)."""
        try:
            # Semantic search
            query_embedding = self.encode_query(query)

            k = min(int(max(top_k, 3)), int(self.index.ntotal))
            if k <= 0:
                return []
            distances, indices = self.index.search(query_embedding, k)

            results: List[Dict[str, Any]] = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if int(idx) < 0:
                    continue
                chunk_id = int(idx)
                meta = self._meta_by_id.get(chunk_id, {})

                results.append({
                    'id': chunk_id,
                    'chunk_id': chunk_id,
                    'corpus_id': meta.get('corpus_id'),
                    'type': meta.get('type'),
                    'number': meta.get('number'),
                    'year': meta.get('year'),
                    'suffix': meta.get('suffix'),
                    'content': meta.get('preview', ''),  # dùng preview làm snippet
                    'preview': meta.get('preview', ''),
                    'score': float(distance)
                })

            # Sắp xếp theo score giảm dần
            results.sort(key=lambda r: r.get('score', 0.0), reverse=True)
            return results[:k]
        except Exception:
            return []

    def get_chunk_content(self, chunk_id: int) -> Optional[str]:
        """Trả về nội dung chunk từ JSONL schema mới (cache theo ID)."""
        try:
            if chunk_id in self._content_cache:
                return self._content_cache.get(chunk_id)
            if not self.jsonl_path.exists():
                return None
            # Đọc tuần tự và cache lazy theo nhu cầu để tiết kiệm RAM
            with open(self.jsonl_path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if int(obj.get('chunk_id', -1)) == int(chunk_id):
                        content = obj.get('content') or ''
                        self._content_cache[chunk_id] = content
                        return content
            return None
        except Exception:
            return None


