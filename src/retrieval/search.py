#!/usr/bin/env python3
"""
Script test retrieval system với FAISS
"""

import os
import sys
sys.path.append('../..')

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import sqlite3
import pandas as pd
from retrieval.service import RetrievalService

class LegalRetriever:
    """Class để retrieval văn bản pháp luật (sử dụng RetrievalService)."""

    def __init__(self, use_gpu: bool = False):
        self.service = RetrievalService(use_gpu=use_gpu)
        self.model_info = self.service.model_info
        self.metadata = self.service.metadata

        print(f"✅ LegalRetriever loaded!")
        print(f"   - Model: {self.model_info['model_name']}")
        print(f"   - Vectors: {self.service.index.ntotal}")
        print(f"   - Dimension: {self.service.index.d}")

    def search(self, query, top_k=5):
        """Tìm kiếm tài liệu liên quan"""

        # Delegate cho service
        results = self.service.retrieve(query, top_k=top_k)
        # Rút gọn schema cho script test
        simplified = [{
            'chunk_id': r.get('chunk_id'),
            'doc_file': r.get('doc_file'),
            'chunk_index': self.service._meta_by_id.get(int(r.get('chunk_id', -1)), {}).get('chunk_index') if r.get('chunk_id') is not None else None,
            'score': r.get('score', 0.0),
            'word_count': self.service._meta_by_id.get(int(r.get('chunk_id', -1)), {}).get('word_count') if r.get('chunk_id') is not None else None
        } for r in results]

        return simplified

    def get_chunk_content(self, chunk_id):
        """Lấy nội dung chunk theo ID từ SQLite/Parquet"""
        return self.service.get_chunk_content(int(chunk_id))
        try:
            processed_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
            sqlite_path = processed_dir / 'smart_chunks_stable.db'
            parquet_path = processed_dir / 'smart_chunks_stable.parquet'

            if sqlite_path.exists():
                conn = sqlite3.connect(str(sqlite_path))
                cur = conn.cursor()
                cur.execute("SELECT content FROM chunks WHERE chunk_id=?", (int(chunk_id),))
                row = cur.fetchone()
                conn.close()
                return row[0] if row and row[0] else None

            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                match = df.loc[df['chunk_id'] == int(chunk_id), 'content']
                if not match.empty:
                    return str(match.iloc[0])
        except Exception:
            return None
        return None

def test_retrieval():
    """Test retrieval system"""

    print("🚀 Test Legal Retrieval System...")

    # Khởi tạo retriever
    retriever = LegalRetriever()

    # Các câu query test
    test_queries = [
        "quyền của công dân",
        "thủ tục ly hôn",
        "quy định về lao động",
        "phạt vi phạm giao thông",
        "quyền sở hữu trí tuệ"
    ]

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)

        # Tìm kiếm
        results = retriever.search(query, top_k=3)

        # Hiển thị kết quả
        for i, result in enumerate(results, 1):
            print(f"{i}. File: {result['doc_file']}")
            print(f"   Score: {result['score']:.4f}")
            print(f"   Words: {result['word_count']}")

            # Lấy và hiển thị nội dung mẫu
            content = retriever.get_chunk_content(result['chunk_id'])
            if content:
                preview = content[:200] + "..." if len(content) > 200 else content
                print(f"   Content: {preview}")

            print()

def main():
    """Hàm chính"""
    test_retrieval()

if __name__ == "__main__":
    main()
