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
from .service import RetrievalService

class LegalRetriever:
    """Class để retrieval văn bản pháp luật (sử dụng RetrievalService)."""

    def __init__(self, use_gpu: bool = False):
        self.service = RetrievalService(use_gpu=use_gpu)
        self.model_info = self.service.model_info

        print(f"✅ LegalRetriever loaded!")
        print(f"   - Model: {self.model_info['model_name']}")
        print(f"   - Vectors: {self.service.index.ntotal}")
        print(f"   - Dimension: {self.service.index.d}")

    def search(self, query, top_k=5):
        """Tìm kiếm tài liệu liên quan"""

        # Delegate cho service
        results = self.service.retrieve(query, top_k=top_k)
        # Trả về kết quả trực tiếp (đã có đầy đủ metadata)
        return results

    def get_chunk_content(self, chunk_id):
        """Lấy nội dung chunk theo ID từ SQLite/Parquet"""
        # Dùng service thống nhất (JSONL schema mới)
        return self.service.get_chunk_content(int(chunk_id))

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
            print(f"{i}. Corpus ID: {result.get('corpus_id', 'N/A')}")
            print(f"   Score: {result.get('score', 0.0):.4f}")
            print(f"   Type: {result.get('type', 'N/A')} | Number: {result.get('number', 'N/A')}")

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
