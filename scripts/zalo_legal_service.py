# -*- coding: utf-8 -*-
"""
DEPRECATED: Service tìm kiếm pháp lý - Legacy wrapper

Sử dụng src.retrieval.service.RetrievalService thay vào đó để tìm kiếm pháp lý.

Entry point:
    python scripts/zalo_legal_service.py
    hoặc
    python -c "from src.retrieval.service import RetrievalService; rs = RetrievalService(); rs.search('query')"
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.service import RetrievalService


def main():
    """
    Legacy interactive search interface.
    Delegates to RetrievalService.
    """
    print("\n=== Retrieval pháp lý Zalo-AI-Legal VN (pipeline mới) ===")
    print("📌 Chú ý: Đây là wrapper legacy cho src.retrieval.service.RetrievalService")
    print("   Khuyến cáo: Sử dụng API hoặc programmatic access thay vào đó\n")
    
    try:
        retriever = RetrievalService()
        print("✅ RetrievalService initialized\n")
    except Exception as e:
        print(f"❌ Lỗi khởi tạo RetrievalService: {e}")
        return 1
    
    while True:
        try:
            query = input("Nhập câu hỏi pháp lý › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break
        
        if not query:
            continue
        
        try:
            results = retriever.retrieve(query, k=5)
            
            if not results:
                print("❌ Không tìm thấy kết quả\n")
                continue
            
            for i, res in enumerate(results, 1):
                print(f"\n[{i}] Văn bản: {res.get('corpus_id', 'N/A')}")
                print(f"    Loại: {res.get('type', 'N/A')} - Số: {res.get('number', 'N/A')} - Năm: {res.get('year', 'N/A')}")
                print(f"    Điểm số: {res.get('score', 'N/A'):.4f}")
                print(f"    Preview: {res.get('preview', 'N/A')[:150]}...")
            print()
        except Exception as e:
            print(f"❌ Lỗi tìm kiếm: {e}")
            import traceback
            traceback.print_exc()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
