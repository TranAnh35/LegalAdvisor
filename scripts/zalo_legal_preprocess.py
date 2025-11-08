#!/usr/bin/env python3
"""
Tiền xử lý schema từ corpus.jsonl cho retrieval - Zalo-AI-Legal

DEPRECATED: Sử dụng src.data_preprocessing.zalo_legal module thay vào đó.

Chạy bằng:
    python -m src.data_preprocessing.zalo_legal
    hoặc
    python scripts/zalo_legal_preprocess.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing.zalo_legal import preprocess_corpus

if __name__ == '__main__':
    """
    Legacy entry point - delegates to main preprocessing module.
    """
    print("📌 Chú ý: Đây là wrapper legacy cho src.data_preprocessing.zalo_legal")
    print("   Khuyến cáo: Sử dụng 'python -m src.data_preprocessing.zalo_legal' trực tiếp\n")
    
    try:
        output_file, num_chunks = preprocess_corpus()
        print(f"\n✅ Thành công: {num_chunks} chunks lưu vào {output_file}")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
