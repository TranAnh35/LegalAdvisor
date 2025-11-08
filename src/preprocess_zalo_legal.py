#!/usr/bin/env python3
"""
LegalAdvisor Data Processing Entry Point

Usage:
    python -m src.preprocess_zalo_legal
    
This script preprocesses Zalo-AI-Legal corpus and prepares it for retrieval.
"""

import sys
from src.data_preprocessing.zalo_legal import preprocess_corpus


def main():
    """Main entry point"""
    try:
        print("🚀 Starting Zalo-AI-Legal corpus preprocessing...\n")
        output_file, num_chunks = preprocess_corpus()
        
        print(f"\n✅ Successfully processed {num_chunks} chunks")
        print(f"📁 Output: {output_file}\n")
        
        return 0
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        print("\n💡 Kiểm tra:")
        print("   - data/raw/zalo_ai_legal_text_retrieval/corpus.jsonl có tồn tại không?")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
