#!/usr/bin/env python3
"""
📦 Process Dataset cho HyperbolicRAG

Xử lý dataset Zalo Legal và tạo các file cần thiết cho HyperbolicRAG.

Input:
    - data/raw/zalo_ai_legal_text_retrieval/corpus.jsonl
    - data/raw/zalo_ai_legal_text_retrieval/queries.jsonl
    - data/raw/zalo_ai_legal_text_retrieval/pairs_train.jsonl
    - data/raw/zalo_ai_legal_text_retrieval/pairs_test.jsonl
    - data/processed/zalo-legal/hierarchy.json (từ build_hierarchy.py)
    - data/processed/zalo-legal/chunk_depths.json

Output:
    - data/processed/zalo-legal/corpus_hyperbolic.jsonl (với depth info)
    - data/processed/zalo-legal/queries.jsonl (deduplicated)
    - data/processed/zalo-legal/train_pairs.jsonl
    - data/processed/zalo-legal/test_pairs.jsonl

Schema corpus_hyperbolic.jsonl:
    {
        "chunk_id": int,
        "corpus_id": str,           # "91/2015/qh13+279"
        "doc_code": str,            # "91/2015/qh13"
        "article": int,             # 279
        "title": str,               # "Điều 279. Thực hiện nghĩa vụ giao vật"
        "content": str,             # Full text content
        "depth": float,             # Poincaré depth (0.65-0.90)
        "issuer": str,              # "Quốc hội"
        "doc_type": str,            # "qh13"
        "year": str                 # "2015"
    }

Usage:
    python scripts/process_dataset.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_corpus_id(corpus_id: str) -> Tuple[str, Optional[int], str, str, str]:
    """
    Parse corpus_id thành các thành phần.
    
    Args:
        corpus_id: ID dạng "91/2015/qh13+279"
    
    Returns:
        (doc_code, article, doc_type, year, doc_number)
    """
    if not corpus_id:
        return "", None, "", "", ""
    
    # Split article number
    parts = corpus_id.split('+')
    if len(parts) == 2:
        doc_code = parts[0].strip()
        try:
            article = int(parts[1].strip())
        except ValueError:
            article = None
    else:
        doc_code = corpus_id.strip()
        article = None
    
    # Parse doc_code: {số}/{năm}/{loại}
    doc_parts = doc_code.split('/')
    if len(doc_parts) >= 3:
        doc_number = doc_parts[0]
        year = doc_parts[1]
        doc_type = '/'.join(doc_parts[2:])
    elif len(doc_parts) == 2:
        doc_number = doc_parts[0]
        year = doc_parts[1]
        doc_type = ""
    else:
        doc_number = ""
        year = ""
        doc_type = ""
    
    return doc_code, article, doc_type, year, doc_number


def get_issuer(doc_type: str) -> str:
    """Xác định cơ quan ban hành từ doc_type."""
    if not doc_type:
        return "Khác"
    
    doc_type_lower = doc_type.lower()
    
    issuer_patterns = {
        "qh": "Quốc hội",
        "ubtvqh": "UBTVQH",
        "cp": "Chính phủ",
        "ttg": "Thủ tướng",
        "btc": "Bộ Tài chính",
        "bct": "Bộ Công Thương",
        "btp": "Bộ Tư pháp",
        "bca": "Bộ Công an",
        "bgtvt": "Bộ GTVT",
        "byt": "Bộ Y tế",
        "bgdđt": "Bộ GD&ĐT",
        "bxd": "Bộ Xây dựng",
        "btnmt": "Bộ TN&MT",
        "bnnptnt": "Bộ NN&PTNT",
        "blđtbxh": "Bộ LĐ-TB&XH",
        "btttt": "Bộ TT&TT",
    }
    
    for pattern, issuer in issuer_patterns.items():
        if pattern in doc_type_lower:
            return issuer
    
    return "Khác"


def load_depth_lookup(depths_path: Path) -> Dict[str, float]:
    """Load chunk_depths.json."""
    if not depths_path.exists():
        print(f"⚠️ Warning: {depths_path} not found. Using default depths.")
        return {}
    
    with open(depths_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_corpus(
    input_path: Path,
    output_path: Path,
    depth_lookup: Dict[str, float]
) -> int:
    """
    Process corpus.jsonl thành corpus_hyperbolic.jsonl.
    """
    print(f"\n📂 Processing corpus...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    processed_count = 0
    errors = []
    
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(tqdm(fin, desc="Processing corpus", unit="lines"), 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append({"line": line_num, "error": str(e)})
                continue
            
            corpus_id = record.get('_id', '')
            title = record.get('title', '')
            content = record.get('text', '')
            
            if not corpus_id:
                continue
            
            # Parse corpus_id
            doc_code, article, doc_type, year, doc_number = parse_corpus_id(corpus_id)
            
            # Get depth
            depth = depth_lookup.get(corpus_id, 0.75)  # Default depth
            
            # Get issuer
            issuer = get_issuer(doc_type)
            
            # Build output record
            output_record = {
                "chunk_id": processed_count,
                "corpus_id": corpus_id,
                "doc_code": doc_code,
                "article": article,
                "title": title,
                "content": content,
                "depth": depth,
                "issuer": issuer,
                "doc_type": doc_type,
                "year": year
            }
            
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            processed_count += 1
    
    if errors:
        print(f"   ⚠️ {len(errors)} parse errors")
    
    print(f"   ✅ Processed {processed_count:,} records")
    return processed_count


def process_queries(input_path: Path, output_path: Path) -> int:
    """
    Process và deduplicate queries.
    """
    print(f"\n📂 Processing queries...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Deduplicate by _id
    unique_queries: "OrderedDict[str, Dict]" = OrderedDict()
    duplicates = 0
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing queries", unit="lines"):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            query_id = record.get('_id', '')
            text = record.get('text', '').strip()
            
            if not query_id or not text:
                continue
            
            if query_id in unique_queries:
                duplicates += 1
                continue
            
            unique_queries[query_id] = {
                "query_id": query_id,
                "text": text
            }
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        for record in unique_queries.values():
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"   ✅ {len(unique_queries):,} unique queries (removed {duplicates} duplicates)")
    return len(unique_queries)


def process_pairs(input_path: Path, output_path: Path) -> int:
    """
    Process pairs (train hoặc test).
    """
    print(f"\n📂 Processing pairs...")
    print(f"   Input: {input_path}")
    print(f"   Output: {output_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in tqdm(fin, desc="Processing pairs", unit="lines"):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            query_id = record.get('query-id', '')
            corpus_id = record.get('corpus-id', '')
            score = record.get('score', 1.0)
            
            if not query_id or not corpus_id:
                continue
            
            output_record = {
                "query_id": query_id,
                "corpus_id": corpus_id,
                "score": score
            }
            
            fout.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            count += 1
    
    print(f"   ✅ Processed {count:,} pairs")
    return count


def main():
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw" / "zalo_ai_legal_text_retrieval"
    processed_dir = project_root / "data" / "processed" / "zalo-legal"
    
    print("=" * 60)
    print("📦 DATASET PROCESSOR FOR HYPERBOLICRAG")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check required files
    required_files = [
        raw_dir / "corpus.jsonl",
        raw_dir / "queries.jsonl",
        raw_dir / "pairs_train.jsonl",
        raw_dir / "pairs_test.jsonl",
    ]
    
    for f in required_files:
        if not f.exists():
            print(f"❌ Error: Required file not found: {f}")
            sys.exit(1)
    
    # Load depth lookup
    depths_path = processed_dir / "chunk_depths.json"
    depth_lookup = load_depth_lookup(depths_path)
    print(f"\n📊 Loaded {len(depth_lookup):,} depth mappings")
    
    # Process corpus
    process_corpus(
        input_path=raw_dir / "corpus.jsonl",
        output_path=processed_dir / "corpus_hyperbolic.jsonl",
        depth_lookup=depth_lookup
    )
    
    # Process queries
    process_queries(
        input_path=raw_dir / "queries.jsonl",
        output_path=processed_dir / "queries.jsonl"
    )
    
    # Process train pairs
    process_pairs(
        input_path=raw_dir / "pairs_train.jsonl",
        output_path=processed_dir / "train_pairs.jsonl"
    )
    
    # Process test pairs
    process_pairs(
        input_path=raw_dir / "pairs_test.jsonl",
        output_path=processed_dir / "test_pairs.jsonl"
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"""
Files created:
  ✅ {processed_dir / 'corpus_hyperbolic.jsonl'}
  ✅ {processed_dir / 'queries.jsonl'}
  ✅ {processed_dir / 'train_pairs.jsonl'}
  ✅ {processed_dir / 'test_pairs.jsonl'}

Existing files:
  📁 {processed_dir / 'hierarchy.json'}
  📁 {processed_dir / 'chunk_depths.json'}

Next steps:
  - Run scripts/train_hyperbolic.py to train Poincaré embeddings
  - Run scripts/build_hyperbolic_index.py to build FAISS index
""")


if __name__ == "__main__":
    main()
