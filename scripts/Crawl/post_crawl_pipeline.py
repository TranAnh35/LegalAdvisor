#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-Crawl Pipeline: Chuyển đổi dữ liệu crawled thành format HyperbolicRAG.

Đây là bước cầu nối giữa Crawler và Training Pipeline.

Input (từ Scrapy):
    data/raw/crawled/documents.jsonl   (Parent nodes)
    data/raw/crawled/articles.jsonl    (Child nodes)

Output (cho HyperbolicRAG):
    data/processed/zalo-legal/corpus_hyperbolic.jsonl   (Unified corpus - updated)
    data/processed/zalo-legal/hierarchy.json             (Hierarchy tree - updated)

Logic:
1. Merge dữ liệu crawled mới với corpus Zalo cũ (nếu có)
2. Gán chunk_id mới, depth, issuer
3. Rebuild hierarchy.json
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import OrderedDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ==============================================================================
# ISSUER DETECTION (từ process_dataset.py hiện tại)
# ==============================================================================
ISSUER_PATTERNS = {
    "qh":       "Quốc hội",
    "ubtvqh":   "UBTVQH",
    "cp":       "Chính phủ",
    "ttg":      "Thủ tướng",
    "btc":      "Bộ Tài chính",
    "bct":      "Bộ Công Thương",
    "btp":      "Bộ Tư pháp",
    "bca":      "Bộ Công an",
    "bgtvt":    "Bộ GTVT",
    "byt":      "Bộ Y tế",
    "bgdđt":    "Bộ GD&ĐT",
    "bxd":      "Bộ Xây dựng",
    "btnmt":    "Bộ TN&MT",
    "bnnptnt":  "Bộ NN&PTNT",
    "blđtbxh":  "Bộ LĐ-TB&XH",
    "btttt":    "Bộ TT&TT",
    "bqp":      "Bộ Quốc phòng",
    "bnv":      "Bộ Nội vụ",
    "nhnn":     "Ngân hàng Nhà nước",
}


def detect_issuer(doc_code: str, doc_type_raw: str = "") -> str:
    """Phát hiện cơ quan ban hành từ doc_code hoặc doc_type."""
    text = (doc_code + " " + doc_type_raw).lower()
    for pattern, issuer in ISSUER_PATTERNS.items():
        if pattern in text:
            return issuer
    return "Khác"


def extract_year(doc_code: str) -> str:
    """Trích xuất năm từ doc_code."""
    parts = doc_code.split("/")
    for p in parts:
        p = p.strip()
        if len(p) == 4 and p.isdigit():
            return p
    return ""


def extract_doc_type_short(doc_code: str) -> str:
    """Trích xuất loại văn bản viết tắt từ doc_code."""
    parts = doc_code.split("/")
    type_parts = [p for p in parts[2:] if p.strip()] if len(parts) >= 3 else []
    return "/".join(type_parts) if type_parts else ""


# ==============================================================================
# DEPTH ASSIGNMENT
# ==============================================================================
DEPTH_MAP = {
    "Quốc hội":     0.60,
    "UBTVQH":       0.62,
    "Chính phủ":    0.65,
    "Thủ tướng":    0.68,
}


def assign_depth(issuer: str, hierarchy_path: list = None) -> float:
    """Gán depth cho Poincaré based on hierarchy và authority."""
    base = DEPTH_MAP.get(issuer, 0.75)
    # Điều luật con nằm sâu hơn document
    if hierarchy_path:
        # Phần sâu thêm: +0.03 mỗi tầng phân cấp
        extra = min(len(hierarchy_path) * 0.03, 0.15)
        return min(base + 0.10 + extra, 0.95)
    return base


# ==============================================================================
# MAIN MERGE LOGIC
# ==============================================================================
def merge_and_build(
    crawled_dir: Path,
    existing_corpus_path: Path,
    output_corpus_path: Path,
    output_hierarchy_path: Path,
):
    """
    Merge crawled data với corpus hiện có → tạo ra corpus_hyperbolic.jsonl mới.
    """
    print("=" * 60)
    print("📦 POST-CRAWL PIPELINE: Merge & Build HyperbolicRAG Data")
    print("=" * 60)

    # --- 1. Load existing corpus ---
    existing_records: Dict[str, Dict] = OrderedDict()
    existing_doc_codes: Set[str] = set()

    if existing_corpus_path.exists():
        print(f"\n📂 Loading existing corpus: {existing_corpus_path}")
        with open(existing_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    cid = record.get("corpus_id", "")
                    if cid:
                        existing_records[cid] = record
                        existing_doc_codes.add(cid.split("+")[0])
                except json.JSONDecodeError:
                    continue
        print(f"   ✅ Loaded {len(existing_records):,} existing articles")
        print(f"   📁 From {len(existing_doc_codes):,} unique documents")

    # --- 2. Load crawled documents ---
    docs_path = crawled_dir / "documents.jsonl"
    articles_path = crawled_dir / "articles.jsonl"

    crawled_docs: Dict[str, Dict] = {}
    if docs_path.exists():
        print(f"\n📂 Loading crawled documents: {docs_path}")
        with open(docs_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    doc_code = doc.get("doc_code", "").strip()
                    if doc_code:
                        crawled_docs[doc_code] = doc
                except json.JSONDecodeError:
                    continue
        print(f"   ✅ {len(crawled_docs):,} crawled documents")

    crawled_articles = []
    new_count = 0
    updated_count = 0

    if articles_path.exists():
        print(f"📂 Loading crawled articles: {articles_path}")
        with open(articles_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    art = json.loads(line)
                    doc_code = art.get("doc_code", "").strip()
                    art_num = art.get("article_number")
                    content = art.get("content", "").strip()

                    if not doc_code or art_num is None or not content:
                        continue

                    corpus_id = f"{doc_code}+{art_num}"

                    # Xây record mới
                    issuer = detect_issuer(
                        doc_code,
                        crawled_docs.get(doc_code, {}).get("doc_type", ""),
                    )
                    year = extract_year(doc_code)
                    doc_type_short = extract_doc_type_short(doc_code)
                    hierarchy_path = art.get("hierarchy_path", [])
                    depth = assign_depth(issuer, hierarchy_path)

                    new_record = {
                        "corpus_id": corpus_id,
                        "doc_code": doc_code,
                        "article": art_num,
                        "title": art.get("title", f"Điều {art_num}"),
                        "content": content,
                        "depth": round(depth, 3),
                        "issuer": issuer,
                        "doc_type": doc_type_short,
                        "year": year,
                        "hierarchy_path": hierarchy_path,
                    }

                    if corpus_id in existing_records:
                        updated_count += 1
                    else:
                        new_count += 1

                    existing_records[corpus_id] = new_record

                except json.JSONDecodeError:
                    continue

        print(f"   ✅ {new_count:,} NEW articles added")
        print(f"   🔄 {updated_count:,} existing articles updated")

    # --- 3. Re-assign chunk_ids và write corpus ---
    print(f"\n📝 Writing merged corpus: {output_corpus_path}")
    output_corpus_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_corpus_path, "w", encoding="utf-8") as f:
        for chunk_id, (corpus_id, record) in enumerate(existing_records.items()):
            record["chunk_id"] = chunk_id
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(existing_records)
    print(f"   ✅ {total:,} total articles in corpus")

    # --- 4. Build hierarchy.json ---
    print(f"\n🌳 Building hierarchy tree: {output_hierarchy_path}")

    hierarchy = {
        "metadata": {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "total_articles": total,
            "total_documents": len(set(
                r["doc_code"] for r in existing_records.values()
            )),
        },
        "nodes": {},
        "edges": [],
        "statistics": {},
    }

    # Root node
    hierarchy["nodes"]["root"] = {
        "id": "root",
        "type": "root",
        "label": "Vietnamese Legal Corpus",
    }

    # Build nodes and edges
    doc_codes_seen = set()
    issuer_nodes = set()

    for corpus_id, record in existing_records.items():
        doc_code = record.get("doc_code", "")
        art_num = record.get("article")
        issuer = record.get("issuer", "Khác")

        # Issuer node
        issuer_id = f"issuer:{issuer}"
        if issuer_id not in issuer_nodes:
            hierarchy["nodes"][issuer_id] = {
                "id": issuer_id,
                "type": "issuer",
                "label": issuer,
            }
            hierarchy["edges"].append({
                "source": "root",
                "target": issuer_id,
                "relation": "has_issuer",
            })
            issuer_nodes.add(issuer_id)

        # Document node
        doc_id = f"doc:{doc_code}"
        if doc_code not in doc_codes_seen:
            doc_title = ""
            if doc_code in crawled_docs:
                doc_title = crawled_docs[doc_code].get("title", "")
            hierarchy["nodes"][doc_id] = {
                "id": doc_id,
                "type": "document",
                "label": doc_title or doc_code,
                "doc_code": doc_code,
            }
            hierarchy["edges"].append({
                "source": issuer_id,
                "target": doc_id,
                "relation": "issued_by",
            })
            doc_codes_seen.add(doc_code)

        # Article node
        art_id = f"article:{corpus_id}"
        hierarchy["nodes"][art_id] = {
            "id": art_id,
            "type": "article",
            "label": record.get("title", f"Điều {art_num}"),
            "doc_code": doc_code,
            "article_num": art_num,
            "corpus_id": corpus_id,
            "chunk_id": record.get("chunk_id"),
        }
        hierarchy["edges"].append({
            "source": doc_id,
            "target": art_id,
            "relation": "has_article",
        })

    # Statistics
    hierarchy["statistics"] = {
        "total_nodes": len(hierarchy["nodes"]),
        "total_edges": len(hierarchy["edges"]),
        "total_issuers": len(issuer_nodes),
        "total_documents": len(doc_codes_seen),
        "total_articles": total,
    }

    with open(output_hierarchy_path, "w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)

    print(f"   ✅ {hierarchy['statistics']['total_nodes']:,} nodes")
    print(f"   ✅ {hierarchy['statistics']['total_edges']:,} edges")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"""
Total corpus articles: {total:,}
Total documents:       {len(doc_codes_seen):,}
New articles added:    {new_count:,}
Updated articles:      {updated_count:,}

Output files:
  ✅ {output_corpus_path}
  ✅ {output_hierarchy_path}

Next steps:
  1. Run train_hyperbolic.py to retrain with new data
  2. Run build_hyperbolic_index.py to rebuild index
""")


def main():
    import argparse

    parser = argparse.ArgumentParser("Post-Crawl Pipeline")
    parser.add_argument(
        "--crawled-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "crawled",
    )
    parser.add_argument(
        "--existing-corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "zalo-legal" / "corpus_hyperbolic.jsonl",
    )
    parser.add_argument(
        "--output-corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "zalo-legal" / "corpus_hyperbolic.jsonl",
    )
    parser.add_argument(
        "--output-hierarchy",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "zalo-legal" / "hierarchy.json",
    )
    args = parser.parse_args()

    merge_and_build(
        crawled_dir=args.crawled_dir,
        existing_corpus_path=args.existing_corpus,
        output_corpus_path=args.output_corpus,
        output_hierarchy_path=args.output_hierarchy,
    )


if __name__ == "__main__":
    main()
