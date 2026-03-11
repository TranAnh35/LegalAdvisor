#!/usr/bin/env python3
"""
🌳 Build Hierarchy for HyperbolicRAG

Xây dựng cấu trúc hierarchy từ kết quả phân tích ID patterns.
Sử dụng data từ document_structure.json (đã được tạo bởi analyze_id_patterns.py).

Hierarchy Structure:
    L0: Root (Legal Corpus)
    L1: Issuer (16 issuers) - depth = 0.1
    L2: Document Type (20 types) - depth = 0.2-0.3
    L3: Document (3271 documents) - depth = 0.4-0.5
    L4: Article (61425 articles) - depth = 0.6-0.9

Output:
    data/processed/zalo-legal/hierarchy.json

Usage:
    # First run analyze_id_patterns.py to generate document_structure.json
    python scripts/analyze_id_patterns.py
    
    # Then build hierarchy
    python scripts/build_hierarchy.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


# Depth assignments for Poincaré Ball
# Lower depth = closer to center = more general
# Higher depth = farther from center = more specific
DEPTH_CONFIG = {
    "root": 0.0,
    "issuer": 0.15,
    "doc_type": 0.30,
    "document": 0.50,
    "article_base": 0.65,
    "article_max": 0.90,
}


def load_document_structure(analysis_dir: Path) -> Dict[str, Any]:
    """Load document structure từ file đã phân tích."""
    structure_file = analysis_dir / "document_structure.json"
    
    if not structure_file.exists():
        print(f"❌ Error: document_structure.json not found!")
        print(f"   Please run: python scripts/analyze_id_patterns.py first")
        sys.exit(1)
    
    with open(structure_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_article_depth(article_num: int, max_article: int) -> float:
    """
    Tính depth cho article dựa trên số thứ tự.
    
    Articles đầu (Điều 1, 2, 3...) thường chứa định nghĩa chung -> depth thấp hơn
    Articles sau thường cụ thể hơn -> depth cao hơn
    """
    base = DEPTH_CONFIG["article_base"]
    max_depth = DEPTH_CONFIG["article_max"]
    
    if max_article <= 1:
        return base
    
    # Linear interpolation based on article position
    ratio = (article_num - 1) / max(1, max_article - 1)
    return base + ratio * (max_depth - base)


def build_hierarchy_from_structure(doc_structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Xây dựng hierarchy từ document structure.
    """
    print("🌳 Building hierarchy...")
    
    documents = doc_structure.get("documents", {})
    
    # Initialize hierarchy
    hierarchy = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "source": "document_structure.json",
            "total_documents": len(documents),
            "depth_config": DEPTH_CONFIG
        },
        "nodes": {},
        "edges": [],
        "statistics": {
            "by_level": defaultdict(int),
            "by_issuer": defaultdict(int),
            "by_doc_type": defaultdict(int)
        }
    }
    
    # Create root node
    hierarchy["nodes"]["root"] = {
        "id": "root",
        "type": "root",
        "label": "Vietnamese Legal Corpus",
        "depth": DEPTH_CONFIG["root"],
        "children_count": 0
    }
    
    # Track unique issuers and doc_types
    issuers = set()
    doc_types = set()
    
    # Process each document
    for doc_code, doc_info in documents.items():
        issuer = doc_info.get("issuer") or "Khác"
        doc_type = doc_info.get("doc_type") or "unknown"
        year = doc_info.get("year") or "unknown"
        articles = doc_info.get("articles", [])
        
        issuers.add(issuer)
        doc_types.add(doc_type)
        
        # Create issuer node if not exists
        issuer_id = f"issuer:{issuer}"
        if issuer_id not in hierarchy["nodes"]:
            hierarchy["nodes"][issuer_id] = {
                "id": issuer_id,
                "type": "issuer",
                "label": issuer,
                "depth": DEPTH_CONFIG["issuer"],
                "children_count": 0
            }
            hierarchy["edges"].append({
                "source": "root",
                "target": issuer_id,
                "relation": "has_issuer"
            })
            hierarchy["nodes"]["root"]["children_count"] += 1
        
        # Create doc_type node (under issuer)
        doctype_id = f"doctype:{issuer}:{doc_type}"
        if doctype_id not in hierarchy["nodes"]:
            hierarchy["nodes"][doctype_id] = {
                "id": doctype_id,
                "type": "doc_type",
                "label": doc_type.upper(),
                "depth": DEPTH_CONFIG["doc_type"],
                "issuer": issuer,
                "children_count": 0
            }
            hierarchy["edges"].append({
                "source": issuer_id,
                "target": doctype_id,
                "relation": "has_doctype"
            })
            hierarchy["nodes"][issuer_id]["children_count"] += 1
        
        # Create document node
        doc_id = f"doc:{doc_code}"
        max_article = max(articles) if articles else 0
        hierarchy["nodes"][doc_id] = {
            "id": doc_id,
            "type": "document",
            "label": doc_code,
            "depth": DEPTH_CONFIG["document"],
            "issuer": issuer,
            "doc_type": doc_type,
            "year": year,
            "article_count": len(articles),
            "children_count": len(articles)
        }
        hierarchy["edges"].append({
            "source": doctype_id,
            "target": doc_id,
            "relation": "has_document"
        })
        hierarchy["nodes"][doctype_id]["children_count"] += 1
        
        # Create article nodes
        for article_num in articles:
            article_id = f"article:{doc_code}+{article_num}"
            article_depth = compute_article_depth(article_num, max_article)
            
            hierarchy["nodes"][article_id] = {
                "id": article_id,
                "type": "article",
                "label": f"Điều {article_num}",
                "depth": round(article_depth, 4),
                "document": doc_code,
                "article_num": article_num,
                "corpus_id": f"{doc_code}+{article_num}"
            }
            hierarchy["edges"].append({
                "source": doc_id,
                "target": article_id,
                "relation": "has_article"
            })
        
        # Update statistics
        hierarchy["statistics"]["by_issuer"][issuer] += len(articles)
        hierarchy["statistics"]["by_doc_type"][doc_type] += len(articles)
    
    # Calculate level statistics
    for node_id, node in hierarchy["nodes"].items():
        node_type = node.get("type", "unknown")
        hierarchy["statistics"]["by_level"][node_type] += 1
    
    # Convert defaultdicts to regular dicts for JSON serialization
    hierarchy["statistics"]["by_level"] = dict(hierarchy["statistics"]["by_level"])
    hierarchy["statistics"]["by_issuer"] = dict(hierarchy["statistics"]["by_issuer"])
    hierarchy["statistics"]["by_doc_type"] = dict(hierarchy["statistics"]["by_doc_type"])
    
    return hierarchy


def create_chunk_depth_lookup(hierarchy: Dict[str, Any]) -> Dict[str, float]:
    """
    Tạo lookup table: corpus_id -> depth
    Dùng cho việc assign depth khi train hyperbolic embeddings.
    """
    lookup = {}
    
    for node_id, node in hierarchy["nodes"].items():
        if node.get("type") == "article":
            corpus_id = node.get("corpus_id")
            if corpus_id:
                lookup[corpus_id] = node.get("depth", 0.7)
    
    return lookup


def print_hierarchy_summary(hierarchy: Dict[str, Any]):
    """In tóm tắt hierarchy."""
    print("\n" + "=" * 60)
    print("📊 HIERARCHY SUMMARY")
    print("=" * 60)
    
    stats = hierarchy.get("statistics", {})
    
    print(f"\n🌲 Nodes by Level:")
    for level, count in stats.get("by_level", {}).items():
        print(f"   {level}: {count:,}")
    
    print(f"\n🏛️ Articles by Top Issuers:")
    by_issuer = stats.get("by_issuer", {})
    for issuer, count in sorted(by_issuer.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {issuer}: {count:,}")
    
    print(f"\n📑 Articles by Top Doc Types:")
    by_type = stats.get("by_doc_type", {})
    for dtype, count in sorted(by_type.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"   {dtype}: {count:,}")
    
    print(f"\n📐 Depth Configuration:")
    for level, depth in DEPTH_CONFIG.items():
        print(f"   {level}: {depth}")
    
    total_nodes = len(hierarchy.get("nodes", {}))
    total_edges = len(hierarchy.get("edges", []))
    print(f"\n📈 Total: {total_nodes:,} nodes, {total_edges:,} edges")


def main():
    project_root = Path(__file__).parent.parent
    analysis_dir = project_root / "data" / "analysis"
    output_dir = project_root / "data" / "processed" / "zalo-legal"
    
    print("=" * 60)
    print("🌳 HIERARCHY BUILDER FOR HYPERBOLICRAG")
    print("=" * 60)
    
    # Load document structure
    print("\n📂 Loading document structure...")
    doc_structure = load_document_structure(analysis_dir)
    print(f"   Found {doc_structure.get('total_documents', 0):,} documents")
    
    # Build hierarchy
    hierarchy = build_hierarchy_from_structure(doc_structure)
    
    # Print summary
    print_hierarchy_summary(hierarchy)
    
    # Save hierarchy
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hierarchy.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Hierarchy saved to: {output_file}")
    
    # Save depth lookup (smaller file for training)
    depth_lookup = create_chunk_depth_lookup(hierarchy)
    lookup_file = output_dir / "chunk_depths.json"
    
    with open(lookup_file, 'w', encoding='utf-8') as f:
        json.dump(depth_lookup, f, ensure_ascii=False)
    print(f"✅ Depth lookup saved to: {lookup_file}")
    print(f"   ({len(depth_lookup):,} corpus_id -> depth mappings)")
    
    # Next steps
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)
    print("""
1. Hierarchy đã sẵn sàng cho HyperbolicRAG training.

2. Files được tạo:
   - hierarchy.json: Full hierarchy với nodes và edges
   - chunk_depths.json: Lookup table corpus_id -> depth

3. Tiếp theo:
   - Chạy train_hyperbolic.py để train Poincaré embeddings
   - Sử dụng chunk_depths.json để assign depth trong training
""")


if __name__ == "__main__":
    main()
