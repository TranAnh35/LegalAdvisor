#!/usr/bin/env python3
"""
📊 Deep Analysis: ID Patterns & Document Structure

Phân tích chi tiết cấu trúc ID và nhóm documents để chuẩn bị cho HyperbolicRAG.

Output:
- data/analysis/id_patterns_analysis.json
- data/analysis/document_structure.json

Usage:
    python scripts/analyze_id_patterns.py
"""

import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))


class IDPatternAnalyzer:
    """Phân tích chi tiết cấu trúc ID của corpus."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_records = 0
        self.id_components = []
        self.documents = defaultdict(lambda: {
            "articles": [],
            "article_count": 0,
            "sample_titles": [],
            "doc_type": None,
            "year": None,
            "issuer": None
        })
        self.doc_types = Counter()
        self.years = Counter()
        self.issuers = Counter()
        self.depth_stats = Counter()
        self.errors = []
    
    def parse_id(self, doc_id: str) -> Dict[str, Any]:
        """
        Parse ID thành các thành phần.
        
        Pattern: {số}/{năm}/{loại-cơ_quan}+{điều}
        Examples:
            "91/2015/qh13+279" -> {doc_code: "91/2015/qh13", article: 279, ...}
            "130/2018/nđ-cp+36" -> {doc_code: "130/2018/nđ-cp", article: 36, ...}
        """
        result = {
            "raw_id": doc_id,
            "doc_code": None,
            "article": None,
            "doc_number": None,
            "year": None,
            "doc_type": None,
            "issuer": None,
            "is_valid": False
        }
        
        if not doc_id:
            return result
        
        # Split by "+" to separate document code and article
        parts = doc_id.split('+')
        if len(parts) == 2:
            result["doc_code"] = parts[0].strip()
            try:
                result["article"] = int(parts[1].strip())
            except ValueError:
                result["article"] = parts[1].strip()
        elif len(parts) == 1:
            result["doc_code"] = parts[0].strip()
        else:
            result["doc_code"] = '+'.join(parts[:-1])
            try:
                result["article"] = int(parts[-1].strip())
            except ValueError:
                result["article"] = parts[-1].strip()
        
        # Parse document code: {số}/{năm}/{loại}
        if result["doc_code"]:
            doc_parts = result["doc_code"].split('/')
            if len(doc_parts) >= 3:
                result["doc_number"] = doc_parts[0]
                result["year"] = doc_parts[1]
                result["doc_type"] = '/'.join(doc_parts[2:])  # Handle complex types
                result["is_valid"] = True
            elif len(doc_parts) == 2:
                result["doc_number"] = doc_parts[0]
                result["year"] = doc_parts[1]
                result["is_valid"] = True
        
        # Extract issuer from doc_type
        if result["doc_type"]:
            result["issuer"] = self._extract_issuer(result["doc_type"])
        
        return result
    
    def _extract_issuer(self, doc_type: str) -> str:
        """Extract cơ quan ban hành từ loại văn bản."""
        doc_type_lower = doc_type.lower()
        
        # Common issuers
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
    
    def get_doc_type_name(self, doc_type: str) -> str:
        """Chuyển mã loại văn bản thành tên đầy đủ."""
        if not doc_type:
            return "Không xác định"
        
        doc_type_lower = doc_type.lower()
        
        type_names = {
            "qh": "Luật",
            "nđ-cp": "Nghị định",
            "nd-cp": "Nghị định",
            "tt-": "Thông tư",
            "ttlt": "Thông tư liên tịch",
            "qđ-ttg": "Quyết định TTg",
            "qd-ttg": "Quyết định TTg",
            "pl-": "Pháp lệnh",
            "nq-": "Nghị quyết",
        }
        
        for pattern, name in type_names.items():
            if pattern in doc_type_lower:
                return name
        
        return doc_type.upper()
    
    def analyze_corpus(self, corpus_path: Path) -> Dict[str, Any]:
        """Phân tích toàn bộ corpus."""
        self.reset()
        
        print(f"📂 Analyzing: {corpus_path}")
        print("   Streaming through corpus...")
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    self._analyze_record(record)
                except json.JSONDecodeError as e:
                    self.errors.append({"line": line_num, "error": str(e)})
                
                if line_num % 10000 == 0:
                    print(f"   Processed: {line_num:,} records...", end='\r')
        
        print(f"   Processed: {self.total_records:,} records total")
        
        return self._compile_analysis()
    
    def _analyze_record(self, record: Dict[str, Any]):
        """Phân tích một record."""
        self.total_records += 1
        
        doc_id = record.get('_id', '')
        title = record.get('title', '')
        
        # Parse ID
        parsed = self.parse_id(doc_id)
        self.id_components.append(parsed)
        
        # Track document
        if parsed["doc_code"]:
            doc = self.documents[parsed["doc_code"]]
            if parsed["article"]:
                doc["articles"].append(parsed["article"])
                doc["article_count"] += 1
            if len(doc["sample_titles"]) < 3:
                doc["sample_titles"].append(title[:100])
            if not doc["doc_type"]:
                doc["doc_type"] = parsed["doc_type"]
            if not doc["year"]:
                doc["year"] = parsed["year"]
            if not doc["issuer"]:
                doc["issuer"] = parsed["issuer"]
        
        # Track doc types
        if parsed["doc_type"]:
            type_name = self.get_doc_type_name(parsed["doc_type"])
            self.doc_types[type_name] += 1
        
        # Track years
        if parsed["year"]:
            self.years[parsed["year"]] += 1
        
        # Track issuers
        if parsed["issuer"]:
            self.issuers[parsed["issuer"]] += 1
        
        # Track depth (based on article existence)
        if parsed["article"]:
            self.depth_stats["article_level"] += 1
        else:
            self.depth_stats["document_level"] += 1
    
    def _compile_analysis(self) -> Dict[str, Any]:
        """Tổng hợp kết quả phân tích."""
        # Calculate document statistics
        doc_stats = []
        for doc_code, info in self.documents.items():
            articles = info["articles"]
            doc_stats.append({
                "doc_code": doc_code,
                "article_count": len(articles),
                "min_article": min(articles) if articles else None,
                "max_article": max(articles) if articles else None,
                "doc_type": info["doc_type"],
                "year": info["year"],
                "issuer": info["issuer"],
                "sample_titles": info["sample_titles"]
            })
        
        # Sort by article count
        doc_stats.sort(key=lambda x: x["article_count"], reverse=True)
        
        # ID format analysis
        valid_ids = sum(1 for p in self.id_components if p["is_valid"])
        
        return {
            "analysis_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_records": self.total_records,
                "unique_documents": len(self.documents),
                "valid_id_format": valid_ids,
                "invalid_id_format": self.total_records - valid_ids,
                "avg_articles_per_doc": round(self.total_records / max(1, len(self.documents)), 2)
            },
            "document_types": dict(self.doc_types.most_common(30)),
            "years": dict(sorted(self.years.items())),
            "issuers": dict(self.issuers.most_common(20)),
            "depth_distribution": dict(self.depth_stats),
            "top_documents_by_articles": doc_stats[:50],
            "document_size_distribution": self._calculate_size_distribution(doc_stats),
            "id_format_examples": self._get_id_examples(),
            "hierarchy_structure": self._analyze_hierarchy_structure(),
            "errors": self.errors[:10]
        }
    
    def _calculate_size_distribution(self, doc_stats: List[Dict]) -> Dict[str, int]:
        """Tính phân bố số điều trên mỗi văn bản."""
        ranges = {
            "1-10 articles": 0,
            "11-50 articles": 0,
            "51-100 articles": 0,
            "101-200 articles": 0,
            "201-500 articles": 0,
            "500+ articles": 0
        }
        
        for doc in doc_stats:
            count = doc["article_count"]
            if count <= 10:
                ranges["1-10 articles"] += 1
            elif count <= 50:
                ranges["11-50 articles"] += 1
            elif count <= 100:
                ranges["51-100 articles"] += 1
            elif count <= 200:
                ranges["101-200 articles"] += 1
            elif count <= 500:
                ranges["201-500 articles"] += 1
            else:
                ranges["500+ articles"] += 1
        
        return ranges
    
    def _get_id_examples(self) -> Dict[str, List[str]]:
        """Lấy ví dụ cho các loại ID."""
        examples = defaultdict(list)
        
        for parsed in self.id_components[:1000]:  # Sample first 1000
            if parsed["doc_type"] and len(examples[parsed["doc_type"]]) < 3:
                examples[parsed["doc_type"]].append(parsed["raw_id"])
        
        return dict(examples)
    
    def _analyze_hierarchy_structure(self) -> Dict[str, Any]:
        """Phân tích cấu trúc hierarchy có thể xây dựng."""
        # Group documents by issuer
        issuer_docs = defaultdict(set)
        for doc_code, info in self.documents.items():
            if info["issuer"]:
                issuer_docs[info["issuer"]].add(doc_code)
        
        # Group by year
        year_docs = defaultdict(set)
        for doc_code, info in self.documents.items():
            if info["year"]:
                year_docs[info["year"]].add(doc_code)
        
        return {
            "proposed_hierarchy": {
                "L0_root": "Legal Corpus",
                "L1_issuer": f"{len(issuer_docs)} unique issuers",
                "L2_doc_type": f"{len(self.doc_types)} document types",
                "L3_document": f"{len(self.documents)} unique documents",
                "L4_article": f"{self.depth_stats.get('article_level', 0)} articles",
            },
            "issuers_summary": {k: len(v) for k, v in issuer_docs.items()},
            "years_range": {
                "min": min(self.years.keys()) if self.years else None,
                "max": max(self.years.keys()) if self.years else None
            }
        }


def print_analysis(analysis: Dict[str, Any]):
    """In kết quả phân tích."""
    print("\n" + "=" * 70)
    print("📋 ID PATTERNS & DOCUMENT STRUCTURE ANALYSIS")
    print("=" * 70)
    
    summary = analysis.get("summary", {})
    print(f"\n📊 SUMMARY:")
    print(f"   Total records: {summary.get('total_records', 0):,}")
    print(f"   Unique documents: {summary.get('unique_documents', 0):,}")
    print(f"   Valid ID format: {summary.get('valid_id_format', 0):,}")
    print(f"   Avg articles/doc: {summary.get('avg_articles_per_doc', 0)}")
    
    print(f"\n📁 DOCUMENT TYPES (top 15):")
    for dtype, count in list(analysis.get("document_types", {}).items())[:15]:
        print(f"   {dtype}: {count:,}")
    
    print(f"\n🏛️ ISSUERS (top 10):")
    for issuer, count in list(analysis.get("issuers", {}).items())[:10]:
        print(f"   {issuer}: {count:,}")
    
    print(f"\n📅 YEARS (showing range):")
    years = analysis.get("years", {})
    if years:
        year_list = sorted(years.keys())
        print(f"   Range: {year_list[0]} - {year_list[-1]}")
        print(f"   Most common: ", end="")
        top_years = sorted(years.items(), key=lambda x: x[1], reverse=True)[:5]
        print(", ".join([f"{y}: {c}" for y, c in top_years]))
    
    print(f"\n📚 TOP DOCUMENTS BY ARTICLE COUNT:")
    for doc in analysis.get("top_documents_by_articles", [])[:10]:
        print(f"   {doc['doc_code']}: {doc['article_count']} articles")
        print(f"      Type: {doc['doc_type']}, Year: {doc['year']}")
    
    print(f"\n📏 DOCUMENT SIZE DISTRIBUTION:")
    for range_name, count in analysis.get("document_size_distribution", {}).items():
        print(f"   {range_name}: {count}")
    
    print(f"\n🌳 PROPOSED HIERARCHY:")
    hierarchy = analysis.get("hierarchy_structure", {}).get("proposed_hierarchy", {})
    for level, desc in hierarchy.items():
        print(f"   {level}: {desc}")
    
    print("\n" + "=" * 70)


def main():
    project_root = Path(__file__).parent.parent
    corpus_path = project_root / "data" / "raw" / "zalo_ai_legal_text_retrieval" / "corpus.jsonl"
    output_dir = project_root / "data" / "analysis"
    
    if not corpus_path.exists():
        print(f"Error: Corpus not found at {corpus_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("📊 ID PATTERNS & DOCUMENT STRUCTURE ANALYZER")
    print("=" * 70)
    
    analyzer = IDPatternAnalyzer()
    analysis = analyzer.analyze_corpus(corpus_path)
    
    # Print results
    print_analysis(analysis)
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "id_patterns_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Analysis saved to: {output_file}")
    
    # Save document structure separately (detailed)
    doc_structure_file = output_dir / "document_structure.json"
    doc_structure = {
        "timestamp": analysis["analysis_timestamp"],
        "total_documents": len(analyzer.documents),
        "documents": {
            doc_code: {
                "article_count": info["article_count"],
                "articles": sorted(set(info["articles"])) if info["articles"] else [],
                "doc_type": info["doc_type"],
                "year": info["year"],
                "issuer": info["issuer"]
            }
            for doc_code, info in analyzer.documents.items()
        }
    }
    with open(doc_structure_file, 'w', encoding='utf-8') as f:
        json.dump(doc_structure, f, ensure_ascii=False, indent=2)
    print(f"✅ Document structure saved to: {doc_structure_file}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("💡 INSIGHTS FOR HYPERBOLICRAG")
    print("=" * 70)
    print(f"""
Dựa trên phân tích ID patterns:

1. ID FORMAT CHUẨN: {{số}}/{{năm}}/{{loại}}+{{điều}}
   - Có thể extract hierarchy trực tiếp từ ID
   - Không cần parse text phức tạp

2. HIERARCHY ĐỀ XUẤT:
   L0: Root (Legal Corpus)
   L1: Issuer ({len(analyzer.issuers)} issuers)
   L2: Document Type ({len(analyzer.doc_types)} types)
   L3: Document ({len(analyzer.documents)} documents)
   L4: Article ({analyzer.depth_stats.get('article_level', 0)} articles)

3. DEPTH ASSIGNMENT:
   - Documents: depth = 0.3 (closer to center)
   - Articles: depth = 0.6-0.8 (farther from center)
   - Specific clauses: depth = 0.9 (edge)

4. NEXT STEP:
   - Xây dựng hierarchy.json từ document_structure.json
   - Không cần phân tích text, chỉ cần parse ID
""")


if __name__ == "__main__":
    main()
