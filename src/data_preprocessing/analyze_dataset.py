#!/usr/bin/env python3
"""
📊 Dataset Analysis Script cho LegalAdvisor

Thực hiện phân tích toàn diện dataset Zalo Legal mà KHÔNG đọc toàn bộ vào RAM.

Chiến lược:
1. Streaming: Đọc từng dòng, không load toàn bộ file
2. Sampling: Lấy mẫu ngẫu nhiên để phân tích chi tiết
3. Statistics: Tính toán thống kê on-the-fly

Output:
- Console: Báo cáo phân tích
- File: data/analysis/dataset_analysis_report.json

Usage:
    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --sample-size 1000
    python scripts/analyze_dataset.py --output-dir data/analysis
"""

import json
import sys
import argparse
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import statistics

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent.parent))


class StreamingAnalyzer:
    """Phân tích file JSONL bằng streaming (không load toàn bộ vào RAM)."""
    
    def __init__(self, sample_size: int = 500):
        self.sample_size = sample_size
        self.reset_stats()
    
    def reset_stats(self):
        """Reset tất cả statistics."""
        self.total_records = 0
        self.total_bytes = 0
        self.field_counts = Counter()
        self.field_types = defaultdict(Counter)
        self.samples = []
        self.content_lengths = []
        self.errors = []
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Phân tích một file JSONL.
        
        Sử dụng reservoir sampling để lấy mẫu ngẫu nhiên.
        """
        self.reset_stats()
        file_size = file_path.stat().st_size
        
        print(f"\n📂 Analyzing: {file_path.name}")
        print(f"   File size: {file_size / 1024 / 1024:.2f} MB")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                self.total_bytes += len(line.encode('utf-8'))
                
                try:
                    record = json.loads(line)
                    self._analyze_record(record, line_num)
                except json.JSONDecodeError as e:
                    self.errors.append({
                        "line": line_num,
                        "error": str(e),
                        "preview": line[:100]
                    })
                
                # Progress indicator for large files
                if line_num % 10000 == 0:
                    print(f"   Processed: {line_num:,} records...", end='\r')
        
        print(f"   Processed: {self.total_records:,} records total")
        
        return self._compile_report(file_path)
    
    def _analyze_record(self, record: Dict[str, Any], line_num: int):
        """Phân tích một record."""
        self.total_records += 1
        
        # Track fields
        for field, value in record.items():
            self.field_counts[field] += 1
            self.field_types[field][type(value).__name__] += 1
        
        # Track content length nếu có
        content = record.get('text') or record.get('content') or record.get('question') or ''
        if content:
            self.content_lengths.append(len(content))
        
        # Reservoir sampling cho samples
        if len(self.samples) < self.sample_size:
            self.samples.append(record)
        else:
            # Replace với xác suất giảm dần
            j = random.randint(0, self.total_records - 1)
            if j < self.sample_size:
                self.samples[j] = record
    
    def _compile_report(self, file_path: Path) -> Dict[str, Any]:
        """Tổng hợp báo cáo phân tích."""
        report = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_size_bytes": file_path.stat().st_size,
            "file_size_mb": round(file_path.stat().st_size / 1024 / 1024, 2),
            "total_records": self.total_records,
            "avg_record_bytes": round(self.total_bytes / max(1, self.total_records), 2),
            "parse_errors": len(self.errors),
            "schema": self._analyze_schema(),
            "content_stats": self._content_statistics(),
            "sample_records": self.samples[:5],  # Chỉ lưu 5 samples vào report
            "errors": self.errors[:10]  # Chỉ lưu 10 errors đầu
        }
        
        return report
    
    def _analyze_schema(self) -> Dict[str, Any]:
        """Phân tích schema của dataset."""
        schema = {}
        for field, count in self.field_counts.items():
            schema[field] = {
                "count": count,
                "presence_rate": round(count / max(1, self.total_records) * 100, 2),
                "types": dict(self.field_types[field])
            }
        return schema
    
    def _content_statistics(self) -> Dict[str, Any]:
        """Thống kê về content/text."""
        if not self.content_lengths:
            return {}
        
        return {
            "min_length": min(self.content_lengths),
            "max_length": max(self.content_lengths),
            "mean_length": round(statistics.mean(self.content_lengths), 2),
            "median_length": round(statistics.median(self.content_lengths), 2),
            "stdev_length": round(statistics.stdev(self.content_lengths), 2) if len(self.content_lengths) > 1 else 0,
            "total_samples": len(self.content_lengths)
        }


class LegalDatasetAnalyzer:
    """Phân tích chuyên sâu cho Zalo Legal Dataset."""
    
    def __init__(self, data_dir: Path, sample_size: int = 500):
        self.data_dir = data_dir
        self.sample_size = sample_size
        self.streaming_analyzer = StreamingAnalyzer(sample_size)
    
    def analyze_corpus(self, file_path: Path) -> Dict[str, Any]:
        """Phân tích corpus.jsonl - file chính chứa văn bản luật."""
        base_report = self.streaming_analyzer.analyze_file(file_path)
        
        # Phân tích chuyên sâu từ samples
        samples = self.streaming_analyzer.samples
        
        # 1. Phân tích loại văn bản
        doc_types = self._analyze_document_types(samples)
        
        # 2. Phân tích cấu trúc hierarchy
        hierarchy_analysis = self._analyze_hierarchy_patterns(samples)
        
        # 3. Phân tích metadata
        metadata_analysis = self._analyze_metadata(samples)
        
        # 4. Phân tích nội dung text
        text_patterns = self._analyze_text_patterns(samples)
        
        base_report["legal_analysis"] = {
            "document_types": doc_types,
            "hierarchy_patterns": hierarchy_analysis,
            "metadata": metadata_analysis,
            "text_patterns": text_patterns
        }
        
        return base_report
    
    def _analyze_document_types(self, samples: List[Dict]) -> Dict[str, Any]:
        """Phân tích các loại văn bản pháp luật."""
        type_counter = Counter()
        id_patterns = Counter()
        
        for record in samples:
            # Tìm ID hoặc corpus_id
            doc_id = record.get('id') or record.get('corpus_id') or ''
            
            # Extract type từ ID
            doc_type = self._extract_doc_type(doc_id)
            type_counter[doc_type] += 1
            
            # Track ID patterns
            pattern = self._get_id_pattern(doc_id)
            id_patterns[pattern] += 1
        
        return {
            "types": dict(type_counter.most_common(20)),
            "id_patterns": dict(id_patterns.most_common(10)),
            "sample_size": len(samples)
        }
    
    def _extract_doc_type(self, doc_id: str) -> str:
        """Extract loại văn bản từ ID."""
        if not doc_id:
            return "unknown"
        
        doc_lower = doc_id.lower()
        
        # Map các patterns thường gặp
        type_patterns = [
            (r'blds|bo-luat-dan-su', 'Bộ luật Dân sự'),
            (r'blhs|bo-luat-hinh-su', 'Bộ luật Hình sự'),
            (r'bllđ|bo-luat-lao-dong', 'Bộ luật Lao động'),
            (r'luat-|luật-', 'Luật'),
            (r'nd-cp|nghi-dinh', 'Nghị định'),
            (r'tt-|thong-tu', 'Thông tư'),
            (r'ttlt', 'Thông tư liên tịch'),
            (r'qd-ttg|quyet-dinh', 'Quyết định'),
            (r'pl-', 'Pháp lệnh'),
        ]
        
        for pattern, doc_type in type_patterns:
            if re.search(pattern, doc_lower):
                return doc_type
        
        return "Khác"
    
    def _get_id_pattern(self, doc_id: str) -> str:
        """Tìm pattern của ID để hiểu cấu trúc."""
        if not doc_id:
            return "empty"
        
        # Thay thế số bằng {N}, chữ bằng {S}
        pattern = re.sub(r'\d+', '{N}', doc_id)
        pattern = re.sub(r'[a-zA-Z]+', '{S}', pattern)
        
        return pattern[:50]  # Truncate nếu quá dài
    
    def _analyze_hierarchy_patterns(self, samples: List[Dict]) -> Dict[str, Any]:
        """Phân tích cấu trúc phân cấp trong văn bản."""
        hierarchy_markers = {
            "phần": 0,
            "chương": 0,
            "mục": 0,
            "điều": 0,
            "khoản": 0,
            "điểm": 0
        }
        
        depth_indicators = Counter()
        
        for record in samples:
            text = record.get('text') or record.get('content') or ''
            text_lower = text.lower()
            
            # Đếm các markers
            for marker in hierarchy_markers:
                if marker in text_lower:
                    hierarchy_markers[marker] += 1
            
            # Phân tích depth từ structure
            first_line = text.split('\n')[0].lower() if text else ''
            depth = self._detect_depth(first_line)
            depth_indicators[depth] += 1
        
        return {
            "hierarchy_markers": hierarchy_markers,
            "depth_distribution": dict(depth_indicators),
            "sample_size": len(samples)
        }
    
    def _detect_depth(self, first_line: str) -> str:
        """Detect depth level từ dòng đầu tiên."""
        patterns = [
            (r'^phần\s+(thứ\s+)?[ivxlcdm\d]+', 'L2_Phần'),
            (r'^chương\s+[ivxlcdm\d]+', 'L3_Chương'),
            (r'^mục\s+\d+', 'L4_Mục'),
            (r'^điều\s+\d+', 'L5_Điều'),
            (r'^\d+[\.\)]\s', 'L6_Khoản'),
            (r'^[a-zđ][\.\)]\s', 'L7_Điểm'),
        ]
        
        for pattern, level in patterns:
            if re.search(pattern, first_line):
                return level
        
        return 'L5_Điều'  # Default
    
    def _analyze_metadata(self, samples: List[Dict]) -> Dict[str, Any]:
        """Phân tích các field metadata."""
        fields_found = set()
        field_examples = defaultdict(list)
        
        for record in samples:
            for field, value in record.items():
                fields_found.add(field)
                
                # Lưu examples (tối đa 3)
                if len(field_examples[field]) < 3:
                    if isinstance(value, str):
                        field_examples[field].append(value[:100])
                    else:
                        field_examples[field].append(str(value)[:100])
        
        return {
            "fields": list(fields_found),
            "field_count": len(fields_found),
            "field_examples": dict(field_examples)
        }
    
    def _analyze_text_patterns(self, samples: List[Dict]) -> Dict[str, Any]:
        """Phân tích patterns trong text content."""
        # Các patterns pháp lý thường gặp
        legal_patterns = {
            "tham_chieu_dieu": r'(theo|căn cứ|quy định tại)\s+điều\s+\d+',
            "tham_chieu_khoan": r'khoản\s+\d+\s+(điều|của)',
            "so_hieu_van_ban": r'\d+/\d{4}/[A-Z\-]+',
            "ngay_thang": r'\d{1,2}/\d{1,2}/\d{4}',
            "phan_tram": r'\d+(\.\d+)?%',
            "tien_te": r'\d+[\.\,]?\d*\s*(đồng|vnđ|vnd)',
        }
        
        pattern_counts = Counter()
        reference_targets = []  # Các điều được tham chiếu
        
        for record in samples:
            text = record.get('text') or record.get('content') or ''
            text_lower = text.lower()
            
            for pattern_name, pattern in legal_patterns.items():
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                pattern_counts[pattern_name] += len(matches) if matches else 0
            
            # Extract các điều được tham chiếu
            refs = re.findall(r'điều\s+(\d+)', text_lower)
            reference_targets.extend(refs[:5])  # Limit per record
        
        # Top referenced articles
        ref_counter = Counter(reference_targets)
        
        return {
            "pattern_counts": dict(pattern_counts),
            "top_referenced_articles": dict(ref_counter.most_common(20)),
            "sample_size": len(samples)
        }
    
    def analyze_queries(self, file_path: Path) -> Dict[str, Any]:
        """Phân tích queries.jsonl."""
        base_report = self.streaming_analyzer.analyze_file(file_path)
        samples = self.streaming_analyzer.samples
        
        # Phân tích câu hỏi
        question_types = Counter()
        question_lengths = []
        
        for record in samples:
            question = record.get('question') or record.get('text') or ''
            question_lengths.append(len(question))
            
            # Phân loại câu hỏi
            q_type = self._classify_question(question)
            question_types[q_type] += 1
        
        base_report["query_analysis"] = {
            "question_types": dict(question_types),
            "avg_question_length": round(statistics.mean(question_lengths), 2) if question_lengths else 0,
            "sample_size": len(samples)
        }
        
        return base_report
    
    def _classify_question(self, question: str) -> str:
        """Phân loại câu hỏi theo intent."""
        q_lower = question.lower()
        
        patterns = [
            (r'^(thế nào|như thế nào|làm sao)', 'how'),
            (r'^(là gì|gì là)', 'what'),
            (r'^(tại sao|vì sao)', 'why'),
            (r'^(khi nào|bao giờ)', 'when'),
            (r'^(ở đâu|nơi nào)', 'where'),
            (r'^(ai |người nào)', 'who'),
            (r'(có được|được không|có thể)', 'can/permission'),
            (r'(phải|bắt buộc|nghĩa vụ)', 'obligation'),
            (r'(quy định|theo luật)', 'regulation'),
            (r'(xử phạt|phạt|hình phạt)', 'penalty'),
        ]
        
        for pattern, q_type in patterns:
            if re.search(pattern, q_lower):
                return q_type
        
        return 'other'
    
    def analyze_pairs(self, file_path: Path) -> Dict[str, Any]:
        """Phân tích pairs (train/test)."""
        base_report = self.streaming_analyzer.analyze_file(file_path)
        samples = self.streaming_analyzer.samples
        
        # Check relationship structure
        relation_types = Counter()
        
        for record in samples:
            # Xác định loại relationship
            if 'query_id' in record and 'corpus_id' in record:
                relation_types['query-corpus'] += 1
            elif 'question' in record and 'answer' in record:
                relation_types['qa'] += 1
            else:
                relation_types['other'] += 1
        
        base_report["pair_analysis"] = {
            "relation_types": dict(relation_types),
            "sample_size": len(samples)
        }
        
        return base_report
    
    def full_analysis(self) -> Dict[str, Any]:
        """Thực hiện phân tích toàn bộ dataset."""
        raw_dir = self.data_dir / "raw" / "zalo_ai_legal_text_retrieval"
        
        print("=" * 70)
        print("📊 ZALO LEGAL DATASET ANALYSIS")
        print("=" * 70)
        print(f"Data directory: {raw_dir}")
        print(f"Sample size per file: {self.sample_size}")
        print("=" * 70)
        
        reports = {
            "analysis_timestamp": datetime.now().isoformat(),
            "sample_size": self.sample_size,
            "files": {}
        }
        
        # Analyze each file
        files_to_analyze = [
            ("corpus.jsonl", self.analyze_corpus),
            ("queries.jsonl", self.analyze_queries),
            ("pairs_train.jsonl", self.analyze_pairs),
            ("pairs_test.jsonl", self.analyze_pairs),
        ]
        
        for filename, analyzer_func in files_to_analyze:
            file_path = raw_dir / filename
            if file_path.exists():
                report = analyzer_func(file_path)
                reports["files"][filename] = report
            else:
                print(f"⚠️ File not found: {filename}")
        
        # Summary
        reports["summary"] = self._generate_summary(reports)
        
        return reports
    
    def _generate_summary(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Tạo summary từ tất cả reports."""
        summary = {
            "total_files": len(reports.get("files", {})),
            "files_analyzed": list(reports.get("files", {}).keys()),
        }
        
        # Extract key stats from corpus
        corpus_report = reports.get("files", {}).get("corpus.jsonl", {})
        if corpus_report:
            summary["corpus"] = {
                "total_documents": corpus_report.get("total_records", 0),
                "file_size_mb": corpus_report.get("file_size_mb", 0),
                "document_types": corpus_report.get("legal_analysis", {}).get("document_types", {}).get("types", {}),
                "hierarchy_patterns": corpus_report.get("legal_analysis", {}).get("hierarchy_patterns", {})
            }
        
        # Extract key stats from queries
        queries_report = reports.get("files", {}).get("queries.jsonl", {})
        if queries_report:
            summary["queries"] = {
                "total_queries": queries_report.get("total_records", 0),
                "question_types": queries_report.get("query_analysis", {}).get("question_types", {})
            }
        
        return summary


def print_report(reports: Dict[str, Any]):
    """In báo cáo ra console một cách dễ đọc."""
    print("\n")
    print("=" * 70)
    print("📋 ANALYSIS REPORT")
    print("=" * 70)
    
    # Summary
    summary = reports.get("summary", {})
    print(f"\n📁 Files analyzed: {summary.get('total_files', 0)}")
    for fname in summary.get("files_analyzed", []):
        print(f"   - {fname}")
    
    # Corpus details
    corpus = summary.get("corpus", {})
    if corpus:
        print(f"\n📚 CORPUS (corpus.jsonl):")
        print(f"   Total documents: {corpus.get('total_documents', 0):,}")
        print(f"   File size: {corpus.get('file_size_mb', 0):.2f} MB")
        
        doc_types = corpus.get("document_types", {})
        if doc_types:
            print(f"\n   Document Types (top 10):")
            for dtype, count in list(doc_types.items())[:10]:
                print(f"      {dtype}: {count}")
        
        hierarchy = corpus.get("hierarchy_patterns", {})
        if hierarchy:
            print(f"\n   Hierarchy Markers:")
            for marker, count in hierarchy.get("hierarchy_markers", {}).items():
                print(f"      {marker}: {count}")
    
    # Queries details
    queries = summary.get("queries", {})
    if queries:
        print(f"\n❓ QUERIES (queries.jsonl):")
        print(f"   Total queries: {queries.get('total_queries', 0):,}")
        
        q_types = queries.get("question_types", {})
        if q_types:
            print(f"\n   Question Types:")
            for qtype, count in q_types.items():
                print(f"      {qtype}: {count}")
    
    # File details
    print(f"\n📊 FILE DETAILS:")
    for fname, freport in reports.get("files", {}).items():
        print(f"\n   {fname}:")
        print(f"      Records: {freport.get('total_records', 0):,}")
        print(f"      Size: {freport.get('file_size_mb', 0):.2f} MB")
        print(f"      Avg record size: {freport.get('avg_record_bytes', 0):.0f} bytes")
        
        schema = freport.get("schema", {})
        if schema:
            print(f"      Fields: {list(schema.keys())}")
        
        content_stats = freport.get("content_stats", {})
        if content_stats:
            print(f"      Content length: min={content_stats.get('min_length', 0)}, "
                  f"max={content_stats.get('max_length', 0)}, "
                  f"avg={content_stats.get('mean_length', 0):.0f}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze Zalo Legal Dataset")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Number of samples to analyze per file (default: 500)")
    parser.add_argument("--output-dir", type=str, default="data/analysis",
                        help="Output directory for analysis report")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save report to file")
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / args.output_dir
    
    # Run analysis
    analyzer = LegalDatasetAnalyzer(data_dir, sample_size=args.sample_size)
    reports = analyzer.full_analysis()
    
    # Print report
    print_report(reports)
    
    # Save report
    if not args.no_save:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "dataset_analysis_report.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(reports, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Report saved to: {output_file}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS FOR HYPERBOLICRAG")
    print("=" * 70)
    print("""
Dựa trên phân tích, các bước tiếp theo:

1. HIERARCHY EXTRACTION:
   - Sử dụng patterns đã phát hiện để xây dựng hierarchy
   - Focus vào: Phần > Chương > Mục > Điều > Khoản > Điểm

2. DOCUMENT LINKING:
   - Extract các tham chiếu chéo (Điều X tham chiếu Điều Y)
   - Build relationship graph

3. DEPTH ASSIGNMENT:
   - Assign depth level cho mỗi record dựa trên patterns
   - Validate với samples

Chạy lại với --sample-size lớn hơn để phân tích chi tiết hơn.
""")


if __name__ == "__main__":
    main()
