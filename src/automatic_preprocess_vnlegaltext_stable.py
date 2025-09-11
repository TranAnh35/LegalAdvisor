#!/usr/bin/env python3
"""
Version ổn định của automatic_preprocess_vnlegaltext.py
- Checkpoint saving: Lưu tiến trình mỗi 100 files
- Memory management: Xử lý theo batch nhỏ
- Error recovery: Skip file lỗi, tiếp tục xử lý
- Resume capability: Có thể tiếp tục từ checkpoint
"""

import json
import re
import time
from pathlib import Path
try:
    # Khi chạy dạng module: python -m src.automatic_preprocess_vnlegaltext_stable
    from src.utils.paths import get_project_root
except Exception:
    # Khi chạy trực tiếp: python src/automatic_preprocess_vnlegaltext_stable.py
    from utils.paths import get_project_root  # type: ignore
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import sqlite3
import argparse

# Sinh smart chunks trực tiếp không qua JSON
try:
    from src.export_chunks_storage import build_chunks_for_document
except Exception:
    from export_chunks_storage import build_chunks_for_document  # type: ignore

# Import PyVi
try:
    from pyvi import ViTokenizer
    HAS_PYVI = True
except ImportError:
    HAS_PYVI = False

class StableVNLegalTextProcessor:
    """Version ổn định với checkpoint và error recovery"""

    def __init__(self, fast: bool = False):
        """Khởi tạo processor"""
        self.fast = fast
        self.patterns = {
            'article': re.compile(r'^\s*Điều\s+(\d+)\s*\.\s*(.*)$', re.MULTILINE),
            'clause': re.compile(r'^\s*(\d+)\s*\.(?:\s+|\s*)(.*)$', re.MULTILINE),
            # Điểm: chấp nhận a) hoặc a ) hoặc (a), bao gồm chữ có dấu (đ, â, ê, ô, ă, ơ, ư ...)
            'point': re.compile(r'^\s*(?:\(|\s*)?([a-zàáạảãăằắặẳẵâầấậẩẫđèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ])\s*\)\s+(.*)$', re.MULTILINE | re.IGNORECASE),
            'chapter': re.compile(r'^\s*CHƯƠNG\s+([IVXLCDM]+|[0-9]+)\b\s*(.*)$', re.MULTILINE | re.IGNORECASE),
            'section': re.compile(r'^\s*Mục\s+(\d+)\b\s*(.*)$', re.MULTILINE | re.IGNORECASE)
        }

        # Mapping số La Mã
        self.roman_numerals = {
            'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5,
            'VI': 6, 'VII': 7, 'VIII': 8, 'IX': 9, 'X': 10,
            'XI': 11, 'XII': 12, 'XIII': 13, 'XIV': 14, 'XV': 15
        }

        # Tải stopwords - BẮT BUỘC
        self.stopwords = self._load_stopwords()

        # Kiểm tra các dependencies bắt buộc
        if not HAS_PYVI:
            raise ImportError("❌ PyVi is required but not available. Please install pyvi: pip install pyvi")

        if not self.stopwords:
            raise ValueError("❌ Stopwords file is required but not available or empty")

    def _load_stopwords(self) -> set:
        """Tải stopwords từ file"""
        try:
            stopwords_file = Path(__file__).parent.parent / "data" / "vietnamese-stopwords-dash.txt"
            if stopwords_file.exists():
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    stopwords = set(line.strip().lower() for line in f if line.strip())
                print(f"✅ Đã tải {len(stopwords)} stopwords")
                return stopwords
            else:
                print("⚠️  Không tìm thấy file stopwords")
                return set()
        except Exception as e:
            print(f"⚠️  Lỗi khi tải stopwords: {e}")
            return set()

    def _load_checkpoint(self, checkpoint_file: Path) -> Dict[str, Any]:
        """Tải checkpoint để resume"""
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️  Không thể tải checkpoint: {e}")
        return {'processed_files': [], 'last_index': 0, 'start_time': time.time()}

    def _save_checkpoint(self, checkpoint_file: Path, processed_data: List[Dict], last_index: int, start_time: float):
        """Lưu checkpoint"""
        try:
            checkpoint = {
                'processed_files': processed_data,
                'last_index': last_index,
                'start_time': start_time,
                'timestamp': time.time()
            }
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️  Lỗi khi lưu checkpoint: {e}")

    def clean_text_with_vietnamese_support(self, text: str, remove_stopwords: bool = False, keep_underscore: bool = True) -> str:
        """Làm sạch văn bản với PyVi.

        - remove_stopwords: True để loại bỏ stopwords (dùng cho UI), False cho index
        - keep_underscore: True để giữ "_" của ViTokenizer (dùng cho index)
        """
        if not text:
            return ""

        try:
            # Xóa các tag XML nhưng thay bằng khoảng trắng để tránh dính từ
            text = re.sub(r'<[^>]+>', ' ', text)

            # Chuẩn hóa khoảng trắng nhưng giữ lại cấu trúc dòng
            text = re.sub(r'[ \t]+', ' ', text)
            # Chèn khoảng trắng trước khi tokenize để tránh dính từ cạnh dấu câu
            text = re.sub(r'([,;:\.!?)(\[\]"\'])', r' \1 ', text)
            text = re.sub(r'\s+', ' ', text)

            # Sử dụng pyvi để tách từ tự động - BẮT BUỘC
            text = ViTokenizer.tokenize(text)

            # Loại bỏ stopwords nếu được yêu cầu
            if remove_stopwords:
                words = text.split()
                words = [word for word in words if word.lower() not in self.stopwords]
                text = ' '.join(words)

            # Xử lý dấu gạch dưới từ ViTokenizer
            if not keep_underscore:
                text = text.replace('_', ' ')

            # Làm sạch nhẹ nhàng - GIỮ LẠI DẤU TIẾNG VIỆT VÀ KÝ HIỆU PHỔ BIẾN
            pattern = r"[^a-zA-Z0-9\sÀ-ỹà-ỹ\.,;:!\?()\[\]\"'\-\n_–—…/§“”‘’]"
            text = re.sub(pattern, '', text)

            return text.strip()

        except Exception as e:
            print(f"⚠️  Lỗi khi xử lý text: {e}")
            return text[:500] if text else ""  # Return truncated text as fallback

    def _normalize_underscore_separated_words(self, text: str) -> str:
        """Deprecated: Không sử dụng trong pipeline chính."""
        return text

    def parse_legal_structure(self, content: str, document_title: str = "") -> List[Dict[str, Any]]:
        """Parse cấu trúc pháp lý (CHƯƠNG → MỤC → Điều → Khoản → Điểm)."""
        sections: List[Dict[str, Any]] = []

        try:
            lines = content.split('\n')

            current_chapter: Optional[Dict[str, Any]] = None
            current_section: Optional[Dict[str, Any]] = None
            current_article: Optional[Dict[str, Any]] = None
            current_clause: Optional[Dict[str, Any]] = None

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue

                # CHƯƠNG
                chapter_match = self.patterns['chapter'].search(line)
                if chapter_match:
                    if current_clause:
                        sections.append(current_clause)
                        current_clause = None
                    if current_article:
                        sections.append(current_article)
                        current_article = None
                    if current_section:
                        sections.append(current_section)
                        current_section = None
                    if current_chapter:
                        sections.append(current_chapter)

                    chap_num = chapter_match.group(1)
                    chap_title = chapter_match.group(2).strip() if chapter_match.lastindex and chapter_match.group(2) else ""
                    current_chapter = {
                        'title': f"CHƯƠNG {chap_num}",
                        'heading': chap_title,
                        'content': "",
                        'section_type': 'chapter',
                        'section_number': chap_num,
                        'level': 0
                    }
                    continue

                # MỤC
                section_match = self.patterns['section'].search(line)
                if section_match:
                    if current_clause:
                        sections.append(current_clause)
                        current_clause = None
                    if current_article:
                        sections.append(current_article)
                        current_article = None
                    if current_section:
                        sections.append(current_section)

                    sec_num = section_match.group(1)
                    sec_title = section_match.group(2).strip() if section_match.lastindex and section_match.group(2) else ""
                    current_section = {
                        'title': f"Mục {sec_num}",
                        'heading': sec_title,
                        'content': "",
                        'section_type': 'section',
                        'section_number': sec_num,
                        'parent_section': current_chapter['title'] if current_chapter else None,
                        'level': 0.5
                    }
                    continue

                # Điều
                article_match = self.patterns['article'].search(line)
                if article_match:
                    if current_clause:
                        sections.append(current_clause)
                        current_clause = None
                    if current_article:
                        sections.append(current_article)

                    article_num = article_match.group(1)
                    article_heading = article_match.group(2).strip() if article_match.lastindex and article_match.group(2) else ""
                    current_article = {
                        'title': f"Điều {article_num}",
                        'heading': article_heading,
                        'content': "",
                        'section_type': 'article',
                        'section_number': article_num,
                        'parent_section': current_section['title'] if current_section else (current_chapter['title'] if current_chapter else None),
                        'level': 1
                    }
                    continue

                # Khoản
                clause_match = self.patterns['clause'].search(line)
                if clause_match and current_article:
                    if current_clause:
                        sections.append(current_clause)

                    clause_num = clause_match.group(1)
                    clause_content = clause_match.group(2).strip()
                    current_clause = {
                        'title': f"Khoản {clause_num}",
                        'content': clause_content,
                        'section_type': 'clause',
                        'section_number': clause_num,
                        'parent_section': current_article['title'],
                        'level': 2
                    }
                    continue

                # Điểm
                point_match = self.patterns['point'].search(line)
                if point_match and current_clause:
                    point_label = point_match.group(1)
                    point_content = point_match.group(2).strip()
                    sections.append({
                        'title': f"Điểm {point_label})",
                        'content': point_content,
                        'section_type': 'point',
                        'section_number': point_label,
                        'parent_section': current_clause['title'],
                        'level': 3
                    })
                    continue

                # Nội dung
                if current_clause:
                    current_clause['content'] = (current_clause['content'] + ' ' + line).strip() if current_clause.get('content') else line
                elif current_article:
                    current_article['content'] = (current_article['content'] + ' ' + line).strip() if current_article.get('content') else line
                elif current_section:
                    current_section['content'] = (current_section['content'] + ' ' + line).strip() if current_section.get('content') else line
                elif current_chapter:
                    current_chapter['content'] = (current_chapter['content'] + ' ' + line).strip() if current_chapter.get('content') else line

            # Flush cuối cùng
            if current_clause:
                sections.append(current_clause)
            if current_article:
                sections.append(current_article)
            if current_section:
                sections.append(current_section)
            if current_chapter:
                sections.append(current_chapter)

        except Exception as e:
            print(f"⚠️  Lỗi khi parse structure: {e}")

        return sections

    def extract_title_and_metadata(self, content: str, raw_text: Optional[str] = None) -> Dict[str, Any]:
        """Trích xuất tiêu đề và metadata cơ bản (heuristic)."""
        try:
            lines = content.split('\n')[:10]  # Chỉ xem phần đầu

            metadata = {
                'title': 'Unknown',
                'document_type': 'legal_document',
                'language': 'vietnamese'
            }

            # Ưu tiên dòng đầu không rỗng làm tiêu đề nếu chứa từ khóa pháp lý
            for line in lines:
                line = line.strip()
                if not line or line.startswith('<'):
                    continue
                low = line.lower()
                if any(k in low for k in ['luật', 'nghị định', 'thông tư', 'quyết định', 'hiến pháp']):
                    metadata['title'] = line
                    break

            # Heuristic từ raw_text cho các trường phổ biến + ngày tháng (hiệu lực / thông qua)
            if raw_text:
                raw_norm = raw_text.replace('_', ' ')

                # Vùng đầu văn bản: số, cơ quan, tình trạng
                head = '\n'.join(raw_norm.splitlines()[:200])
                patterns = {
                    'reference_number': r'\bSố[:\s]+([^\n]+)',
                    'issuing_body': r'\bCơ\s*quan\s*ban\s*hành[:\s]+([^\n]+)',
                    'status': r'\bTình\s*trạng[:\s]+([^\n]+)'
                }
                for key, pat in patterns.items():
                    m = re.search(pat, head, flags=re.IGNORECASE)
                    if m:
                        metadata[key] = m.group(1).strip()

                # Tìm hiệu lực thi hành (ưu tiên điều Hiệu_lực thi_hành)
                eff_match = re.search(r'hiệu\s*lực[^\n]*?từ\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', raw_norm, flags=re.IGNORECASE)
                if not eff_match:
                    eff_match = re.search(r'(\d{1,2})[\/-](\d{1,2})[\/-](\d{4}).{0,40}hiệu\s*lực', raw_norm, flags=re.IGNORECASE)
                if eff_match:
                    try:
                        d, mth, y = int(eff_match.group(1)), int(eff_match.group(2)), int(eff_match.group(3))
                        metadata['effective_date'] = f"{y:04d}-{mth:02d}-{d:02d}"
                        metadata['effective_year'] = y
                    except Exception:
                        pass

                # Tìm ngày thông qua ở phần cuối
                tail = '\n'.join(raw_norm.splitlines()[-200:])
                prom_match = re.search(r'thông\s*qua\s+ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', tail, flags=re.IGNORECASE)
                if prom_match:
                    try:
                        d, mth, y = int(prom_match.group(1)), int(prom_match.group(2)), int(prom_match.group(3))
                        metadata['promulgation_date'] = f"{y:04d}-{mth:02d}-{d:02d}"
                        metadata['promulgation_year'] = y
                    except Exception:
                        pass

                # Trích citations từ tag <L|HP>
                cites = []
                for m in re.finditer(r'<(L|HP)([^>]*)>(.*?)</\1>', raw_text, flags=re.IGNORECASE | re.DOTALL):
                    tag = m.group(1)
                    attrs = m.group(2) or ''
                    txt = re.sub(r'<[^>]+>', ' ', m.group(3)).strip()
                    rel_m = re.search(r'rel\s*=\s*"([^"]+)"', attrs)
                    cites.append({'type': tag.upper(), 'rel': rel_m.group(1) if rel_m else None, 'text': txt})
                if cites:
                    metadata['citations'] = cites

            return metadata

        except Exception as e:
            print(f"⚠️  Lỗi khi extract metadata: {e}")
            return {'title': 'Error', 'document_type': 'unknown', 'language': 'vietnamese'}

    def process_single_file(self, file_path: Path) -> Dict[str, Any]:
        """Xử lý một file XML (version ổn định)"""
        try:
            # Đọc file với error handling
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()
            except UnicodeDecodeError:
                # Thử encoding khác
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    raw_content = f.read()
            except Exception as e:
                return {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'error': f"Cannot read file: {e}",
                    'error_type': 'file_read_error'
                }

            # Giới hạn kích thước file (tránh memory issue)
            if len(raw_content) > 10 * 1024 * 1024:  # 10MB
                return {
                    'file_name': file_path.name,
                    'file_path': str(file_path),
                    'error': "File too large (>10MB)",
                    'error_type': 'file_too_large'
                }

            # Làm sạch văn bản (hai bản: cho index và cho UI)
            cleaned_for_index = self.clean_text_with_vietnamese_support(
                raw_content,
                remove_stopwords=False,
                keep_underscore=True
            )
            cleaned_for_ui = None
            if not self.fast:
                cleaned_for_ui = self.clean_text_with_vietnamese_support(
                    raw_content,
                    remove_stopwords=True,
                    keep_underscore=False
                )

            # Trích xuất metadata
            metadata = self.extract_title_and_metadata(cleaned_for_index, raw_text=raw_content)

            # Parse cấu trúc pháp lý
            document_title = metadata.get('title', file_path.name.replace('.xml', ''))
            sections = self.parse_legal_structure(cleaned_for_index, document_title)

            # Tạo kết quả
            result = {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'metadata': metadata,
                'cleaned_content': cleaned_for_index,
                'cleaned_content_index': cleaned_for_index,
                'cleaned_content_ui': cleaned_for_ui if cleaned_for_ui is not None else '',
                'sections': sections,
                'stats': {} if self.fast else {
                    'total_sections': len(sections),
                    'articles_count': len([s for s in sections if s['section_type'] == 'article']),
                    'clauses_count': len([s for s in sections if s['section_type'] == 'clause']),
                    'points_count': len([s for s in sections if s['section_type'] == 'point']),
                    'total_words': len(cleaned_for_index.split())
                },
                'processing_info': {
                    'used_pyvi': True,
                    'stopwords_removed_for_index': False,
                    'stopwords_removed_for_ui': (cleaned_for_ui is not None),
                    'processing_time': time.time(),
                    'file_size': len(raw_content)
                }
            }

            return result

        except Exception as e:
            return {
                'file_name': file_path.name,
                'file_path': str(file_path),
                'error': str(e),
                'error_type': type(e).__name__
            }

    def process_all_files_stable(self, input_dir: Path, output_path: Path, batch_size: int = 50, skip_parquet: bool = False):
        """Xử lý tất cả files và xuất thẳng smart chunks vào SQLite/Parquet, không tạo JSON tổng."""
        # Tạo thư mục output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Đích xuất tối ưu
        processed_dir = output_path.parent
        db_path = processed_dir / 'smart_chunks_stable.db'
        parquet_path = processed_dir / 'smart_chunks_stable.parquet'

        # Xóa output cũ để luôn tạo DB/Parquet mới (tránh phình file khi reprocess)
        try:
            if db_path.exists():
                db_path.unlink()
            if parquet_path.exists():
                parquet_path.unlink()
        except Exception as e:
            print(f"⚠️  Không thể xóa output cũ: {e}. Sẽ ghi đè bảng thay thế.")

        # Khởi tạo SQLite (tạo mới bảng)
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        # PRAGMA tối ưu ghi nhanh (đánh đổi độ bền trong phiên xử lý batch)
        try:
            cur.execute("PRAGMA journal_mode=OFF;")
            cur.execute("PRAGMA synchronous=OFF;")
            cur.execute("PRAGMA temp_store=MEMORY;")
            cur.execute("PRAGMA cache_size=-200000;")
        except Exception:
            pass
        cur.execute('DROP TABLE IF EXISTS chunks')
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id      INTEGER PRIMARY KEY,
                doc_file      TEXT,
                doc_title     TEXT,
                chapter       TEXT,
                section       TEXT,
                article       TEXT,
                article_heading TEXT,
                clause        TEXT,
                point         TEXT,
                chunk_index   INTEGER,
                content       TEXT,
                word_count    INTEGER,
                chunk_type    TEXT,
                effective_date TEXT,
                effective_year INTEGER,
                promulgation_date TEXT,
                promulgation_year INTEGER,
                citations     TEXT
            )
            """
        )

        def to_int(value):
            if value is None:
                return None
            try:
                return int(value)
            except Exception:
                return None

        def to_text(value):
            if value is None:
                return None
            try:
                return str(value)
            except Exception:
                return None

        total_chunks = 0

        # Lấy danh sách file XML
        xml_files = list(input_dir.glob("*.xml"))

        if not xml_files:
            print(f"❌ Không tìm thấy file XML nào trong {input_dir}")
            # Đóng kết nối nếu đã mở
            conn.close()
            return

        print(f"📁 Tìm thấy {len(xml_files)} file XML")
        print(f"🤖 Sử dụng xử lý ổn định: PyVi {'Có' if HAS_PYVI else 'Không'}")
        print(f"📊 Batch size: {batch_size} files")

        # Không sử dụng JSON checkpoint lớn; theo dõi tối thiểu bằng biến đếm
        start_time = time.time()
        total_processed = 0

        remaining_files = xml_files
        print(f"🔄 Bắt đầu xử lý {len(xml_files)} files")

        if not remaining_files:
            print("✅ Tất cả files đã được xử lý!")
            return

        # Xử lý theo batch
        # Bắt đầu đếm từ 0
        # total_processed đã được khởi tạo ở trên

        with tqdm(total=len(remaining_files), desc="Processing XML files", unit="file") as pbar:
            for i in range(0, len(remaining_files), batch_size):
                batch = remaining_files[i:i + batch_size]

                for xml_file in batch:
                    try:
                        result = self.process_single_file(xml_file)

                        # Cập nhật progress
                        total_processed += 1
                        pbar.update(1)

                        # Bỏ qua file lỗi
                        if 'error' in result:
                            continue

                        # Xây smart chunks cho tài liệu này
                        doc_chunks = build_chunks_for_document(result, start_chunk_id=total_chunks)

                        # Lấy ngày tháng/citations từ metadata (nếu có)
                        eff_date = result.get('metadata', {}).get('effective_date')
                        eff_year = result.get('metadata', {}).get('effective_year')
                        prom_date = result.get('metadata', {}).get('promulgation_date')
                        prom_year = result.get('metadata', {}).get('promulgation_year')
                        citations = result.get('metadata', {}).get('citations')
                        citations_json = json.dumps(citations, ensure_ascii=False) if citations else None

                        # Chuẩn hóa và chèn vào SQLite
                        rows = []
                        for ch in doc_chunks:
                            chunk_id = to_int(ch.get('chunk_id'))
                            doc_file = to_text(ch.get('doc_file'))
                            # Rút gọn doc_title để giảm size: cắt tối đa 200 ký tự
                            _title = to_text(ch.get('doc_title'))
                            doc_title = (_title[:200] if _title and len(_title) > 200 else _title)
                            chapter = to_text(ch.get('chapter'))
                            section = to_text(ch.get('section'))
                            article = to_text(ch.get('article'))
                            article_heading = to_text(ch.get('article_heading'))
                            clause = to_text(ch.get('clause'))
                            point = to_text(ch.get('point'))
                            chunk_index = to_int(ch.get('chunk_index'))
                            # Nén content nhẹ: bỏ khoảng trắng thừa
                            _content = to_text(ch.get('content'))
                            content = ' '.join(_content.split()) if _content else None
                            word_count = ch.get('word_count')
                            if word_count is None and content is not None:
                                word_count = len(content.split())
                            word_count = to_int(word_count)
                            chunk_type = to_text(ch.get('chunk_type'))

                            rows.append((
                                chunk_id,
                                doc_file,
                                doc_title,
                                chapter,
                                section,
                                article,
                                article_heading,
                                clause,
                                point,
                                chunk_index,
                                content,
                                word_count,
                                chunk_type,
                                eff_date,
                                to_int(eff_year),
                                prom_date,
                                to_int(prom_year),
                                citations_json,
                            ))

                        if rows:
                            cur.executemany(
                                'INSERT INTO chunks (chunk_id, doc_file, doc_title, chapter, section, article, article_heading, clause, point, chunk_index, content, word_count, chunk_type, effective_date, effective_year, promulgation_date, promulgation_year, citations) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                rows
                            )
                            total_chunks += len(rows)
                            # Commit định kỳ để an toàn
                            if total_chunks % 5000 == 0:
                                conn.commit()

                        # Hiển thị status mỗi 10 files
                        if total_processed % 10 == 0:
                            elapsed_time = time.time() - start_time
                            files_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
                            pbar.set_postfix({
                                'processed': f"{total_processed}/{len(xml_files)}",
                                'speed': f"{files_per_second:.1f} files/s",
                                'chunks': f"{total_chunks:,}"
                            })

                    except Exception as e:
                        print(f"❌ Lỗi xử lý {xml_file.name}: {e}")
                        total_processed += 1
                        pbar.update(1)

                # Commit mỗi batch
                conn.commit()
                print(f"💾 Đã commit sau {total_processed} files, tổng chunks: {total_chunks:,}")

        # Hoàn tất SQLite
        conn.commit()
        conn.close()

        # Tạo Parquet từ SQLite (nếu có pyarrow và không skip)
        parquet_ok = False
        if not skip_parquet:
            try:
                import pandas as pd  # type: ignore
                try:
                    import pyarrow  # noqa: F401
                    has_pa = True
                except Exception:
                    has_pa = False
                if has_pa:
                    conn = sqlite3.connect(str(db_path))
                    df = pd.read_sql_query(
                        'SELECT chunk_id, doc_file, doc_title, chapter, section, article, article_heading, clause, point, chunk_index, content, word_count, chunk_type, effective_date, effective_year, promulgation_date, promulgation_year, citations FROM chunks ORDER BY chunk_id',
                        conn
                    )
                    df.to_parquet(parquet_path, engine='pyarrow', index=False)
                    conn.close()
                    parquet_ok = True
            except Exception as e:
                print(f"⚠️  Không thể xuất Parquet: {e}")

        elapsed_time = time.time() - start_time
        files_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0

        print("\n✅ HOÀN THÀNH XỬ LÝ ỔN ĐỊNH!")
        print(f"📊 KẾT QUẢ TỔNG QUAN:")
        print(f"   - Tổng thời gian: {elapsed_time:.1f} giây")
        print(f"   - Tổng chunks: {total_chunks:,}")

        print(f"💾 SQLite: {db_path}")
        if parquet_ok:
            print(f"💾 Parquet: {parquet_path}")

def main():
    """Hàm chính"""
    print("🚀 VNLEGAL TEXT PROCESSOR - VERSION ỔN ĐỊNH")
    print("=" * 60)

    parser = argparse.ArgumentParser(description='Stable VNLegalText Preprocessor')
    parser.add_argument('--batch-size', type=int, default=100, help='Số file xử lý mỗi batch (mặc định: 100)')
    parser.add_argument('--skip-parquet', action='store_true', help='Bỏ qua bước xuất Parquet để tăng tốc')
    parser.add_argument('--fast', action='store_true', help='Chế độ nhanh: bỏ cleaned_for_ui và thống kê')
    args = parser.parse_args()

    try:
        processor = StableVNLegalTextProcessor(fast=args.fast)
        print("✅ Khởi tạo processor thành công")
        print("✅ PyVi: Đã sẵn sàng")
        print(f"✅ Stopwords: {len(processor.stopwords)} từ")
    except ImportError as e:
        print(f"❌ LỖI KHỞI TẠO: {e}")
        return
    except ValueError as e:
        print(f"❌ LỖI KHỞI TẠO: {e}")
        return

    # Đường dẫn
    root = get_project_root()
    vnlegal_path = root / "data" / "raw" / "VNLegalText"
    # Giữ biến output_path cho tương thích, nhưng không xuất JSON nữa
    output_path = root / "data" / "processed" / "vnlegaltext_stable.json"

    # Cấu hình batch size
    batch_size = int(args.batch_size)

    print(f"📂 Input: {vnlegal_path}")
    print(f"📂 Output (DB/Parquet sẽ nằm cùng thư mục): {output_path.parent}")
    print(f"🔢 Batch size: {batch_size}")
    if args.fast:
        print("⚡ Fast mode: Bỏ cleaned_for_ui và thống kê")
    if args.skip_parquet:
        print("🧪 Skip Parquet: Chỉ xuất SQLite")
    print("📋 Xử lý: PyVi + Stopwords removal (BẮT BUỘC)")
    print()

    processor.process_all_files_stable(vnlegal_path, output_path, batch_size, skip_parquet=args.skip_parquet)

if __name__ == "__main__":
    main()
