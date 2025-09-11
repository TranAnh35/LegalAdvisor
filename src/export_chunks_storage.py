#!/usr/bin/env python3
"""
Tiện ích xuất smart chunks sang định dạng tối ưu: Parquet (nếu có pyarrow) và SQLite.

Đầu ra:
- data/processed/smart_chunks_stable.parquet (nếu có pyarrow)
- data/processed/smart_chunks_stable.db (SQLite, bảng 'chunks')
"""

import sqlite3
from pathlib import Path
try:
    from src.utils.paths import get_processed_data_dir
except Exception:
    from utils.paths import get_processed_data_dir  # type: ignore
from typing import Any, Dict, List, Optional


def _normalize_spaces(text: str) -> str:
    return ' '.join((text or '').split()).strip()


def _compose_article_text(article_num: str, heading: str, content: str) -> str:
    header = f"Điều {article_num}. {heading}".strip()
    header = header.rstrip('.') if header.endswith('..') else header
    body = content or ''
    return (header + "\n" + body).strip()


def _compose_clause_text(article_num: str, clause_num: str, clause_content: str) -> str:
    prefix = f"Điều {article_num} - Khoản {clause_num}. "
    return (prefix + (clause_content or '')).strip()


def _compose_point_text(article_num: str, clause_num: str, point_label: str, point_content: str) -> str:
    prefix = f"Điều {article_num} - Khoản {clause_num} - Điểm {point_label}) "
    return (prefix + (point_content or '')).strip()


def build_chunks_for_document(doc: Dict[str, Any], start_chunk_id: int) -> List[Dict[str, Any]]:
    """Sinh smart chunks từ cấu trúc pháp lý đã parse (chapter/section/article/clause/point).

    Hàm này được dùng bởi automatic_preprocess_vnlegaltext_stable để ghi trực tiếp
    sang SQLite/Parquet thông qua export_*.
    """

    chunks: List[Dict[str, Any]] = []
    sections: List[Dict[str, Any]] = doc.get('sections') or []

    def normalize_for_db(text: str) -> str:
        if not text:
            return ''
        text = text.replace('_', ' ')
        return ' '.join(text.split()).strip()

    if not sections:
        # Fallback: chia thô theo độ dài nếu không có cấu trúc
        text = doc.get('cleaned_content_index') or doc.get('cleaned_content') or ''
        words = text.split()
        chunk_size, overlap = 400, 50
        idx = 0
        per_doc_index = 0
        while idx < len(words):
            part = words[idx: idx + chunk_size]
            if len(part) >= 80:
                content = ' '.join(part)
                content = normalize_for_db(content)
                chunks.append({
                    'chunk_id': start_chunk_id + len(chunks),
                    'doc_file': doc.get('file_name'),
                    'doc_title': (doc.get('metadata') or {}).get('title'),
                    'chunk_index': per_doc_index,
                    'content': content,
                    'word_count': len(part),
                    'chunk_type': 'fallback'
                })
                per_doc_index += 1
            idx += (chunk_size - overlap)
        return chunks

    # Duyệt tuyến tính để gom theo cấu trúc
    current_chapter: Optional[str] = None
    current_section: Optional[str] = None
    article_ctx: Optional[Dict[str, Any]] = None
    clause_ctx: Optional[Dict[str, Any]] = None

    articles: List[Dict[str, Any]] = []

    for s in sections:
        stype = s.get('section_type')
        if stype == 'chapter':
            current_chapter = s.get('title')
            current_section = None
            article_ctx = None
            clause_ctx = None
        elif stype == 'section':
            current_section = s.get('title')
            article_ctx = None
            clause_ctx = None
        elif stype == 'article':
            article_ctx = {
                'article_number': s.get('section_number'),
                'article_title': s.get('title'),
                'article_heading': s.get('heading', ''),
                'chapter_title': current_chapter,
                'section_title': current_section,
                'article_content': s.get('content', ''),
                'clauses': []
            }
            articles.append(article_ctx)
            clause_ctx = None
        elif stype == 'clause' and article_ctx is not None:
            clause_ctx = {
                'clause_number': s.get('section_number'),
                'content': s.get('content', ''),
                'points': []
            }
            article_ctx['clauses'].append(clause_ctx)
        elif stype == 'point' and clause_ctx is not None:
            clause_ctx['points'].append({
                'point_label': s.get('section_number'),
                'content': s.get('content', '')
            })

    # Tạo chunks
    per_doc_index = 0
    for art in articles:
        art_num = art.get('article_number')
        art_heading = _normalize_spaces(art.get('article_heading') or '')
        art_content = _normalize_spaces(art.get('article_content') or '')

        # Chunk cấp Điều (nếu có nội dung và KHÔNG có clause nào) để tránh trùng lặp với Khoản/Điểm
        has_any_clause = bool(art.get('clauses'))
        if art_content and not has_any_clause:
            content = _compose_article_text(art_num or '', art_heading, art_content)
            content = normalize_for_db(content)
            chunks.append({
                'chunk_id': start_chunk_id + len(chunks),
                'doc_file': doc.get('file_name'),
                'doc_title': (doc.get('metadata') or {}).get('title'),
                'chapter': art.get('chapter_title'),
                'section': art.get('section_title'),
                'article': art_num,
                'article_heading': art_heading,
                'clause': None,
                'point': None,
                'chunk_index': per_doc_index,
                'content': content,
                'word_count': len(content.split()),
                'chunk_type': 'article'
            })
            per_doc_index += 1

        # Chunks cấp Khoản và Điểm
        for cl in art.get('clauses') or []:
            clause_num = cl.get('clause_number')
            clause_content = _normalize_spaces(cl.get('content') or '')

            if clause_content:
                content = _compose_clause_text(art_num or '', clause_num or '', clause_content)
                content = normalize_for_db(content)
                chunks.append({
                    'chunk_id': start_chunk_id + len(chunks),
                    'doc_file': doc.get('file_name'),
                    'doc_title': (doc.get('metadata') or {}).get('title'),
                    'chapter': art.get('chapter_title'),
                    'section': art.get('section_title'),
                    'article': art_num,
                    'article_heading': art_heading,
                    'clause': clause_num,
                    'point': None,
                    'chunk_index': per_doc_index,
                    'content': content,
                    'word_count': len(content.split()),
                    'chunk_type': 'clause'
                })
                per_doc_index += 1

            # Mỗi điểm là một chunk riêng
            for pt in cl.get('points') or []:
                label = pt.get('point_label')
                pcontent = _normalize_spaces(pt.get('content') or '')
                if not pcontent:
                    continue
                content = _compose_point_text(art_num or '', clause_num or '', label or '', pcontent)
                content = normalize_for_db(content)
                chunks.append({
                    'chunk_id': start_chunk_id + len(chunks),
                    'doc_file': doc.get('file_name'),
                    'doc_title': (doc.get('metadata') or {}).get('title'),
                    'chapter': art.get('chapter_title'),
                    'section': art.get('section_title'),
                    'article': art_num,
                    'article_heading': art_heading,
                    'clause': clause_num,
                    'point': label,
                    'chunk_index': per_doc_index,
                    'content': content,
                    'word_count': len(content.split()),
                    'chunk_type': 'point'
                })
                per_doc_index += 1

    return chunks

def export_to_parquet(chunks: List[Dict[str, Any]]) -> bool:
    try:
        import pandas as pd  # type: ignore
        # Yêu cầu pyarrow
        try:
            import pyarrow  # noqa: F401
        except Exception:
            print('ℹ️  Bỏ qua Parquet: thiếu pyarrow. Cài đặt: pip install pyarrow')
            return False

        df = pd.DataFrame(chunks)
        # Chỉ giữ các cột cần thiết để giảm size
        keep_cols = [
            'chunk_id', 'doc_file', 'doc_title', 'chapter', 'section', 'article',
            'article_heading', 'clause', 'point', 'chunk_index', 'content',
            'word_count', 'chunk_type'
        ]
        for col in list(df.columns):
            if col not in keep_cols:
                df.drop(columns=[col], inplace=True)

        base_dir = get_processed_data_dir()
        out_path = base_dir / 'smart_chunks_stable.parquet'
        # Xóa parquet cũ nếu tồn tại để đảm bảo ghi đè sạch
        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, engine='pyarrow', index=False, compression='zstd', compression_level=9)
        print(f'✅ Đã xuất Parquet: {out_path}')
        print(f'   → Cột: {list(df.columns)} | Số dòng: {len(df)}')
        return True
    except Exception as e:
        print(f'⚠️  Lỗi khi xuất Parquet: {e}')
        return False


def export_to_sqlite(chunks: List[Dict[str, Any]]) -> bool:
    try:
        base_dir = get_processed_data_dir()
        db_path = base_dir / 'smart_chunks_stable.db'
        # Xóa db cũ nếu tồn tại để đảm bảo tạo mới sạch
        try:
            if db_path.exists():
                db_path.unlink()
        except Exception:
            pass
        db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # Tối ưu ghi hàng loạt (chỉ hiệu lực trong phiên này)
        try:
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.execute("PRAGMA synchronous=OFF;")
            cur.execute("PRAGMA temp_store=MEMORY;")
        except Exception:
            pass

        # Đảm bảo schema đúng mỗi lần export (tránh xung đột schema cũ)
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
                chunk_type    TEXT
            )
            """
        )

        def to_int(value):
            if value is None:
                return None
            try:
                return int(value)
            except Exception:
                # Ép lỗi về NULL để tránh datatype mismatch
                return None

        def to_text(value):
            if value is None:
                return None
            try:
                return str(value)
            except Exception:
                return None

        rows = []
        for ch in chunks:
            # Chuẩn hóa kiểu dữ liệu an toàn cho SQLite
            chunk_id = to_int(ch.get('chunk_id'))
            doc_file = to_text(ch.get('doc_file'))
            doc_title = to_text(ch.get('doc_title'))
            chapter = to_text(ch.get('chapter'))
            section = to_text(ch.get('section'))
            article = to_text(ch.get('article'))
            article_heading = to_text(ch.get('article_heading'))
            clause = to_text(ch.get('clause'))
            point = to_text(ch.get('point'))
            chunk_index = to_int(ch.get('chunk_index'))
            content = to_text(ch.get('content'))
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
            ))

        cur.executemany(
            'INSERT INTO chunks (chunk_id, doc_file, doc_title, chapter, section, article, article_heading, clause, point, chunk_index, content, word_count, chunk_type) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
            rows
        )
        conn.commit()
        conn.close()

        print(f'✅ Đã xuất SQLite: {db_path} (rows={len(rows)})')
        return True
    except Exception as e:
        print(f'⚠️  Lỗi khi xuất SQLite: {e}')
        return False


pass



