# -*- coding: utf-8 -*-
"""
Scrapy Pipelines cho Legal Crawler.
Cập nhật: Tối ưu dữ liệu lớn, xử lý trùng lặp tự động (Upsert).
"""

import json
import re
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from scrapy.exceptions import DropItem
from legal_crawler.items import LegalDocumentItem, LegalArticleItem

class ProgressPipeline:
    """Theo dõi tiến độ cào bằng tqdm."""
    def __init__(self):
        self.pbar = None

    def open_spider(self, spider):
        pass

    def close_spider(self, spider):
        if hasattr(spider, 'pbar') and spider.pbar:
            spider.pbar.close()

    def process_item(self, item, spider):
        if isinstance(item, LegalDocumentItem):
            if hasattr(spider, 'pbar') and spider.pbar:
                spider.pbar.update(1)
        return item

class DeduplicationPipeline:
    def __init__(self):
        self.seen_docs = set()
        self.seen_articles = set()

    def process_item(self, item, spider):
        if isinstance(item, LegalDocumentItem):
            key = item.get("doc_code", "")
            if key in self.seen_docs:
                raise DropItem(f"Duplicate document: {key}")
            self.seen_docs.add(key)
        elif isinstance(item, LegalArticleItem):
            key = f"{item.get('doc_code')}+{item.get('article_number')}+{item.get('clause_number')}+{item.get('point_id')}"
            if key in self.seen_articles:
                raise DropItem(f"Duplicate content unit: {key}")
            self.seen_articles.add(key)
        return item

class CleanTextPipeline:
    NOISE_PATTERNS = [
        (re.compile(r"\s+"), " "),
        (re.compile(r"\n{3,}"), "\n\n"),
        (re.compile(r"[ \t]+\n"), "\n"),
        (re.compile(r"\n[ \t]+"), "\n"),
    ]

    def process_item(self, item, spider):
        if isinstance(item, LegalArticleItem):
            content = item.get("content", "")
            if content:
                for pattern, replacement in self.NOISE_PATTERNS:
                    content = pattern.sub(replacement, content)
                item["content"] = content.strip()
            if item.get("title"):
                item["title"] = item["title"].strip()
        elif isinstance(item, LegalDocumentItem):
            title = item.get("title", "")
            if title:
                title = re.sub(r"[\n\t\r]+", " ", title)
                title = re.sub(r"\s+", " ", title)
                item["title"] = title.strip()
        return item

class SQLitePipeline:
    def __init__(self):
        self.conn = None
        self.cursor = None

    def open_spider(self, spider):
        db_dir = Path(spider.settings.get("PROJECT_ROOT", ".")) / "data/database"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "legal_data.db"
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        
        # 1. Bảng documents
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_code TEXT PRIMARY KEY,
                title TEXT,
                doc_type TEXT,
                issuer TEXT,
                issue_date TEXT,
                effective_date TEXT,
                status TEXT,
                source_url TEXT,
                crawled_at TEXT
            )
        """)
        
        # 2. Bảng articles với UNIQUE constraint để tự động xử lý trùng lặp
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                node_type TEXT,
                doc_code TEXT,
                article_number INTEGER,
                clause_number INTEGER,
                point_id TEXT,
                title TEXT,
                content TEXT,
                hierarchy_path TEXT,
                FOREIGN KEY (doc_code) REFERENCES documents (doc_code),
                UNIQUE(doc_code, article_number, clause_number, point_id)
            )
        """)
        self.conn.commit()

    def close_spider(self, spider):
        if self.conn:
            self.conn.close()

    def process_item(self, item, spider):
        try:
            if isinstance(item, LegalDocumentItem):
                self.cursor.execute("""
                    INSERT OR REPLACE INTO documents 
                    (doc_code, title, doc_type, issuer, issue_date, effective_date, status, source_url, crawled_at)
                    VALUES (?,?,?,?,?,?,?,?,?)
                """, (
                    item.get("doc_code"), item.get("title"), item.get("doc_type"),
                    item.get("issuer"), item.get("issue_date"), item.get("effective_date"),
                    item.get("status"), item.get("source_url"),
                    item.get("crawled_at") or datetime.utcnow().isoformat() + "Z"
                ))
            elif isinstance(item, LegalArticleItem):
                # Sử dụng INSERT OR REPLACE nhờ vào UNIQUE constraint ở trên
                self.cursor.execute("""
                    INSERT OR REPLACE INTO articles 
                    (node_type, doc_code, article_number, clause_number, point_id, title, content, hierarchy_path)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    item.get("node_type"), item.get("doc_code"), item.get("article_number"),
                    item.get("clause_number"), item.get("point_id"),
                    item.get("title"), item.get("content"),
                    json.dumps(item.get("hierarchy_path", []), ensure_ascii=False)
                ))
            self.conn.commit()
        except Exception as e:
            spider.logger.error(f"Lỗi SQLite: {e}")
            
        return item

class JsonlExportPipeline:
    def __init__(self):
        self.doc_file = None
        self.article_file = None

    def open_spider(self, spider):
        # Đảm bảo file được mở ở chế độ append
        output_dir = Path(spider.settings.get("PROJECT_ROOT", ".")) / "data/raw/crawled"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.doc_file = open(output_dir / "documents.jsonl", "a", encoding="utf-8")
        self.article_file = open(output_dir / "articles.jsonl", "a", encoding="utf-8")

    def close_spider(self, spider):
        if self.doc_file: self.doc_file.close()
        if self.article_file: self.article_file.close()

    def process_item(self, item, spider):
        record = dict(item)
        if isinstance(item, LegalDocumentItem):
            if "crawled_at" not in record: record["crawled_at"] = datetime.utcnow().isoformat() + "Z"
            self.doc_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.doc_file.flush()
        elif isinstance(item, LegalArticleItem):
            self.article_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.article_file.flush()
        return item
