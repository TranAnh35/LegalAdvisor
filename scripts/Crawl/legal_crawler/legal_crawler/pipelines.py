# -*- coding: utf-8 -*-
"""
Scrapy Pipelines cho Legal Crawler - Scrapy 2.13+ compatible.
"""

import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from scrapy.exceptions import DropItem
from legal_crawler.items import LegalDocumentItem, LegalArticleItem


class ProgressPipeline:
    """Theo dõi tiến độ cào bằng tqdm."""

    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        return o

    def open_spider(self):
        pass  # tqdm được khởi tạo trong spider.start()

    def close_spider(self):
        spider = self.crawler.spider
        if hasattr(spider, 'pbar') and spider.pbar is not None:
            spider.pbar.close()
            tqdm.write("✅ Cào hoàn tất!", file=sys.stderr)

    def process_item(self, item):
        spider = self.crawler.spider
        if isinstance(item, LegalDocumentItem):
            if hasattr(spider, 'pbar') and spider.pbar is not None:
                spider.pbar.update(1)
        return item


class DeduplicationPipeline:
    """Loại bỏ Item trùng lặp trong cùng một phiên cào."""

    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        return o

    def open_spider(self):
        self.seen_docs = set()
        self.seen_articles = set()

    def process_item(self, item):
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
    """Làm sạch nội dung văn bản luật."""

    NOISE_PATTERNS = [
        (re.compile(r"\s+"), " "),
        (re.compile(r"\n{3,}"), "\n\n"),
        (re.compile(r"[ \t]+\n"), "\n"),
        (re.compile(r"\n[ \t]+"), "\n"),
    ]

    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        return o

    def process_item(self, item):
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
    """Lưu trữ Items vào SQLite Database."""

    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        return o

    def open_spider(self):
        spider = self.crawler.spider
        db_dir = Path(spider.settings.get("PROJECT_ROOT", ".")) / "data/database"
        db_dir.mkdir(parents=True, exist_ok=True)
        db_path = db_dir / "legal_data.db"
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

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

    def close_spider(self):
        if self.conn:
            self.conn.close()

    def process_item(self, item):
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
            self.crawler.spider.logger.error(f"Lỗi SQLite: {e}")
        return item


class JsonlExportPipeline:
    """Ghi Items ra file JSONL (backup)."""

    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        return o

    def open_spider(self):
        spider = self.crawler.spider
        output_dir = Path(spider.settings.get("PROJECT_ROOT", ".")) / "data/raw/crawled"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.doc_file = open(output_dir / "documents.jsonl", "a", encoding="utf-8")
        self.article_file = open(output_dir / "articles.jsonl", "a", encoding="utf-8")

    def close_spider(self):
        if self.doc_file:
            self.doc_file.close()
        if self.article_file:
            self.article_file.close()

    def process_item(self, item):
        record = dict(item)
        if isinstance(item, LegalDocumentItem):
            if "crawled_at" not in record:
                record["crawled_at"] = datetime.utcnow().isoformat() + "Z"
            self.doc_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.doc_file.flush()
        elif isinstance(item, LegalArticleItem):
            self.article_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.article_file.flush()
        return item
