# -*- coding: utf-8 -*-
import json
import re
import sqlite3
import sys
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scrapy.exceptions import DropItem
from legal_crawler.items import LegalDocumentItem, LegalArticleItem

class SQLitePipeline:
    """Lưu trữ metadata vào SQLite."""
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

        # Bảng documents (Không lưu full text)
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
                crawled_at TEXT,
                parse_status TEXT,
                parse_log TEXT
            )
        """)
        # Bảng articles
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS articles (
                node_id TEXT PRIMARY KEY,
                node_type TEXT,
                doc_code TEXT,
                parent_id TEXT,
                article_number INTEGER,
                clause_number INTEGER,
                point_id TEXT,
                title TEXT,
                content TEXT,
                hierarchy_path TEXT,
                is_amendment INTEGER DEFAULT 0,
                FOREIGN KEY (doc_code) REFERENCES documents (doc_code)
            )
        """)
        self.conn.commit()

    def process_item(self, item):
        if isinstance(item, LegalDocumentItem):
            self.cursor.execute("""
                INSERT OR REPLACE INTO documents 
                (doc_code, title, doc_type, issuer, issue_date, effective_date, status, source_url, crawled_at, parse_status, parse_log)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                item.get("doc_code"), item.get("title"), item.get("doc_type"),
                item.get("issuer"), item.get("issue_date"), item.get("effective_date"),
                item.get("status"), item.get("source_url"),
                item.get("crawled_at") or datetime.utcnow().isoformat() + "Z",
                item.get("parse_status"), item.get("parse_log")
            ))
            self.conn.commit()
        return item

class MarkdownStoragePipeline:
    """Lưu Full Markdown ra file vật lý."""
    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        return o

    def process_item(self, item):
        if isinstance(item, LegalDocumentItem) and item.get("full_markdown"):
            doc_code = item["doc_code"]
            md_dir = Path(self.crawler.spider.settings.get("PROJECT_ROOT", ".")) / "data/raw/markdown"
            md_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = md_dir / f"{doc_code.replace('/', '_')}.md"
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"--- metadata ---\nTitle: {item['title']}\nSource: {item['source_url']}\n---\n\n")
                f.write(item["full_markdown"])
        return item

class CleanTextPipeline:
    """Tiền xử lý văn bản."""
    END_PUNCTUATION = ('.', '?', '!', '”', '/', ':', ';')
    NEW_SECTION_AGNOS = re.compile(r'^(\*\*|#|Điều|Khoản|\d+\.|[a-z]\))', re.I)

    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        return o

    def clean_markdown(self, text):
        if not text: return ""
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        lines = text.split('\n')
        cleaned_lines = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                cleaned_lines.append(""); i += 1; continue
            line = re.sub(r'\*\*\s+(.*?)\s+\*\*', r'**\1**', line)
            while i + 1 < len(lines):
                next_line = lines[i+1].strip()
                if not next_line: break
                if (not line.endswith(self.END_PUNCTUATION) or re.search(r'\s(và|hoặc|nhưng|như|của|tại|theo)$', line, re.I)) and not self.NEW_SECTION_AGNOS.match(next_line):
                    line = line + " " + next_line; i += 1
                else: break
            if line.startswith("**Điều") or line.startswith("Điều"): cleaned_lines.append("")
            cleaned_lines.append(line); i += 1
        return re.sub(r'\n{3,}', '\n\n', "\n".join(cleaned_lines)).strip()

    def process_item(self, item):
        if isinstance(item, LegalDocumentItem) and item.get("full_markdown"):
            item["full_markdown"] = self.clean_markdown(item["full_markdown"])
        return item

class LegalAIPipeline:
    """Pipeline thực hiện bóc tách AI với định dạng Output bị khóa chặt bằng JSON Schema."""
    
    # Định nghĩa Schema cho Mục lục (Catalog)
    CATALOG_SCHEMA = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "num": {"type": "string"},
                        "title": {"type": "string"}
                    },
                    "required": ["num", "title"]
                }
            }
        },
        "required": ["items"]
    }

    # Định nghĩa Schema cho Chi tiết Điều/Khoản
    DETAIL_SCHEMA = {
        "type": "object",
        "properties": {
            "metadata": {
                "type": "object",
                "properties": {
                    "official_title": {"type": "string"},
                    "doc_code": {"type": "string"}
                }
            },
            "articles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "num": {"type": "string"},
                        "title": {"type": "string"},
                        "content": {"type": "string"},
                        "sub_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {"type": "string", "enum": ["clause", "point"]},
                                    "num": {"type": "string"},
                                    "content": {"type": "string"},
                                    "is_amendment": {"type": "boolean"}
                                },
                                "required": ["type", "num", "content"]
                            }
                        }
                    },
                    "required": ["num", "content"]
                }
            }
        },
        "required": ["articles"]
    }

    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        o.api_key = crawler.settings.get("GEMINI_API_KEY")
        if o.api_key:
            import google.generativeai as genai
            genai.configure(api_key=o.api_key)
            o.model_catalog = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json", "response_schema": cls.CATALOG_SCHEMA})
            o.model_detail = genai.GenerativeModel('gemini-1.5-flash', generation_config={"response_mime_type": "application/json", "response_schema": cls.DETAIL_SCHEMA})
        return o

    def process_item(self, item, spider):
        if not isinstance(item, LegalDocumentItem) or not item.get("full_markdown") or not hasattr(self, 'model_catalog'):
            return item

        doc_code = item["doc_code"]
        md_content = item["full_markdown"]
        spider.logger.info(f"🤖 Bắt đầu bóc tách AI (Strict Mode) cho: {doc_code}")
        
        try:
            import time
            import sys, os
            # Dùng chung Prompt chuẩn từ auto_pilot để đảm bảo nhất quán
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
            from auto_pilot import PROMPT_CATALOG, PROMPT_STRUCTURE

            # Stage 1: Catalog - Nhận diện cấu trúc văn bản
            cat_resp = self.model_catalog.generate_content(PROMPT_CATALOG.format(content=md_content))
            cat_data = json.loads(cat_resp.text)
            
            # Tương thích với cấu trúc mới {"doc_type": ..., "items": [...]}
            if isinstance(cat_data, dict):
                catalog = cat_data.get('items', [])
                doc_type = cat_data.get('doc_type', 'unknown')
            else:
                catalog = cat_data
                doc_type = 'unknown'
            
            spider.logger.info(f"📄 Loại văn bản: {doc_type} | Tổng số Điều: {len(catalog)}")
            
            db_path = Path(spider.settings.get("PROJECT_ROOT", ".")) / "data/database/legal_data.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            chunk_size = 5
            for i in range(0, len(catalog), chunk_size):
                chunk = catalog[i:i + chunk_size]
                nums = [c['num'] if isinstance(c, dict) else str(c) for c in chunk]
                spider.logger.info(f"   -> Đang xử lý nhóm Điều: {', '.join(nums)}")
                
                # Stage 2: Detailed - Bóc tách chi tiết
                res = self.model_detail.generate_content(PROMPT_STRUCTURE.format(target_articles=nums, content=md_content))
                result = json.loads(res.text)

                
                # Cập nhật metadata
                meta = result.get('metadata', {})
                if meta and meta.get('official_title'):
                    cursor.execute("UPDATE documents SET title = ? WHERE doc_code = ?", (meta['official_title'], doc_code))

                for art in result.get('articles', []):
                    art_num = art.get('num')
                    node_id = f"{doc_code}_a_{art_num}"
                    cursor.execute("INSERT OR REPLACE INTO articles (node_id, node_type, doc_code, parent_id, article_number, title, content) VALUES (?, 'article', ?, ?, ?, ?, ?)", (node_id, doc_code, doc_code, art_num, art.get('title'), art.get('content')))
                    
                    for sub in art.get('sub_items', []):
                        st, sn = sub.get('type', 'clause'), sub.get('num')
                        sid = f"{node_id}_{st[0]}_{sn}"
                        cursor.execute("INSERT OR REPLACE INTO articles (node_id, node_type, doc_code, parent_id, article_number, clause_number, point_id, content, is_amendment) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (sid, st, doc_code, node_id, art_num, sn if st == 'clause' else None, sn if st == 'point' else None, sub.get('content'), 1 if sub.get('is_amendment') else 0))
                conn.commit()
                time.sleep(4)

            cursor.execute("UPDATE documents SET parse_status = 'SUCCESS' WHERE doc_code = ?", (doc_code,))
            conn.commit(); conn.close()
            spider.logger.info(f"✅ Xong AI: {doc_code}")
        except Exception as e:
            spider.logger.error(f"❌ Lỗi AI Processing: {e}")
        return item

class ProgressPipeline:
    @classmethod
    def from_crawler(cls, crawler):
        o = cls()
        o.crawler = crawler
        return o

    def process_item(self, item, spider):
        if isinstance(item, LegalDocumentItem):
            if hasattr(spider, 'pbar') and spider.pbar:
                spider.pbar.update(1)
        return item
