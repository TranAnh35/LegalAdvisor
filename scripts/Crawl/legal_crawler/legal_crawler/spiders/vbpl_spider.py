# -*- coding: utf-8 -*-
"""
Spider: VBPL.vn (Cơ sở dữ liệu Quốc gia về Văn bản Pháp luật)
Cải tiến (v6 - Incremental & Check DB):
- Tích hợp kiểm tra Database để tránh tải lại nội dung đã có.
- Hỗ trợ cào tăng trưởng cho luật mới.
"""

import scrapy
from urllib.parse import urlparse, urljoin
from datetime import datetime
import re
import sqlite3
from pathlib import Path
from legal_crawler.items import LegalDocumentItem, LegalArticleItem
from tqdm import tqdm

class VbplSpider(scrapy.Spider):
    name = "vbpl"
    allowed_domains = ["vbpl.vn"]
    BASE_URL = "https://vbpl.vn"
    
    SEARCH_API_URL = (
        "https://vbpl.vn/VBQPPL_UserControls/Publishing/TimKiem/pKetQuaTimKiem.aspx?"
        "dvid=13&IsVietNamese=True&type=1&stemp=1&TimTrong1=VBPQFulltext&TimTrong1=Title&"
        "order=VBPQNgayBanHanh&TypeOfOrder=False&TrangThaiHieuLuc=2"
    )

    def __init__(self, keyword=None, item_id=None, max_pages=None, incremental=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword = keyword
        self.item_id = item_id
        self.max_pages = int(max_pages) if max_pages else None
        self.incremental = str(incremental).lower() in ["true", "1", "yes"]
        self.pbar = None
        self.total_docs = 0
        
        # Kết nối DB để check nhanh
        db_path = Path(self.settings.get("PROJECT_ROOT", ".")) / "data/database/legal_data.db"
        self.db_exists = db_path.exists()
        self.existing_ids = set()
        if self.db_exists and self.incremental:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                # Lấy item_id từ source_url trong bảng documents
                cursor.execute("SELECT source_url FROM documents")
                for (url,) in cursor.fetchall():
                    match = re.search(r"ItemID=(\d+)", url, re.I)
                    if match: self.existing_ids.add(match.group(1))
                conn.close()
                self.logger.info(f"Đã tải {len(self.existing_ids)} mã văn bản hiện có từ Database để cào tăng trưởng.")
            except Exception as e:
                self.logger.error(f"Lỗi tải cache DB: {e}")

    def start_requests(self):
        if self.item_id:
            url = f"{self.BASE_URL}/TW/Pages/vbpq-thuoctinh.aspx?ItemID={self.item_id}"
            yield scrapy.Request(url, callback=self.parse_metadata, meta={'item_id': self.item_id})
        elif self.keyword:
            search_url = f"{self.BASE_URL}/TW/Pages/timkiem.aspx?Keyword={self.keyword}"
            yield scrapy.Request(search_url, callback=self.parse_search_results)
        else:
            url = f"{self.SEARCH_API_URL}&Page=1"
            yield scrapy.Request(url, callback=self.parse_catalog, meta={'page': 1})

    def parse_catalog(self, response):
        page = response.meta.get('page', 1)
        
        if page == 1:
            try:
                total_str = response.css(".selected span strong b::text").get()
                if total_str:
                    clean_total = re.sub(r"\D", "", total_str)
                    self.total_docs = int(clean_total)
                    self.pbar = tqdm(total=self.total_docs, desc="Progress", unit="doc")
                    self.logger.info(f"Tổng số văn bản phát hiện trên hệ thống: {self.total_docs}")
            except Exception as e:
                self.logger.warning(f"Không thể khởi tạo tqdm: {e}")

        doc_links = response.css("div.item p.title a::attr(href)").getall()
        if not doc_links:
            doc_links = response.css("a[href*='ItemID']::attr(href)").getall()

        for link in doc_links:
            item_id = self._extract_item_id(link)
            if item_id:
                # KIỂM TRA TĂNG TRƯỞNG: Nếu văn bản đã có trong kho và đang ở chế độ incremental=True
                if self.incremental and item_id in self.existing_ids:
                    if self.pbar: self.pbar.update(1)
                    continue
                
                meta_url = f"{self.BASE_URL}/TW/Pages/vbpq-thuoctinh.aspx?ItemID={item_id}"
                yield scrapy.Request(meta_url, callback=self.parse_metadata, meta={'item_id': item_id})

        if self.max_pages and page >= self.max_pages:
            return

        if len(doc_links) > 0:
            next_page = page + 1
            next_url = f"{self.SEARCH_API_URL}&Page={next_page}"
            yield scrapy.Request(next_url, callback=self.parse_catalog, meta={'page': next_page})

    def parse_metadata(self, response):
        item_id = response.meta['item_id']
        def get_val(label):
            texts = response.xpath(f"//td[contains(normalize-space(), '{label}')]/following-sibling::td[1]//text()").getall()
            return " ".join([t.strip() for t in texts if t.strip()]).strip()

        # Kiểm tra nhanh: Nếu item_id này đã tồn tại trong session hiện tại (Duplicate request)
        # (Scrapy có sẵn DUPEFILTER, nhưng đây là check logic nghiệp vụ)

        status_raw = response.xpath("//td[contains(normalize-space(), 'Tình trạng hiệu lực')]//text()").get("")
        status = status_raw.split(":")[-1].strip() if ":" in status_raw else status_raw.strip()
        
        # Chỉ cào văn bản Còn hiệu lực
        if status and "Còn hiệu lực" not in status:
            if self.pbar: self.pbar.update(1)
            return

        doc_code = get_val("Số ký hiệu") or f"unknown_{item_id}"
        doc_code_norm = doc_code.lower().replace(" ", "")

        metadata = {
            'doc_code': doc_code_norm,
            'doc_type': get_val("Loại văn bản"),
            'issuer': get_val("Cơ quan ban hành"),
            'issue_date': get_val("Ngày ban hành"),
            'effective_date': get_val("Ngày có hiệu lực"),
            'status': status,
            'item_id': item_id
        }
        full_text_url = f"{self.BASE_URL}/TW/Pages/vbpq-toanvan.aspx?ItemID={item_id}"
        yield scrapy.Request(full_text_url, callback=self.parse_full_text, meta=metadata)

    def parse_full_text(self, response):
        doc_code = response.meta.get('doc_code')
        content_div = response.css("div.toanvancontent") or response.css("div#contentDoc")
        if not content_div: 
            if self.pbar: self.pbar.update(1)
            return

        # 1. Title Extraction
        title_parts = []
        for p in content_div.xpath("./p"):
            align = p.xpath("./@align").get() or p.xpath("./@style").get() or ""
            text = "".join(p.xpath(".//text()").getall()).strip()
            if not text: continue
            if re.match(r"^(PHẦN|Phần|CHƯƠNG|Chương|ĐIỀU|Điều)\s+", text, re.I): break
            if "center" in align.lower(): title_parts.append(text)
            elif len(title_parts) > 0: break
        
        full_title = " ".join(title_parts).strip() or f"{response.meta['doc_type']} {response.meta['doc_code']}".strip()

        yield LegalDocumentItem(
            node_type="document", doc_code=doc_code, title=full_title,
            doc_type=response.meta["doc_type"], issuer=response.meta["issuer"],
            issue_date=response.meta["issue_date"], effective_date=response.meta["effective_date"],
            status=response.meta["status"], source_url=response.url,
            crawled_at=datetime.utcnow().isoformat() + "Z"
        )

        all_blocks = content_div.xpath("./*[self::p or self::h3 or self::h4 or self::div[@class='article']]")
        
        cur_h = []
        cur_article = None; cur_clause = None; cur_point = None
        cur_text_lines = []; cur_title = ""

        def _get_item():
            nonlocal cur_text_lines
            content = "\n".join(cur_text_lines).strip()
            if not content: return None
            h_path = list(cur_h)
            if cur_article: h_path.append(f"Điều {cur_article}")
            if cur_clause: h_path.append(f"Khoản {cur_clause}")
            
            return LegalArticleItem(
                node_type="point" if cur_point else ("clause" if cur_clause else "article"),
                doc_code=doc_code, article_number=cur_article, clause_number=cur_clause,
                point_id=cur_point, title=cur_title, content=content, hierarchy_path=h_path
            )

        for el in all_blocks:
            text = "".join(el.xpath(".//text()").getall()).strip()
            if not text: continue
            
            part_match = re.match(r"^(PHẦN|Phần|CHƯƠNG|Chương|MỤC|Mục)\s+(.*)", text, re.I)
            if part_match:
                item = _get_item(); 
                if item: yield item
                label = part_match.group(1).capitalize()
                val = part_match.group(2).strip()
                if label == "Phần": cur_h = [f"Phần {val}"]
                elif label == "Chương": cur_h = cur_h[:1] + [f"Chương {val}"] if cur_h and "Phần" in cur_h[0] else [f"Chương {val}"]
                else: cur_h.append(f"{label} {val}")
                cur_article = cur_clause = cur_point = None
                cur_text_lines = []; cur_title = ""
                continue

            art_match = re.match(r"^(Điều|ĐIỀU)\s+(\d+)[\.:\s]*(.*)", text, re.I)
            if art_match:
                item = _get_item(); 
                if item: yield item
                cur_article = int(art_match.group(2))
                cur_title = art_match.group(3).strip()
                cur_clause = cur_point = None
                cur_text_lines = []
                if cur_title and len(cur_title) > 60:
                    cur_text_lines.append(cur_title); cur_title = ""
                continue

            clause_match = re.match(r"^(\d+)\.\s+(.*)", text)
            if clause_match:
                item = _get_item()
                if item: yield item
                cur_clause = int(clause_match.group(1)); cur_point = None
                cur_text_lines = [clause_match.group(2).strip()]
                continue

            point_match = re.match(r"^([a-z])[\)\.]\s+(.*)", text, re.I)
            if point_match:
                item = _get_item()
                if item: yield item
                cur_point = point_match.group(1).lower()
                cur_text_lines = [point_match.group(2).strip()]
                continue

            if cur_article:
                cur_text_lines.append(text)

        item = _get_item(); 
        if item: yield item

    def _extract_item_id(self, url):
        match = re.search(r"ItemID=(\d+)", url, re.I)
        return match.group(1) if match else None
