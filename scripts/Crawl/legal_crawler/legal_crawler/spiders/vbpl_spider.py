# -*- coding: utf-8 -*-
import scrapy
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from markdownify import markdownify as md
from tqdm import tqdm
from legal_crawler.items import LegalDocumentItem

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
        self.item_id = item_id
        self.max_pages = int(max_pages) if max_pages else None
        self.incremental = str(incremental).lower() in ["true", "1", "yes"]
        self.pbar = None
        self.total_docs = 0
        self.existing_ids = set()

    def _load_existing_ids(self):
        db_path = Path(self.settings.get("PROJECT_ROOT", ".")) / "data/database/legal_data.db"
        if not db_path.exists(): return
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT doc_code FROM documents")
            for (code,) in cursor.fetchall(): self.existing_ids.add(code)
            conn.close()
        except: pass

    async def start(self):
        self.logger.info("🚀 Spider đã khởi động thành công!")
        if self.incremental: self._load_existing_ids()
        self.pbar = tqdm(total=0, desc="⏳ Đang tải", unit="doc", file=sys.stderr)
        
        headers = {'Referer': self.BASE_URL}
        
        if self.item_id:
            attr_url = f"{self.BASE_URL}/TW/Pages/vbpq-thuoctinh.aspx?ItemID={self.item_id}"
            self.logger.info(f"🔍 Đang cào ID: {self.item_id}")
            yield scrapy.Request(attr_url, callback=self.parse_metadata, meta={'item_id': self.item_id}, headers=headers)
        else:
            self.logger.info(f"🌐 Đấu nối vào Catalog trang 1...")
            yield scrapy.Request(f"{self.SEARCH_API_URL}&Page=1", callback=self.parse_catalog, meta={'page': 1}, headers=headers)

    def parse_catalog(self, response):
        page = response.meta.get('page', 1)
        self.logger.info(f"📄 Đang đọc danh mục trang {page}...")
        
        if page == 1:
            total_str = response.css(".selected span strong b::text").get()
            if total_str:
                self.total_docs = int(re.sub(r"\D", "", total_str))
                self.pbar.total = self.total_docs
                self.pbar.refresh()

        doc_links = response.css("div.item p.title a::attr(href)").getall()
        self.logger.info(f"🔗 Tìm thấy {len(doc_links)} văn bản tại trang này.")
        
        for link in doc_links:
            item_id = re.search(r"ItemID=(\d+)", link, re.I)
            if item_id:
                iid = item_id.group(1)
                yield scrapy.Request(f"{self.BASE_URL}/TW/Pages/vbpq-thuoctinh.aspx?ItemID={iid}", 
                                     callback=self.parse_metadata, meta={'item_id': iid})

        if (not self.max_pages or page < self.max_pages) and doc_links:
            yield scrapy.Request(f"{self.SEARCH_API_URL}&Page={page+1}", callback=self.parse_catalog, meta={'page': page+1})

    def parse_metadata(self, response):
        def get_val(label):
            return " ".join(response.xpath(f"//td[contains(normalize-space(), '{label}')]/following-sibling::td[1]//text()").getall()).strip()

        item_id = response.meta['item_id']
        doc_code = (get_val("Số ký hiệu") or f"unknown_{item_id}").lower().strip().replace(" ", "")
        
        if self.incremental and doc_code in self.existing_ids:
            self.pbar.update(1)
            return

        self.logger.info(f"📑 Đang xử lý metadata cho: {doc_code}")
        metadata = {
            'doc_code': doc_code,
            'doc_type': get_val("Loại văn bản"),
            'issuer': get_val("Cơ quan ban hành"),
            'issue_date': get_val("Ngày ban hành"),
            'effective_date': get_val("Ngày có hiệu lực"),
            'status': response.xpath("//td[contains(normalize-space(), 'Tình trạng hiệu lực')]//text()").get("").split(":")[-1].strip(),
            'item_id': item_id
        }
        yield scrapy.Request(f"{self.BASE_URL}/TW/Pages/vbpq-toanvan.aspx?ItemID={item_id}", 
                             callback=self.parse_full_text, meta=metadata)

    def parse_full_text(self, response):
        self.logger.info(f"🖋️ Đang tải toàn văn: {response.meta['doc_code']}")
        content_div = response.css("div.toanvancontent") or response.css("div#contentDoc")
        if not content_div:
            self.pbar.update(1)
            return

        html_raw = content_div.get()
        full_md = md(html_raw, heading_style="ATX", strip=['a', 'span'])
        
        title_match = re.search(r"^#*\s*(.*?)(?=\n\n|\n---|\n\*\*|Ngày|$)", full_md, re.S)
        title = title_match.group(1).strip() if title_match else response.meta['doc_type']

        yield LegalDocumentItem(
            doc_code=response.meta['doc_code'], title=title,
            doc_type=response.meta['doc_type'], issuer=response.meta['issuer'],
            issue_date=response.meta['issue_date'], effective_date=response.meta['effective_date'],
            status=response.meta['status'], source_url=response.url, 
            full_markdown=full_md,
            parse_status="PENDING_AI",
            parse_log="Waiting for AI extraction"
        )
        self.pbar.update(1)
