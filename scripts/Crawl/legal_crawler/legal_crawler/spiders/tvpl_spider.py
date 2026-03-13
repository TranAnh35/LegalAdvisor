# -*- coding: utf-8 -*-
"""
Spider: Thư viện Pháp luật (thuvienphapluat.vn)
Nguồn dữ liệu HTML cực kỳ tin cậy, thay thế cho PDF parsing.

Đặc điểm:
- Cấu trúc Điều/Khoản được định danh bằng các thẻ <a> neo.
- Phân cấp (Phần, Chương) rõ ràng trong cây DOM.
- Chế độ bảo mật cao: Cần headers giả lập trình duyệt tốt và delay hợp lý.
"""

import scrapy
from scrapy.http import HtmlResponse
from urllib.parse import urljoin, urlparse
from datetime import datetime
import re
from legal_crawler.items import LegalDocumentItem, LegalArticleItem

class TvplSpider(scrapy.Spider):
    name = "tvpl"
    allowed_domains = ["m.thuvienphapluat.vn", "thuvienphapluat.vn"]
    
    # URL cơ sở - Ưu tiên Mobile vì ít bị block hơn và HTML sạch hơn
    BASE_URL = "https://m.thuvienphapluat.vn"
    SEARCH_TEMPLATE = "https://m.thuvienphapluat.vn/tim-van-ban.aspx?keyword={}"

    custom_settings = {
        "DOWNLOAD_DELAY": 3.0,          # Delay lâu hơn để tránh block
        "CONCURRENT_REQUESTS": 1,        # Chạy tuần tự an toàn
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 3,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 0.5,
        "DEFAULT_REQUEST_HEADERS": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "Sec-Ch-UA": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
            "Upgrade-Insecure-Requests": "1",
        }
    }

    def __init__(self, keyword=None, seed_file=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keyword = keyword
        self.seed_file = seed_file

    def start_requests(self):
        if self.seed_file:
            # TODO: Đọc file seed
            pass
        elif self.keyword:
            yield scrapy.Request(
                url=self.SEARCH_TEMPLATE.format(self.keyword),
                callback=self.parse_search_results
            )
        else:
            # Mặc định cào trang chủ văn bản mới
            yield scrapy.Request(
                url=f"{self.BASE_URL}/van-ban-moi.aspx",
                callback=self.parse_catalog
            )

    def parse_catalog(self, response):
        """Duyệt danh mục văn bản mới."""
        links = response.css("a.NQLink::attr(href)").getall()
        for link in links:
            yield response.follow(link, callback=self.parse_document_detail)

    def parse_search_results(self, response):
        """Xử lý kết quả tìm kiếm."""
        # Lấy link đầu tiên khớp nhất
        first_result = response.css("p.nqTitle a::attr(href)").get()
        if first_result:
            yield response.follow(first_result, callback=self.parse_document_detail)

    def parse_document_detail(self, response):
        """
        Phân tích nội dung HTML của văn bản từ Thư viện Pháp luật.
        """
        # 1. Trích xuất Metadata
        title = response.css("h1.title-vb::text").get() or response.css("div.content-html h1::text").get()
        
        # Số hiệu thường nằm trong bảng hoặc text
        doc_code_raw = response.xpath("//td[contains(text(), 'Số ký hiệu')]/following-sibling::td/text()").get()
        if not doc_code_raw:
             # Dự phòng nếu không có bảng
             match = re.search(r"Số:\s*([\w\-/]+)", response.text)
             doc_code_raw = match.group(1) if match else "unknown"

        doc_code_norm = doc_code_raw.strip().lower().replace(" ", "")

        # Lưu Document Item
        doc_item = LegalDocumentItem()
        doc_item["node_type"] = "document"
        doc_item["doc_code"] = doc_code_norm
        doc_item["title"] = title.strip() if title else ""
        doc_item["source_url"] = response.url
        doc_item["crawled_at"] = datetime.utcnow().isoformat() + "Z"
        
        # Trích xuất thêm metadata từ bảng tóm tắt nếu có
        metadata_table = response.xpath("//table[contains(@class, 'table-tomtat')]")
        if metadata_table:
            doc_item["issuer"] = metadata_table.xpath(".//td[contains(text(), 'Cơ quan')]/following-sibling::td/text()").get()
            doc_item["doc_type"] = metadata_table.xpath(".//td[contains(text(), 'Loại văn bản')]/following-sibling::td/text()").get()
            doc_item["issue_date"] = metadata_table.xpath(".//td[contains(text(), 'Ngày ban hành')]/following-sibling::td/text()").get()
            doc_item["effective_date"] = metadata_table.xpath(".//td[contains(text(), 'Ngày có hiệu lực')]/following-sibling::td/text()").get()
            doc_item["status"] = metadata_table.xpath(".//td[contains(text(), 'Tình trạng')]/following-sibling::td/span/text()").get()

        yield doc_item

        # 2. Phân rã nội dung (HTML Parsing)
        # Mobile version dùng #content-document-detail
        content_div = response.css("#content-document-detail")
        if not content_div:
            # Fallback desktop selectors
            content_div = response.css("div.content-html")
        if not content_div:
            content_div = response.css("div#divNoiDung")

        if not content_div:
            self.logger.warning(f"Không tìm thấy vùng nội dung tại {response.url}")
            return

        # Chiến thuật TVPL: Duyệt cây phẳng và gom nhóm theo Điều
        all_elements = content_div.xpath("./*")
        
        current_hierarchy = []
        current_article_num = None
        current_article_title = ""
        current_article_lines = []

        def _flush():
            nonlocal current_article_num, current_article_title, current_article_lines
            if current_article_num is not None:
                art = LegalArticleItem()
                art["node_type"] = "article"
                art["doc_code"] = doc_code_norm
                art["article_number"] = current_article_num
                art["title"] = current_article_title
                art["content"] = "\n".join(current_article_lines).strip()
                art["hierarchy_path"] = list(current_hierarchy)
                art["source_url"] = response.url
                art["crawled_at"] = datetime.utcnow().isoformat() + "Z"
                return art
            return None

        for idx, el in enumerate(all_elements):
            text = "".join(el.xpath(".//text()").getall()).strip()
            if not text: continue

            # Kiểm tra Phần/Chương
            if re.match(r"^(PHẦN|Phần)\s+", text, re.I):
                _flush_item = _flush()
                if _flush_item: yield _flush_item
                current_hierarchy = [text]
                current_article_num = None
                continue
            
            if re.match(r"^(CHƯƠNG|Chương)\s+", text, re.I):
                _flush_item = _flush()
                if _flush_item: yield _flush_item
                # Giữ Phần, thay Chương
                if current_hierarchy and current_hierarchy[0].lower().startswith("phần"):
                    current_hierarchy = current_hierarchy[:1] + [text]
                else:
                    current_hierarchy = [text]
                current_article_num = None
                continue

            # Kiểm tra Điều
            art_match = re.match(r"^(Điều|ĐIỀU)\s+(\d+)[\.:\s]*(.*)", text, re.I)
            if art_match:
                # Lưu điều cũ
                _flush_item = _flush()
                if _flush_item: yield _flush_item
                
                # Khởi tạo điều mới
                current_article_num = int(art_match.group(2))
                current_article_title = art_match.group(3).strip()
                current_article_lines = []
                continue

            # Nếu đang trong một Điều, gom nội dung
            if current_article_num is not None:
                current_article_lines.append(text)

        # Flush Điều cuối cùng
        _flush_item = _flush()
        if _flush_item: yield _flush_item
