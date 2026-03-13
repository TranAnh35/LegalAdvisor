# -*- coding: utf-8 -*-
"""
Spider: Cào toàn văn pháp luật từ Cổng TTĐT Chính phủ (vanban.chinhphu.vn).

Quy trình hai giai đoạn:
  Stage 1: Duyệt danh mục văn bản theo loại (Luật, Nghị định, Thông tư...)
            → Thu thập URL chi tiết từng văn bản.
  Stage 2: Truy cập trang chi tiết → Phân rã nội dung theo Điều.

Cách chạy:
  # Cào từ danh mục (mặc mặc định)
  scrapy crawl chinhphu

  # Cào từ danh sách doc_codes có sẵn (file JSON)
  scrapy crawl chinhphu -a seed_file=data/registry/act_codes_unique.json

  # Giới hạn số trang danh mục
  scrapy crawl chinhphu -a max_pages=5

  # Chỉ định output
  scrapy crawl chinhphu -a output_dir=data/raw/crawled
"""

import json
import re
import io
import pdfplumber
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlencode, urljoin

import scrapy
from scrapy.http import HtmlResponse

from legal_crawler.items import LegalDocumentItem, LegalArticleItem


class ChinhphuSpider(scrapy.Spider):
    name = "chinhphu"
    allowed_domains = ["vanban.chinhphu.vn", "datafiles.chinhphu.vn"]

    # URL cơ sở
    BASE_URL = "https://vanban.chinhphu.vn"

    # Các loại văn bản cần cào (có thể mở rộng)
    DOC_TYPE_IDS = {
        "1":  "Luật",
        "2":  "Pháp lệnh",
        "3":  "Nghị quyết",
        "4":  "Nghị định",
        "6":  "Quyết định",
        "7":  "Chỉ thị",
        "8":  "Thông tư",
        "9":  "Thông tư liên tịch",
    }

    # ==============================================================================
    # Custom settings cho spider cụ thể này
    # ==============================================================================
    custom_settings = {
        "DOWNLOAD_DELAY": 2.5,
        "CONCURRENT_REQUESTS": 2,
    }

    def __init__(
        self,
        seed_file: str = None,
        max_pages: int = 0,
        output_dir: str = None,
        doc_types: str = None,         # VD: "1,4,8" (Luật, Nghị định, Thông tư)
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed_file = seed_file
        self.max_pages = int(max_pages) if max_pages else 0
        self.output_dir = output_dir

        # Cho phép lọc loại văn bản
        if doc_types:
            self.target_doc_types = {
                t.strip()
                for t in doc_types.split(",")
                if t.strip() in self.DOC_TYPE_IDS
            }
        else:
            self.target_doc_types = set(self.DOC_TYPE_IDS.keys())

        self.crawled_codes = set()  # Theo dõi doc_code đã crawl

    # ==============================================================================
    # START REQUESTS
    # ==============================================================================
    def start_requests(self):
        """Khởi tạo quá trình crawl từ seed file hoặc từ danh mục."""
        if self.seed_file:
            yield from self._start_from_seed()
        else:
            yield from self._start_from_catalog()

    def _start_from_seed(self):
        """Đọc danh sách doc_codes từ file JSON và tìm kiếm từng mã."""
        seed_path = Path(self.seed_file)
        if not seed_path.exists():
            self.logger.error(f"Seed file not found: {seed_path}")
            return

        with open(seed_path, "r", encoding="utf-8") as f:
            doc_codes = json.load(f)

        self.logger.info(f"Loaded {len(doc_codes)} doc_codes from seed file")

        for code in doc_codes:
            code = str(code).strip()
            if not code:
                continue
            # Tìm kiếm trên vanban.chinhphu.vn
            search_url = f"{self.BASE_URL}/he-thong-van-ban?s={code}"
            yield scrapy.Request(
                url=search_url,
                callback=self.parse_search_results,
                cb_kwargs={"target_code": code},
                meta={"dont_cache": False},
            )

    def _start_from_catalog(self):
        """Duyệt danh mục văn bản theo loại."""
        for type_id in self.target_doc_types:
            type_name = self.DOC_TYPE_IDS.get(type_id, "unknown")
            catalog_url = (
                f"{self.BASE_URL}/he-thong-van-ban?"
                f"loaivanban={type_id}&page=1"
            )
            self.logger.info(f"Starting catalog crawl: {type_name}")
            yield scrapy.Request(
                url=catalog_url,
                callback=self.parse_catalog,
                cb_kwargs={
                    "doc_type_id": type_id,
                    "doc_type_name": type_name,
                    "current_page": 1,
                },
            )

    # ==============================================================================
    # PARSE: CATALOG LISTING
    # ==============================================================================
    def parse_catalog(self, response: HtmlResponse, doc_type_id: str, doc_type_name: str, current_page: int):
        """Parse trang danh sách văn bản → lấy link chi tiết."""
        # Tìm các link văn bản trong bảng kết quả
        doc_links = response.css("table.search-result tr a[href*='docid=']::attr(href)").getall()

        if not doc_links:
            # Thử selector khác
            doc_links = response.css("a[href*='pageid=27160']::attr(href)").getall()

        if not doc_links:
            self.logger.info(
                f"No more results for {doc_type_name} at page {current_page}"
            )
            return

        self.logger.info(
            f"[{doc_type_name}] Page {current_page}: found {len(doc_links)} documents"
        )

        for href in doc_links:
            detail_url = urljoin(self.BASE_URL, href)
            yield scrapy.Request(
                url=detail_url,
                callback=self.parse_document_detail,
                cb_kwargs={"doc_type_name": doc_type_name},
            )

        # Phân trang
        if self.max_pages and current_page >= self.max_pages:
            return

        next_page = current_page + 1
        next_url = (
            f"{self.BASE_URL}/he-thong-van-ban?"
            f"loaivanban={doc_type_id}&page={next_page}"
        )
        yield scrapy.Request(
            url=next_url,
            callback=self.parse_catalog,
            cb_kwargs={
                "doc_type_id": doc_type_id,
                "doc_type_name": doc_type_name,
                "current_page": next_page,
            },
        )

    # ==============================================================================
    # PARSE: SEARCH RESULTS (cho chế độ seed)
    # ==============================================================================
    def parse_search_results(self, response: HtmlResponse, target_code: str):
        """Parse kết quả tìm kiếm → tìm link chi tiết khớp với target_code."""
        target_norm = target_code.replace(" ", "").upper()

        rows = response.css("table.search-result tr")
        for row in rows:
            code_text = row.css("span.code::text").get("")
            code_norm = code_text.strip().replace(" ", "").upper()

            if code_norm == target_norm:
                href = row.css("a[href*='docid=']::attr(href)").get()
                if href:
                    detail_url = urljoin(self.BASE_URL, href)
                    yield scrapy.Request(
                        url=detail_url,
                        callback=self.parse_document_detail,
                        cb_kwargs={"doc_type_name": ""},
                    )
                    return

        self.logger.warning(f"No exact match found for: {target_code}")

    # ==============================================================================
    # PARSE: DOCUMENT DETAIL (Trang chi tiết văn bản)
    # ==============================================================================
    def parse_document_detail(self, response: HtmlResponse, doc_type_name: str = ""):
        """
        Parse trang chi tiết văn bản:
        1. Trích xuất metadata (số hiệu, cơ quan, ngày ban hành...)
        2. Phân rã nội dung theo Điều luật
        """
        # --- 1. Metadata từ bảng thông tin ---
        metadata = {}
        for row in response.css("tr"):
            cells = row.css("td")
            if len(cells) == 2:
                key = cells[0].css("::text").get("").strip().rstrip(":")
                val = cells[1].css("::text").get("").strip()
                if not val:
                    val = cells[1].css("*::text").getall()
                    val = " ".join(v.strip() for v in val if v.strip())

                if "Số ký hiệu" in key or "Số/Ký hiệu" in key:
                    metadata["so_hieu"] = val
                elif "Loại văn bản" in key:
                    metadata["loai_van_ban"] = val
                elif "Cơ quan ban hành" in key:
                    metadata["co_quan"] = val
                elif "Trích yếu" in key:
                    metadata["trich_yeu"] = val
                elif "Ngày ban hành" in key:
                    metadata["ngay_ban_hanh"] = val
                elif "Ngày có hiệu lực" in key or "Ngày hiệu lực" in key:
                    metadata["ngay_hieu_luc"] = val
                elif "Tình trạng" in key:
                    metadata["tinh_trang"] = val

        doc_code = metadata.get("so_hieu", "").strip()
        if not doc_code:
            self.logger.warning(f"No doc_code found at {response.url}")
            return

        # Chuẩn hóa doc_code
        doc_code_norm = doc_code.replace(" ", "").lower()

        # Tránh crawl trùng
        if doc_code_norm in self.crawled_codes:
            return
        self.crawled_codes.add(doc_code_norm)

        # --- Yield Document Item (Parent Node) ---
        doc_item = LegalDocumentItem()
        doc_item["node_type"] = "document"
        doc_item["doc_code"] = doc_code_norm
        doc_item["title"] = metadata.get("trich_yeu", "")
        doc_item["doc_type"] = metadata.get("loai_van_ban", doc_type_name)
        doc_item["issuer"] = metadata.get("co_quan", "")
        doc_item["issue_date"] = metadata.get("ngay_ban_hanh", "")
        doc_item["effective_date"] = metadata.get("ngay_hieu_luc", "")
        doc_item["status"] = metadata.get("tinh_trang", "")
        doc_item["source_url"] = response.url
        doc_item["crawled_at"] = datetime.utcnow().isoformat() + "Z"

        yield doc_item

        # --- 2. Kiểm tra PDF hoặc HTML ---
        # Kiểm tra xem có file PDF để tải về không (thường là link .pdf)
        pdf_links = response.css("a[href$='.pdf']::attr(href)").getall()
        # Ưu tiên các file PDF nằm trong vùng nội dung
        content_pdf_links = response.css("div.content-detail a[href$='.pdf']::attr(href)").getall()
        
        target_pdf = content_pdf_links[0] if content_pdf_links else (pdf_links[0] if pdf_links else None)

        if target_pdf:
            pdf_url = urljoin(self.BASE_URL, target_pdf)
            self.logger.info(f"PDF detected for {doc_code}: {pdf_url}")
            yield scrapy.Request(
                url=pdf_url,
                callback=self.parse_pdf,
                cb_kwargs={
                    "doc_code_norm": doc_code_norm,
                    "source_url": response.url
                },
                meta={"doc_item": doc_item} # Pass metadata
            )
            return # Đợi callback xử lý PDF

        # --- 3. Nếu không có PDF, thử phân rã từ HTML (hiện tại) ---
        # Tìm vùng nội dung chính
        content_area = response.css("div.content-detail")
        if not content_area:
            content_area = response.css("div.fulltext")
        if not content_area:
            content_area = response.css("div.doc-content")
        if not content_area:
            # Fallback: lấy toàn bộ body
            content_area = response

        # Lấy toàn bộ text từ content area
        full_text = content_area.css("*::text").getall()
        full_text = "\n".join(t.strip() for t in full_text if t.strip())

        if not full_text:
            self.logger.warning(f"No content found for {doc_code}")
            return

        # Parse từng Điều
        articles = self._split_into_articles(full_text)

        self.logger.info(
            f"Parsed HTML {doc_code}: {len(articles)} articles extracted"
        )

        for art_num, art_title, art_content, hierarchy in articles:
            article_item = LegalArticleItem()
            article_item["node_type"] = "article"
            article_item["doc_code"] = doc_code_norm
            article_item["article_number"] = art_num
            article_item["title"] = art_title
            article_item["content"] = art_content
            article_item["hierarchy_path"] = hierarchy
            article_item["source_url"] = response.url
            article_item["crawled_at"] = datetime.utcnow().isoformat() + "Z"

            yield article_item

    def parse_pdf(self, response, doc_code_norm: str, source_url: str):
        """Xử lý file PDF tải về."""
        self.logger.info(f"Processing PDF content for: {doc_code_norm}")
        
        try:
            full_text = ""
            with pdfplumber.open(io.BytesIO(response.body)) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text + "\n"

            if not full_text:
                self.logger.warning(f"Could not extract text from PDF: {response.url}")
                return

            articles = self._split_into_articles(full_text)
            self.logger.info(f"Parsed PDF {doc_code_norm}: {len(articles)} articles extracted")

            for art_num, art_title, art_content, hierarchy in articles:
                article_item = LegalArticleItem()
                article_item["node_type"] = "article"
                article_item["doc_code"] = doc_code_norm
                article_item["article_number"] = art_num
                article_item["title"] = art_title
                article_item["content"] = art_content
                article_item["hierarchy_path"] = hierarchy
                article_item["source_url"] = source_url
                article_item["crawled_at"] = datetime.utcnow().isoformat() + "Z"
                yield article_item

        except Exception as e:
            self.logger.error(f"Error processing PDF {response.url}: {str(e)}")


    # ==============================================================================
    # ARTICLE SPLITTER: Phân rã văn bản thành từng Điều
    # ==============================================================================
    def _split_into_articles(self, full_text: str) -> list:
        """
        Phân rã toàn văn thành các Điều riêng lẻ.

        Returns:
            list of (article_number, title, content, hierarchy_path)
        """
        articles = []
        current_hierarchy = []

        # Pattern nhận diện cấu trúc phân cấp (linh hoạt hơn với khoảng trắng)
        phan_re = re.compile(
            r"^\s*(?:PHẦN|Phần)\s+([IVXLCDM]+|[0-9]+)[\.\s:]?\s*(.*)",
            re.MULTILINE | re.IGNORECASE
        )
        chuong_re = re.compile(
            r"^\s*(?:CHƯƠNG|Chương)\s+([IVXLCDM]+|[0-9]+)[\.\s:]?\s*(.*)",
            re.MULTILINE | re.IGNORECASE
        )
        muc_re = re.compile(
            r"^\s*(?:MỤC|Mục)\s+([0-9]+)[\.\s:]?\s*(.*)",
            re.MULTILINE | re.IGNORECASE
        )

        # Pattern nhận diện Điều: "Điều X." hoặc "Điều X:" hoặc "ĐIỀU X"
        dieu_re = re.compile(
            r"^\s*(?:Điều|ĐIỀU)\s+(\d+)[\.:\s]*\s*(.*)",
            re.MULTILINE
        )

        lines = full_text.split("\n")
        current_article_num = None
        current_title = ""
        current_content_lines = []

        def _flush_article():
            """Lưu Điều hiện tại vào danh sách."""
            nonlocal current_article_num, current_title, current_content_lines
            if current_article_num is not None:
                content = "\n".join(current_content_lines).strip()
                if content:
                    articles.append((
                        current_article_num,
                        current_title.strip(),
                        content,
                        list(current_hierarchy),  # Snapshot hierarchy
                    ))
                current_article_num = None
                current_title = ""
                current_content_lines = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_article_num is not None:
                    current_content_lines.append("")
                continue

            # Kiểm tra Phần
            m = phan_re.match(line_stripped)
            if m:
                _flush_article()
                phan_label = f"Phần {m.group(1)}"
                phan_title = m.group(2).strip() if m.group(2) else ""
                current_hierarchy = [
                    f"{phan_label}. {phan_title}" if phan_title else phan_label
                ]
                continue

            # Kiểm tra Chương
            m = chuong_re.match(line_stripped)
            if m:
                _flush_article()
                chuong_label = f"Chương {m.group(1)}"
                chuong_title = m.group(2).strip() if m.group(2) else ""
                # Giữ Phần, cập nhật Chương
                if current_hierarchy and current_hierarchy[0].startswith("Phần"):
                    current_hierarchy = current_hierarchy[:1]
                else:
                    current_hierarchy = []
                current_hierarchy.append(
                    f"{chuong_label}. {chuong_title}" if chuong_title else chuong_label
                )
                continue

            # Kiểm tra Mục
            m = muc_re.match(line_stripped)
            if m:
                _flush_article()
                muc_label = f"Mục {m.group(1)}"
                muc_title = m.group(2).strip() if m.group(2) else ""
                # Giữ Phần + Chương, cập nhật Mục
                keep = []
                for h in current_hierarchy:
                    keep.append(h)
                    if h.startswith("Chương"):
                        break
                current_hierarchy = keep
                current_hierarchy.append(
                    f"{muc_label}. {muc_title}" if muc_title else muc_label
                )
                continue

            # Kiểm tra Điều
            m = dieu_re.match(line_stripped)
            if m:
                _flush_article()
                current_article_num = int(m.group(1))
                current_title = m.group(2).strip() if m.group(2) else ""
                # Dòng đầu tiên của điều cũng là nội dung
                # (một số điều có nội dung ngay sau tiêu đề trên cùng dòng)
                continue

            # Nội dung thuộc Điều hiện tại
            if current_article_num is not None:
                current_content_lines.append(line_stripped)

        # Flush điều cuối cùng
        _flush_article()

        return articles
