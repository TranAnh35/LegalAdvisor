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
        self.existing_ids = set()
        # NOTE: self.settings không có sẵn ở đây (gán sau bởi from_crawler)
        # → Việc đọc DB sẽ thực hiện trong start_requests()

    def _load_existing_ids(self):
        """Tải danh sách ItemID đã có trong DB để hỗ trợ Incremental mode."""
        db_path = Path(self.settings.get("PROJECT_ROOT", ".")) / "data/database/legal_data.db"
        if not db_path.exists():
            self.logger.info("Chưa có DB, sẽ cào toàn bộ từ đầu.")
            return
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT source_url FROM documents WHERE source_url IS NOT NULL")
            for (url,) in cursor.fetchall():
                match = re.search(r"ItemID=(\d+)", url or "", re.I)
                if match:
                    self.existing_ids.add(match.group(1))
            conn.close()
            self.logger.info(f"Incremental mode: đã tải {len(self.existing_ids)} văn bản hiện có từ DB.")
        except Exception as e:
            self.logger.error(f"Lỗi tải cache DB: {e}")

    async def start(self):
        import sys
        # Tải danh sách văn bản hiện có nếu ở chế độ incremental
        if self.incremental:
            self._load_existing_ids()
            tqdm.write(f"[Incremental] Đã bỏ qua {len(self.existing_ids)} văn bản đã có trong DB.", file=sys.stderr)

        # Tiền khởi tạo tqdm ngay lập tức
        self.pbar = tqdm(total=None, desc="⏳ Đang kết nối server", unit="doc",
                         dynamic_ncols=True, file=sys.stderr)
        tqdm.write("🚀 Spider vbpl khởi động...", file=sys.stderr)

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
                import sys
                total_str = response.css(".selected span strong b::text").get()
                if total_str:
                    clean_total = re.sub(r"\D", "", total_str)
                    self.total_docs = int(clean_total)
                    # Đóng thanh indeterminate, tạo lại với total thực tế
                    if self.pbar is not None:
                        self.pbar.close()
                    self.pbar = tqdm(
                        total=self.total_docs,
                        desc="📥 Cào văn bản",
                        unit="doc",
                        dynamic_ncols=True,
                        file=sys.stderr,
                        initial=len(self.existing_ids)  # Bắt đầu từ số đã có nếu incremental
                    )
                    tqdm.write(f"✅ Tổng {self.total_docs:,} văn bản | Đã có: {len(self.existing_ids):,} | Cần cào: {self.total_docs - len(self.existing_ids):,}", file=sys.stderr)
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
                    if self.pbar is not None: self.pbar.update(1)
                    continue
                
                meta_url = f"{self.BASE_URL}/TW/Pages/vbpq-thuoctinh.aspx?ItemID={item_id}"
                yield scrapy.Request(meta_url, callback=self.parse_metadata, meta={'item_id': item_id})

        if self.max_pages and page >= self.max_pages:
            return

        if len(doc_links) > 0:
            next_page = page + 1
            next_url = f"{self.SEARCH_API_URL}&Page={next_page}"
            yield scrapy.Request(next_url, callback=self.parse_catalog, meta={'page': next_page})

    def _is_host_structure(self, raw_text, p_class, el):
        """Kiểm tra xem một thẻ HTML có chắc chắn thuộc về cấu trúc văn bản hiện tại (Host) không."""
        # 1. Dấu hiệu Class HTML từ Bộ Tư pháp (Host Article luôn có class dieu-p/dieu-h)
        if "dieu-p" in p_class or bool(el.xpath(".//span[contains(@class, 'dieu-h')]")):
            return True
        
        # 2. Dấu hiệu các mục lớn (thường không nằm trong quote sửa đổi chi tiết Điều)
        if re.match(r"^(PHẦN|Phần|CHƯƠNG|Chương|MỤC|Mục)\s+", raw_text, re.I):
            return True
            
        # 3. Dấu hiệu câu dẫn sửa đổi đặc trưng của văn bản hiện hành (Host lead-in)
        if re.match(r"^\d+\.\s+", raw_text):
            lower_text = raw_text.lower()
            
            # Loại bỏ các từ ghép dễ gây nhận diện nhầm cấu trúc pháp lý (Bug 1: địa điểm, điều kiện...)
            clean_text = re.sub(r"(địa điểm|thời điểm|quan điểm|đặc điểm|điều kiện|điều chỉnh|điều hành|điều tra|tài khoản)", "", lower_text)
            
            action_pattern = r"\b(sửa đổi|bổ sung|bãi bỏ|thay thế|hủy bỏ|thay)\b"
            target_pattern = r"\b(điều|khoản|điểm|cụm từ|đoạn|mục|chương)\b"
            
            has_action = bool(re.search(action_pattern, clean_text))
            has_target = bool(re.search(target_pattern, clean_text))
            
            # Chỉ coi là Host nếu dòng đó chứa cả hành động và đối tượng (ví dụ: "1. Khoản 2 Điều 4 được sửa đổi...")
            if (has_action and has_target) or ("như sau:" in clean_text and has_target):
                return True
        return False

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
            if self.pbar is not None: self.pbar.update(1)
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
            if self.pbar is not None: self.pbar.update(1)
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
        # Cờ trạng thái: Đánh dấu đang nằm trong vùng nội dung sửa đổi/bổ sung (trích dẫn)
        in_amendment_block = False

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
            raw_text = "".join(el.xpath(".//text()").getall()).strip()
            if not raw_text: continue

            # Lấy class của thẻ p và các span bên trong để phân biệt Chủ (văn bản hiện tại) và Khách (văn bản bị sửa)
            p_class = el.xpath("./@class").get() or ""
            is_host_article = "dieu-p" in p_class or bool(el.xpath(".//span[contains(@class, 'dieu-h')]"))
            
            # --- XỬ LÝ TRẠNG THÁI KHỐI TRÍCH DẪN (AMENDMENT BLOCK) ---
            
            # Thoát quote nếu gặp cấu trúc chủ chắc chắn (phòng hờ gõ thiếu nháy đóng)
            if self._is_host_structure(raw_text, p_class, el):
                in_amendment_block = False

            # Nhận diện class của Điều khách (Guest Article)
            is_guest_article = bool(el.xpath(".//span[contains(@class, 'dieuchar-h')]"))
            
            # Nếu dòng bắt đầu bằng nháy kép hoặc là Điều của văn bản bị sửa (Guest)
            # (Chúng ta ưu tiên check startswith để xử lý các khối thụt lề)
            if not in_amendment_block and (raw_text.startswith(('“', '"')) or is_guest_article):
                in_amendment_block = True

            # Nếu đang ở trong vùng trích dẫn, gộp toàn bộ vào text của Điều/Khoản chủ hiện tại
            if in_amendment_block:
                if cur_article is not None:
                    cur_text_lines.append(raw_text)
                
                # Nếu kết thúc bằng nháy kép đóng, thoát vùng trích dẫn cho dòng sau
                if raw_text.endswith(('”', '"')):
                    in_amendment_block = False
                continue

            # --- BÓC TÁCH CẤU TRÚC (CHỈ CHẠY KHI KHÔNG Ở TRONG VÙNG TRÍCH DẪN) ---
            matched_structure = False

            # 1. Tách Điều (Chỉ nhận những thẻ được VBPL đánh dấu là Điều chủ)
            if is_host_article:
                art_match = re.match(r"^(Điều|ĐIỀU)\s+(\d+[a-zA-Z]*)[\.:\s]*(.*)", raw_text, re.I)
                if art_match:
                    item = _get_item()
                    if item: yield item
                    cur_article = art_match.group(2)
                    cur_title = art_match.group(3).strip()
                    cur_clause = cur_point = None
                    cur_text_lines = []
                    # Nếu title quá dài, đẩy bớt vào nội dung
                    if cur_title and len(cur_title) > 100:
                        cur_text_lines.append(cur_title); cur_title = ""
                    matched_structure = True

            # 2. Tách Phần/Chương/Mục
            if not matched_structure:
                part_match = re.match(r"^(PHẦN|Phần|CHƯƠNG|Chương|MỤC|Mục)\s+(.*)", raw_text, re.I)
                if part_match:
                    item = _get_item() 
                    if item: yield item
                    label = part_match.group(1).capitalize()
                    val = part_match.group(2).strip()
                    if label == "Phần": cur_h = [f"Phần {val}"]
                    elif label == "Chương": cur_h = cur_h[:1] + [f"Chương {val}"] if cur_h and "Phần" in cur_h[0] else [f"Chương {val}"]
                    else: cur_h.append(f"{label} {val}")
                    cur_article = cur_clause = cur_point = None
                    cur_text_lines = []; cur_title = ""
                    matched_structure = True

            # 3. Tách Khoản (Thỏa mãn tiêu chuẩn số thứ tự đầu dòng)
            if not matched_structure:
                clause_match = re.match(r"^(\d+)\.\s+(.*)", raw_text)
                if clause_match:
                    item = _get_item()
                    if item: yield item
                    cur_clause = int(clause_match.group(1)); cur_point = None
                    cur_text_lines = [raw_text]  # Giữ nguyên text gốc bao gồm số thứ tự
                    matched_structure = True

            # 4. Tách Điểm
            if not matched_structure:
                point_match = re.match(r"^([a-zđ])[\)\.]\s+(.*)", raw_text, re.I)
                if point_match:
                    item = _get_item()
                    if item: yield item
                    cur_point = point_match.group(1).lower()
                    cur_text_lines = [raw_text]  # Giữ nguyên text gốc bao gồm ký hiệu điểm
                    matched_structure = True

            # 5. Ghi nhận nội dung thường
            if not matched_structure and cur_article is not None:
                cur_text_lines.append(raw_text)

            # --- XỬ LÝ QUOTE ĐA DÒNG (Multi-line Quote State Management) ---
            # Chỉ bật cờ vùng trích dẫn đa dòng nếu có dấu hiệu ngoặc kép MỞ mà không ĐÓNG (Bug 2)
            if not in_amendment_block:
                open_q = raw_text.count('“')
                close_q = raw_text.count('”')
                
                # Trích dẫn cong: Mở nhiều hơn đóng -> Bắt đầu khối đa dòng
                if open_q > close_q:
                    in_amendment_block = True
                # Trích dẫn thẳng (ASCII): Đếm nếu số lượng ngoặc là lẻ
                elif raw_text.count('"') % 2 != 0:
                    in_amendment_block = True

        item = _get_item(); 
        if item: yield item

    def _extract_item_id(self, url):
        match = re.search(r"ItemID=(\d+)", url, re.I)
        return match.group(1) if match else None
