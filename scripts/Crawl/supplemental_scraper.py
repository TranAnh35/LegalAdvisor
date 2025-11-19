# -*- coding: utf-8 -*-
"""
Script bổ sung để tìm kiếm các văn bản không thành công từ nguồn chính
trên một trang web thay thế (luatminhkhue.vn).
ĐÃ ĐƯỢC CẢI TIẾN ĐỂ TRÁNH LỖI 403 VÀ LÀM PARSER LINH HOẠT HƠN (KHÔNG DỰA VÀO CLASS CỤ THỂ).
"""

import json
import time
import re
import cloudscraper
from bs4 import BeautifulSoup

# ==============================================================================
# CẤU HÌNH
# ==============================================================================

FAILED_INPUT_FILE = "scripts\\Crawl\\output_failed.json"
SUCCESS_OUTPUT_FILE = "scripts\\Crawl\\output_success.json"

BASE_URL = "https://luatminhkhue.vn"

REQUEST_DELAY = 2.0  # Delay để tránh bị block

DOC_TYPE_MAP = {
    'TT': 'thong-tu', 'TTLT': 'thong-tu-lien-tich', 'NĐ': 'nghi-dinh',
    'NĐ-CP': 'nghi-dinh', 'NQ': 'nghi-quyet', 'QH': 'luat', 'QĐ': 'quyet-dinh',
    'CT': 'chi-thi', 'L': 'lenh', 'PL': 'phap-lenh',
}

# ==============================================================================
# KHỞI TẠO SCRAPER (BỎ QUA CLOUDFLARE)
# ==============================================================================

def create_scraper():
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'windows',
            'mobile': False
        },
        delay=10,
    )
    print("Đã khởi tạo cloudscraper - Bỏ qua Cloudflare...")
    # Test trang chủ
    try:
        resp = scraper.get(BASE_URL)
        print(f"Trang chủ: {resp.status_code}")
        if resp.status_code == 200:
            print("Kết nối thành công!")
        time.sleep(1)
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
    return scraper

# ==============================================================================
# TẠO URL
# ==============================================================================

def generate_minhkhue_url(doc_id: str) -> str:
    slug = doc_id.replace('/', '-').lower()
    last_part = doc_id.split('/')[-1].upper()
    match = re.match(r'([A-Z]+)', last_part)
    if not match:
        raise ValueError("Không xác định loại văn bản.")
    doc_type_abbr = match.group(1)
    doc_type_slug = DOC_TYPE_MAP.get(doc_type_abbr)
    if not doc_type_slug:
        raise ValueError(f"Loại văn bản '{doc_type_abbr}' chưa định nghĩa.")
    return f"{BASE_URL}/van-ban/{doc_type_slug}-{slug}.aspx"

# ==============================================================================
# CRAWL TRANG CHI TIẾT (PARSER LINH HOẠT HƠN)
# ==============================================================================

def scrape_minhkhue_page(url: str, scraper, doc_id: str) -> dict:
    try:
        response = scraper.get(url, timeout=30)
        print(f"  -> Status: {response.status_code}")

        if response.status_code == 403:
            raise Exception("403 - Cloudflare vẫn chặn (hiếm)")
        if response.status_code == 404:
            raise ValueError("404 - Không tồn tại")
        if "Tài liệu bạn tìm kiếm không tồn tại!" in response.text:
            raise ValueError("Trang không tồn tại (nội dung)")

        soup = BeautifulSoup(response.content, "html.parser", from_encoding="utf-8")

        info = {"url": url}

        # Tìm h1 linh hoạt (không dùng class cụ thể, lấy h1 đầu tiên)
        title_tag = soup.find('h1')
        if not title_tag:
            # Nếu không tìm thấy h1, thử tìm div hoặc span với class chứa 'title'
            title_tag = soup.find(lambda tag: tag.name in ['div', 'span', 'h2'] and 'title' in tag.get('class', []))
            if not title_tag:
                raise ValueError("Không tìm thấy tiêu đề (h1 hoặc tương tự)")
        info['trich_yeu'] = title_tag.get_text(strip=True).strip()

        # Tìm table linh hoạt: tìm table chứa 'Số hiệu văn bản'
        info_table = None
        for table in soup.find_all('table'):
            if 'Số hiệu văn bản' in table.get_text():
                info_table = table
                break
        if not info_table:
            # Nếu không tìm thấy, thử tìm table đầu tiên
            info_table = soup.find('table')
            if not info_table:
                raise ValueError("Không tìm thấy bảng thông tin")

        # Debug: In ra bảng để kiểm tra
        print("  -> Bảng thông tin tìm thấy: OK")

        for row in info_table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) >= 2:  # Linh hoạt hơn, >=2 thay vì ==2
                key = cells[0].get_text(strip=True).lower()
                val = ''.join(c.get_text(strip=True) for c in cells[1:])  # Kết hợp nếu nhiều td
                if 'số hiệu văn bản' in key or 'so hieu' in key: info['so_hieu'] = val
                elif 'loại văn bản' in key or 'loai van ban' in key: info['loai_van_ban'] = val
                elif 'cơ quan ban hành' in key or 'co quan ban hanh' in key: info['co_quan_ban_hanh'] = val

        if not all(k in info for k in ['so_hieu', 'trich_yeu']):
            # Debug: Nếu thiếu, in HTML để kiểm tra
            debug_filename = f"debug_{doc_id.replace('/', '_')}.html"
            with open(debug_filename, 'w', encoding='utf-8') as f:
                f.write(response.text)
            raise ValueError(f"Thiếu thông tin bắt buộc (đã lưu HTML debug: {debug_filename})")

        return info

    except Exception as e:
        raise e

# ==============================================================================
# XỬ LÝ CHÍNH
# ==============================================================================

def process_supplemental_batch():
    print("="*70)
    print("BẮT ĐẦU CRAWL BỔ SUNG TỪ LUATMINHKHUE.VN (DÙNG cloudscraper)")
    print(f"  - Thất bại: {FAILED_INPUT_FILE}")
    print(f"  - Thành công: {SUCCESS_OUTPUT_FILE}")
    print("="*70)

    # Đọc file thất bại
    try:
        with open(FAILED_INPUT_FILE, 'r', encoding='utf-8') as f:
            failed_ids = json.load(f)
        if not failed_ids:
            print("Không có dữ liệu để xử lý.")
            return
        print(f"Đang xử lý {len(failed_ids)} văn bản...")
    except FileNotFoundError:
        print(f"KHÔNG TÌM THẤY: {FAILED_INPUT_FILE}")
        return
    except json.JSONDecodeError as e:
        print(f"JSON lỗi: {e}")
        return

    scraper = create_scraper()
    newly_found_docs = []
    still_failed_ids = []
    total = len(failed_ids)

    for i, doc_id in enumerate(failed_ids, 1):
        print(f"\n--- [{i}/{total}] Xử lý: '{doc_id}' ---")
        try:
            url = generate_minhkhue_url(doc_id)
            print(f"  -> URL: {url}")
            info = scrape_minhkhue_page(url, scraper, doc_id)
            newly_found_docs.append(info)
            print(f"THÀNH CÔNG: '{doc_id}'")
        except Exception as e:
            still_failed_ids.append(doc_id)
            print(f"THẤT BẠI: '{doc_id}' | Lỗi: {e}")
        time.sleep(REQUEST_DELAY)

    # === GHI KẾT QUẢ ===
    print("\n" + "="*70)
    print("HOÀN TẤT")
    print(f"  + Thành công: {len(newly_found_docs)}")
    print(f"  - Thất bại: {len(still_failed_ids)}")

    if newly_found_docs:
        try:
            with open(SUCCESS_OUTPUT_FILE, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except:
            existing = []
        existing.extend(newly_found_docs)
        with open(SUCCESS_OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing, f, ensure_ascii=False, indent=4)
        print(f"Đã ghi {len(newly_found_docs)} bản ghi mới")

    with open(FAILED_INPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(still_failed_ids, f, ensure_ascii=False, indent=4)
    print(f"Cập nhật file thất bại: {len(still_failed_ids)} còn lại")
    print("="*70)


# ==============================================================================
# CHẠY
# ==============================================================================

if __name__ == "__main__":
    process_supplemental_batch()