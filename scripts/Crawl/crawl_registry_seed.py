# -*- coding: utf-8 -*-
"""
Script xử lý hàng loạt để lấy thông tin văn bản pháp luật từ
Cổng thông tin điện tử Chính phủ (vanban.chinhphu.vn).

Cách hoạt động:
1. Đọc danh sách các số hiệu văn bản từ một file JSON đầu vào.
2. Lần lượt tìm kiếm và trích xuất thông tin cho từng số hiệu.
3. Ghi các kết quả thành công vào một file JSON đầu ra.
4. Ghi các số hiệu không thể xử lý (lỗi) vào một file JSON khác.
"""

import json
import time
import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ==============================================================================
# ĐỊNH NGHĨA CÁC HẰNG SỐ VÀ CẤU HÌNH
# ==============================================================================
BASE_URL = "https://vanban.chinhphu.vn"
SEARCH_PATH = "/he-thong-van-ban"
SEARCH_URL = f"{BASE_URL}{SEARCH_PATH}"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
    "Connection": "keep-alive",
    "Referer": BASE_URL,
}

# Thời gian nghỉ (giây) giữa các lần request để tránh làm quá tải server
REQUEST_DELAY = 1

# ==============================================================================
# HÀM LẤY THÔNG TIN VĂN BẢN (Không thay đổi so với phiên bản trước)
# ==============================================================================
def _normalize_id(s: str) -> str:
    """Chuẩn hóa số hiệu để so sánh/đối chiếu (bỏ khoảng trắng, viết hoa)."""
    return (s or "").replace(" ", "").upper()

def fetch_document_info(doc_id: str, session: requests.Session, timeout: int = 30) -> dict:
    """
    Tìm kiếm và trích xuất thông tin chi tiết của một văn bản.
    Sử dụng lại session đã có để tăng hiệu quả.
    """
    cleaned_doc_id = doc_id.replace(" ", "").upper()
    
    # Bước 1 & 2: Gửi yêu cầu tìm kiếm
    resp_initial = session.get(SEARCH_URL, timeout=timeout)
    resp_initial.raise_for_status()
    soup_initial = BeautifulSoup(resp_initial.content, "html.parser")
    
    form = soup_initial.find("form", {"method": "post"}) or soup_initial.find("form")
    if not form:
        raise ValueError("Không tìm thấy thẻ <form> trên trang tìm kiếm.")
        
    payload = {
        inp.get("name"): inp.get("value", "")
        for inp in form.find_all("input") if inp.get("name")
    }
    search_input_name = next(
        (inp.get("name") for inp in form.find_all("input") if "txtSearchKeyword" in inp.get("name", "")), None
    )
    if not search_input_name:
        raise ValueError("Không tìm thấy ô nhập từ khóa tìm kiếm.")
    payload[search_input_name] = doc_id

    resp_search = session.post(SEARCH_URL, data=payload, timeout=timeout)
    resp_search.raise_for_status()
    soup_search = BeautifulSoup(resp_search.content, "html.parser")

    # Bước 3: Phân tích kết quả
    result_table = soup_search.find("table", class_="search-result")
    if not result_table:
        raise ValueError(f"Không có bảng kết quả cho số hiệu '{doc_id}'.")

    document_url = None
    for row in result_table.find_all("tr")[1:]:
        code_tag = row.find("span", class_="code")
        if not code_tag: continue
        
        current_doc_id = code_tag.get_text(strip=True)
        if cleaned_doc_id == current_doc_id.replace(" ", "").upper():
            link_tag = row.find("a", href=lambda h: h and "docid=" in h)
            if link_tag:
                raw_href = link_tag["href"]
                document_url = f"{BASE_URL}{raw_href}" if raw_href.startswith("/") else raw_href
                break
    
    if not document_url:
        raise ValueError(f"Không tìm thấy văn bản khớp chính xác số hiệu '{doc_id}'.")

    # Bước 4 & 5: Lấy thông tin chi tiết và xác thực
    resp_detail = session.get(document_url, timeout=timeout)
    resp_detail.raise_for_status()
    soup_detail = BeautifulSoup(resp_detail.content, "html.parser")

    info = {"url": document_url}
    for row in soup_detail.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 2:
            key = cells[0].get_text(strip=True).replace(":", "").strip()
            val = cells[1].get_text(strip=True).strip()
            if "Số ký hiệu" in key: info["so_hieu"] = val
            elif "Loại văn bản" in key: info["loai_van_ban"] = val
            elif "Cơ quan ban hành" in key: info["co_quan_ban_hanh"] = val
            elif "Trích yếu" in key: info["trich_yeu"] = val
    
    if not info.get("so_hieu") or cleaned_doc_id != info["so_hieu"].replace(" ", "").upper():
        raise ValueError(f"Lỗi xác thực thông tin trên trang chi tiết cho '{doc_id}'.")

    return info

# ==============================================================================
# HÀM CHÍNH ĐỂ ĐIỀU PHỐI QUÁ TRÌNH
# ==============================================================================
def process_batch(input_filepath: str, success_filepath: str, failed_filepath: str):
    """
    Hàm chính để đọc file input, xử lý hàng loạt và ghi kết quả ra các file output.
    """
    print(f"Bắt đầu quá trình xử lý hàng loạt...")
    print(f"  - File đầu vào: '{input_filepath}'")
    
    # 1. Đọc file JSON đầu vào
    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            doc_ids = json.load(f)
        if not isinstance(doc_ids, list):
            raise TypeError("File JSON đầu vào phải chứa một danh sách (list) các số hiệu.")
    except FileNotFoundError:
        print(f"❌ LỖI: Không tìm thấy file đầu vào '{input_filepath}'. Vui lòng kiểm tra lại.")
        return
    except (json.JSONDecodeError, TypeError) as e:
        print(f"❌ LỖI: File '{input_filepath}' không phải là file JSON hợp lệ hoặc có cấu trúc sai. {e}")
        return

    # 2. Chuẩn bị cho quá trình xử lý
    successful_results = []  # Lưu trong bộ nhớ để tiện tổng kết cuối
    failed_ids = []          # Id thất bại
    session = requests.Session()
    session.headers.update(HEADERS)

    # Đảm bảo thư mục tồn tại
    for fp in (success_filepath, failed_filepath):
        parent = os.path.dirname(fp)
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    # Nếu file đã tồn tại, nạp lại (cho phép resume thô sơ); nếu chưa tồn tại, tạo file rỗng [] để rõ ràng.
    if os.path.exists(success_filepath):
        try:
            with open(success_filepath, 'r', encoding='utf-8') as f:
                existing_success = json.load(f)
            if isinstance(existing_success, list):
                successful_results.extend(existing_success)
        except Exception:
            pass  # Bỏ qua nếu hỏng định dạng để không chặn tiến trình
    else:
        with open(success_filepath, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)

    if os.path.exists(failed_filepath):
        try:
            with open(failed_filepath, 'r', encoding='utf-8') as f:
                existing_failed = json.load(f)
            if isinstance(existing_failed, list):
                failed_ids.extend(existing_failed)
        except Exception:
            pass
    else:
        with open(failed_filepath, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False)

    # Tạo tập hợp các ID đã xử lý (từ file cũ) để hỗ trợ resume
    processed_success_ids = set()
    for item in successful_results:
        if isinstance(item, dict) and item.get("so_hieu"):
            processed_success_ids.add(_normalize_id(item.get("so_hieu")))
    processed_failed_ids = { _normalize_id(x) for x in failed_ids }
    processed_all = processed_success_ids | processed_failed_ids

    # Lọc danh sách cần xử lý (bỏ các id đã có trong success/failed)
    pending_doc_ids = [d for d in doc_ids if _normalize_id(d) not in processed_all]
    total_all = len(doc_ids)
    total_pending = len(pending_doc_ids)

    print(f"Tìm thấy {total_all} số hiệu, sẽ xử lý {total_pending} số hiệu còn lại (resume).")

    # 3. Lặp qua từng số hiệu và xử lý (hiển thị tiến trình với tqdm)
    for doc_id in tqdm(pending_doc_ids, total=total_pending, desc="Đang xử lý", unit="văn bản"):
        try:
            result = fetch_document_info(doc_id, session)
            # Tránh trùng lặp nếu input chứa trùng
            norm = _normalize_id(result.get("so_hieu", doc_id))
            if norm not in processed_success_ids:
                successful_results.append(result)
                processed_success_ids.add(norm)
            tqdm.write(f"✅ Thành công: '{doc_id}'")
            # Ghi ngay kết quả thành công (ghi toàn bộ danh sách để giữ JSON hợp lệ)
            try:
                with open(success_filepath, 'w', encoding='utf-8') as f_success:
                    json.dump(successful_results, f_success, ensure_ascii=False, indent=4)
            except Exception as write_err:
                tqdm.write(f"⚠️ Lỗi ghi file thành công: {write_err}")
        except (requests.exceptions.RequestException, ValueError) as e:
            norm = _normalize_id(doc_id)
            # Chỉ thêm vào danh sách thất bại nếu chưa có và chưa thành công trước đó
            if norm not in processed_failed_ids and norm not in processed_success_ids:
                failed_ids.append(doc_id)
                processed_failed_ids.add(norm)
            tqdm.write(f"❌ Thất bại: '{doc_id}'. Lý do: {e}")
            # Ghi ngay danh sách thất bại
            try:
                with open(failed_filepath, 'w', encoding='utf-8') as f_failed:
                    json.dump(failed_ids, f_failed, ensure_ascii=False, indent=4)
            except Exception as write_err:
                tqdm.write(f"⚠️ Lỗi ghi file thất bại: {write_err}")

        # Nghỉ một chút trước khi tiếp tục
        time.sleep(REQUEST_DELAY)

    # 4. Ghi kết quả ra các file JSON
    print("\n" + "="*50)
    print("QUÁ TRÌNH XỬ LÝ HOÀN TẤT!")
    print(f"  - Thành công: {len(successful_results)}")
    print(f"  - Thất bại: {len(failed_ids)}")

    # Hai file đã được cập nhật liên tục trong quá trình. Thông báo tồn tại.
    # Đảm bảo đồng bộ cuối cùng (ghi lại một lần nữa, đề phòng gián đoạn giữa chừng)
    try:
        with open(success_filepath, 'w', encoding='utf-8') as f_success:
            json.dump(successful_results, f_success, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ Không thể ghi cuối cùng file thành công: {e}")
    try:
        with open(failed_filepath, 'w', encoding='utf-8') as f_failed:
            json.dump(failed_ids, f_failed, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"⚠️ Không thể ghi cuối cùng file thất bại: {e}")

    print(f"✔️ File thành công: '{success_filepath}' (số mục: {len(successful_results)})")
    print(f"✔️ File thất bại: '{failed_filepath}' (số mục: {len(failed_ids)})")
    print("="*50)


# ==============================================================================
# ĐIỂM BẮT ĐẦU CHẠY SCRIPT
# ==============================================================================
if __name__ == "__main__":
    # Đặt tên cho các file
    INPUT_FILE = "data/registry/act_codes_unique.json"
    SUCCESS_FILE = "data/registry/law_registry.json"
    FAILED_FILE = "data/registry/law_registry_missing.json"

    process_batch(
        input_filepath=INPUT_FILE,
        success_filepath=SUCCESS_FILE,
        failed_filepath=FAILED_FILE
    )