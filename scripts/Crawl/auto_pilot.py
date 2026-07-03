# -*- coding: utf-8 -*-
import sqlite3
import json
import time
import os
import google.generativeai as genai
from pathlib import Path
from datetime import datetime

# --- CẤU HÌNH ---
DB_PATH = "e:/Project/LegalAdvisor/data/database/legal_data.db"
GEMINI_API_KEY = "AIzaSyBOhOS7tAKp6IUDnRCMGWursvXSaqX2B2c"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash', 
                              generation_config={"response_mime_type": "application/json"})

# =============================================================================
# PROMPTS - Được thiết kế để xử lý CHÍNH XÁC cả văn bản gốc & văn bản sửa đổi
# =============================================================================

PROMPT_CATALOG = """
Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam.

NHIỆM VỤ: Xác định cấu trúc cấp cao nhất và liệt kê CHÍNH XÁC các Điều của văn bản.

## QUY TẮC QUAN TRỌNG:

### Quy tắc 1: Phân biệt Điều GỐC và Điều ĐƯỢC SỬA ĐỔI
- Điều GỐC (cần liệt kê): là các dòng `**Điều X.` hoặc `Điều X.` ở ĐẦU DÒNG, và KHÔNG nằm bên trong dấu ngoặc kép.
- Điều ĐƯỢC SỬA ĐỔI (KHÔNG liệt kê): là các Điều xuất hiện bên trong dấu ngoặc kép `"..."`. Chúng chỉ là NỘI DUNG của một Khoản thuộc Điều gốc.

### Quy tắc 2: Ví dụ minh họa với văn bản SỬA ĐỔI
```
Điều 1. Sửa đổi, bổ sung một số điều của Luật XYZ   <- Điều GỐC, cần liệt kê
  4. Sửa đổi, bổ sung Điều 28 như sau:               <- Khoản 4 của Điều 1
  "Điều 28. Nhiệm vụ của Viện kiểm sát..."           <- Nội dung sửa đổi, KHÔNG phải Điều gốc
Điều 2. Hiệu lực thi hành                            <- Điều GỐC, cần liệt kê
```
Kết quả đúng: chỉ có Điều 1 và Điều 2.

TRẢ VỀ JSON:
{{
  "doc_type": "original hoặc amendment hoặc guidance",
  "items": [
    {{"num": "1", "title": "Tiêu đề Điều 1"}},
    {{"num": "2", "title": "Tiêu đề Điều 2"}}
  ]
}}

Nội dung văn bản:
{content}
"""

PROMPT_STRUCTURE = """
Bạn là chuyên gia pháp luật Việt Nam. Hãy bóc tách chi tiết CÁC ĐIỀU sau: {target_articles}

## QUY TẮC BÓC TÁCH:

### 1. Với văn bản GỐC (doc_type=original)
Cấu trúc đa tầng: Điều → Khoản (clause) → Điểm (point).

### 2. Với văn bản SỬA ĐỔI/BỔ SUNG (doc_type=amendment)
- Cấu trúc thường gặp: Câu dẫn lệnh (VD: "[Số]. Sửa đổi Điều 28 như sau:") và phần Nội dung mới (nằm trong dấu ngoặc kép `"..."`).
- LƯU Ý TỐI QUAN TRỌNG: Bạn PHẢI NGHIÊM TÚC LƯU TOÀN BỘ câu dẫn lệnh VÀ toàn bộ nội dung mới vào duy nhất trường `content`. TUYỆT ĐỐI KHÔNG BỎ SÓT NỘI DUNG NGOẶC KÉP!
- Nếu bên trong Khoản có các Điểm nhỏ hơn (a, b, c...) để sửa đổi, hãy trích xuất chúng thành `sub_items` bên trong Khoản đó với `type="point"`.

### 3. Quy tắc bắt buộc
- GIỮ NGUYÊN 100% nội dung chữ gốc, không tóm tắt hay cắt gọt.
- `num` của Điều, Khoản, Điểm LUÔN là String ("1", "2", "a", "b").
- Xét `is_amendment` = true đối với mọi thành phần mang tính chất sửa đổi, bổ sung, thay thế.

DỮ LIỆU ĐẦU RA JSON BẮT BUỘC:
{{
  "metadata": {{
    "official_title": "Tên chính thức của văn bản",
    "doc_code": "Số ký hiệu (VD: 82/2025/QH15)",
    "doc_type": "original hoặc amendment"
  }},
  "articles": [
    {{
      "num": "1",
      "title": "Tiêu đề Điều 1",
      "content": "Phần nội dung dẫn nhập (ví dụ: 'Luật này sửa đổi các điều sau đây:')",
      "sub_items": [
        {{
          "type": "clause",
          "num": "5",
          "content": "5. Sửa đổi Điều 28 như sau: \"Điều 28. Nhiệm vụ của Viện kiểm sát...\"",
          "is_amendment": true
        }}
      ]
    }}
  ]
}}

Nội dung Markdown:
---
{content}
---
"""


def autopilot_single_doc(doc_code, md_content, conn, api_key=None):
    """Trích xuất chi tiết cho MỘT văn bản dùng AI với cơ chế xử lý Rate Limit."""
    if api_key:
        genai.configure(api_key=api_key)
    
    cursor = conn.cursor()
    
    def safe_generate(prompt_text):
        """Hàm gọi AI an toàn, tự động đợi nếu chạm giới hạn 15 RPM."""
        for attempt in range(5):
            try:
                return model.generate_content(prompt_text)
            except Exception as e:
                if "429" in str(e):
                    wait_time = (attempt + 1) * 10
                    print(f"⏳ Chạm giới hạn (429). Đang đợi {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
        return None

    try:
        print(f"📡 Stage 1: Nhận diện cấu trúc văn bản {doc_code}...")
        cat_resp = safe_generate(PROMPT_CATALOG.format(content=md_content))
        if not cat_resp: return
        
        cat_data = json.loads(cat_resp.text.strip('`json\n '))
        
        # Mới: catalog trả về {"doc_type": ..., "items": [...]}
        if isinstance(cat_data, dict):
            catalog = cat_data.get('items', [])
            doc_type = cat_data.get('doc_type', 'unknown')
        else:
            # Fallback nếu AI vẫn trả về list cũ
            catalog = cat_data
            doc_type = 'unknown'
        
        print(f"📄 Loại văn bản: {doc_type} | Tổng số Điều: {len(catalog)}")
        print(f"🚀 Stage 2: Bóc tách cuốn chiếu {len(catalog)} Điều...")
        chunk_size = 5
        for i in range(0, len(catalog), chunk_size):
            chunk = catalog[i:i + chunk_size]
            target_nums = []
            for c in chunk:
                if isinstance(c, dict):
                    target_nums.append(str(c.get('num', '')))
                else:
                    target_nums.append(str(c))

            
            prompt = PROMPT_STRUCTURE.format(target_articles=target_nums, content=md_content)
            response = safe_generate(prompt)
            if not response: continue
            
            # Làm sạch JSON cực kỳ nghiêm ngặt
            json_text = response.text.strip()
            if json_text.startswith('```json'): json_text = json_text[7:]
            if json_text.endswith('```'): json_text = json_text[:-3]
            json_text = json_text.strip()
            
            try:
                result = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"⚠️ JSON lỗi tại nhóm {target_nums}. Đang cố gắng tự sửa lỗi format...")
                # Giải pháp: Nếu AI quên escape dấu ngoặc kép trong content, ta thử replace
                # Với trường hợp này, tốt nhất là yêu cầu AI sinh lại một phần nhỏ hoặc log lại để check
                try:
                    # Thử một trick đơn giản: thay thế các dấu ngoặc kép không được escape
                    cleaned = re.sub(r'(?<!\\)"', '\\"', json_text) # Escape toàn bộ
                    # Nhưng cách này sẽ làm hỏng ngoặc kép của KEY. 
                    # Vì vậy, ta sẽ dùng prompt hướng dẫn AI chi tiết hơn ở trên.
                    raise e
                except:
                    print(f"❌ Không thể tự sửa JSON: {e}")
                    continue
            
            # Cập nhật Metadata (nếu có)
            meta = result.get('metadata', {})
            if meta.get('official_title'):
                cursor.execute("UPDATE documents SET title = ? WHERE doc_code = ?", (meta['official_title'], doc_code))

            for art in result.get('articles', []):
                art_num = art.get('num')
                node_id = f"{doc_code}_a_{art_num}"
                cursor.execute("""
                    INSERT OR REPLACE INTO articles (node_id, node_type, doc_code, parent_id, article_number, title, content)
                    VALUES (?, 'article', ?, ?, ?, ?, ?)
                """, (node_id, doc_code, doc_code, art_num, art.get('title'), art.get('content')))
                
                for item in art.get('sub_items', []):
                    i_type = item.get('type')
                    i_num = item.get('num')
                    i_node_id = f"{node_id}_{i_type[0]}_{i_num}"
                    cursor.execute("""
                        INSERT OR REPLACE INTO articles 
                        (node_id, node_type, doc_code, parent_id, article_number, clause_number, point_id, content, is_amendment)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (i_node_id, i_type, doc_code, node_id, art_num, i_num if i_type == 'clause' else None, i_num if i_type == 'point' else None, item.get('content'), 1 if item.get('is_amendment') else 0))
            
            conn.commit()
            # Nghỉ 4s để duy trì trung bình 15 requests/phút
            time.sleep(4)

        cursor.execute("UPDATE documents SET parse_status = 'SUCCESS' WHERE doc_code = ?", (doc_code,))
        conn.commit()
    except Exception as e:
        print(f"❌ Lỗi xử lý AI cho {doc_code}: {e}")
        conn.rollback()

def autopilot_run():
    """Hàm chạy theo mẻ cho các văn bản cũ, đọc MD từ file."""
    if not Path(DB_PATH).exists(): return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Lấy danh sách cần xử lý
    cursor.execute("SELECT doc_code FROM documents WHERE parse_status = 'PENDING_AI'")
    tasks = cursor.fetchall()
    
    MD_DIR = "e:/Project/LegalAdvisor/data/raw/markdown"
    
    for (doc_code,) in tasks:
        # Ưu tiên đọc từ file MD vì Database không còn lưu full text
        md_file = Path(MD_DIR) / f"{doc_code.replace('/', '_')}.md"
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                md_content = f.read()
            autopilot_single_doc(doc_code, md_content, conn)
        else:
            print(f"⚠️ Không tìm thấy file Markdown cho {doc_code}")
            
    conn.close()


if __name__ == "__main__":
    autopilot_run()
