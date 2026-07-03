# -*- coding: utf-8 -*-
"""
Script: AI Refiner (Gemini 1.5 Flash)
Mục đích: Đọc bản Markdown và sử dụng LLM để bóc tách cấu trúc cực kỳ chính xác 
cho các văn bản luật phức tạp (như luật sửa đổi 2025).
"""

import os
import sqlite3
import json
import google.generativeai as genai
from pathlib import Path

# --- Cấu hình ---
API_KEY = "AIzaSyBOhOS7tAKp6IUDnRCMGWursvXSaqX2B2c" # Hoặc đọc từ settings.py
DB_PATH = "data/database/legal_data.db"
MD_DIR = "data/raw/markdown"

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash', 
                              generation_config={"response_mime_type": "application/json"})

PROMPT_TEMPLATE = """
Bạn là một chuyên gia pháp luật và xử lý dữ liệu. 
Tôi sẽ cung cấp cho bạn nội dung một văn bản luật Việt Nam dưới dạng Markdown.
Nhiệm vụ của bạn: Phân tích và trích xuất cấu trúc Điều, Khoản, Điểm của văn bản này.

Đặc biệt lưu ý:
1. Nếu là luật sửa đổi, phải nhận diện chính xác nội dung trích dẫn (ví dụ: Điều 1 sửa đổi Điều 5 của luật cũ).
2. Kết quả trả về phải là một danh sách JSON, mỗi phần tử gồm:
   - type: 'article', 'clause', hoặc 'point'
   - num: Số điều/khoản/điểm (ví dụ: "1", "a")
   - title: Tiêu đề của Điều (nếu có)
   - content: Nội dung văn bản
   - parent_num: Số hiệu của node cha (để xây dựng Namespace ID)

Nội dung Markdown:
---
{content}
---

Chỉ trả về JSON, không giải thích gì thêm.
"""

PROMPT_CATALOG = "Hãy liệt kê danh sách các Điều (num, title) dưới dạng JSON từ văn bản sau: {content}"
PROMPT_STRUCTURE = """
Bóc tách chi tiết JSON cho các Điều: {target_articles}
Schema: {{"articles": [{{ "num": "..", "title": "..", "content": "..", "sub_items": [..] }}]}}
Nội dung: {content}
"""

def refine_document(doc_code):
    md_file = Path(MD_DIR) / f"{doc_code.replace('/', '_')}.md"
    if not md_file.exists():
        print(f"❌ Không tìm thấy file Markdown.")
        return

    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()

    print(f"📡 Stage 1: Phân tích mục lục...")
    cat_resp = model.generate_content(PROMPT_CATALOG.format(content=md_content[:50000]))
    catalog = json.loads(cat_resp.text.strip('`json\n '))
    
    all_results = []
    chunk_size = 5
    for i in range(0, len(catalog), chunk_size):
        chunk = catalog[i:i+chunk_size]
        nums = [c['num'] for c in chunk]
        print(f"📡 Stage 2: Đang bóc tách Điều {', '.join(nums)}...")
        
        response = model.generate_content(
            PROMPT_STRUCTURE.format(target_articles=nums, content=md_content[:500000])
        )
        data = json.loads(response.text.strip('`json\n '))
        all_results.extend(data.get('articles', []))
        time.sleep(1)
        
    return all_results

if __name__ == "__main__":
    # Ví dụ: Thử nghiệm với luật 79/2025/QH15
    # refine_document("79/2025/qh15")
    print("Script đã sẵn sàng. Hãy nhập API Key và gọi hàm refine_document(doc_code) cho các ca khó.")
