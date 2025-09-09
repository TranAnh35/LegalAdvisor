# Hướng Dẫn Sử Dụng LegalAdvisor với Google Gemini

## Tổng Quan

LegalAdvisor sử dụng Google Gemini làm model sinh văn bản duy nhất. Hệ thống không còn hỗ trợ GPT-2 local nữa để đảm bảo chất lượng và tính nhất quán cao nhất.

## Yêu Cầu

### 1. Cài Đặt Dependencies

```bash
pip install google-generativeai
```

### 2. Thiết Lập Google Gemini API Key

1. Truy cập [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Tạo API key mới
3. Sao chép API key

### 3. Cấu Hình Environment Variables

1. Mở file `.env` (đã được tạo tự động)
2. Thay thế `your_google_gemini_api_key_here` bằng API key thực tế của bạn:

```env
# Google Gemini API Key
GOOGLE_API_KEY=AIzaSyD...your_actual_api_key_here
```

## Cách Sử Dụng

### 1. Khởi Động Hệ Thống

```bash
# Khởi động với Gemini (bắt buộc)
python launcher.py
```

### 2. API Server

```bash
# Chạy API server với Gemini
python src/app/api.py --host 0.0.0.0 --port 8000
```

### 3. Test Hệ Thống

Sau khi khởi động, hệ thống sẽ hiển thị:

```
🤖 Sử dụng Google Gemini cho text generation (bắt buộc)
🤖 Initializing RealLegalRAG with Gemini support only
✅ Google Gemini client loaded successfully
```

## Tính Năng Gemini

| Tính năng | Mô tả |
|-----------|--------|
| Chất lượng câu trả lời | ⭐⭐⭐⭐⭐ Xuất sắc với khả năng hiểu ngữ cảnh phức tạp |
| Tốc độ | ⭐⭐⭐⭐ Nhanh chóng với API tối ưu |
| Chi phí | 💰 Phụ thuộc vào số lượng API calls |
| Offline | ❌ Cần kết nối internet |
| Hỗ trợ tiếng Việt | ⭐⭐⭐⭐⭐ Hoàn hảo với khả năng xử lý ngôn ngữ tự nhiên |
| Cần API key | ✅ Bắt buộc để sử dụng |
| Khả năng RAG | ⭐⭐⭐⭐⭐ Tích hợp tốt với retrieval system |

## Xử Lý Lỗi

### Lỗi: "google-generativeai not installed"

```bash
pip install google-generativeai
```

### Lỗi: "GOOGLE_API_KEY not found"

1. Kiểm tra file `.env` có tồn tại không
2. Đảm bảo `GOOGLE_API_KEY` được thiết lập đúng
3. Restart server sau khi thay đổi

### Lỗi: "Failed to load Gemini client"

1. ❌ Gemini lỗi → Hệ thống không thể khởi động
2. 🔍 Kiểm tra API key và kết nối internet
3. 📝 Xem log chi tiết để biết lỗi cụ thể

## Tips Sử Dụng

1. **API Key Bảo Mật**: Không commit file `.env` vào Git
2. **Quota**: Gemini có giới hạn API calls, theo dõi usage trên Google AI Studio
3. **Cost**: Mỗi request có chi phí nhỏ, tính toán dựa trên usage
4. **Performance**: Gemini cung cấp chất lượng và tốc độ tối ưu cho việc trả lời câu hỏi pháp lý

## Troubleshooting

### Kiểm tra trạng thái Gemini

```python
# Trong Python console
from src.rag.real_rag import LegalRAG
rag = LegalRAG()  # Gemini is now mandatory
print("Gemini model:", rag.gemini_model)
```

### Test Gemini connection

```python
import google.generativeai as genai
import os

api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content("Xin chào! Bạn có thể trả lời bằng tiếng Việt không?")
    print(response.text)
else:
    print("API key not found - kiểm tra file .env")
```
