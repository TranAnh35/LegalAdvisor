# Hướng dẫn Sử dụng LegalAdvisor

Tài liệu này hướng dẫn chi tiết cách sử dụng hệ thống **LegalAdvisor** thông qua giao diện đồ họa (Web UI) và giao diện lập trình ứng dụng (API).

## 1. Giao diện Chat (Web UI)

Giao diện web được xây dựng trên Streamlit, tập trung vào sự đơn giản và hiệu quả.

**Truy cập:** [http://localhost:8501](http://localhost:8501)

### Các chức năng chính:

1.  **Đặt câu hỏi**:
    *   Nhập câu hỏi pháp luật bằng tiếng Việt vào ô chat ở dưới cùng.
    *   Ví dụ: *"Lái xe máy không đội mũ bảo hiểm phạt bao nhiêu?"* hoặc *"Thủ tục đăng ký kết hôn cần giấy tờ gì?"*

2.  **Xem câu trả lời**:
    *   Hệ thống sẽ hiển thị câu trả lời tổng hợp từ AI.
    *   Phần **"Căn cứ pháp lý"** liệt kê các Điều luật cụ thể được sử dụng để trả lời.

3.  **Kiểm tra nguồn tin (Fact-checking)**:
    *   Bên dưới mỗi câu trả lời, bạn có thể mở rộng phần **"Chi tiết nguồn tham khảo"**.
    *   Tại đây hiển thị nội dung gốc của các Điều luật mà AI đã tìm thấy. Bạn có thể so sánh để đảm bảo AI không "bịa" thông tin.

---

## 2. Sử dụng API (Dành cho Developer)

Hệ thống cung cấp RESTful API đầy đủ để tích hợp vào website hoặc ứng dụng mobile khác.

**Base URL:** `http://localhost:8000`
**Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)

### A. Endpoint: Hỏi đáp (`/ask`)

Dùng để gửi câu hỏi và nhận câu trả lời.

*   **Method**: `POST`
*   **URL**: `/ask`
*   **Body**:
    ```json
    {
      "question": "Tuổi nghỉ hưu của lao động nam là bao nhiêu?",
      "top_k": 3
    }
    ```
*   **Response**:
    ```json
    {
      "data": {
        "answer": "Theo quy định tại Bộ luật Lao động 2019...",
        "sources": [
          {
            "id": 1234,
            "corpus_id": "45/2019/QH14+169",
            "content": "Điều 169. Tuổi nghỉ hưu...",
            "score": 0.85
          }
        ],
        "process_time": 1.25
      }
    }
    ```

### B. Endpoint: Tra cứu nguồn (`/sources/{chunk_id}`)

Lấy nội dung chi tiết của một chunk (Điều luật) cụ thể.

*   **Method**: `GET`
*   **URL**: `/sources/1234` (với 1234 là `id` trả về từ API `/ask`)

### C. Endpoint: Kiểm tra hệ thống (`/health`)

Kiểm tra xem hệ thống có đang hoạt động bình thường không.

*   **Method**: `GET`
*   **URL**: `/health`
*   **Response**: `{"status": "ok"}`

---

## 3. Mẹo đặt câu hỏi hiệu quả

Để nhận được câu trả lời tốt nhất từ LegalAdvisor:

1.  **Rõ ràng ngữ cảnh**: Thay vì hỏi *"Phạt bao nhiêu?"*, hãy hỏi *"Mức phạt cho hành vi vượt đèn đỏ đối với xe máy là bao nhiêu?"*.
2.  **Cụ thể**: Nếu bạn biết tên luật (ví dụ Luật Đất đai), hãy nhắc đến trong câu hỏi.
3.  **Một vấn đề một lúc**: Tránh hỏi dồn quá nhiều vấn đề không liên quan trong cùng một câu lệnh.

## 4. Xử lý sự cố thường gặp

*   **Lỗi "Connection refused"**: Kiểm tra xem bạn đã chạy `launcher.py` chưa.
*   **Lỗi "API Key invalid"**: Kiểm tra biến môi trường `GOOGLE_API_KEY`.