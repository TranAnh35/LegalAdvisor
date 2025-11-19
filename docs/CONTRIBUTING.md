# 📘 Coding Guidelines – LegalAdvisor

Chào mừng bạn đến với LegalAdvisor! Dưới đây là các quy tắc quy ước để đảm bảo codebase luôn sạch sẽ, dễ bảo trì và mở rộng.

## 1. 📂 Cấu trúc thư mục

Code được tổ chức trong thư mục `src/` theo các **module chức năng**:

*   `src/retrieval/`: Logic tìm kiếm, đánh chỉ mục (Indexing) và mã hóa văn bản.
*   `src/rag/`: Pipeline RAG, tích hợp LLM (Gemini) để sinh câu trả lời.
*   `src/app/`: Chứa API Backend (FastAPI) và Giao diện Frontend (Streamlit).
*   `src/utils/`: Các hàm tiện ích dùng chung (logger, path helper...).
*   `src/data_preprocessing/`: Script xử lý dữ liệu thô.

**Lưu ý**:
*   Dữ liệu đặt trong `data/` (chia thành `raw/` và `processed/`).
*   Notebook nghiên cứu đặt trong `notebooks/`.
*   Các script chạy một lần hoặc benchmark đặt trong `scripts/`.

---

## 2. 📝 Quy tắc đặt tên

*   **Biến & hàm**: Dùng `snake_case`.
    ```python
    def load_dataset(path: str) -> List[Dict]: ...
    ```
*   **Class**: Dùng `PascalCase`.
    ```python
    class RetrievalService: ...
    ```
*   **Tên file**: `snake_case.py` (ví dụ: `build_index.py`, `gemini_rag.py`).
*   **Hằng số**: `UPPER_CASE` (ví dụ: `MAX_SEQ_LENGTH = 512`).

---

## 3. 🔧 Yêu cầu về Code

1.  **Type Hinting**: Bắt buộc sử dụng type hint cho arguments và return type.
    ```python
    def search(query: str, top_k: int = 5) -> List[Dict[str, Any]]: ...
    ```

2.  **Docstring**: Sử dụng định dạng Google Style cho các hàm quan trọng.
    ```python
    def preprocess_text(text: str) -> str:
        """
        Chuẩn hóa văn bản luật (lowercase, bỏ ký tự đặc biệt).

        Args:
            text (str): Văn bản đầu vào.

        Returns:
            str: Văn bản đã chuẩn hóa.
        """
    ```

3.  **Logging**: Sử dụng `src.utils.logger` thay vì `print()`.
    ```python
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Starting retrieval process...")
    ```

---

## 4. ⚙️ Cấu hình & Môi trường

*   **Environment Variables**: Sử dụng file `.env` để quản lý cấu hình (API Key, đường dẫn model, tham số hệ thống).
*   **Không Hard-code**: Tuyệt đối không hard-code đường dẫn tuyệt đối hoặc API Key trong code. Sử dụng `os.getenv()` hoặc `pathlib`.

---

## 5. 📦 Quản lý Dependencies

*   Thư viện phụ thuộc được liệt kê trong `requirements.txt`.
*   Môi trường khuyến nghị: **Conda** (Python 3.10+).
*   Khi thêm thư viện mới, hãy cập nhật `requirements.txt` ngay lập tức.

---

## 6. 🧪 Testing

*   Unit test đặt trong thư mục `tests/`.
*   Sử dụng framework `pytest`.
*   Tên file test bắt đầu bằng `test_` (ví dụ: `test_api.py`).
*   Đảm bảo chạy pass tất cả test trước khi tạo Pull Request.

---

## 7. 🔀 Git & Commit

*   **Branch**:
    *   `main`: Code ổn định, ready-to-deploy.
    *   `dev` hoặc `feature/...`: Code đang phát triển.
*   **Commit Message**: Rõ ràng, mô tả ngắn gọn thay đổi.
    *   `feat: ...` (Tính năng mới)
    *   `fix: ...` (Sửa lỗi)
    *   `docs: ...` (Cập nhật tài liệu)
    *   `refactor: ...` (Cấu trúc lại code)

---

## 8. 📖 Documentation

*   Cập nhật `README.md` nếu có thay đổi về cách cài đặt/sử dụng.
*   Cập nhật `docs/` nếu thay đổi về kiến trúc hoặc dữ liệu.
