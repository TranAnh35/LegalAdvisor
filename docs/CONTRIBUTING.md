# 📘 Coding Guidelines – LegalAdvisor

## 1. 📂 Cấu trúc thư mục

* Toàn bộ code chính nằm trong thư mục `src/`, chia theo **module chức năng**:

  * `retrieval/` → code liên quan đến tìm kiếm.
  * `reader/` → code QA model.
  * `rag/` → pipeline kết hợp.
  * `app/` → API + UI.
* Notebook để trong `notebooks/`, không để notebook trong `src/`.
* Dữ liệu để trong `data/`, chia `raw/` và `processed/`.

---

## 2. 📝 Quy tắc đặt tên

* **Biến & hàm**: dùng `snake_case`.

  ```python
  def load_dataset(path: str) -> List[Dict]:
      ...
  ```
* **Class**: dùng `PascalCase`.

  ```python
  class LegalRetriever:
      ...
  ```
* **Tên file**: `snake_case.py` (ví dụ: `build_index.py`, `train_reader.py`).
* **Tên module**: rõ nghĩa theo chức năng (`retrieval`, `reader`, `rag`).

---

## 3. 🔧 Yêu cầu về code

* Code phải **PEP8 compliant**.
* Dùng **type hinting** cho hàm và class.

  ```python
  def search(query: str, top_k: int = 5) -> List[str]:
      ...
  ```
* Dùng **docstring chuẩn Google style**:

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
* Mỗi file Python phải có `if __name__ == "__main__":` để test local.

---

## 4. 🧪 Testing

* Unit test để trong `tests/`.
* Tên test function: `test_<tên_hàm>()`.
* Test chính bằng `pytest`.
* Ví dụ:

  ```python
  def test_preprocess_text():
      assert preprocess_text("Điều 1. ABC...") == "dieu 1 abc"
  ```

---

## 5. 📊 Logging & Config

* Dùng `logging` thay vì `print()`.
* Config để trong file `config.yaml` hoặc `config.json`.
* Code không hard-code đường dẫn dataset → dùng biến ENV hoặc config file.

---

## 6. 📦 Quản lý dependency

* Toàn bộ thư viện ghi trong `requirements.txt`.
* Cài bằng `pip install -r requirements.txt`.
* Không commit thư viện, chỉ commit danh sách.

---

## 7. 🔀 Git & Commit

* Branch chính: `main`.
* Feature branch: `feature/<tên_mô-đun>`.
* Commit message theo convention:

  * `feat:` → thêm chức năng.
  * `fix:` → sửa lỗi.
  * `refactor:` → cải tiến code.
  * `docs:` → tài liệu.
  * `test:` → thêm/sửa test.
* Ví dụ:

  ```
  feat: add FAISS retriever module
  fix: correct path in dataset loader
  docs: update README with usage example
  ```

---

## 8. 📖 Documentation

* Mỗi module (`retrieval`, `reader`, `rag`) có file `README.md` riêng.
* Code có docstring đầy đủ.
* Có thể auto-generate docs bằng `pdoc` hoặc `sphinx`.

---

## 9. ⚡ Style chung

* Giữ code **ngắn gọn, modular, tái sử dụng được**.
* Tránh viết hàm quá 50 dòng.
* Chia logic thành các hàm nhỏ, dễ test.

---