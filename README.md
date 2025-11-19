# ⚖️ LegalAdvisor v1.0

**LegalAdvisor** là trợ lý pháp lý AI thông minh dành cho người Việt, kết hợp công nghệ **Retrieval-Augmented Generation (RAG)** tiên tiến để trả lời câu hỏi pháp luật một cách chính xác, có trích dẫn nguồn cụ thể.

Dự án được xây dựng hoàn thiện, sẵn sàng triển khai và sử dụng.

---

## 🌟 Tính năng nổi bật

- **Hỏi đáp pháp luật tự nhiên**: Trả lời câu hỏi dựa trên ngữ cảnh luật pháp Việt Nam.
- **Trích dẫn chính xác**: Mọi câu trả lời đều kèm theo nguồn dẫn cụ thể (Điều, Khoản, Văn bản luật).
- **Truy hồi thông minh**:
  - **Semantic Search**: Sử dụng mô hình **`intfloat/multilingual-e5-small`** đã được fine-tune chuyên biệt cho pháp luật Việt Nam.
  - **Phân đoạn thông minh**: Xử lý văn bản luật dài thành các đoạn nhỏ (chunks) tối ưu cho việc tìm kiếm.
- **Giao diện trực quan**:
  - **Web UI**: Giao diện Chat thân thiện (Streamlit).
  - **API RESTful**: Endpoint đầy đủ cho tích hợp hệ thống khác (FastAPI).
- **Hiệu năng cao**: Hỗ trợ tăng tốc GPU, caching thông minh và tối ưu hóa độ trễ.

---

## 🛠️ Yêu cầu hệ thống

- **OS**: Windows
- **Python**: 3.11+
- **Conda**: Khuyến nghị sử dụng để quản lý môi trường.
- **API Key**: Cần có **Google Gemini API Key** (miễn phí hoặc trả phí).

---

## ⚡ Cài đặt & Chạy chương trình

### 1. Thiết lập môi trường

```bash
# Tạo và kích hoạt môi trường conda
conda create -n LegalAdvisor python=3.11
conda activate LegalAdvisor

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Cấu hình API Key

Bạn cần thiết lập biến môi trường `GOOGLE_API_KEY` để sử dụng mô hình Gemini.

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY = "AIzaSy..."
```

**Linux/macOS:**
```bash
export GOOGLE_API_KEY="AIzaSy..."
```

### 3. Khởi chạy hệ thống (All-in-One)

Sử dụng script `launcher.py` để tự động kiểm tra hệ thống và khởi chạy cả API lẫn giao diện Web.

```bash
python launcher.py
```

Sau khi khởi động thành công:
- **Giao diện Chat (Web UI)**: [http://localhost:8501](http://localhost:8501)
- **API Backend**: [http://localhost:8000](http://localhost:8000)
- **Tài liệu API (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🧠 Quy trình Huấn luyện & Cập nhật Dữ liệu

Dưới đây là hướng dẫn đầy đủ để tái tạo lại hệ thống từ dữ liệu gốc (Raw Data) đến khi có Index và Model hoàn chỉnh.

### Bước 1: Tải dữ liệu gốc (Download)

Tải dataset Zalo Legal từ Hugging Face về thư mục `data/raw`.

```bash
python scripts/zalo_legal_download.py
```
*Kết quả: Dữ liệu thô được lưu tại `data/raw/zalo_ai_legal_text_retrieval/`.*

### Bước 2: Tiền xử lý dữ liệu (Preprocessing)

Thực hiện chuẩn hóa Unicode, tạo schema chuẩn và làm giàu dữ liệu train.

1. **Chuẩn hóa Corpus cho ứng dụng (Application Schema):**
   ```bash
   python -m src.data_preprocessing.zalo_legal
   ```
   *Tạo: `data/processed/zalo-legal/chunks_schema.jsonl`*

2. **Chuẩn hóa Corpus cho huấn luyện (Unicode Fix):**
   ```bash
   python scripts/normalize_zalo_corpus.py
   ```
   *Tạo: `data/processed/zalo-legal/corpus_cleaned.jsonl`*

3. **Xử lý cặp câu hỏi - văn bản (Pairs Enrichment):**
   ```bash
   python scripts/build_enriched_pairs.py
   ```
   *Tạo: `queries_dedup.jsonl` và `train_pairs_enriched.jsonl`*

### Bước 3: Tạo dữ liệu huấn luyện (Hard Negatives)

Sử dụng BM25 để tìm các văn bản "gây nhiễu" (hard negatives) giúp model học tốt hơn.

```bash
python scripts/build_triplets.py
```
*Kết quả: `data/processed/zalo-legal/triplets_train.jsonl`*

### Bước 4: Huấn luyện Model Retrieval

Fine-tune mô hình `intfloat/multilingual-e5-small` trên dữ liệu luật Việt Nam.

```bash
python scripts/train_retrieval.py \
  --base-model intfloat/multilingual-e5-small \
  --output-dir models/retrieval/vi_legal_finetuned \
  --batch-size 32 \
  --epochs 4
```
*Kết quả: Model mới được lưu tại `models/retrieval/vi_legal_finetuned`.*

### Bước 5: Xây dựng Index tìm kiếm (Build Index)

Tạo FAISS Index từ model đã fine-tune để sử dụng trong ứng dụng.

```bash
python src/retrieval/build_index.py \
  --chunks data/processed/zalo-legal/chunks_schema.jsonl \
  --model-dir models/retrieval/vi_legal_finetuned \
  --output-dir models/retrieval/index_v2
```

**Lưu ý:** Sau khi chạy xong Bước 5, hệ thống khi chạy `launcher.py` sẽ tự động nhận diện index mới trong `models/retrieval/index_v2`.

---

## 📂 Cấu trúc dữ liệu & Model

Hệ thống sử dụng bộ dữ liệu **Zalo Legal** đã được chuẩn hóa:

- **Lưu trữ**: `data/processed/zalo-legal/chunks_schema.jsonl` (JSONL format).
- **Index**: FAISS Index (`models/retrieval/index_v2`) sử dụng model `intfloat/multilingual-e5-small` (fine-tuned).

---

## 🔍 Hướng dẫn sử dụng nâng cao

### Chạy riêng lẻ từng thành phần

**Chạy API Server:**
```bash
python -m src.app.api
```

**Chạy Giao diện Web:**
```bash
streamlit run src/app/ui.py
```

## 🤝 Đóng góp

Dự án đã hoàn thiện phiên bản v1.0. Mọi đóng góp vui lòng xem tại [CONTRIBUTING.md](docs/CONTRIBUTING.md).
