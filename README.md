# LegalAdvisor Mini

LegalAdvisor Mini là chatbot pháp luật tiếng Việt áp dụng kiến trúc RAG: câu hỏi được mã hóa bằng SentenceTransformer, tìm kiếm văn bản liên quan bằng FAISS, sau đó gửi ngữ cảnh cho Groq để tạo câu trả lời có nguồn tham khảo rõ ràng.

## Thành phần chính

- `src/retrieval/service.py`: Tải mô hình mã hóa (encoder), FAISS index và truy hồi tài liệu.
- `src/retrieval/build_index.py`: Xây dựng FAISS index từ `chunks_schema.jsonl`.
- `src/rag/groq_rag.py`: Ghép ngữ cảnh và gọi API Groq.
- `src/app/api.py`: FastAPI backend.
- `src/app/ui.py`: Giao diện người dùng Streamlit.

## Cài đặt

```bash
pip install -r requirements.txt
```

Tạo tệp `.env` từ `.env.sample` và điền khóa `GROQ_API_KEY`.

## Chuẩn bị dữ liệu và build index

Nếu clone dự án mới và chưa có thư mục `data/` hoặc `models/`, hãy chạy pipeline tối thiểu theo thứ tự sau:

```bash
python scripts/dataset/download.py
python scripts/zalo_legal_preprocess.py
python src/retrieval/build_index.py ^
  --chunks data/processed/zalo-legal/chunks_schema.jsonl ^
  --base-model intfloat/multilingual-e5-small ^
  --model-dir models/retrieval/base_encoder ^
  --output-dir models/retrieval/index
python scripts/utils/build_law_registry.py
```

Trong đó:

- `download.py`: Tải dữ liệu thô (raw corpus) Zalo Legal về thư mục `data/raw/zalo_ai_legal_text_retrieval/`.
- `zalo_legal_preprocess.py`: Tiền xử lý dữ liệu và tạo tệp `data/processed/zalo-legal/chunks_schema.jsonl`.
- `build_index.py`: Tạo FAISS index và `id_map.jsonl` từ chính tệp chunks ở trên.
- `build_law_registry.py`: Tạo tệp `data/registry/law_registry.json` hỗ trợ hiển thị nguồn/trích dẫn.

Quan trọng: Không trộn lẫn index cũ với tệp `chunks_schema.jsonl` mới. Mỗi khi tạo lại chunks, hãy xây dựng lại index.

Nếu máy cá nhân đã có mô hình mã hóa local trong thư mục `models/retrieval/base_encoder` và muốn xây dựng index bằng mô hình đó, hãy thêm tham số `--use-local-model` vào lệnh chạy `build_index.py`.

## Build index bằng base model

Bản nộp mặc định sử dụng base model `intfloat/multilingual-e5-small`. Nếu đã có tệp `chunks_schema.jsonl` và chỉ cần xây dựng lại index, hãy chạy:

```bash
python src/retrieval/build_index.py ^
  --chunks data/processed/zalo-legal/chunks_schema.jsonl ^
  --base-model intfloat/multilingual-e5-small ^
  --model-dir models/retrieval/base_encoder ^
  --output-dir models/retrieval/index
```

Mặc định, script chỉ lưu các artifact cần thiết cho quá trình chạy runtime: `chunks_index.faiss`, `id_map.jsonl`, `metadata.json`, `model_info.json`. Nếu cần debug vector thô, hãy thêm tham số `--save-embeddings`.

## Chạy ứng dụng

```bash
python launcher.py
```

Hoặc chạy riêng lẻ từng thành phần:

```bash
python -m src.app.api
streamlit run src/app/ui.py
```

## Cấu hình local riêng

Nếu máy cá nhân đã có encoder/index cục bộ trong các thư mục mặc định, tệp `.env` có thể trỏ trực tiếp tới:

```env
LEGALADVISOR_EMBEDDING_MODEL_DIR=models/retrieval/base_encoder
LEGALADVISOR_INDEX_DIR=models/retrieval/index
```

Không cần gửi các artifact local này trong bản nộp, nhưng người clone mới phải chạy pipeline chuẩn bị dữ liệu và build index ở trên.