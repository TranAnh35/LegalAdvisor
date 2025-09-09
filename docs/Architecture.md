LegalAdvisor/
│── data/                     # Dữ liệu (raw + processed)
│   ├── raw/                   # Dataset gốc (ViQuAD, VNLegalText)
│   ├── processed/             # Dữ liệu đã tiền xử lý (smart_chunks_stable.db/parquet)
│
│── notebooks/                 # Notebook thử nghiệm
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_train_retriever.ipynb
│   ├── 03_train_reader.ipynb
│
│── src/                       # Code chính
│   ├── retrieval/             # Module tìm kiếm văn bản (thống nhất)
│   │   ├── build_index.py     # Xây FAISS index từ SQLite/Parquet
│   │   ├── search.py          # Script test sử dụng RetrievalService
│   │   ├── service.py         # RetrievalService: FAISS + metadata + content store
│   │
│   ├── tools/                 # CLI dữ liệu hợp nhất
│   │   ├── data_tools.py      # download-viquad / split-chunks / export-txt
│   │
│   ├── rag/                   # RAG (generate): Gemini-only
│   │   ├── gemini_rag.py      # Dùng RetrievalService + Gemini generate
│   │
│   ├── app/                   # Ứng dụng demo
│   │   ├── api.py             # FastAPI backend (Gemini-only)
│   │   ├── ui.py              # Streamlit UI
│
│── docs/                      # Tài liệu
│── tests/                     # Unit tests (tùy chọn)
│
│── requirements.txt           # Thư viện cần thiết
│── README.md                  # Giới thiệu project
│── TODO.md                    # Danh sách công việc
│── LICENSE                    # License (MIT/GPL/Apache)
