# ⚖️ LegalAdvisor

LegalAdvisor là một hệ thống **Hỏi – Đáp pháp luật bằng tiếng Việt**, được xây dựng trên nền tảng **NLP + LLM**.  
Dự án được thực hiện trong khuôn khổ môn học **Xử lý ngôn ngữ tự nhiên (NLP)**.  

---

## 🚀 Mục tiêu
- Trả lời câu hỏi pháp luật tiếng Việt dựa trên dữ liệu luật.  
- Kết hợp **Retrieval-Augmented Generation (RAG)** để vừa chính xác, vừa mượt mà.  
- Hỗ trợ người dùng tham khảo luật một cách dễ dàng.  

---

## 🗂️ Dataset
- **ViQuAD** → dùng để train QA Reader.  
- **ViLegalText** → tập văn bản luật, dùng cho Retrieval corpus.  

---

## 🛠️ Kiến trúc hệ thống
1. **Retrieval**: FAISS/ElasticSearch để tìm văn bản luật liên quan.  
2. **Reader**: PhoBERT fine-tuned trên ViQuAD để trích xuất câu trả lời.  
3. **LLM Post-processing**: GPT/LLaMA để sinh câu trả lời tự nhiên.  
4. **UI Demo**: FastAPI + Streamlit.  

---

## 📅 Roadmap (2 tháng)
- Tuần 1–2: Chuẩn bị & tiền xử lý dữ liệu.  
- Tuần 3–4: Train Retrieval + Reader.  
- Tuần 5–6: Tích hợp LLM (RAG pipeline).  
- Tuần 7: Xây dựng API + UI.  
- Tuần 8: Báo cáo + Demo.  

---

## 📂 Cấu trúc repo
Xem chi tiết trong phần `tree` ở trên.

---

## ⚡ Hướng dẫn cài đặt
```bash
git clone https://github.com/username/LegalAdvisor.git
cd LegalAdvisor

# Tạo môi trường conda
conda create -n LegalAdvisor python=3.8
conda activate LegalAdvisor

# Cài đặt dependencies
pip install -r requirements.txt
```

### 🚀 GPU Support (Khuyến nghị)

LegalAdvisor hỗ trợ **GPU acceleration** để tăng hiệu suất lên đến **15x**!

#### Kiểm tra GPU
```bash
python check_gpu.py
```

#### Cài đặt GPU (Tùy chọn)
Xem hướng dẫn chi tiết trong [README_GPU.md](docs/README_GPU.md)

**Yêu cầu**: NVIDIA GPU với CUDA 11.8+ và 8GB VRAM

**Lợi ích**:
- ⚡ Xử lý câu hỏi chỉ trong **1-2 giây** thay vì 10-15 giây
- 🎯 Embedding nhanh hơn 20x
- 🤖 Generation nhanh hơn 10x
- 🔍 Search nhanh hơn 30x

# Setup Google Gemini
# Tạo file .env và thêm GOOGLE_API_KEY; xem [GEMINI_SETUP.md](docs/GEMINI_SETUP.md)
```

## 🚀 Chạy demo nhanh
```bash
# Yêu cầu: đặt GOOGLE_API_KEY để sử dụng Gemini
# PowerShell (Windows):
$env:GOOGLE_API_KEY = "<your_key_here>"
python launcher.py
```

## ▶️ Chạy từng phần
### 1. Chuẩn bị dữ liệu
```bash
# Tải ViQuAD (hoặc tạo mock nếu không tải được)
python -m src.tools.data_tools download-viquad

# Xử lý VNLegalText → tạo smart_chunks_stable.db/parquet
python src/automatic_preprocess_vnlegaltext_stable.py

# Tạo FAISS index
python src/retrieval/build_index.py
```

### 2. Test retrieval
```bash
python src/retrieval/search.py
```

### 3. Chạy hệ thống
#### Cách 1: Chạy tự động (Khuyến nghị)
```bash
python launcher.py
```

#### Cách 2: Chạy riêng lẻ
##### Backend (FastAPI)
```bash
# PowerShell (Windows): đảm bảo có GOOGLE_API_KEY
$env:GOOGLE_API_KEY = "<your_key_here>"
python src/app/api.py
# hoặc
uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload
```

##### UI (Streamlit)
```bash
streamlit run src/app/ui.py
```

#### Cách 3: Dừng servers
```bash
# Đơn giản: Chỉ cần nhấn Ctrl+C trong terminal
# Hệ thống sẽ tự động dừng tất cả servers
```

### 📊 Kết quả đạt được

✅ **Đã hoàn thành:**
- Pipeline RAG với retrieval và generation
- FAISS index cho 29,234 document chunks
- Fine-tuned QA model trên dataset ViQuAD
- FastAPI backend với logging và monitoring
- Streamlit UI với giao diện thân thiện
- Unit tests và comprehensive logging
- **Launcher đơn giản** - khởi động/dừng servers dễ dàng
- **Signal handling tốt** - dừng với Ctrl+C
- **Health check tự động** - đảm bảo API sẵn sàng trước khi khởi động UI
- Tích hợp Google Gemini (tùy chọn)

## 🎯 Tính năng chính

- **Hỏi đáp pháp luật** bằng tiếng Việt
- **Retrieval-Augmented Generation (RAG)**
- **Tìm kiếm ngữ nghĩa** trong 12.8M từ văn bản luật
- **API RESTful** với FastAPI
- **Giao diện web** với Streamlit
- **Logging và monitoring** đầy đủ
- **Unit tests** và validation

## 📈 Metrics

- **29,234 chunks** văn bản pháp luật
- **12.8 triệu từ** đã xử lý
- **Retrieval accuracy**: ~70-80% relevant results
- **Response time**: < 2 giây per query
- **Model size**: ~500MB (FAISS + transformers)