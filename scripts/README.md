# LegalAdvisor Scripts

Thư mục này chứa các công cụ dòng lệnh (CLI) để chuẩn bị dữ liệu, huấn luyện mô hình và đánh giá hệ thống.

## 📂 Cấu trúc thư mục

### 1. `dataset/` - Chuẩn bị dữ liệu
Các script dùng để tải, làm sạch và tạo dữ liệu huấn luyện.

*   `download.py`: Tải dataset Zalo Legal từ HuggingFace về `data/raw/`.
*   `normalize.py`: Chuẩn hóa lỗi font/Unicode trong corpus `data/raw/` -> `data/processed/`.
*   `build_enriched_pairs.py`: Kết hợp câu hỏi và văn bản luật để tạo cặp training (`queries_dedup.jsonl`, `train_pairs_enriched.jsonl`).
*   `build_triplets.py`: Tạo dữ liệu huấn luyện nâng cao với hard negatives (BM25) -> `triplets_train.jsonl`.

### 2. `training/` - Huấn luyện mô hình
*   `train_retrieval.py`: Fine-tune mô hình SentenceTransformer (Retrieval) trên dữ liệu luật Việt Nam.

### 3. `evaluation/` - Đánh giá & Benchmark
Các công cụ đo lường hiệu năng và độ chính xác.

*   `sanity_check.py`: Test nhanh khả năng tìm kiếm (Smoke test) với vài câu hỏi mẫu.
*   `eval_retrieval.py`: Đánh giá chỉ số Recall@K, MRR@K trên tập test.
*   `compare_retrieval_models.py`: So sánh hiệu năng giữa các mô hình (MiniLM vs E5-Base vs E5-Finetune).
*   `benchmark_pipeline.py`: Đo độ trễ (latency) End-to-End của toàn bộ hệ thống (Retrieve -> Rerank -> Gen).
*   `benchmark_optimization_tier1.py`: Kiểm tra hiệu quả của các tối ưu hóa (Cache, Parallel Fetching).

### 4. `utils/` - Tiện ích
*   `extract_citations.py`: Công cụ trích xuất trích dẫn luật từ văn bản (regex debug).
*   `export_act_codes.py`: Thống kê danh sách mã văn bản luật có trong dataset.

### 5. `Crawl/` - Thu thập dữ liệu
*   Chứa các script crawler bổ sung (nếu có).

---

## 🚀 Hướng dẫn chạy (Ví dụ)

**Lưu ý**: Luôn chạy từ thư mục gốc của dự án (LegalAdvisor/).

```bash
# 1. Chuẩn bị dữ liệu
python scripts/dataset/download.py
python scripts/dataset/normalize.py

# 2. Huấn luyện
python scripts/training/train_retrieval.py --epochs 4

# 3. Đánh giá
python scripts/evaluation/sanity_check.py
```

