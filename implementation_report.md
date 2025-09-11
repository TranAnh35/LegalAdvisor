## Báo cáo triển khai nâng cấp RAG – LegalAdvisor

### 1) Tóm tắt công việc đã thực hiện
- Tạo và hoàn thiện pipeline dữ liệu retriever với các nguồn có thể tải:
  - Script tải dữ liệu đa nguồn: `scripts/download_datasets.py` (ưu tiên VietnameseLegalQA; có fallback snapshot; loại bỏ ViRHE4QA).
  - Script chuẩn hóa retriever: `src/datasets/retrieval_prepare.py` (hỗ trợ dữ liệu chỉ có query; ánh xạ query/positive_text → chunk_id qua FAISS; sinh hard negatives).
  - Script chuyển đổi ViQuAD (tuỳ chọn bootstrap): `scripts/prepare_viquad_retrieval.py` (đọc nhiều biến thể schema).
- Enrich thông tin hiệu lực văn bản:
  - `scripts/enrich_effective_year.py` (độc lập, tự định vị `--root`; quét XML VNLegalText, regex ngày hiệu lực; cập nhật `effective_date`, `effective_year` vào `models/retrieval/metadata.json`).
- Cập nhật tài liệu kế hoạch: `docs/TODO.md` (loại bộ dữ liệu không khả dụng; thay bằng VietnameseLegalQA + VNLegalText; cập nhật checklist retriever).

### 2) Chi tiết kỹ thuật & cách chạy

#### 2.1 Tải queries phục vụ retriever
```powershell
conda activate LegalAdvisor
python scripts\download_datasets.py --datasets vlegalqa --output-dir data\raw --limit 0
```
- Đầu ra: `data\raw\VietnameseLegalQA.jsonl`

#### 2.2 Chuẩn hóa dữ liệu retriever (tạo retrieval_train.jsonl)
```powershell
python -m src.datasets.retrieval_prepare --input data\raw\VietnameseLegalQA.jsonl --output data\processed\retrieval_train.jsonl --hard-negatives 15 --dense-top-k 64 --hn-top-k 64
```
- Logic: nếu chỉ có query → ánh xạ query → `positive_id` bằng FAISS; sau đó mine hard negatives.

#### 2.3 Xây FAISS (nếu chưa có index)
```powershell
python -m src.retrieval.build_index
```
- Sinh `models\retrieval\{faiss_index.bin, metadata.json, model_info.json}`.

#### 2.4 Enrich năm hiệu lực (tùy chọn)
```powershell
python scripts\enrich_effective_year.py --root . --raw-xml data\raw\VNLegalText --metadata models\retrieval\metadata.json --dry-run
python scripts\enrich_effective_year.py --root . --raw-xml data\raw\VNLegalText --metadata models\retrieval\metadata.json
```

### 3) Quyết định kiến trúc dữ liệu
- Retriever dùng VNLegalText làm kho passage (đã có nội bộ) + queries từ VietnameseLegalQA để mining/đánh giá.
- Loại hoàn toàn ViRHE4QA và giảm phụ thuộc VNLAWQC/VNSynLawQC do khó tải.
- Bổ sung enrich `effective_year` để hỗ trợ lọc theo hiệu lực khi cần.

### 4) Rủi ro & xử lý
- Dữ liệu query không có positive: đã xử lý bằng ánh xạ trực tiếp query → chunk gần nhất.
- Repo HF lỗi Arrow/loader: script có fallback snapshot + parser JSON thủ công.
- Enrich ngày hiệu lực không đầy đủ: có fallback suy luận theo năm trong tiêu đề.

### 5) Việc tiếp theo (được đề xuất)
- Huấn luyện bi-encoder (mE5-small) với `retrieval_train.jsonl` và đánh giá nDCG/Recall.
- Triển khai hybrid BM25+dense và (tùy chọn) reranker.
- Chuẩn hóa QA dataset (VLQA/ViBidLQA) cho bước generator.


