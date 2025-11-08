# Kế hoạch triển khai (tập trung Zalo Legal + ViQuAD)

Mục tiêu: triển khai nhanh vòng lặp huấn luyện–đánh giá cho retrieval trên Zalo Legal, kèm đánh giá QA bằng ViQuAD. Tất cả chạy trong Conda env `LegalAdvisor` (Windows).

## Sprint 1 (Retrieval Zalo Legal v1)

### 0) Chuẩn bị môi trường
- [X] Xác nhận Conda env `LegalAdvisor` hoạt động; cài đặt bổ sung: `ftfy`, `rank-bm25`, `sentence-transformers` (nếu thiếu).
- [ ] Thiết lập ENV: `LEGALADVISOR_EMBEDDING_MODEL_DIR` (đường dẫn model fine-tune, sẽ thêm sau).

### 1) Chuẩn hoá dữ liệu (Zalo Legal)
- [x] Deduplicate `queries.jsonl` → `data/processed/zalo-legal/queries_dedup.jsonl`.
- [x] Enrich train pairs: join `pairs_train.jsonl` + `queries` → `train_pairs_enriched.jsonl` (thêm `query_text`, parse `type/year/suffix` từ `corpus-id`).
- [x] Normalize Unicode `corpus.jsonl` (streaming + `ftfy.fix_text`) → `corpus_cleaned.jsonl`.
- [x] Kiểm tra toàn vẹn: % query trong pairs có mặt ở queries, % corpus-id hợp lệ.

Deliverables:
- `data/processed/zalo-legal/queries_dedup.jsonl`
- `data/processed/zalo-legal/train_pairs_enriched.jsonl`
- (tuỳ chọn) `data/processed/zalo-legal/corpus_cleaned.jsonl`

### 2) Hard negatives & Triplets
- [x] Tạo negatives bằng FAISS hoặc BM25 (top 50) loại positives; chọn 8–10 negatives/query.
- [x] Sinh `triplets_train.jsonl` với `{query, positive, negatives}`.
- [x] Ghi thống kê: % query có đủ negatives, phân phối số negatives.

Deliverables:
- `data/processed/zalo-legal/triplets_train.jsonl`
- `results/retrieval/negatives_stats.json`

### 3) Huấn luyện embedding (v1)
- [ ] Script huấn luyện `scripts/train_retrieval.py` (SentenceTransformers + MultipleNegativesRankingLoss): epochs=4, batch=32 (có thể dùng gradient accumulation), lr=2e-5, warmup=10%.
- [ ] Theo dõi loss/Recall@10 trên dev (tách một phần từ pairs_train).
- [ ] Lưu model → `models/retrieval/zalo_v1/` + `model_card` ngắn.

Deliverables:
- `models/retrieval/zalo_v1/` (model + config + `model_info.json`)
- `results/retrieval/train_loss.csv`
- `docs/model_card_retrieval_zalo_v1.md` (sau Sprint 2 cũng được)

### 4) Build index & tích hợp
- [ ] Sửa `src/retrieval/build_index.py` nhận `--model-dir`; build FAISS với model v1.
- [ ] Cập nhật `src/retrieval/service.py` đọc ENV `LEGALADVISOR_EMBEDDING_MODEL_DIR` (fallback pretrain nếu thiếu).
- [ ] Sanity-check: chạy tìm kiếm top-k, đảm bảo không lỗi, latency ổn.

Deliverables:
- `models/retrieval/metadata.json` cập nhật
- Index FAISS mới và log build

### 5) Đánh giá retrieval
- [ ] `scripts/eval_retrieval.py`: dùng `pairs_test.jsonl` làm ground-truth.
- [ ] Tính Recall@5/10/20/50, MRR@10, nDCG@10 → `results/retrieval/zalo_v1_metrics.json`.
- [ ] So sánh baseline (pretrain) → `results/retrieval/baseline_vs_v1.md`.

Definition of Done Sprint 1:
- [ ] Recall@10 tăng ≥ 8% vs baseline
- [ ] ≥ 95% query train có ≥ 8 negatives
- [ ] Build & tích hợp index mới không lỗi

## Sprint 2 (Reranker + Benchmark + QA Eval)

### 6) (Tuỳ chọn) Reranker
- [ ] `scripts/rerank.py`: tích hợp cross-encoder (ví dụ `BAAI/bge-reranker-large` hoặc đa ngôn ngữ nhỏ hơn).
- [ ] Pipeline: retrieve top 200 → rerank → top K cuối; đo latency.
- [ ] Đánh giá lại MRR@10, nDCG@10.

Deliverables:
- `results/retrieval/zalo_v1_rerank_metrics.json`
- `results/benchmark/rerank_latency.json`

### 7) Benchmark end-to-end
- [ ] `scripts/benchmark_pipeline.py`: đo `retrieval_ms`, `rerank_ms`, `generation_ms` (Gemini) và tổng.
- [ ] Heuristic hallucination rate (lexical overlap nguồn vs answer).

Deliverables:
- `results/benchmark/zalo_v1_latency.json`

### 8) Đánh giá QA bằng ViQuAD
- [x] `scripts/eval_qa_viquad.py` (đã tạo)
- [x] `scripts/viquad_to_sft.py` (đã tạo)
- [ ] Tạo baseline EM/F1: sinh `predictions.json` cho `validation.json` (có thể dùng Gemini hoặc mẫu).
- [ ] Chạy evaluator → `results/qa/viquad_eval.json`.

### 9) Fine-tune model sinh (mở rộng)
- [ ] Chọn SLM phù hợp (≤2B tham số hoặc bản lượng tử) cho tiếng Việt.
- [ ] Chuẩn hoá ViQuAD → định dạng instruction (`scripts/viquad_to_sft.py` hoặc tương đương).
- [ ] Huấn luyện LoRA/QLoRA trên hạ tầng GPU ngoài máy cá nhân.
- [ ] Chuẩn bị bản lượng tử/inference (GGUF/GPTQ) để chạy với 4GB VRAM.
- [ ] Đánh giá EM/F1 trên ViQuAD validation, lưu `results/qa/viquad_slm_metrics.json`.

## Rủi ro & phương án
- Unicode corpus lỗi hiển thị → dùng `ftfy.fix_text` + kiểm tra mẫu.
- Ít hard negatives → tăng topN hoặc thêm BM25 song song.
- Overfit embedding → early stopping theo dev Recall.
- Latency tăng với reranker → giảm candidate hoặc dùng model nhỏ.

## Lộ trình file/script cần tạo
- `scripts/build_enriched_pairs.py`
- `scripts/build_triplets.py`
- `scripts/train_retrieval.py`
- `scripts/eval_retrieval.py`
- (tuỳ chọn) `scripts/rerank.py`, `scripts/benchmark_pipeline.py`

## Ghi chú môi trường (Windows, Conda)
- Luôn chạy trong env `LegalAdvisor`.
- Có thể bổ sung `ftfy`, `rank-bm25` nếu chưa có.

---
Trang này sẽ cập nhật theo tiến độ; ưu tiên chạy Sprint 1 trước (khoảng 3–5 ngày làm việc tuỳ tài nguyên GPU).
