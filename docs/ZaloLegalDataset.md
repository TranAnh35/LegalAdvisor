# Đánh giá và Chuẩn hoá Dataset Zalo Legal

## 1. Mục tiêu
Phân tích cấu trúc bộ dữ liệu nội bộ `zalo_ai_legal_text_retrieval` để:
- Khẳng định khả năng sử dụng độc lập (không cần ViQuAD / ALQAC / ViBidLQA).
- Chuẩn hoá schema phục vụ: huấn luyện retrieval, đánh giá, mở rộng RAG.
- Xác định các bước làm sạch / tiền xử lý cần thiết.

## 2. Thành phần dữ liệu
Thư mục: `data/raw/zalo_ai_legal_text_retrieval/`

| File | Mô tả | Số dòng | Ghi chú sơ bộ |
|------|-------|---------|---------------|
| corpus.jsonl | Kho văn bản pháp lý đã chunk hoá (mỗi `_id` chứa pattern `number/year/type+suffix`) | (rất lớn, >50MB – cần đọc streaming) | Chứa `title`, `text`; có dấu bị lỗi encoding Unicode hiển thị (cần chuẩn hoá). |
| queries.jsonl | Danh sách câu hỏi tiếng Việt dạng tự nhiên | 3298 | Một số câu hỏi trùng `_id` (duplicate). |
| pairs_train.jsonl | Mapping `query-id` → `corpus-id` (positive pairs) cho train | 2505 | Có trường `score=1.0` (nhãn dương). Có query xuất hiện nhiều lần (nhiều positive). |
| pairs_test.jsonl | Mapping tương tự cho test | 793 | Dùng làm ground-truth đánh giá retrieval. |

## 3. Schema quan sát
### corpus.jsonl (mẫu dòng)
```json
{"_id": "01/2009/tt-bnn+1", "title": "…", "text": "…"}
```
Trường phân tích từ `_id`:
- `number/year/type+suffix` → tách thành: `number_year`, `year`, `type`, `suffix`.
- Có thể sử dụng hàm `parse_corpus_id()` đã có để tái dụng.

### queries.jsonl
```json
{"_id": "0637bf82c8b290c7875c5bfddbf91df5", "text": "Công an xã xử phạt lỗi không mang bằng lái xe có đúng không?"}
```
Trường: `_id`, `text`.

### pairs_* .jsonl
```json
{"query-id": "0637…", "corpus-id": "47/2011/tt-bca+7", "score": 1.0}
```
=> Đây là nhãn positive. Không có negative rõ ràng.

## 4. Đánh giá mức độ đầy đủ
| Khía cạnh | Đánh giá | Kết luận |
|-----------|----------|----------|
| Retrieval positives | Có mapping trực tiếp (pairs_train/test). | Đủ cho fine-tune embedding. |
| Negatives | Chưa có. | Cần sinh hard negatives bằng FAISS/BM25. |
| Câu hỏi (query) | Đa lĩnh vực pháp luật, ~3.3K câu. | Phù hợp giai đoạn đầu. |
| Văn bản nguồn | Pháp lý đa dạng, chunk hoá có suffix (Điều). | Đủ làm corpus tìm kiếm. |
| Chất lượng text | Lỗi encoding Unicode hiển thị (ký tự biến dạng). | Cần normalize (utf-8 decode + unicodedata normalize + sửa artifacts). |
| Metadata bổ sung | Chưa có phân loại lĩnh vực (domain tag). | Có thể gán heuristic sau. |
| Trùng lặp | Có duplicate dòng query. | Cần lọc trước huấn luyện. |
| Đánh giá test | Có test pairs. | Dùng cho Recall@K, MRR, nDCG. |

=> Bộ Zalo Legal đủ làm nguồn duy nhất cho giai đoạn huấn luyện retrieval & đánh giá, với điều kiện bổ sung pipeline sinh negative và làm sạch.

## 5. Các bước xử lý đề xuất
1. Đọc streaming `corpus.jsonl` → tạo `metadata.json` chuẩn (đã có một phần) + thêm trường phân tách: `type`, `year`, `suffix`.
2. Chuẩn hoá text:
   - Fix lỗi hiển thị do tái mã hoá (các ký tự như `Äiá»u` → `Điều`).
   - Dùng bảng thay thế dựa trên mapping heuristics hoặc thư viện `ftfy`.
3. Loại bỏ trùng lặp query (`queries.jsonl`). Giữ bản duy nhất theo `_id` hoặc theo lowercase text.
4. Join `queries` với `pairs_train` → tạo `train_pairs_enriched.jsonl` gồm: `{query_id, query_text, corpus_id, score}`.
5. Sinh hard negatives:
   - Với mỗi query: lấy top N (ví dụ 50) từ FAISS/ BM25.
   - Loại bỏ các `corpus-id` là positive.
   - Chọn random một phần + chọn những doc có điểm gần positive.
6. Tạo file training triplets: `{query, positive_text, negative_texts: [...]}`.
7. Fine-tune embedding (MultipleNegativesRankingLoss).
8. Rebuild index với model mới.
9. Đánh giá: dùng `pairs_test` làm ground-truth.
10. Lưu kết quả vào `results/retrieval/zalo_v1_metrics.json`.

## 6. Schema chuẩn hoá đề xuất
### Enriched train pairs
```json
{"query_id": "...", "query_text": "...", "corpus_id": "47/2011/tt-bca+7", "type": "tt-bca", "year": "2011", "suffix": "7", "score": 1.0}
```
### Triplets cho huấn luyện
```json
{"query": "...", "positive": "...", "negatives": ["...", "..."]}
```
### Ground truth evaluation row
```json
{"query_id": "...", "positive_corpus_ids": ["47/2011/tt-bca+7", "..."], "retrieved": ["...", "..."], "ranks_found": [1, 4], "top_k": 20}
```

## 7. Metrics áp dụng
Retrieval:
- Recall@K (K = 5,10,20,50)
- MRR@K
- nDCG@K
- MAP (tuỳ chọn)

Sau khi RAG hoạt động ổn định có thể bổ sung đánh giá answer (nếu sinh câu trả lời) bằng heuristic overlap pháp lý.

## 8. Rủi ro & Xử lý
| Vấn đề | Giải pháp |
|--------|-----------|
| Lỗi Unicode (tiếng Việt bị vỡ) | Dùng `ftfy` hoặc tái mã hoá: `bytes(text, 'latin1').decode('utf-8')` nếu pattern khớp; mapping thủ công ký tự. |
| Suffix không phải luôn là số (có Điều, Khoản khác nhau) | Giữ nguyên khi không parse được; thêm field `suffix_raw`. |
| Thiếu negative thực sự khó | Thu thập hard negatives qua reranking BM25 + cosine similarity gần positive. |
| Nhiều positives cho một query | Xử lý thành multi-label; trong eval: bất kỳ positive nào khớp coi là hit. |
| Trùng `_id` query (dòng y hệt) | Deduplicate trước khi build. |

## 9. Kế hoạch thư mục đầu ra
```
data/processed/zalo-legal/
  queries_dedup.jsonl
  train_pairs_enriched.jsonl
  triplets_train.jsonl
  corpus_cleaned.jsonl  (nếu normalize text)
models/retrieval/zalo_v1/
  sentence_transformer/...
  model_info.json
results/retrieval/
  zalo_v1_metrics.json
scripts/
  build_enriched_pairs.py
  build_triplets.py
  train_retrieval.py
  eval_retrieval.py
```

## 10. Tiêu chí hoàn thành (Definition of Done)
- Có `queries_dedup.jsonl` giảm ≥5% dòng trùng.
- Triplets sinh ra cho ≥90% query (số negatives ≥5 mỗi query).
- Huấn luyện hoàn tất với log loss giảm ổn định.
- Recall@10 tăng ≥ +8% so với baseline pretrain.
- Lưu đầy đủ artifacts + metrics.

## 11. Các bước tiếp theo (Action Items)
1. Viết script `scripts/build_enriched_pairs.py`.
2. Viết script `scripts/build_triplets.py` (FAISS + BM25 pipeline).
3. Thêm yêu cầu vào `requirements.txt` nếu cần (`rank-bm25`, `ftfy`).
4. Huấn luyện và ghi lại metrics.
5. Cập nhật README phần “Huấn luyện Retrieval (Zalo Legal v1)”.

## 12. Ghi chú encoding
Các ký tự như `Äiá»u` xuất hiện thường do double-encoding UTF-8 → Latin1. Cần pipeline kiểm tra:
```python
import ftfy
clean = ftfy.fix_text(raw_text)
```
Nếu vẫn lỗi: thử heuristics từng pattern.

---
Tài liệu này phục vụ thay đổi hướng: chỉ dùng Zalo Legal làm nguồn dữ liệu duy nhất cho Phase huấn luyện retrieval.
