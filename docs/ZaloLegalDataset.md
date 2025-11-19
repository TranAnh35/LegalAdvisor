# Bộ Dữ liệu Pháp luật Zalo Legal (Đã xử lý)

Tài liệu này mô tả chi tiết bộ dữ liệu pháp luật được sử dụng trong **LegalAdvisor**, có nguồn gốc từ cuộc thi **Zalo AI Challenge**, đã qua các bước làm sạch, chuẩn hoá và làm giàu thông tin.

## 1. Tổng quan

*   **Nguồn gốc**: Zalo AI Challenge (Legal Text Retrieval).
*   **Quy mô**: ~61,425 chunks (đoạn văn bản).
*   **Đơn vị lưu trữ**: Mỗi chunk tương ứng với một **Điều luật** (Article-level).
*   **Định dạng**: JSONL (JSON Lines).

## 2. Quy trình Xử lý (Preprocessing Pipeline)

Dữ liệu thô ban đầu gặp nhiều vấn đề về encoding, nhiễu và thiếu cấu trúc. Quy trình xử lý đã thực hiện:

1.  **Unicode Normalization**: Sửa lỗi font (mojibake), chuẩn hoá về Unicode dựng sẵn (NFC).
2.  **Metadata Extraction**: Phân tích ID và nội dung để tách các trường:
    *   Số hiệu văn bản (Number).
    *   Năm ban hành (Year).
    *   Loại văn bản (Type: Luật, Nghị định, Thông tư...).
    *   Số thứ tự Điều (Suffix).
3.  **Text Cleaning**: Loại bỏ các ký tự thừa, khoảng trắng thừa, chuẩn hoá định dạng "Điều X".

## 3. Cấu trúc Dữ liệu (Schema)

File dữ liệu chính: `data/processed/zalo-legal/chunks_schema.jsonl`

Mỗi dòng là một đối tượng JSON với cấu trúc sau:

```json
{
  "chunk_id": 12345,
  "corpus_id": "47/2011/TT-BCA+7",
  "title": "Thông tư 47/2011/TT-BCA...",
  "content": "Điều 7. Trách nhiệm của...",
  "preview": "Điều 7. Trách nhiệm của...",
  "type": "tt-bca",
  "number": "47",
  "year": 2011,
  "suffix": "7"
}
```

### Giải thích trường thông tin

| Trường | Kiểu dữ liệu | Mô tả |
| :--- | :--- | :--- |
| `chunk_id` | Integer | ID định danh duy nhất của chunk trong hệ thống (dùng cho Indexing). |
| `corpus_id` | String | ID gốc từ Zalo, thường có dạng `<số_hiệu>+<số_điều>`. |
| `title` | String | Tiêu đề văn bản pháp luật. |
| `content` | String | Nội dung đầy đủ của Điều luật. |
| `preview` | String | Nội dung rút gọn (thường giống content) để hiển thị nhanh. |
| `type` | String | Loại văn bản (viết tắt), ví dụ: `nd-cp` (Nghị định), `tt-bca` (Thông tư). |
| `number` | String | Số hiệu văn bản. |
| `year` | Integer | Năm ban hành. |
| `suffix` | String | Số thứ tự của Điều trong văn bản (dùng để sắp xếp). |

## 4. Dữ liệu Huấn luyện & Đánh giá

Ngoài kho ngữ liệu (corpus) chính, hệ thống còn bao gồm các tập dữ liệu phục vụ huấn luyện mô hình tìm kiếm (Retriever):

*   **`train_pairs_enriched.jsonl`**: Các cặp câu hỏi - điều luật tương ứng (Positive pairs) dùng để fine-tune model.
*   **`triplets_train.jsonl`**: Bộ 3 `{Query, Positive, Negative}` dùng cho loss function (MultipleNegativesRankingLoss). Negative được sinh bằng cách chọn các văn bản có điểm BM25 cao nhưng không phải đáp án đúng (Hard Negatives).
*   **`queries_dedup.jsonl`**: Danh sách câu hỏi pháp luật tự nhiên đã loại bỏ trùng lặp.

## 5. Thống kê

*   **Tổng số văn bản**: ~61,000+ Điều luật.
*   **Phạm vi**: Bao phủ nhiều lĩnh vực (Dân sự, Hình sự, Hành chính, Giao thông...).
*   **Chất lượng**: Đã được kiểm tra thủ công ngẫu nhiên để đảm bảo không còn lỗi hiển thị tiếng Việt.

## 6. Cách sử dụng trong Code

Hệ thống sử dụng Class `RetrievalService` (`src/retrieval/service.py`) để đọc dữ liệu này:

*   Khi khởi động, service load file JSONL vào bộ nhớ đệm (Indexed Cache) để truy xuất O(1).
*   Mapping từ kết quả tìm kiếm Vector (FAISS ID) sang nội dung text được thực hiện qua `chunk_id`.
