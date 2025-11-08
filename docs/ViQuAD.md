# ViQuAD: Chuẩn hoá & Sử dụng cho Generative QA

## 1. Tổng quan
- Định dạng: SQuAD-style JSON (array các example)
- Trường chính mỗi example:
  - `id`, `uit_id`
  - `title`
  - `context`
  - `question`
  - `answers`: { `text`: [..], `answer_start`: [..] }
  - `is_impossible` (có thể có)

## 2. Mục đích sử dụng trong dự án
- Đánh giá khả năng sinh câu trả lời (QA) bằng các chỉ số EM/F1.
- (Tuỳ chọn) Tạo dữ liệu instruction-tuning (SFT) cho mô hình generative open-source (LoRA).
- Lưu ý: Nội dung ViQuAD là bách khoa tổng quát, không phải pháp luật — phù hợp để đánh giá/nắn style sinh câu trả lời tiếng Việt; sau đó có thể chuyển sang dữ liệu pháp luật tự xây.

## 3. Công cụ đã tạo
- `scripts/eval_qa_viquad.py` — Trình đánh giá EM/F1 trên file dự đoán offline.
  - Input: `--dataset data/raw/ViQuAD/validation.json`
  - Input: `--pred predictions.json` (map id → answer)
  - Output: `results/qa/viquad_eval.json`

- `scripts/viquad_to_sft.py` — Converter sang JSONL cho SFT.
  - Input: `--input data/raw/ViQuAD/train.json`
  - Output: `--output data/processed/viquad_sft_train.jsonl`
  - Tham số `--max-context-chars` để cắt ngắn context nếu cần.

## 4. Gợi ý quy trình khi dùng ViQuAD
1. Chạy converter tạo SFT JSONL cho train/valid/test (tuỳ mục đích huấn luyện/đánh giá).
2. Sinh dự đoán từ mô hình (Gemini hoặc model nội bộ) cho `validation.json` → tạo `predictions.json` (id → answer text).
3. Chạy evaluator để tính EM/F1.

## 5. Lưu ý kỹ thuật
- Chuẩn hoá chuỗi trước khi so khớp: lowercase, unicode normalize (NFKC), bỏ punctuation, rút gọn whitespace (đã có trong evaluator).
- Với câu hỏi `is_impossible=true`, script hiện bỏ qua (không có answers). Có thể mở rộng tính No-Answer F1/EM nếu cần.
- Khi fine-tune SFT, nên giữ prompt có cả context và hướng dẫn rõ “trả lời CHỈ dựa vào đoạn văn”.

## 6. Kế hoạch mở rộng
- Thêm mode online (gọi Gemini) cho evaluator với rate-limit & retry (ẩn API key trong log).
- Bổ sung đo ROUGE-L, BLEU nếu muốn đánh giá văn phong tổng quát (không chỉ QA extractive).
