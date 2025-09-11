#!/usr/bin/env python3
"""
Tạo tập đánh giá retrieval từ VNLegalText (VNLegalText-only).

- Lấy danh sách chunk_id từ metadata và/hoặc FAISS qua RetrievalService
- Lấy content/preview và cắt đoạn làm query
- Ghi JSONL: {query, gold_chunk_id, doc_file, article, clause, point, effective_year}

Cách chạy (PowerShell):
  conda activate LegalAdvisor
  python scripts\create_retrieval_eval.py --num-samples 1000 --output data\processed\retrieval_eval.jsonl --min-len 80 --max-len 320
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

try:
    # Khi chạy từ root
    from src.retrieval.service import RetrievalService
except Exception:
    # Khi chạy trực tiếp
    import sys
    THIS = Path(__file__).resolve()
    sys.path.append(str(THIS.parent.parent / "src"))
    from retrieval.service import RetrievalService  # type: ignore


def write_jsonl(path: Path, records: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo retrieval_eval.jsonl từ VNLegalText")
    parser.add_argument("--num-samples", type=int, default=1000, help="Số mẫu eval (ngẫu nhiên)")
    parser.add_argument("--output", type=str, default="data/processed/retrieval_eval.jsonl", help="Đường dẫn JSONL đầu ra")
    parser.add_argument("--min-len", type=int, default=80, help="Độ dài tối thiểu của query")
    parser.add_argument("--max-len", type=int, default=320, help="Độ dài tối đa của query")
    parser.add_argument("--seed", type=int, default=42, help="Seed chọn mẫu ngẫu nhiên")
    parser.add_argument("--use-gpu", action="store_true", help="Dùng GPU nếu có")
    args = parser.parse_args()

    random.seed(int(args.seed))

    service = RetrievalService(use_gpu=args.use_gpu)
    meta_by_id = getattr(service, "_meta_by_id", {})
    all_ids: List[int] = list(meta_by_id.keys())
    if not all_ids:
        print("❌ Không tìm thấy metadata để tạo eval")
        return

    n = max(1, min(int(args.num_samples), len(all_ids)))
    sample_ids = random.sample(all_ids, n)

    def gen():
        for cid in sample_ids:
            meta = meta_by_id.get(cid, {})
            text = service.get_chunk_content(cid) or meta.get("preview") or ""
            if not text:
                continue
            t = text.replace("_", " ").strip()
            if len(t) < args.min_len:
                continue
            q = t[: args.max_len]
            yield {
                "query": q,
                "gold_chunk_id": int(cid),
                "doc_file": meta.get("doc_file"),
                "article": meta.get("article"),
                "clause": meta.get("clause"),
                "point": meta.get("point"),
                "effective_year": meta.get("effective_year"),
            }

    out = Path(args.output)
    total = write_jsonl(out, gen())
    print(f"✅ Đã tạo {total} mẫu eval → {out}")


if __name__ == "__main__":
    main()


