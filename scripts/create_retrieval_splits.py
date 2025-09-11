#!/usr/bin/env python3
"""
Tạo bộ train/eval cho retrieval từ VNLegalText (VNLegalText-only) với chia tách hợp lý.

Nguyên tắc chia:
- Holdout theo cấp Điều: chọn ngẫu nhiên một tỷ lệ bài Điều trong mỗi Luật (doc_file) cho eval
- Mọi chunk thuộc các Điều holdout → eval; phần còn lại → train
- Đảm bảo phân bố theo chunk_type (article/clause/point) giữ nguyên tương đối
- Lọc bỏ chunk quá ngắn (min_words)

Đầu ra:
- data/processed/retrieval_train_splits.jsonl (mỗi dòng: {query, positive_id, ...})
- data/processed/retrieval_eval.jsonl

Cách chạy (PowerShell):
  conda activate LegalAdvisor
  python scripts\create_retrieval_splits.py \
    --chunks-db data\processed\smart_chunks_stable.db \
    --eval-article-ratio 0.2 --min-words 10 --max-len 320 --seed 42 \
    --train-out data\processed\retrieval_train_splits.jsonl \
    --eval-out data\processed\retrieval_eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def write_jsonl(path: Path, records: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return text.replace("_", " ").strip()


def load_chunks(sqlite_path: Path) -> List[Dict]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, doc_file, doc_title, chapter, section, article, article_heading,
                   clause, point, chunk_index, content, word_count, chunk_type,
                   COALESCE(effective_year, 0) as effective_year
            FROM chunks
            ORDER BY doc_file, article, clause, point
            """
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    keys = [
        "chunk_id", "doc_file", "doc_title", "chapter", "section", "article", "article_heading",
        "clause", "point", "chunk_index", "content", "word_count", "chunk_type", "effective_year"
    ]
    out: List[Dict] = []
    for row in rows:
        d = {k: row[i] for i, k in enumerate(keys)}
        d["content"] = normalize_text(d.get("content") or "")
        out.append(d)
    return out


def make_splits(chunks: List[Dict], eval_article_ratio: float, min_words: int, max_len: int, seed: int) -> Tuple[List[Dict], List[Dict]]:
    rng = random.Random(seed)
    # Tạo map: doc_file → article → list chunks
    doc_to_article: Dict[str, Dict[str, List[Dict]]] = {}
    for ch in chunks:
        if not ch.get("content"):
            continue
        if int(ch.get("word_count") or 0) < min_words:
            continue
        doc = ch.get("doc_file") or ""
        art = str(ch.get("article") or "")
        if not art:
            art = "_no_article_"
        doc_to_article.setdefault(doc, {}).setdefault(art, []).append(ch)

    def record_from_chunk(ch: Dict) -> Dict:
        q = (ch.get("content") or "")[:max_len]
        return {
            "query": q,
            "positive_id": int(ch.get("chunk_id")),
            "doc_file": ch.get("doc_file"),
            "doc_title": ch.get("doc_title"),
            "article": ch.get("article"),
            "clause": ch.get("clause"),
            "point": ch.get("point"),
            "chunk_type": ch.get("chunk_type"),
            "effective_year": ch.get("effective_year"),
        }

    # Mặc định: nếu hầu hết không có Điều → dùng holdout theo doc_file
    has_article = any(a != "_no_article_" for arts in doc_to_article.values() for a in arts.keys())

    train_records: List[Dict] = []
    eval_records: List[Dict] = []

    if has_article:
        # Holdout theo Điều trong từng luật
        for doc, art_map in doc_to_article.items():
            arts = [a for a in art_map.keys() if a != "_no_article_"]
            rng.shuffle(arts)
            k_eval = max(1, int(eval_article_ratio * len(arts))) if arts else 0
            eval_set = set(arts[:k_eval]) if k_eval > 0 else set()

            for art, lst in art_map.items():
                target = eval_records if (art in eval_set) else train_records
                for ch in lst:
                    target.append(record_from_chunk(ch))
    else:
        # Fallback: holdout theo doc_file
        docs = list(doc_to_article.keys())
        rng.shuffle(docs)
        k_eval_docs = max(1, int(0.2 * len(docs)))
        eval_docs = set(docs[:k_eval_docs])
        for doc, art_map in doc_to_article.items():
            target = eval_records if (doc in eval_docs) else train_records
            for lst in art_map.values():
                for ch in lst:
                    target.append(record_from_chunk(ch))

    return train_records, eval_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Tạo train/eval splits cho retrieval từ VNLegalText")
    parser.add_argument("--chunks-db", type=str, default="data/processed/smart_chunks_stable.db")
    parser.add_argument("--eval-article-ratio", type=float, default=0.2)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-out", type=str, default="data/processed/retrieval_train_splits.jsonl")
    parser.add_argument("--eval-out", type=str, default="data/processed/retrieval_eval.jsonl")
    args = parser.parse_args()

    chunks = load_chunks(Path(args.chunks_db))
    if not chunks:
        print("❌ Không tải được chunks từ SQLite")
        return

    train_recs, eval_recs = make_splits(
        chunks,
        eval_article_ratio=float(args.eval_article_ratio),
        min_words=int(args.min_words),
        max_len=int(args.max_len),
        seed=int(args.seed),
    )

    n_train = write_jsonl(Path(args.train_out), train_recs)
    n_eval = write_jsonl(Path(args.eval_out), eval_recs)
    print(f"✅ Train samples: {n_train} → {args.train_out}")
    print(f"✅ Eval  samples: {n_eval} → {args.eval_out}")


if __name__ == "__main__":
    main()


