#!/usr/bin/env python3
"""
Huấn luyện bi-encoder cho retrieval (VNLegalText-only) bằng Sentence-Transformers.

Đặc điểm:
- Đọc `data/processed/retrieval_train.jsonl` (schema: {query, positive_id, hard_negatives: [ids...]})
- Map positive_id → text từ SQLite `data/processed/smart_chunks_stable.db`
- Loss: MultipleNegativesRankingLoss (in-batch negatives) + hỗ trợ hard negatives qua sampling
- Evaluator: InformationRetrievalEvaluator (self-built từ dev split)
- Lưu model: `models/embeddings/<model_name>-finetuned/`

Cách chạy (PowerShell):
  conda activate LegalAdvisor
  python -m src.retrieval.train_biencoder \
    --train data/processed/retrieval_train.jsonl \
    --chunks-db data/processed/smart_chunks_stable.db \
    --base-model intfloat/multilingual-e5-small \
    --output-dir models/embeddings/legal-multilingual-e5-small-finetuned \
    --epochs 3 --batch-size 64 --max-seq-length 512 --lr 2e-5 --warmup 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
from sentence_transformers import SentenceTransformer, InputExample, losses  # type: ignore
from sentence_transformers.evaluation import InformationRetrievalEvaluator  # type: ignore


def read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return str(text).replace("_", " ").strip()


def load_chunk_store(sqlite_path: Path) -> Dict[int, str]:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        cur = conn.cursor()
        cur.execute("SELECT chunk_id, content FROM chunks ORDER BY chunk_id")
        rows = cur.fetchall()
    finally:
        conn.close()
    store: Dict[int, str] = {}
    for cid, content in rows:
        try:
            store[int(cid)] = normalize_text(content)[:4000]
        except Exception:
            continue
    return store


def build_train_dev_examples(
    train_jsonl: Path,
    chunk_store: Dict[int, str],
    dev_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[InputExample], Dict[str, str], Dict[str, Dict[str, int]], Dict[str, str]]:
    """Tạo train InputExamples và dev evaluator inputs.

    Trả về:
    - train_examples: List[InputExample(texts=[query, positive_text])]
    - dev_corpus: {doc_id(str): text}
    - dev_relevant_docs: {query_id(str): {doc_id(str): 1}}
    - dev_queries: {query_id(str): query}
    """
    rng = random.Random(seed)
    all_rows = list(read_jsonl(train_jsonl))
    rng.shuffle(all_rows)

    split = int((1.0 - dev_ratio) * len(all_rows))
    train_rows = all_rows[:split]
    dev_rows = all_rows[split:] if dev_ratio > 0 else []

    # Train examples (MNRL in-batch negatives)
    train_examples: List[InputExample] = []
    for r in train_rows:
        q = normalize_text(r.get("query"))
        pid = r.get("positive_id")
        if not q or pid is None:
            continue
        pos_text = chunk_store.get(int(pid))
        if not pos_text:
            continue
        train_examples.append(InputExample(texts=[q, pos_text]))

    # Dev evaluator structures
    dev_corpus: Dict[str, str] = {}
    dev_relevant_docs: Dict[str, Dict[str, int]] = {}
    dev_queries: Dict[str, str] = {}

    # Build a small corpus: include all positive ids from dev
    for i, r in enumerate(dev_rows):
        q = normalize_text(r.get("query"))
        pid = r.get("positive_id")
        if not q or pid is None:
            continue
        qid = f"q{i}"
        dev_queries[qid] = q
        doc_id = str(int(pid))
        pos_text = chunk_store.get(int(pid))
        if pos_text:
            dev_corpus[doc_id] = pos_text
            dev_relevant_docs[qid] = {doc_id: 1}

    return train_examples, dev_corpus, dev_relevant_docs, dev_queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bi-encoder for VNLegalText retrieval")
    parser.add_argument("--train", type=str, default="data/processed/retrieval_train.jsonl", help="JSONL train file")
    parser.add_argument("--chunks-db", type=str, default="data/processed/smart_chunks_stable.db", help="SQLite chunks store")
    parser.add_argument("--base-model", type=str, default="intfloat/multilingual-e5-small", help="Base sentence-transformers model")
    parser.add_argument("--output-dir", type=str, default="models/embeddings/legal-multilingual-e5-small-finetuned", help="Output dir to save model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=float, default=0.1, help="Warmup ratio (0-1)")
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train)
    chunks_db = Path(args.chunks_db)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load chunk store
    print(f"📖 Load chunks store from: {chunks_db}")
    chunk_store = load_chunk_store(chunks_db)
    print(f"📊 Total chunks loaded: {len(chunk_store):,}")

    # Build datasets
    print(f"📥 Read train jsonl: {train_path}")
    train_examples, dev_corpus, dev_rel, dev_queries = build_train_dev_examples(
        train_path, chunk_store, dev_ratio=float(args.dev_ratio), seed=int(args.seed)
    )
    print(f"🧾 Train examples: {len(train_examples):,}")
    print(f"🧪 Dev queries: {len(dev_queries):,} | Dev corpus docs: {len(dev_corpus):,}")

    # Model & loss
    print(f"🤖 Load base model: {args.base_model}")
    model = SentenceTransformer(str(args.base_model))
    model.max_seq_length = int(args.max_seq_length)

    train_dataloader = DataLoader(train_examples, batch_size=int(args.batch_size), shuffle=True, drop_last=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    warmup_steps = int(len(train_dataloader) * int(args.epochs) * float(args.warmup))

    # Evaluator
    evaluator: Optional[InformationRetrievalEvaluator] = None
    if dev_queries and dev_corpus:
        evaluator = InformationRetrievalEvaluator(
            dev_queries, dev_corpus, dev_rel,
            name="vnlegal_dev",
            show_progress_bar=True,
            corpus_chunk_size=50000,
        )

    print("🚀 Start training...")
    model.fit(
        train_objectives=[(train_dataloader, loss)],
        epochs=int(args.epochs),
        warmup_steps=warmup_steps,
        optimizer_params={"lr": float(args.lr)},
        evaluator=evaluator,
        evaluation_steps=max(100, len(train_dataloader)),
        output_path=str(out_dir),
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"✅ Model saved to: {out_dir}")


if __name__ == "__main__":
    main()


