#!/usr/bin/env python3
"""
Đánh giá hệ thống Retrieval (VNLegalText-only).

Chức năng:
- Tải FAISS + metadata qua RetrievalService
- Đọc queries từ JSONL (mỗi dòng: {"query": str, "gold_chunk_id": Optional[int]})
- Nếu không cung cấp queries → tự sinh bộ self-eval: chọn ngẫu nhiên N chunk, dùng preview/content làm query, gold = chunk_id
- Tính các metric: Recall@k, MRR@k, nDCG@k

Cách chạy (PowerShell):
  conda activate LegalAdvisor
  # Tự sinh self-eval 500 mẫu
  python scripts\\eval_retrieval.py --num-samples 500 --k 1 3 5 10

  # Đánh giá theo file queries
  python scripts\\eval_retrieval.py --queries data\\processed\\retrieval_eval.jsonl --k 1 3 5 10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    # Module import khi chạy từ root
    from src.retrieval.service import RetrievalService
except Exception:
    # Chạy trực tiếp
    import sys
    THIS = Path(__file__).resolve()
    sys.path.append(str(THIS.parent.parent / "src"))
    from retrieval.service import RetrievalService  # type: ignore


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


def compute_metrics(pred_indices: List[List[int]], gold_indices: List[int], ks: List[int]) -> Dict[str, Dict[int, float]]:
    """Tính Recall@k, MRR@k, nDCG@k (binary relevance: gold=1, else 0).

    - pred_indices[i] = list topK chunk_ids cho query i (đã sắp theo score giảm dần)
    - gold_indices[i] = gold chunk_id
    """
    n = len(gold_indices)
    results: Dict[str, Dict[int, float]] = {"recall": {}, "mrr": {}, "ndcg": {}}

    for k in ks:
        recall_sum = 0.0
        mrr_sum = 0.0
        ndcg_sum = 0.0

        for i in range(n):
            gold = int(gold_indices[i])
            preds = pred_indices[i][:k]

            # Recall@k (binary: 1 nếu có gold trong top-k)
            hit_positions = [j for j, pid in enumerate(preds) if int(pid) == gold]
            hit = 1.0 if hit_positions else 0.0
            recall_sum += hit

            # MRR@k
            if hit_positions:
                rank = hit_positions[0] + 1
                mrr_sum += 1.0 / rank

            # nDCG@k (binary relevance: DCG = 1/log2(1+rank) khi trúng)
            if hit_positions:
                rank = hit_positions[0] + 1
                dcg = 1.0 / (1.0 + (rank).bit_length() - 1)  # xấp xỉ log2 via bit_length
                # IDCG của binary 1 relevant = 1/log2(1+1) = 1.0
                ndcg_sum += dcg

        results["recall"][k] = recall_sum / n if n else 0.0
        results["mrr"][k] = mrr_sum / n if n else 0.0
        results["ndcg"][k] = ndcg_sum / n if n else 0.0

    return results


def self_eval_samples(service: RetrievalService, num_samples: int) -> Tuple[List[str], List[int]]:
    """Sinh bộ data tự đánh giá từ metadata: dùng preview/content làm query, gold = chunk_id."""
    # Lấy danh sách chunk_id có trong metadata map
    all_ids = list(service._meta_by_id.keys())
    if not all_ids:
        return [], []
    if num_samples <= 0 or num_samples > len(all_ids):
        num_samples = min(1000, len(all_ids))
    sample_ids = random.sample(all_ids, num_samples)

    queries: List[str] = []
    golds: List[int] = []
    for cid in sample_ids:
        text = service.get_chunk_content(cid) or service._meta_by_id.get(cid, {}).get("preview", "")
        if not text:
            continue
        q = text[:400]
        queries.append(q)
        golds.append(cid)
    return queries, golds


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá retriever: Recall/MRR/nDCG")
    parser.add_argument("--queries", type=str, default="", help="File JSONL: {query, gold_chunk_id}")
    parser.add_argument("--num-samples", type=int, default=500, help="Self-eval: số mẫu nếu không có --queries")
    parser.add_argument("--k", nargs="*", type=int, default=[1, 3, 5, 10], help="Các giá trị k để tính metric")
    parser.add_argument("--top-k", type=int, default=10, help="TopK truy hồi mỗi query")
    parser.add_argument("--use-gpu", action="store_true", help="Dùng GPU nếu có")
    args = parser.parse_args()

    service = RetrievalService(use_gpu=args.use_gpu)

    # Chuẩn bị queries & golds
    if args.queries:
        qpath = Path(args.queries)
        if not qpath.exists():
            print(f"❌ Không tìm thấy file queries: {qpath}")
            return
        queries: List[str] = []
        golds: List[int] = []
        for obj in read_jsonl(qpath):
            q = obj.get("query")
            gid = obj.get("gold_chunk_id")
            if not q or gid is None:
                continue
            queries.append(str(q))
            golds.append(int(gid))
    else:
        queries, golds = self_eval_samples(service, num_samples=args.num_samples)

    if not queries or not golds:
        print("❌ Không có queries/golds để đánh giá")
        return

    # Truy hồi và thu thập top-k ids
    pred_ids: List[List[int]] = []
    for q in queries:
        results = service.retrieve(q, top_k=max(args.top_k, max(args.k)))
        pred_ids.append([int(r.get("chunk_id")) for r in results])

    metrics = compute_metrics(pred_ids, golds, ks=args.k)

    print("\n===== Retrieval Evaluation =====")
    for name, table in metrics.items():
        row = ", ".join([f"@{k}: {table[k]:.4f}" for k in sorted(table.keys())])
        print(f"{name.upper():<8} {row}")


if __name__ == "__main__":
    main()


