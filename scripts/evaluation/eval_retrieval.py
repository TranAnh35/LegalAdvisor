#!/usr/bin/env python3
"""Evaluate retrieval metrics for the Zalo Legal dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.retrieval.service import RetrievalService
from src.utils.paths import get_project_root

PROJECT_ROOT = get_project_root()
DEFAULT_PAIRS = PROJECT_ROOT / "data" / "raw" / "zalo_ai_legal_text_retrieval" / "pairs_test.jsonl"
DEFAULT_QUERIES = PROJECT_ROOT / "data" / "raw" / "zalo_ai_legal_text_retrieval" / "queries.jsonl"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "retrieval" / "zalo_v1"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "retrieval" / "zalo_v1_metrics.json"
DETAILS_OUTPUT = PROJECT_ROOT / "results" / "retrieval" / "zalo_v1_eval_details.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Recall/MRR/nDCG for retrieval")
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS, help="Path to pairs_test.jsonl")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES, help="Path to queries.jsonl or equivalent")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="SentenceTransformer model directory")
    parser.add_argument("--top-k", type=int, default=50, help="Maximum top-k for retrieval")
    parser.add_argument("--mrr-k", type=int, default=10, help="Cutoff k for MRR")
    parser.add_argument("--ndcg-k", type=int, default=10, help="Cutoff k for nDCG")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Encoding device")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output metrics JSON path")
    parser.add_argument("--details", type=Path, default=DETAILS_OUTPUT, help="JSONL file for per-query details")
    parser.add_argument(
        "--group-strategy",
        type=str,
        default="topm_mean",
        choices=["mean", "max", "topm_mean"],
        help="Chiến lược gộp điểm ở cấp Điều/văn bản (mean/max/topm_mean).",
    )
    parser.add_argument(
        "--group-topm",
        type=int,
        default=2,
        help="Giá trị m khi dùng chiến lược topm_mean (mặc định 2).",
    )
    parser.add_argument("--skip-details", action="store_true", help="Skip writing per-query detail file")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def load_pairs(path: Path) -> Dict[str, Set[str]]:
    mapping: Dict[str, Set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skip malformed JSON line in pairs file")
                continue
            query_id = record.get("query-id") or record.get("query_id")
            corpus_id = record.get("corpus-id") or record.get("corpus_id")
            if not query_id or not corpus_id:
                continue
            mapping[str(query_id)].add(str(corpus_id))
    if not mapping:
        raise ValueError(f"Failed to load pairs from {path}")
    return mapping


def load_queries(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logging.warning("Skip malformed JSON line in queries file")
                continue
            query_id = record.get("query_id") or record.get("_id") or record.get("id")
            text = record.get("query_text") or record.get("text") or record.get("query")
            if not query_id or not text:
                continue
            mapping[str(query_id)] = str(text)
    if not mapping:
        raise ValueError(f"Failed to load queries from {path}")
    return mapping


def pick_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    return arg


def ensure_model_env(model_dir: Path) -> None:
    os.environ["LEGALADVISOR_EMBEDDING_MODEL_DIR"] = str(model_dir.resolve())


def compute_idcg(rel_count: int, k: int) -> float:
    limit = min(rel_count, k)
    if limit <= 0:
        return 0.0
    return sum(1.0 / math.log2(i + 2) for i in range(limit))


def evaluate(
    service: RetrievalService,
    eval_pairs: Dict[str, Set[str]],
    query_texts: Dict[str, str],
    top_k: int,
    mrr_k: int,
    ndcg_k: int,
    store_details: bool,
    details_path: Path,
) -> Dict[str, object]:
    recall_sums: Dict[int, float] = {5: 0.0, 10: 0.0, 20: 0.0, 50: 0.0}
    hit_sums: Dict[int, float] = {k: 0.0 for k in recall_sums}
    mrr_total = 0.0
    ndcg_total = 0.0
    per_query: List[Dict[str, object]] = []

    max_needed = max(max(recall_sums), mrr_k, ndcg_k, top_k)
    logging.info("Starting retrieval for %d queries (top-k=%d)", len(eval_pairs), max_needed)

    for idx, (query_id, positives) in enumerate(eval_pairs.items(), 1):
        text = query_texts.get(query_id)
        if not text:
            logging.debug("Missing text for query %s", query_id)
            continue
        results = service.retrieve(text, top_k=max_needed)
        ranked_ids = [str(item.get("corpus_id")) for item in results]
        scores = [float(item.get("score", 0.0)) for item in results]
        positive_count = max(len(positives), 1)

        for k in recall_sums:
            truncated = ranked_ids[:k]
            hits = sum(1 for doc in truncated if doc in positives)
            recall_sums[k] += hits / positive_count
            hit_sums[k] += 1.0 if hits > 0 else 0.0

        rr = 0.0
        for rank, doc_id in enumerate(ranked_ids[:mrr_k], 1):
            if doc_id in positives:
                rr = 1.0 / rank
                break
        mrr_total += rr

        dcg = 0.0
        for rank, doc_id in enumerate(ranked_ids[:ndcg_k], 1):
            if doc_id in positives:
                dcg += 1.0 / math.log2(rank + 1)
        idcg = compute_idcg(len(positives), ndcg_k)
        ndcg_total += (dcg / idcg) if idcg > 0 else 0.0

        if store_details:
            per_query.append(
                {
                    "query_id": query_id,
                    "query_text": text,
                    "positives": sorted(positives),
                    "retrieved": ranked_ids,
                    "scores": scores,
                    "rr_at_%d" % mrr_k: rr,
                }
            )

        if idx % 50 == 0:
            logging.info("Processed %d/%d queries", idx, len(eval_pairs))

    total_queries = len(eval_pairs)
    if total_queries == 0:
        raise ValueError("Không có truy vấn để đánh giá")

    metrics = {
        "recall": {str(k): recall_sums[k] / total_queries for k in recall_sums},
        "hit_rate": {str(k): hit_sums[k] / total_queries for k in hit_sums},
        "mrr@%d" % mrr_k: mrr_total / total_queries,
        "ndcg@%d" % ndcg_k: ndcg_total / total_queries,
    }

    if store_details:
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as handle:
            for record in per_query:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info("Wrote per-query details to %s", details_path)

    return {"metrics": metrics, "total_queries": total_queries}


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    pairs = load_pairs(args.pairs)
    queries = load_queries(args.queries)

    device = pick_device(args.device)
    ensure_model_env(args.model_dir)
    # Đồng bộ chiến lược group doc-level với RetrievalService/GeminiRAG qua ENV
    os.environ["LEGALADVISOR_ARTICLE_GROUP_STRATEGY"] = args.group_strategy
    os.environ["LEGALADVISOR_ARTICLE_GROUP_TOPM"] = str(max(1, int(args.group_topm)))
    service = RetrievalService(use_gpu=(device == "cuda"))

    results = evaluate(
        service=service,
        eval_pairs=pairs,
        query_texts=queries,
        top_k=args.top_k,
        mrr_k=args.mrr_k,
        ndcg_k=args.ndcg_k,
        store_details=not args.skip_details,
        details_path=args.details,
    )

    output_dir = args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pairs_path": str(args.pairs.resolve()),
        "queries_path": str(args.queries.resolve()),
        "model_dir": str(args.model_dir.resolve()),
        "device": device,
        **results,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info("Saved metrics to %s", args.output)


if __name__ == "__main__":
    main()
