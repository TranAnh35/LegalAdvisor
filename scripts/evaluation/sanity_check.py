#!/usr/bin/env python3
"""Chạy sanity-check cho pipeline truy hồi Zalo Legal."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any

from src.retrieval.service import RetrievalService
from src.utils.paths import get_project_root


DEFAULT_MODEL_DIR = get_project_root() / "models" / "retrieval" / "zalo_v1"


def ensure_model_env() -> None:
    """Đặt biến môi trường LEGALADVISOR_EMBEDDING_MODEL_DIR nếu chưa có."""
    model_dir = os.getenv("LEGALADVISOR_EMBEDDING_MODEL_DIR")
    if model_dir:
        return
    os.environ["LEGALADVISOR_EMBEDDING_MODEL_DIR"] = str(DEFAULT_MODEL_DIR.resolve())


def run_queries(queries: Iterable[str], top_k: int = 5) -> None:
    ensure_model_env()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    service = RetrievalService(use_gpu=False)
    collected: List[Dict[str, Any]] = []
    start = time.perf_counter()
    for query in queries:
        results = service.retrieve(query, top_k=top_k)
        print(f"\nQuery: {query}")
        print(f"Results ({len(results)} hits):")
        if not results:
            print("  (Không có kết quả)")
            collected.append({"query": query, "results": []})
            continue
        for rank, hit in enumerate(results, 1):
            preview = (hit.get("preview") or "")[:100]
            score = hit.get("score")
            corpus_id = hit.get("corpus_id")
            print(f"  {rank:02d}. {corpus_id} | score={score:.4f} | preview={preview}")
        collected.append({
            "query": query,
            "results": results,
        })
    elapsed = time.perf_counter() - start
    print(f"\nTotal elapsed: {elapsed:.2f}s")

    output_dir = get_project_root() / "results" / "retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sanity_check.json"
    output_data = {
        "queries": collected,
        "top_k": top_k,
        "elapsed_seconds": elapsed,
        "model_dir": os.getenv("LEGALADVISOR_EMBEDDING_MODEL_DIR"),
    }
    output_path.write_text(
        json.dumps(output_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Lưu kết quả tại {output_path}")


if __name__ == "__main__":
    SAMPLE_QUERIES = [
        "trách nhiệm thông báo kết quả xét nghiệm HIV",
        "thủ tục treo quốc kỳ tại cơ quan đại diện",
        "mùa lũ tuần tra canh gác đê",
    ]
    run_queries(SAMPLE_QUERIES, top_k=5)
