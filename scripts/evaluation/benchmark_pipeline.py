#!/usr/bin/env python3
"""Benchmark end-to-end pipeline latency cho LegalAdvisor.

Đo thời gian của các bước:
- Retrieval (Semantic search FAISS)
- Rerank (nếu có, hiện tại giữ nguyên kết quả và đo thời gian = 0)
- Generation (Gemini, có thể bỏ qua bằng --skip-generation)

Ngoài ra tính heuristic hallucination rate dựa trên mức độ chồng lấp từ vựng
giữa câu trả lời và ngữ cảnh được cung cấp.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.service import RetrievalService
from src.rag.gemini_rag import format_retrieved_docs, GEMINI_MODEL


@dataclass
class StageLatency:
    retrieval_ms: float
    rerank_ms: float
    generation_ms: Optional[float]
    total_ms: float


@dataclass
class QueryBenchmark:
    query_id: str
    query_text: str
    retrieval_count: int
    latencies: StageLatency
    hallucination_score: Optional[float]
    hallucinated: Optional[bool]


class GeminiGenerator:
    """Trình bao bọc đơn giản cho Gemini để dùng trong benchmark."""

    def __init__(self) -> None:
        load_dotenv()
        try:
            import google.generativeai as genai  # type: ignore
        except Exception as exc:  # pragma: no cover - thư viện bên ngoài
            raise RuntimeError(
                "Không thể import google.generativeai. Cài đặt gói google-generativeai trước."
            ) from exc

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Chưa thiết lập GOOGLE_API_KEY, bỏ qua generation hoặc cung cấp khóa.")

        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 2048,
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        self.model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=generation_config,
            safety_settings=safety_settings,
        )

    def generate(self, prompt: str) -> str:
        response = self.model.generate_content(prompt)
        return getattr(response, "text", "")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark pipeline retrieval → rerank → generation")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("data/raw/zalo_ai_legal_text_retrieval/queries.jsonl"),
        help="Đường dẫn tới queries JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Số lượng truy vấn lấy mẫu để benchmark",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Số lượng tài liệu retrieve",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark/zalo_v1_latency.json"),
        help="Đường dẫn file JSON lưu kết quả",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Bỏ qua bước generation (Gemini)",
    )
    parser.add_argument(
        "--mock-generation",
        action="store_true",
        help="Sinh câu trả lời giả lập từ snippet đầu tiên (dùng khi không có API key)",
    )
    parser.add_argument(
        "--hallucination-threshold",
        type=float,
        default=0.2,
        help="Ngưỡng overlap (0-1) dưới đó sẽ coi là hallucination",
    )
    return parser.parse_args()


def load_queries(path: Path, limit: int) -> List[Dict[str, Any]]:
    queries: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(queries) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" not in obj:
                continue
            queries.append({"id": obj.get("_id", f"idx-{len(queries)}"), "text": obj["text"]})
    return queries


def lexical_overlap_ratio(answer: str, context: str) -> float:
    if not answer or not context:
        return 0.0
    token_pattern = re.compile(r"\w+", re.UNICODE)
    answer_tokens = {tok.lower() for tok in token_pattern.findall(answer) if len(tok) >= 3}
    if not answer_tokens:
        return 0.0
    context_tokens = {tok.lower() for tok in token_pattern.findall(context) if len(tok) >= 3}
    if not context_tokens:
        return 0.0
    intersection = answer_tokens.intersection(context_tokens)
    return len(intersection) / len(answer_tokens)


def build_prompt(query: str, context: str) -> str:
    return (
        "Bạn là trợ lý pháp lý tiếng Việt. Trả lời ngắn gọn dựa hoàn toàn vào ngữ cảnh.\n"
        "Nếu ngữ cảnh không đủ, hãy nói rõ.\n\n"
        f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}"
    )


def benchmark_queries(
    queries: Iterable[Dict[str, Any]],
    retriever: RetrievalService,
    top_k: int,
    generator: Optional[GeminiGenerator],
    hallucination_threshold: float,
    mock_generation: bool,
) -> List[QueryBenchmark]:
    results: List[QueryBenchmark] = []
    for item in queries:
        query_id = str(item.get("id", "unknown"))
        query_text = str(item.get("text", ""))

        t0 = time.perf_counter()
        retrieved = retriever.retrieve(query_text, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        t_rerank = time.perf_counter()
        reranked = retrieved  # Chưa có reranker, giữ nguyên
        rerank_ms = (time.perf_counter() - t_rerank) * 1000

        generation_ms: Optional[float]
        hallucination_score: Optional[float]
        hallucinated: Optional[bool]
        context: str = ""
        if reranked:
            context = format_retrieved_docs(reranked)

        if generator and reranked:
            prompt = build_prompt(query_text, context)
            t_start = time.perf_counter()
            try:
                answer = generator.generate(prompt)
                generation_ms = (time.perf_counter() - t_start) * 1000
                hallucination_score = lexical_overlap_ratio(answer, context)
                hallucinated = hallucination_score < hallucination_threshold
            except Exception as exc:
                print(f"⚠️  Lỗi khi gọi Gemini: {exc}. Chuyển sang chế độ mock.")
                generator = None
                if mock_generation and context:
                    answer = reranked[0].get("content", "") or context
                    generation_ms = 0.0
                    hallucination_score = lexical_overlap_ratio(answer, context)
                    hallucinated = hallucination_score < hallucination_threshold
                else:
                    generation_ms = None
                    hallucination_score = None
                    hallucinated = None
        elif mock_generation and context:
            answer = reranked[0].get("content", "") or context
            generation_ms = 0.0
            hallucination_score = lexical_overlap_ratio(answer, context)
            hallucinated = hallucination_score < hallucination_threshold
        else:
            generation_ms = None
            hallucination_score = None
            hallucinated = None

        total_components = [retrieval_ms, rerank_ms]
        if generation_ms is not None:
            total_components.append(generation_ms)
        total_ms = sum(total_components)

        results.append(
            QueryBenchmark(
                query_id=query_id,
                query_text=query_text,
                retrieval_count=len(retrieved),
                latencies=StageLatency(
                    retrieval_ms=retrieval_ms,
                    rerank_ms=rerank_ms,
                    generation_ms=generation_ms,
                    total_ms=total_ms,
                ),
                hallucination_score=hallucination_score,
                hallucinated=hallucinated,
            )
        )

    return results


def aggregate_latencies(items: List[QueryBenchmark]) -> Dict[str, Any]:
    def collect(fn) -> List[float]:
        vals: List[float] = []
        for elem in items:
            value = fn(elem)
            if value is not None:
                vals.append(value)
        return vals

    retrieval_vals = collect(lambda x: x.latencies.retrieval_ms)
    rerank_vals = collect(lambda x: x.latencies.rerank_ms)
    generation_vals = collect(lambda x: x.latencies.generation_ms)
    total_vals = collect(lambda x: x.latencies.total_ms)

    def summary(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        return {
            "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p90": statistics.quantiles(values, n=10)[8] if len(values) >= 10 else max(values),
            "max": max(values),
        }

    hallucination_scores = collect(lambda x: x.hallucination_score)
    hallucinated_flags = [x.hallucinated for x in items if x.hallucinated is not None]
    hallucination_rate = None
    if hallucinated_flags:
        hallucination_rate = sum(1 for flag in hallucinated_flags if flag) / len(hallucinated_flags)

    return {
        "retrieval_ms": summary(retrieval_vals),
        "rerank_ms": summary(rerank_vals),
        "generation_ms": summary(generation_vals),
        "total_ms": summary(total_vals),
        "hallucination": {
            "scores": summary(hallucination_scores),
            "rate": hallucination_rate,
        },
        "count": len(items),
    }


def save_results(output_path: Path, summary: Dict[str, Any], per_query: List[QueryBenchmark]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "details": [
            {
                **{
                    "query_id": item.query_id,
                    "query_text": item.query_text,
                    "retrieval_count": item.retrieval_count,
                    "hallucination_score": item.hallucination_score,
                    "hallucinated": item.hallucinated,
                },
                **asdict(item.latencies),
            }
            for item in per_query
        ],
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    if args.skip_generation and args.mock_generation:
        raise SystemExit("Không thể đồng thời --skip-generation và --mock-generation")

    queries = load_queries(args.queries, args.limit)
    if not queries:
        raise SystemExit(f"Không đọc được query nào từ {args.queries}")

    retriever = RetrievalService(use_gpu=False)

    generator: Optional[GeminiGenerator]
    if args.mock_generation or args.skip_generation:
        generator = None
    else:
        try:
            generator = GeminiGenerator()
        except Exception as exc:
            print(f"⚠️  Không thể khởi tạo Gemini: {exc}. Sẽ bỏ qua bước generation.")
            generator = None

    per_query = benchmark_queries(
        queries=queries,
        retriever=retriever,
        top_k=args.top_k,
        generator=generator,
        hallucination_threshold=args.hallucination_threshold,
        mock_generation=args.mock_generation,
    )

    summary = aggregate_latencies(per_query)
    save_results(args.output, summary, per_query)

    print(f"Đã ghi kết quả benchmark vào {args.output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
