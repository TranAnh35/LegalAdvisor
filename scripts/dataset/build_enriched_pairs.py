"""Zalo Legal dataset preparation utilities.

Tạo file queries đã loại trùng và pairs train đã enrich để phục vụ huấn luyện retrieval.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.zalo_legal import parse_corpus_id


JsonDict = Dict[str, object]


@dataclass
class QueryDedupResult:
    unique_queries: Dict[str, JsonDict]
    duplicate_count: int
    total: int


def read_jsonl(path: Path) -> Iterator[Tuple[int, JsonDict]]:
    """Yield (line_no, record) from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Lỗi JSON ở dòng {line_no} của {path}: {exc}") from exc


def write_jsonl(path: Path, records: Iterable[JsonDict]) -> int:
    """Write records ra JSONL, trả về số dòng."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        count = 0
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def deduplicate_queries(input_path: Path) -> QueryDedupResult:
    """Loại trùng queries theo _id, ưu tiên lần xuất hiện đầu."""

    unique: "OrderedDict[str, JsonDict]" = OrderedDict()
    duplicates = 0

    for line_no, record in tqdm(
        read_jsonl(input_path),
        desc="Doc queries",
        unit="line",
    ):
        query_id = record.get("_id")
        text = (record.get("text") or "").strip()
        if not query_id or not text:
            raise ValueError(f"Dòng {line_no} thiếu _id hoặc text")

        if query_id in unique:
            duplicates += 1
            continue

        unique[query_id] = {"_id": query_id, "text": text}

    return QueryDedupResult(unique_queries=dict(unique), duplicate_count=duplicates, total=len(unique) + duplicates)


def build_enriched_pairs(pairs_path: Path, query_map: Dict[str, JsonDict]) -> List[JsonDict]:
    """Ghép query text và metadata corpus vào pairs train."""

    enriched: List[JsonDict] = []
    missing_queries = 0

    for _, record in tqdm(
        read_jsonl(pairs_path),
        desc="Doc pairs",
        unit="line",
    ):
        query_id = record.get("query-id")
        corpus_id = record.get("corpus-id")
        score = record.get("score", 1.0)

        if query_id not in query_map:
            missing_queries += 1
            continue

        query_text = query_map[query_id]["text"]
        type_, number, year, suffix = parse_corpus_id(str(corpus_id)) if corpus_id else ("", "", "", "")

        enriched.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "corpus_id": corpus_id,
                "score": score,
                "doc_type": type_,
                "doc_number": number,
                "doc_year": year,
                "doc_suffix": suffix,
            }
        )

    if missing_queries:
        raise ValueError(f"Có {missing_queries} pairs không tìm thấy query tương ứng. Kiểm tra dữ liệu đầu vào.")

    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn hóa dữ liệu Zalo Legal (dedup queries + enrich pairs)")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw/zalo_ai_legal_text_retrieval"),
        help="Thư mục chứa queries.jsonl, pairs_train.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/zalo-legal"),
        help="Thư mục lưu kết quả",
    )

    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    output_dir: Path = args.output_dir
    queries_path = raw_dir / "queries.jsonl"
    pairs_path = raw_dir / "pairs_train.jsonl"
    queries_out_path = output_dir / "queries_dedup.jsonl"
    pairs_out_path = output_dir / "train_pairs_enriched.jsonl"

    if not queries_path.exists() or not pairs_path.exists():
        raise FileNotFoundError("Không tìm thấy queries.jsonl hoặc pairs_train.jsonl trong thư mục raw.")

    print("[1/2] Bat dau deduplicate queries")
    dedup_result = deduplicate_queries(queries_path)
    unique_queries = dedup_result.unique_queries
    written_queries = write_jsonl(queries_out_path, unique_queries.values())

    print(
        f"OK - Ghi {written_queries} query (tong {dedup_result.total}, loai trung {dedup_result.duplicate_count})."
    )

    print("[2/2] Bat dau enrich pairs train")
    enriched_pairs = build_enriched_pairs(pairs_path, unique_queries)
    written_pairs = write_jsonl(pairs_out_path, enriched_pairs)
    print(f"OK - Ghi {written_pairs} dong pairs enrich.")

    print("Hoan tat. File dau ra:")
    print(f"  - {queries_out_path}")
    print(f"  - {pairs_out_path}")


if __name__ == "__main__":
    main()