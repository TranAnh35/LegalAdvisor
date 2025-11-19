"""Generate hard negatives and triplets for Zalo Legal retrieval training."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from rank_bm25 import BM25Okapi
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


JsonDict = Dict[str, object]


@dataclass
class TripletConfig:
    bm25_top_k: int = 50
    max_negatives: int = 10
    min_negatives: int = 8


@dataclass
class TripletSample:
    query_id: str
    query_text: str
    positive_id: str
    positive_text: str
    negative_ids: Sequence[str]
    negative_texts: Sequence[str]


TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return TOKEN_PATTERN.findall(text)


def read_jsonl(path: Path) -> Iterator[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover
                raise ValueError(f"Loi JSON o dong {line_no} cua {path}: {exc}") from exc


def compose_doc_text(record: JsonDict) -> str:
    title = (record.get("title") or "").strip()
    body = (record.get("text") or "").strip()
    if title and body:
        return f"{title}\n{body}"
    return title or body


def load_corpus(corpus_path: Path) -> Tuple[List[str], List[List[str]], Dict[str, str]]:
    doc_ids: List[str] = []
    tokenized: List[List[str]] = []
    doc_texts: Dict[str, str] = {}

    for record in tqdm(read_jsonl(corpus_path), desc="Doc corpus", unit="line"):
        doc_id = record.get("_id")
        if not doc_id:
            continue
        text = compose_doc_text(record)
        if not text:
            continue

        tokens = tokenize(text)
        if not tokens:
            continue

        doc_ids.append(doc_id)
        tokenized.append(tokens)
        doc_texts[doc_id] = text

    if not doc_ids:
        raise ValueError("Corpus khong co tai lieu hop le")

    return doc_ids, tokenized, doc_texts


def load_enriched_pairs(pairs_path: Path) -> Tuple[Dict[str, str], Dict[str, set], Dict[str, List[JsonDict]]]:
    query_texts: Dict[str, str] = {}
    positives: Dict[str, set] = defaultdict(set)
    rows_by_query: Dict[str, List[JsonDict]] = defaultdict(list)

    for record in tqdm(read_jsonl(pairs_path), desc="Doc pairs", unit="line"):
        query_id = record.get("query_id")
        query_text = record.get("query_text")
        corpus_id = record.get("corpus_id")

        if not query_id or not query_text or not corpus_id:
            continue

        query_texts.setdefault(query_id, query_text)
        positives[query_id].add(corpus_id)
        rows_by_query[query_id].append(record)

    if not rows_by_query:
        raise ValueError("pairs_train_enriched.jsonl rong hoac khong hop le")

    return query_texts, positives, rows_by_query


def select_negatives(
    bm25: BM25Okapi,
    doc_ids: Sequence[str],
    doc_texts: Dict[str, str],
    query_text: str,
    positive_ids: set,
    config: TripletConfig,
) -> List[str]:
    tokens = tokenize(query_text)
    if not tokens:
        return []

    scores = bm25.get_scores(tokens)
    ranked_idx = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

    negatives: List[str] = []
    for idx in ranked_idx[: config.bm25_top_k]:
        doc_id = doc_ids[idx]
        if doc_id in positive_ids:
            continue
        if doc_id not in doc_texts:
            continue
        negatives.append(doc_id)
        if len(negatives) >= config.max_negatives:
            break

    return negatives


def build_triplets(
    pairs_by_query: Dict[str, List[JsonDict]],
    query_texts: Dict[str, str],
    positives: Dict[str, set],
    doc_texts: Dict[str, str],
    doc_ids: Sequence[str],
    bm25: BM25Okapi,
    config: TripletConfig,
) -> Tuple[List[TripletSample], Dict[str, float]]:
    triplets: List[TripletSample] = []
    neg_counts: List[int] = []
    insufficient_queries = 0

    for query_id, positive_ids in tqdm(positives.items(), desc="Build triplets", unit="query"):
        query_text = query_texts.get(query_id)
        if not query_text:
            continue

        rows = pairs_by_query.get(query_id)
        if not rows:
            continue

        negatives = select_negatives(bm25, doc_ids, doc_texts, query_text, positive_ids, config)
        neg_counts.append(len(negatives))
        if len(negatives) < config.min_negatives:
            insufficient_queries += 1

        negative_texts = [doc_texts[nid] for nid in negatives]

        for row in rows:
            positive_id = row.get("corpus_id")
            if positive_id not in doc_texts:
                continue
            positive_text = doc_texts[positive_id]
            triplets.append(
                TripletSample(
                    query_id=query_id,
                    query_text=query_text,
                    positive_id=positive_id,
                    positive_text=positive_text,
                    negative_ids=negatives,
                    negative_texts=negative_texts,
                )
            )

    if not triplets:
        raise ValueError("Khong tao duoc triplet nao. Kiem tra du lieu va tham so.")

    total_queries = len(positives)
    with_min_neg = sum(1 for count in neg_counts if count >= config.min_negatives)
    stats = {
        "queries_total": total_queries,
        "queries_with_min_neg": with_min_neg,
        "queries_with_min_neg_ratio": with_min_neg / total_queries if total_queries else 0.0,
        "avg_negatives": statistics.mean(neg_counts) if neg_counts else 0.0,
        "median_negatives": statistics.median(neg_counts) if neg_counts else 0.0,
        "min_negatives": min(neg_counts) if neg_counts else 0,
        "max_negatives": max(neg_counts) if neg_counts else 0,
        "insufficient_queries": insufficient_queries,
        "min_required": config.min_negatives,
        "max_target": config.max_negatives,
        "bm25_top_k": config.bm25_top_k,
    }

    return triplets, stats


def triplet_to_json(sample: TripletSample) -> JsonDict:
    return {
        "query_id": sample.query_id,
        "query_text": sample.query_text,
        "positive_id": sample.positive_id,
        "positive_text": sample.positive_text,
        "negative_ids": list(sample.negative_ids),
        "negative_texts": list(sample.negative_texts),
    }


def write_jsonl(path: Path, records: Iterable[JsonDict], *, desc: str | None = None, total: int | None = None) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    iterator: Iterable[JsonDict] = records
    if desc:
        iterator = tqdm(records, desc=desc, total=total, unit="row")

    with path.open("w", encoding="utf-8") as handle:
        count = 0
        for record in iterator:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Tao hard negatives va triplets cho Zalo Legal")
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed/zalo-legal"))
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/processed/zalo-legal/corpus_cleaned.jsonl"),
        help="Path corpus cleaned (mac dinh dung ban da normalize)",
    )
    parser.add_argument(
        "--bm25-top-k",
        type=int,
        default=50,
        help="So luong tai lieu lay tu BM25 truoc khi loc",
    )
    parser.add_argument(
        "--max-negatives",
        type=int,
        default=10,
        help="So negatives toi da moi query",
    )
    parser.add_argument(
        "--min-negatives",
        type=int,
        default=8,
        help="So negatives toi thieu moi query",
    )

    args = parser.parse_args()

    processed_dir: Path = args.processed_dir
    pairs_path = processed_dir / "train_pairs_enriched.jsonl"
    queries_path = processed_dir / "queries_dedup.jsonl"
    corpus_path: Path = args.corpus

    if not pairs_path.exists():
        raise FileNotFoundError(f"Khong tim thay {pairs_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Khong tim thay {queries_path}")
    if not corpus_path.exists():
        raise FileNotFoundError(f"Khong tim thay {corpus_path}")

    config = TripletConfig(
        bm25_top_k=args.bm25_top_k,
        max_negatives=args.max_negatives,
        min_negatives=args.min_negatives,
    )

    print("[1/4] Doc corpus va build BM25")
    doc_ids, tokenized_docs, doc_texts = load_corpus(corpus_path)
    bm25 = BM25Okapi(tokenized_docs)

    print("[2/4] Doc pairs va truy vet query text")
    query_texts, positives, pairs_by_query = load_enriched_pairs(pairs_path)

    print("[3/4] Build triplets voi tham so")
    triplets, stats = build_triplets(pairs_by_query, query_texts, positives, doc_texts, doc_ids, bm25, config)

    triplets_path = processed_dir / "triplets_train.jsonl"
    stats_path = Path("results/retrieval/negatives_stats.json")

    print("[4/4] Ghi ket qua")
    triplet_count = write_jsonl(
        triplets_path,
        (triplet_to_json(t) for t in triplets),
        desc="Write triplets",
        total=len(triplets),
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"OK - Triplets: {triplet_count} dong -> {triplets_path}")
    print(f"OK - Stats -> {stats_path}")


if __name__ == "__main__":
    main()