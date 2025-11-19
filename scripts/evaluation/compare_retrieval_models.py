#!/usr/bin/env python3
"""
So sánh hiệu năng retrieval của 3 mô hình:
- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- intfloat/multilingual-e5-small (base)
- intfloat/multilingual-e5-small đã fine-tune (đường dẫn local)

Đầu ra: JSON chứa metrics recall/mrr ở các top-K cho từng model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "retrieval"


def read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON lỗi tại dòng {line_no} ở {path}: {exc}") from exc
    return rows


def load_pairs(path: Path) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    query_texts: Dict[str, str] = {}
    positives: Dict[str, Set[str]] = {}
    for record in read_jsonl(path):
        qid = str(record.get("query_id") or "").strip()
        qtext = str(record.get("query_text") or "").strip()
        cid = str(record.get("corpus_id") or "").strip()
        if not qid or not qtext or not cid:
            continue
        query_texts.setdefault(qid, qtext)
        positives.setdefault(qid, set()).add(cid)
    if not query_texts:
        raise ValueError(f"pairs file {path} rỗng hoặc không hợp lệ")
    return query_texts, positives


def compose_doc_text(record: Dict[str, object]) -> str:
    text = str(record.get("content") or record.get("text") or "").strip()
    title = str(record.get("title") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def load_corpus(path: Path) -> Tuple[List[str], List[str]]:
    ids: List[str] = []
    texts: List[str] = []
    for record in read_jsonl(path):
        doc_id = str(record.get("_id") or record.get("doc_id") or record.get("id") or "").strip()
        if not doc_id:
            continue
        text = compose_doc_text(record)
        if not text:
            continue
        ids.append(doc_id)
        texts.append(text)
    if not ids:
        raise ValueError(f"corpus file {path} rỗng hoặc không hợp lệ")
    return ids, texts


def adjust_max_seq_len(model: SentenceTransformer, max_seq_len: int) -> None:
    try:
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = max(8, int(max_seq_len))
        try:
            first = model[0]
        except Exception:
            first = None
        if first is not None and hasattr(first, "max_seq_length"):
            setattr(first, "max_seq_length", max(8, int(max_seq_len)))
    except Exception:
        pass


def encode_batch(model: SentenceTransformer, texts: List[str], batch_size: int) -> torch.Tensor:
    # sentence-transformers < 3.0 không hỗ trợ progress_bar_desc, nên chỉ bật progress mặc định
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )


def evaluate_model(
    model: SentenceTransformer,
    doc_ids: List[str],
    doc_texts: List[str],
    query_texts: Dict[str, str],
    positives: Dict[str, Set[str]],
    top_ks: List[int],
    batch_size: int,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        print("   → Encode corpus")
        doc_emb = encode_batch(model, doc_texts, batch_size)
        q_ids = list(query_texts.keys())
        queries = [query_texts[qid] for qid in q_ids]
        print("   → Encode queries")
        q_emb = encode_batch(model, queries, batch_size)
        scores = torch.matmul(q_emb, doc_emb.T).cpu().numpy()

    metrics: Dict[str, Dict[str, float]] = {}
    for k in tqdm(top_ks, desc="Tính metrics tổng"):
        k = min(k, scores.shape[1])
        top_idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        partial = np.take_along_axis(scores, top_idx, axis=1)
        order = np.argsort(-partial, axis=1)
        ranked = np.take_along_axis(top_idx, order, axis=1)

        hits = 0
        reciprocal_ranks: List[float] = []
        for row, doc_indices in enumerate(
            tqdm(ranked, desc=f"K={k}", leave=False, total=len(ranked))
        ):
            pos = positives.get(q_ids[row], set())
            found_rank = None
            for rank, doc_idx in enumerate(doc_indices, start=1):
                doc_id = doc_ids[int(doc_idx)]
                if doc_id in pos:
                    hits += 1
                    found_rank = rank
                    break
            if found_rank:
                reciprocal_ranks.append(1.0 / found_rank)

        recall = hits / len(q_ids)
        mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
        metrics[f"K={k}"] = {"recall": recall, "mrr": mrr}
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="So sánh 3 model retrieval (MiniLM vs e5-small vs e5-small finetune)")
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/zalo-legal/corpus_cleaned.jsonl"))
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/zalo-legal/train_pairs_enriched.jsonl"))
    parser.add_argument(
        "--finetuned-dir",
        type=Path,
        required=True,
        help="Đường dẫn thư mục model e5-small đã fine-tune",
    )
    parser.add_argument("--output", type=Path, default=Path("results/retrieval/compare_three_models.json"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-ks", type=str, default="5,10,20")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--limit-queries", type=int, default=0, help="Giới hạn số query (0 = tất cả)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--minilm-model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Tên model HF cho MiniLM",
    )
    parser.add_argument(
        "--e5-small-model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Tên model HF cho e5-small base",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Đã yêu cầu CUDA nhưng không có GPU khả dụng")
    return torch.device(arg)


def main() -> None:
    args = parse_args()
    corpus_path = resolve_path(args.corpus)
    pairs_path = resolve_path(args.pairs)
    finetuned_dir = resolve_path(args.finetuned_dir)
    output_path = resolve_path(args.output)

    if not corpus_path.exists():
        raise FileNotFoundError(f"Không tìm thấy corpus: {corpus_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Không tìm thấy pairs: {pairs_path}")
    if not finetuned_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy model fine-tune: {finetuned_dir}")

    query_texts, positives = load_pairs(pairs_path)
    doc_ids, doc_texts = load_corpus(corpus_path)

    if args.limit_queries > 0:
        limited = list(query_texts.keys())[: args.limit_queries]
        query_texts = {qid: query_texts[qid] for qid in limited}
        positives = {qid: positives.get(qid, set()) for qid in limited}

    device = pick_device(args.device)
    top_ks = [int(k.strip()) for k in args.top_ks.split(",") if k.strip().isdigit()]
    top_ks = [k for k in top_ks if k > 0]
    if not top_ks:
        top_ks = [5, 10]

    model_specs = [
        {"label": "MiniLM", "path": args.minilm_model, "source": "hf"},
        {"label": "e5-small-base", "path": args.e5_small_model, "source": "hf"},
        {"label": "e5-small-finetune", "path": str(finetuned_dir), "source": "local"},
    ]

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for spec in tqdm(model_specs, desc="Benchmark từng model"):
        print(f"\n▶️  Đang benchmark model: {spec['label']} ({spec['path']})")
        model = SentenceTransformer(spec["path"], device=device)
        adjust_max_seq_len(model, args.max_seq_len)
        metrics = evaluate_model(
            model,
            doc_ids,
            doc_texts,
            query_texts,
            positives,
            top_ks,
            args.batch_size,
        )
        results[spec["label"]] = metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "top_ks": top_ks,
        "query_count": len(query_texts),
        "models": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Đã lưu kết quả so sánh tại {output_path}")


if __name__ == "__main__":
    main()

