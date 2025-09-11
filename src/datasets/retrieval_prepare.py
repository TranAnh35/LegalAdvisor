#!/usr/bin/env python3
"""
Chuẩn hóa dữ liệu cho Retriever (VNLAWQC, VNSynLawQC):

- Trích (query, positive_text) từ dữ liệu thô, ánh xạ positive_text → chunk_id bằng FAISS hiện có
- Sinh hard negatives bằng cách truy hồi topK rồi loại bỏ positive
- Ghi ra JSONL: {query, positive_id, hard_negatives, positive_score, source}

Chạy ví dụ (PowerShell, Windows):

  conda activate LegalAdvisor
  python -m src.datasets.retrieval_prepare \
    --input data/raw/VNLAWQC.jsonl data/raw/VNSynLawQC.jsonl \
    --output data/processed/retrieval_train.jsonl --hard-negatives 15 --dense-top-k 64

Nếu không truyền --input, script sẽ cố gắng dò các file phổ biến trong data/raw/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Bảo đảm import được service/paths
try:
    # Khi chạy dưới dạng module: python -m src.datasets.retrieval_prepare
    from src.retrieval.service import RetrievalService
    from src.utils.paths import get_project_root, get_processed_data_dir
except Exception:
    # Khi chạy trực tiếp: python src/datasets/retrieval_prepare.py
    THIS_FILE = Path(__file__).resolve()
    SRC_DIR = THIS_FILE.parent.parent
    sys.path.append(str(SRC_DIR))
    from retrieval.service import RetrievalService  # type: ignore
    from utils.paths import get_project_root, get_processed_data_dir  # type: ignore


# -----------------------------
# Tiện ích xử lý văn bản
# -----------------------------

def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    # Thay '_' do tokenizer thành ' ', cắt khoảng trắng
    return text.replace("_", " ").strip()


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard đơn giản (token set)."""
    if not a or not b:
        return 0.0
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return inter / union


# -----------------------------
# Đọc dữ liệu đầu vào linh hoạt
# -----------------------------

def _read_json_or_jsonl(path: Path) -> Iterable[Dict]:
    """Đọc file JSON (list) hoặc JSONL (mỗi dòng một JSON)."""
    if not path.exists():
        return []
    if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
        # Đơn giản: không xử lý gz ở đây để tránh phụ thuộc
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue
    else:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            elif isinstance(data, dict):
                # Một số bộ có dạng {"data":[...]}
                items = data.get("data") or data.get("examples") or []
                if isinstance(items, list):
                    for obj in items:
                        if isinstance(obj, dict):
                            yield obj
        except Exception:
            return []


def _extract_query(obj: Dict) -> Optional[str]:
    for key in ("query", "question", "q", "prompt"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_text(val)
    return None


def _extract_positive_texts(obj: Dict) -> List[str]:
    """Cố gắng lấy positive passages từ nhiều schema khác nhau.

    Ưu tiên: context/positive_ctxs → text; nếu không có, trả []
    """
    positives: List[str] = []

    # Trường đơn lẻ
    for key in ("context", "positive", "positive_text"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            positives.append(normalize_text(val))

    # Danh sách passages
    for key in ("positive_ctxs", "positive_passages", "ctxs", "passages", "contexts"):
        arr = obj.get(key)
        if isinstance(arr, list):
            for item in arr:
                if isinstance(item, dict):
                    # Các tên trường khả dĩ
                    txt = item.get("text") or item.get("content") or item.get("passage")
                    # Một số dataset đánh dấu positive
                    flag = item.get("is_positive") or item.get("label") == "positive"
                    if txt and (flag or key in ("positive_ctxs", "positive_passages")):
                        positives.append(normalize_text(str(txt)))
    # Loại rỗng & trùng
    positives = [p for p in positives if p]
    if positives:
        # Cắt ngắn để tránh đoạn quá dài
        positives = [p[:4000] for p in positives]
    return positives


@dataclass
class Example:
    query: str
    positive_text: Optional[str]
    source: str
    positive_id: Optional[int] = None  # Hỗ trợ đầu vào đã có sẵn gold chunk id


def iter_examples_from_path(path: Path) -> Iterable[Example]:
    source_name = path.stem
    for obj in _read_json_or_jsonl(path):
        q = _extract_query(obj)
        if not q:
            continue
        positives = _extract_positive_texts(obj)
        # Thử đọc sẵn positive_id/gold_chunk_id nếu có
        pos_id: Optional[int] = None
        for key in ("positive_id", "gold_chunk_id", "gold_id", "positive_chunk_id"):
            if key in obj:
                try:
                    pos_id = int(obj.get(key))  # type: ignore
                    break
                except Exception:
                    pos_id = None
        
        if positives:
            # Có thể lấy nhiều positives; để cân bằng, chỉ lấy 1 cái đầu
            yield Example(query=q, positive_text=positives[0], source=source_name, positive_id=pos_id)
        else:
            # Trường hợp chỉ có query: để None, sẽ ánh xạ positive bằng query
            yield Example(query=q, positive_text=None, source=source_name, positive_id=pos_id)


def auto_find_raw_files(raw_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    if not raw_dir.exists():
        return candidates
    names = [
        "VNLAWQC.jsonl", "VNLAWQC.json",
        "VNSynLawQC.jsonl", "VNSynLawQC.json",
        "vlqa.json", "vlqa.jsonl",
    ]
    lower_to_path: Dict[str, Path] = {}
    for p in raw_dir.glob("**/*"):
        if p.suffix.lower() not in {".json", ".jsonl"}:
            continue
        lower_to_path[p.name.lower()] = p
    for n in names:
        p = lower_to_path.get(n.lower())
        if p:
            candidates.append(p)
    return candidates


# -----------------------------
# Ánh xạ passage → chunk_id bằng FAISS
# -----------------------------

class PositiveMapper:
    def __init__(self, use_gpu: bool = False) -> None:
        self.retriever = RetrievalService(use_gpu=use_gpu)

    def map_text_to_chunk(self, text: str, dense_top_k: int = 8) -> Optional[Tuple[int, float]]:
        """Dùng encoder + FAISS để tìm chunk gần nhất cho một đoạn văn hoặc chính query."""
        text = normalize_text(text)
        if not text:
            return None
        try:
            query_vec = self.retriever.encode_query(text)
            k = max(1, min(dense_top_k, int(self.retriever.index.ntotal)))
            distances, indices = self.retriever.index.search(query_vec, k)
            if k <= 0 or len(indices[0]) == 0:
                return None
            # Chọn phần tử tốt nhất (giá trị distance cao nhất vì dùng IP + normalized)
            best_idx = int(indices[0][0])
            best_score = float(distances[0][0])
            return best_idx, best_score
        except Exception:
            return None

    def get_hard_negatives(self, query: str, positive_id: Optional[int], top_k: int = 32, max_return: int = 15) -> List[int]:
        results = self.retriever.retrieve(query, top_k=top_k)
        ids: List[int] = []
        for r in results:
            cid = int(r.get("chunk_id")) if r.get("chunk_id") is not None else -1
            if cid < 0:
                continue
            if positive_id is not None and cid == positive_id:
                continue
            ids.append(cid)
            if len(ids) >= max_return:
                break
        return ids


# -----------------------------
# Ghi JSONL
# -----------------------------

def write_jsonl(output_path: Path, records: Iterable[Dict]) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


# -----------------------------
# Main pipeline
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Chuẩn hóa dữ liệu retriever → retrieval_train.jsonl")
    parser.add_argument("--input", nargs="*", default=[], help="Đường dẫn file JSON/JSONL (VNLAWQC, VNSynLawQC)")
    parser.add_argument("--output", default=str(get_processed_data_dir() / "retrieval_train.jsonl"), help="File JSONL đầu ra")
    parser.add_argument("--dense-top-k", type=int, default=8, help="TopK khi ánh xạ positive_text → chunk_id")
    parser.add_argument("--hard-negatives", type=int, default=15, help="Số lượng hard negatives mỗi ví dụ")
    parser.add_argument("--hn-top-k", type=int, default=64, help="TopK truy hồi để rút hard negatives")
    parser.add_argument("--min-positive-score", type=float, default=0.20, help="Ngưỡng distance tối thiểu để chấp nhận positive mapping")
    parser.add_argument("--skip-jaccard", action="store_true", help="Bỏ qua kiểm tra Jaccard khi positive được map tự động (tăng tốc)")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu/tập (0 = không giới hạn)")
    parser.add_argument("--use-gpu", action="store_true", help="Dùng GPU nếu có sẵn")
    args = parser.parse_args()

    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"

    # Tìm input nếu không truyền
    input_paths: List[Path] = [Path(p) for p in args.input]
    if not input_paths:
        input_paths = auto_find_raw_files(raw_dir)

    if not input_paths:
        print("❌ Không tìm thấy bất kỳ file đầu vào nào trong --input hoặc data/raw/.")
        print("   Vui lòng chỉ định --input các file VNLAWQC/VNSynLawQC.")
        sys.exit(1)

    print("🚀 Chuẩn hóa dữ liệu retriever...")
    print(f"📂 Input files: {', '.join(str(p) for p in input_paths)}")
    print(f"💾 Output: {args.output}")

    # Khởi tạo mapper sử dụng FAISS hiện tại
    mapper = PositiveMapper(use_gpu=args.use_gpu)

    # Tạo generator ghi JSONL dần dần
    def gen_records() -> Iterable[Dict]:
        for path in input_paths:
            count = 0
            for ex in iter_examples_from_path(path):
                # Nếu input đã có sẵn positive_id thì dùng luôn (tăng tốc cực mạnh)
                if ex.positive_id is not None:
                    positive_id = int(ex.positive_id)
                    positive_score = 1.0
                else:
                    # Ánh xạ positive: nếu có positive_text dùng nó, nếu không dùng chính query
                    basis_text = ex.positive_text if ex.positive_text else ex.query
                    mapped = mapper.map_text_to_chunk(basis_text, dense_top_k=args.dense_top_k)
                    if not mapped:
                        continue
                    positive_id, positive_score = mapped

                    # Kiểm tra ngưỡng (chỉ áp dụng khi phải tự map)
                    if positive_score < args.min_positive_score and not args.skip_jaccard:
                        # Thử một thủ thuật: so Jaccard với content top1 để lọc nhẹ
                        top_content = mapper.retriever.get_chunk_content(positive_id) or ""
                        if jaccard_similarity(ex.positive_text, top_content) < 0.08:
                            continue

                # Sinh hard negatives theo truy vấn gốc
                hard_negs = mapper.get_hard_negatives(
                    query=ex.query,
                    positive_id=positive_id,
                    top_k=args.hn_top_k,
                    max_return=args.hard_negatives,
                )

                yield {
                    "query": ex.query,
                    "positive_id": int(positive_id),
                    "hard_negatives": [int(x) for x in hard_negs],
                    "positive_score": float(positive_score),
                    "source": ex.source,
                }

                count += 1
                if args.limit and count >= int(args.limit):
                    break

    out_path = Path(args.output)
    total = write_jsonl(out_path, gen_records())

    print("✅ Hoàn thành chuẩn hóa dữ liệu retriever!")
    print(f"📊 Số dòng ghi: {total}")
    print(f"📁 File: {out_path}")


if __name__ == "__main__":
    main()


