#!/usr/bin/env python3
"""
Tải và chuẩn hóa một phần dữ liệu retriever vào data/raw/:

- VNLAWQC, VNSynLawQC từ Hugging Face (bkai-foundation-models/crosslingual)

Đầu ra mỗi bộ: JSONL trong data/raw/, mỗi dòng là một object tối thiểu có:
  { "query": str, "positive_ctxs": [{"text": str, "is_positive": true}], "source": str, "raw": {...} }

Cách chạy (PowerShell, Windows):

  conda activate LegalAdvisor
  python scripts\download_datasets.py --datasets vnlawqc vnsynlawqc --output-dir data\raw --limit 0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from huggingface_hub import snapshot_download  # type: ignore


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return str(text).replace("_", " ").strip()


def _read_json_or_jsonl(path: Path) -> Iterable[Dict]:
    """Đọc JSON/JSONL đơn giản, dùng trong fallback snapshot."""
    try:
        if path.suffix.lower() in {".jsonl", ".jsonl.gz"}:
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
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            elif isinstance(data, dict):
                arr = data.get("data") or data.get("examples") or []
                if isinstance(arr, list):
                    for obj in arr:
                        if isinstance(obj, dict):
                            yield obj
    except Exception:
        return []


def extract_query(obj: Dict) -> Optional[str]:
    for key in ("query", "question", "q", "prompt", "input"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return normalize_text(val)
    return None


def extract_positives(obj: Dict) -> List[str]:
    positives: List[str] = []

    # String fields
    for k in ("positive", "positive_text", "context", "passage", "document", "target_text"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            positives.append(normalize_text(v))

    # List fields with dict items
    for k in ("positive_ctxs", "positive_passages", "ctxs", "passages", "contexts", "positives", "pos"):
        v = obj.get(k)
        if isinstance(v, list):
            for it in v:
                if isinstance(it, dict):
                    txt = it.get("text") or it.get("content") or it.get("passage")
                    flag = it.get("is_positive") or it.get("label") == "positive"
                    if txt and (flag or k in ("positive_ctxs", "positive_passages", "positives", "pos")):
                        positives.append(normalize_text(txt))
                elif isinstance(it, str) and it.strip():
                    positives.append(normalize_text(it))

    # Dedup & truncate
    seen = set()
    uniq: List[str] = []
    for p in positives:
        if not p:
            continue
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p[:4000])
    return uniq


def to_jsonl(records: Iterable[Dict], out_path: Path) -> int:
    ensure_dir(out_path.parent)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def download_crosslingual_splits(want: List[str], limit: int = 0, splits: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """Tải từ hf dataset bkai-foundation-models/crosslingual.

    Cố gắng tải theo config nếu có (vd: vnlawqc, vnsynlawqc). Nếu không, tải mặc định
    rồi lọc theo trường nhận diện khi có thể.
    """
    from datasets import load_dataset  # lazy import

    results: Dict[str, List[Dict]] = {name.lower(): [] for name in want}

    # Chỉ tải các split cụ thể để tránh đọc phần eval gây lỗi pyarrow
    candidate_splits = splits or ["train", "validation", "dev"]

    for name in want:
        key = name.lower()
        loaded_any = False
        for sp in candidate_splits:
            try:
                ds = load_dataset(
                    "bkai-foundation-models/crosslingual",
                    name=key,
                    split=sp,
                    ignore_verifications=True,
                )
                count = 0
                for ex in ds:
                    results[key].append(ex)
                    count += 1
                    if limit and count >= limit:
                        break
                if count > 0:
                    loaded_any = True
            except Exception:
                # Bỏ qua split lỗi
                continue

        if not loaded_any:
            # Fallback: cố gắng tải toàn bộ rồi lọc theo tag nguồn
            try:
                ds_all = load_dataset(
                    "bkai-foundation-models/crosslingual",
                    split="train",
                    ignore_verifications=True,
                )
                for ex in ds_all:
                    tag = (str(ex.get("dataset_name") or ex.get("source") or "").lower())
                    if key in tag:
                        results[key].append(ex)
                        if limit and len(results[key]) >= limit:
                            break
            except Exception:
                pass

    return results


def download_crosslingual_via_repo(want: List[str], limit: int = 0) -> Dict[str, List[Dict]]:
    """Fallback: tải toàn bộ repo files rồi tự đọc JSON/JSONL theo heuristic.

    - Tìm các file có tên chứa từ khóa của từng config (vnlaw, vnsyn)
    - Chỉ đọc JSON/JSONL; hợp nhất thành list mẫu thô để normalize sau
    """
    results: Dict[str, List[Dict]] = {name.lower(): [] for name in want}
    try:
        repo_dir = Path(snapshot_download(repo_id="bkai-foundation-models/crosslingual"))
    except Exception:
        return results

    # Thử parse cặp file eval/filtered_corpus.json + data_eval_law_dev_queries.json nếu có
    def parse_eval_pair(max_return: int) -> List[Dict]:
        try:
            # Ưu tiên các file trong thư mục 'eval'
            corpus_files = list(repo_dir.rglob("eval/*corpus*.json")) or list(repo_dir.rglob("*corpus*.json"))
            queries_files = list(repo_dir.rglob("eval/*queries*.json")) or list(repo_dir.rglob("*queries*.json"))
            if not corpus_files or not queries_files:
                return []

            # Chọn file đầu tiên theo ưu tiên
            corpus_path = sorted(corpus_files, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)[0]
            queries_path = sorted(queries_files, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)[0]

            # Đọc corpus
            id_to_text: Dict[str, str] = {}
            for obj in _read_json_or_jsonl(corpus_path):
                if not isinstance(obj, dict):
                    continue
                doc_id = obj.get("doc_id") or obj.get("id") or obj.get("pid") or obj.get("docid")
                text = obj.get("text") or obj.get("contents") or obj.get("document") or obj.get("passage")
                if doc_id is None or not text:
                    continue
                id_to_text[str(doc_id)] = normalize_text(str(text))

            if not id_to_text:
                return []

            # Đọc queries
            parsed: List[Dict] = []
            for qobj in _read_json_or_jsonl(queries_path):
                if not isinstance(qobj, dict):
                    continue
                q = qobj.get("query") or qobj.get("question") or qobj.get("q") or qobj.get("text")
                if not q or not str(q).strip():
                    continue
                # Nhiều biến thể key cho id dương
                pos_ids = (
                    qobj.get("positive_doc_ids")
                    or qobj.get("relevant_doc_ids")
                    or qobj.get("doc_ids")
                    or qobj.get("positive_ids")
                    or qobj.get("gold_ids")
                )
                if isinstance(pos_ids, list) and pos_ids:
                    pid = str(pos_ids[0])
                else:
                    # Thử single id
                    single_id = qobj.get("doc_id") or qobj.get("positive_id") or qobj.get("gold_id")
                    pid = str(single_id) if single_id is not None else None

                if not pid or pid not in id_to_text:
                    continue

                parsed.append({
                    "query": normalize_text(str(q)),
                    "positive_ctxs": [{"text": id_to_text[pid], "is_positive": True}],
                    "source": "vnlawqc_eval"
                })
                if max_return and len(parsed) >= max_return:
                    break

            return parsed
        except Exception:
            return []

    # Nếu có thể parse cặp eval chuẩn, ưu tiên dùng cho 'vnlawqc'
    eval_records = parse_eval_pair(max_return=limit or 0)
    if eval_records and any(n.startswith("vnlaw") for n in want):
        results["vnlawqc"] = eval_records

    all_files: List[Path] = list(repo_dir.rglob("*.json")) + list(repo_dir.rglob("*.jsonl"))
    for name in want:
        key = name.lower()
        keywords = ["vnlaw" if key.startswith("vnlaw") else "vnsyn"]
        picked: List[Path] = []
        for p in all_files:
            fname = p.name.lower()
            parent = p.parent.name.lower()
            if any(kw in fname or kw in parent for kw in keywords):
                picked.append(p)
        # Nếu không bắt được theo tên, lấy một số file nhỏ có "query" trong nội dung
        if not picked:
            picked = [p for p in all_files if "query" in p.name.lower()][:5]

        # Đọc các file được chọn
        count = 0
        for p in picked:
            try:
                for obj in _read_json_or_jsonl(p):
                    results[key].append(obj)
                    count += 1
                    if limit and count >= limit:
                        break
                if limit and count >= limit:
                    break
            except Exception:
                continue
    return results


def normalize_records(raw_examples: List[Dict], source: str, limit: int = 0) -> List[Dict]:
    norm: List[Dict] = []
    for ex in raw_examples:
        q = extract_query(ex)
        if not q:
            continue
        positives = extract_positives(ex)
        if not positives:
            continue
        record = {
            "query": q,
            "positive_ctxs": [{"text": p, "is_positive": True} for p in positives[:1]],
            "source": source,
            "raw": ex,
        }
        norm.append(record)
        if limit and len(norm) >= limit:
            break
    return norm


def download_vietnamese_legal_qa(limit: int = 0) -> List[Dict]:
    """Tải queries từ HF dataset 'nqdhocai/vietnamese-legal-qa'.

    Trả về list bản ghi dạng {"query": str, "answers": Optional[List[str]], "source": "VietnameseLegalQA"}
    """
    try:
        from datasets import load_dataset  # lazy import
        # cố gắng các split phổ biến
        for sp in ["train", "validation", "test", None]:
            try:
                if sp is None:
                    ds = load_dataset("nqdhocai/vietnamese-legal-qa")
                    # lấy split đầu tiên
                    split_name = next(iter(ds.keys()))
                    ds = ds[split_name]
                else:
                    ds = load_dataset("nqdhocai/vietnamese-legal-qa", split=sp)
                records: List[Dict] = []
                for ex in ds:
                    q = ex.get("question") or ex.get("query") or ex.get("q") or ex.get("text")
                    if not q or not str(q).strip():
                        continue
                    answers = ex.get("answers") or ex.get("answer") or None
                    if isinstance(answers, str):
                        answers = [answers]
                    records.append({
                        "query": str(q).strip(),
                        "answers": answers if isinstance(answers, list) else None,
                        "source": "VietnameseLegalQA"
                    })
                    if limit and len(records) >= limit:
                        break
                if records:
                    return records
            except Exception:
                continue
    except Exception:
        pass

    # Fallback: snapshot repo và tìm json/jsonl
    try:
        repo_dir = Path(snapshot_download(repo_id="nqdhocai/vietnamese-legal-qa"))
        out: List[Dict] = []
        for p in list(repo_dir.rglob("*.json")) + list(repo_dir.rglob("*.jsonl")):
            for obj in _read_json_or_jsonl(p):
                if not isinstance(obj, dict):
                    continue
                q = obj.get("question") or obj.get("query") or obj.get("q") or obj.get("text")
                if not q or not str(q).strip():
                    continue
                answers = obj.get("answers") or obj.get("answer") or None
                if isinstance(answers, str):
                    answers = [answers]
                out.append({
                    "query": str(q).strip(),
                    "answers": answers if isinstance(answers, list) else None,
                    "source": "VietnameseLegalQA"
                })
                if limit and len(out) >= limit:
                    break
            if limit and len(out) >= limit:
                break
        return out
    except Exception:
        return []


def write_queries_jsonl(records: List[Dict], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải dataset phục vụ retriever")
    parser.add_argument("--datasets", nargs="*", default=["vnlawqc", "vnsynlawqc", "vlegalqa"], help="Các bộ cần tải (vnlawqc, vnsynlawqc, vlegalqa)")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Thư mục đầu ra")
    parser.add_argument("--splits", nargs="*", default=["train", "validation"], help="Các split muốn tải")
    parser.add_argument("--limit", type=int, default=0, help="Giới hạn số mẫu mỗi bộ (0 = tất cả)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ensure_dir(out_dir)

    want = [s.lower() for s in args.datasets]
    # Tải các bộ crosslingual nếu được yêu cầu
    cross_want = [w for w in want if w in ("vnlawqc", "vnsynlawqc")]
    raw_by_name = download_crosslingual_splits(want=cross_want, limit=args.limit, splits=args.splits) if cross_want else {}
    # Fallback: nếu rỗng, thử tải qua snapshot repo và tự parse JSON
    if all(len(v) == 0 for v in raw_by_name.values()):
        print("ℹ️  Fallback: tải toàn bộ repo và tự parse JSON...")
        raw_by_name = download_crosslingual_via_repo(want=want, limit=args.limit)

    # Ghi từng bộ
    for name, raw_examples in raw_by_name.items():
        if not raw_examples:
            print(f"⚠️  Không lấy được dữ liệu cho: {name}")
            continue
        norm = normalize_records(raw_examples, source=name, limit=args.limit)
        if not norm:
            print(f"⚠️  Không trích được bản ghi hợp lệ cho: {name}")
            continue
        out_path = out_dir / ("VNLAWQC.jsonl" if name.startswith("vnlaw") else "VNSynLawQC.jsonl")
        n = to_jsonl(norm, out_path)
        print(f"✅ Ghi {n} dòng → {out_path}")

    # Tải Vietnamese Legal QA (queries) nếu được yêu cầu
    if "vlegalqa" in want:
        legalqa = download_vietnamese_legal_qa(limit=args.limit)
        if legalqa:
            out_path = out_dir / "VietnameseLegalQA.jsonl"
            n = write_queries_jsonl(legalqa, out_path)
            print(f"✅ Ghi {n} dòng queries → {out_path}")
        else:
            print("⚠️  Không tải được Vietnamese Legal QA từ Hugging Face")

    # Không còn xử lý ViRHE4QA


if __name__ == "__main__":
    main()


