#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import sentence_transformers
import torch
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "zalo-legal" / "chunks_schema.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tạo FAISS index cho hệ thống retrieval LegalAdvisor")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Đường dẫn tới file chunks_schema.jsonl",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Tên model SentenceTransformer trên HuggingFace. Nếu cung cấp, sẽ tải và lưu vào --model-dir",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models/retrieval/zalo_v1"),
        help="Đường dẫn thư mục model SentenceTransformer đã fine-tune",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/retrieval/index"),
        help="Thư mục output cho index và metadata",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size khi encode")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Thiết bị encode: auto/cpu/cuda",
    )
    parser.add_argument("--verbose", action="store_true", help="Bật logging chi tiết")
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_chunks(path: Path) -> Tuple[List[str], List[dict]]:
    texts: List[str] = []
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                logging.warning("Bỏ qua dòng JSONL lỗi: %s", err)
                continue
            text = str(record.get("content") or "").strip()
            if not text:
                logging.debug("Chunk thiếu content: %s", record.get("chunk_id"))
            texts.append(text)
            records.append(record)
    if not records:
        raise ValueError(f"File chunks {path} không chứa dữ liệu hợp lệ")
    logging.info("Đọc %d chunks từ %s", len(records), path)
    return texts, records


def pick_device(device_arg: str) -> str:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        logging.info("CUDA không khả dụng, chuyển sang CPU")
        return "cpu"

    if device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Đã yêu cầu device 'cuda' nhưng PyTorch hiện không hỗ trợ CUDA. "
                "Hãy cài đặt PyTorch bản CUDA hoặc chạy lại với '--device cpu'."
            )
        return device_arg

    if device_arg != "cpu":
        logging.warning("Thiết bị '%s' không được nhận diện, sử dụng CPU", device_arg)
    return "cpu"


def load_or_prepare_model(base_model: str | None, model_dir: Path, device: str) -> SentenceTransformer:
    """Load model theo hai chế độ:
    - Nếu base_model được cung cấp: tải từ HF, sau đó lưu vào model_dir để dùng lâu dài.
    - Nếu không: load từ model_dir (đã fine-tune).
    """
    if base_model:
        logging.info("Tải model từ HuggingFace: %s", base_model)
        model = SentenceTransformer(base_model, device=device)
        # Lưu lại local để đảm bảo runtime/index đồng bộ
        model_dir.mkdir(parents=True, exist_ok=True)
        logging.info("Lưu model vào %s", model_dir)
        model.save(str(model_dir))
        return model
    # Fallback: load từ model_dir
    if not model_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_dir}. Hãy cung cấp --base-model hoặc tạo model_dir trước.")
    logging.info("Load model SentenceTransformer từ %s", model_dir)
    return SentenceTransformer(str(model_dir), device=device)


def _get_tokenizer(model: SentenceTransformer):
    """Lấy tokenizer bên trong SentenceTransformer nếu có.

    Ưu tiên module Transformer đầu tiên. Trả về None nếu không tìm thấy.
    """
    try:
        first = model[0]
    except Exception:
        first = None
    if first is not None:
        tok = getattr(first, "tokenizer", None)
        if tok is not None:
            return tok
    # Fallback: một số phiên bản có thuộc tính trực tiếp
    return getattr(model, "tokenizer", None)


def _get_segment_config(model: SentenceTransformer) -> Tuple[int, int, bool]:
    """Xác định chính sách segment (độ dài L, overlap) nhất quán với encoder.

    - L mặc định lấy từ LEGALADVISOR_ENCODER_MAX_SEQ_LENGTH (mặc định 256),
      nhưng không vượt quá max_position_embeddings của backbone nếu có.
    - overlap mặc định 64 token (có thể chỉnh qua LEGALADVISOR_SEGMENT_OVERLAP_TOKENS).
    """
    # Độ dài mong muốn từ ENV (giữ sync với pipeline encode_query / train)
    try:
        max_len_env = int(os.getenv("LEGALADVISOR_ENCODER_MAX_SEQ_LENGTH", "256"))
    except Exception:
        max_len_env = 256

    # Giới hạn theo cấu hình backbone nếu có
    model_max: int | None = None
    try:
        if hasattr(model, "max_seq_length"):
            val = int(getattr(model, "max_seq_length") or 0)
            if val > 0:
                model_max = val
    except Exception:
        model_max = None

    try:
        first = model[0]
    except Exception:
        first = None
    if first is not None:
        auto_model = getattr(first, "auto_model", None)
        cfg = getattr(auto_model, "config", None)
        if cfg is not None and hasattr(cfg, "max_position_embeddings"):
            try:
                pos_max = int(getattr(cfg, "max_position_embeddings") or 0)
                if pos_max > 0:
                    if model_max is None or pos_max < model_max:
                        model_max = pos_max
            except Exception:
                pass

    if model_max is None or model_max <= 0:
        model_max = max_len_env

    # Giảm nhẹ để chừa chỗ cho token đặc biệt nếu cần
    L = int(min(max_len_env, model_max))
    L = max(32, L)  # ít nhất 32 token để tránh đoạn quá ngắn

    # Overlap
    try:
        overlap_env = int(os.getenv("LEGALADVISOR_SEGMENT_OVERLAP_TOKENS", "64"))
    except Exception:
        overlap_env = 64
    overlap = max(0, min(overlap_env, L // 2))

    return L, overlap, True


def segment_text_for_representation(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap: int,
) -> List[Dict[str, int | str]]:
    """Cắt một Điều thành nhiều đoạn biểu diễn theo token, chỉ phục vụ encoder/FAISS.

    - Không thay đổi đơn vị nghiệp vụ (mỗi Điều vẫn giữ nguyên chunk_id/corpus_id).
    - Nếu số token ≤ max_tokens -> 1 đoạn duy nhất (toàn văn).
    - Nếu > max_tokens -> sliding window với overlap token, decode về text cho từng đoạn.
    """
    text = (text or "").strip()
    if not text:
        return []

    # Nếu không có tokenizer (edge-case), fallback theo độ dài ký tự ~4 chars/token
    if tokenizer is None:
        approx_tokens = max_tokens * 4
        if len(text) <= approx_tokens:
            return [{"text": text, "token_start": 0, "token_end": 0}]
        segments: List[Dict[str, int | str]] = []
        stride = max(approx_tokens - overlap * 4, 1)
        start = 0
        while start < len(text):
            end = min(len(text), start + approx_tokens)
            seg = text[start:end].strip()
            if seg:
                segments.append({"text": seg, "token_start": 0, "token_end": 0})
            if end >= len(text):
                break
            start += stride
        return segments

    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception:
        # Nếu encode lỗi, fallback 1 đoạn toàn văn
        return [{"text": text, "token_start": 0, "token_end": 0}]

    total_tokens = len(token_ids)
    if total_tokens <= max_tokens:
        return [{"text": text, "token_start": 0, "token_end": total_tokens}]

    segments: List[Dict[str, int | str]] = []
    stride = max(1, max_tokens - overlap)
    start = 0

    while start < total_tokens:
        end = min(total_tokens, start + max_tokens)
        window_ids = token_ids[start:end]
        if not window_ids:
            break
        # Decode sang text của đoạn
        try:
            seg_text = tokenizer.decode(window_ids, skip_special_tokens=True).strip()
        except Exception:
            seg_text = text  # fallback an toàn
        if seg_text:
            segments.append(
                {
                    "text": seg_text,
                    "token_start": int(start),
                    "token_end": int(end),
                }
            )
        if end >= total_tokens:
            break
        start += stride

    if not segments:
        return [{"text": text, "token_start": 0, "token_end": total_tokens}]
    return segments


def build_segments_for_index(
    records: Sequence[dict],
    model: SentenceTransformer,
) -> Tuple[List[str], List[dict], np.ndarray, Dict[str, int | bool]]:
    """Từ danh sách Điều (records) sinh ra các đoạn biểu diễn để index.

    Trả về:
    - segment_texts: List[str] các đoạn để encode.
    - id_rows: List[dict] cho id_map.jsonl (mỗi đoạn một hàng, có part, span_token_*).
    - faiss_ids: np.ndarray[int64] id cho FAISS.
    - segment_meta: thông tin L/overlap/segmented.
    """
    tokenizer = _get_tokenizer(model)
    seg_len, overlap, segmented = _get_segment_config(model)

    segment_texts: List[str] = []
    id_rows: List[dict] = []
    faiss_ids: List[int] = []
    next_faiss_id: int = 0

    for rec in records:
        content = str(rec.get("content") or "").strip()
        if not content:
            continue
        segments = segment_text_for_representation(content, tokenizer, seg_len, overlap)
        if not segments:
            segments = [{"text": content, "token_start": 0, "token_end": 0}]

        for part_idx, seg in enumerate(segments, start=1):
            text = str(seg.get("text") or "").strip()
            if not text:
                continue
            faiss_id = next_faiss_id
            next_faiss_id += 1

            segment_texts.append(text)
            faiss_ids.append(faiss_id)

            id_rows.append(
                {
                    "faiss_id": int(faiss_id),
                    "chunk_id": rec.get("chunk_id"),
                    "corpus_id": rec.get("corpus_id"),
                    "type": rec.get("type"),
                    "number": rec.get("number"),
                    "year": rec.get("year"),
                    "suffix": rec.get("suffix"),  # Điều
                    "part": int(part_idx),
                    "span_token_start": int(seg.get("token_start") or 0),
                    "span_token_end": int(seg.get("token_end") or 0),
                }
            )

    if not segment_texts:
        raise ValueError("Không sinh được đoạn biểu diễn nào từ records đầu vào")

    faiss_ids_arr = np.asarray(faiss_ids, dtype=np.int64)
    segment_meta: Dict[str, int | bool] = {
        "segment_length": int(seg_len),
        "segment_overlap": int(overlap),
        "segmented": bool(segmented),
    }
    return segment_texts, id_rows, faiss_ids_arr, segment_meta


def encode_chunks(
    texts: Sequence[str],
    model: SentenceTransformer,
    batch_size: int,
    device: str,
    max_seq_length: int | None = None,
) -> np.ndarray:
    # Đồng bộ max_seq_length theo ENV hoặc tham số explicit để nhất quán với search/train
    try:
        if max_seq_length is None:
            try:
                max_len_env = int(os.getenv("LEGALADVISOR_ENCODER_MAX_SEQ_LENGTH", "256"))
            except Exception:
                max_len_env = 256
            max_seq_length = max_len_env
        max_seq_length = max(8, int(max_seq_length))
        if hasattr(model, "max_seq_length"):
            model.max_seq_length = max_seq_length
        try:
            first_mod = model[0]
            if hasattr(first_mod, "max_seq_length"):
                setattr(first_mod, "max_seq_length", max_seq_length)
        except Exception:
            pass
    except Exception:
        pass
    logging.info("Encode %d chunks (batch_size=%d, device=%s)", len(texts), batch_size, device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


def normalize_embeddings(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def build_index(vectors: np.ndarray, ids: np.ndarray) -> Tuple[faiss.IndexIDMap, np.ndarray]:
    logging.info("Chuẩn hóa embeddings và xây FAISS index")
    normalized = normalize_embeddings(vectors.astype(np.float32))
    base_index = faiss.IndexFlatIP(normalized.shape[1])
    index = faiss.IndexIDMap(base_index)
    index.add_with_ids(normalized, ids)
    logging.info("Index đã thêm %d vectors", index.ntotal)
    return index, normalized


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_artifacts(
    raw_embeddings: np.ndarray,
    normalized_embeddings: np.ndarray,
    index: faiss.Index,
    id_rows: Sequence[dict],
    model_dir: Path,
    output_dir: Path,
    model_name: str,
    segment_meta: Dict[str, int | bool] | None = None,
    num_source_chunks: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / "vectors.npy"
    logging.info("Lưu vectors gốc tại %s", raw_path)
    np.save(raw_path, raw_embeddings.astype(np.float32))

    normalized_path = output_dir / "chunk_embeddings.npy"
    logging.info("Lưu vectors chuẩn hóa tại %s", normalized_path)
    np.save(normalized_path, normalized_embeddings.astype(np.float32))

    index_path = output_dir / "chunks_index.faiss"
    logging.info("Lưu FAISS index tại %s", index_path)
    faiss.write_index(index, str(index_path))

    id_map_path = output_dir / "id_map.jsonl"
    logging.info("Lưu mapping chunk_id <-> faiss_id tại %s", id_map_path)
    write_jsonl(id_map_path, id_rows)

    metadata_path = output_dir / "metadata.json"
    # total_chunks: số Điều nguồn (chunk_id duy nhất); total_segments: số vector (index.ntotal)
    if num_source_chunks is None:
        try:
            unique_chunks = {row.get("chunk_id") for row in id_rows if row.get("chunk_id") is not None}
            num_source_chunks = len(unique_chunks)
        except Exception:
            num_source_chunks = index.ntotal
    metadata = {
        "total_chunks": int(num_source_chunks),
        "total_segments": int(index.ntotal),
        "chunk_fields": sorted({key for record in id_rows for key in record.keys()}),
    }
    logging.info("Lưu metadata tại %s", metadata_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    model_info_path = output_dir / "model_info.json"
    model_info = {
        "model_path": str(model_dir.resolve()),
        "model_name": model_name or "unknown",
        "embedding_dim": int(index.d),
        "num_chunks": int(num_source_chunks),
        "num_segments": int(index.ntotal),
        "uses_id_map": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sentence_transformers_version": sentence_transformers.__version__,
    }
    if segment_meta:
        try:
            model_info["segment_length"] = int(segment_meta.get("segment_length", 0))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            model_info["segment_overlap"] = int(segment_meta.get("segment_overlap", 0))  # type: ignore[arg-type]
        except Exception:
            pass
        try:
            model_info["segmented"] = bool(segment_meta.get("segmented", True))  # type: ignore[arg-type]
        except Exception:
            model_info["segmented"] = True
    logging.info("Lưu model_info tại %s", model_info_path)
    model_info_path.write_text(json.dumps(model_info, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    chunks_path = resolve_path(args.chunks)
    model_dir = resolve_path(args.model_dir)
    output_dir = resolve_path(args.output_dir)

    if not chunks_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file chunks: {chunks_path}")
    texts, records = read_chunks(chunks_path)
    device = pick_device(args.device)
    model = load_or_prepare_model(args.base_model, model_dir, device)
    # Xây dựng các đoạn biểu diễn (multi-vector per Điều) cho index
    segment_texts, id_rows, faiss_ids, segment_meta = build_segments_for_index(records, model)
    raw_embeddings = encode_chunks(segment_texts, model, max(1, args.batch_size), device, max_seq_length=int(segment_meta.get("segment_length", 256)))

    index, normalized_embeddings = build_index(raw_embeddings, faiss_ids)

    # Lấy tên model đã dùng (nếu có thuộc tính) để ghi vào model_info
    try:
        # sentence_transformers >=2.x thường có thuộc tính model_name_or_path ở module đầu
        first = model[0]
        used_name = getattr(first, 'model_name_or_path', None) or args.base_model or str(model_dir)
    except Exception:
        used_name = args.base_model or str(model_dir)

    # Số Điều nguồn (mỗi record là một Điều/chunk nghiệp vụ)
    num_source_chunks = len(records)
    save_artifacts(
        raw_embeddings,
        normalized_embeddings,
        index,
        id_rows,
        model_dir,
        output_dir,
        used_name,
        segment_meta=segment_meta,
        num_source_chunks=num_source_chunks,
    )
    logging.info("Hoàn tất build index. Output: %s", output_dir)


if __name__ == "__main__":
    main()
