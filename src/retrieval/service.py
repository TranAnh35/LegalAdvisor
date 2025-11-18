#!/usr/bin/env python3
"""
Dịch vụ truy hồi tài liệu thống nhất cho LegalAdvisor (Zalo-Legal schema).

- Load FAISS index + metadata từ models/retrieval
- Mã hóa truy vấn bằng SentenceTransformer theo model trong model_info
- Trả về kết quả dạng corpus_id/type/number/year và preview (không còn đọc SQLite/Parquet)
"""

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import re
import faiss  # type: ignore
import numpy as np  # type: ignore
import torch  # type: ignore
import unicodedata  # type: ignore

from sentence_transformers import SentenceTransformer, models  # type: ignore

from .build_index import _get_segment_config, _get_tokenizer, segment_text_for_representation  # type: ignore
from ..utils.logger import get_logger
from ..utils.paths import get_models_retrieval_dir, get_processed_data_dir


class RetrievalService:
    """Dịch vụ truy hồi tài liệu dựa trên FAISS + metadata (Zalo-Legal)."""

    def __init__(self, use_gpu: bool = False) -> None:
        # Thư mục models/retrieval
        self.model_dir: Path = get_models_retrieval_dir()
        self.index_dir: Path = self.model_dir / "index_v2"
        self.use_gpu: bool = bool(use_gpu)
        self._logger = get_logger("legaladvisor.retrieval")
        # Thiết bị mục tiêu cho encoder (GPU nếu được yêu cầu và khả dụng)
        try:
            self._device: str = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
        except Exception:
            self._device = "cpu"

        # Load model_info: ưu tiên từ index/model_info.json (mới nhất từ build_index)
        # Nếu không có, fallback tới models/retrieval/model_info.json (cũ)
        self.model_info: Dict[str, Any] = {}
        
        # Try to load from index first (created by build_index)
        index_model_info_path = self.index_dir / "model_info.json"
        if index_model_info_path.exists():
            with open(index_model_info_path, 'r', encoding='utf-8') as f:
                self.model_info = json.load(f)
        
        # Fallback to old location if needed
        if not self.model_info:
            old_model_info_path = self.model_dir / "model_info.json"
            if old_model_info_path.exists():
                with open(old_model_info_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Map old field names to new ones
                    self.model_info = {
                        "model_path": str(self.model_dir / "zalo_v1"),
                        "model_name": data.get("base_model", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
                        **{k: v for k, v in data.items() if k not in ["base_model"]}
                    }

        # Thông tin index (nếu có) - dùng làm mirror
        self.index_info: Dict[str, Any] = self.model_info.copy()
        index_info_path = self.index_dir / "model_info.json"
        if index_info_path.exists() and not self.index_info:
            with open(index_info_path, 'r', encoding='utf-8') as f:
                self.index_info = json.load(f)

        # Load metadata và bảng tra theo chunk_id
        metadata_path = self.model_dir / "metadata.json"
        self._meta_by_id: Dict[int, Dict[str, Any]] = {}
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata: List[Dict[str, Any]] = json.load(f)
            for entry in metadata:
                chunk_id = entry.get('chunk_id')
                if chunk_id is None:
                    continue
                try:
                    self._meta_by_id[int(chunk_id)] = entry
                except (TypeError, ValueError):
                    continue

        # Load FAISS index
        index_path = self.index_dir / "chunks_index.faiss"
        if not index_path.exists():
            index_path = self.model_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))

        # Map FAISS id -> chunk_id (khi dùng IndexIDMap)
        self._id_map: Dict[int, int] = {}
        # Thuộc tính ở cấp chunk (một entry cho mỗi chunk_id)
        self._id_attrs: Dict[int, Dict[str, Any]] = {}
        # Thuộc tính ở cấp vector/segment (một entry cho mỗi faiss_id)
        self._faiss_attrs: Dict[int, Dict[str, Any]] = {}
        id_map_path = self.index_dir / "id_map.jsonl"
        if id_map_path.exists():
            with open(id_map_path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    faiss_id = int(row.get("faiss_id", -1))
                    chunk_id = row.get("chunk_id")
                    if faiss_id >= 0 and chunk_id is not None:
                        chunk_id_int = int(chunk_id)
                        self._id_map[faiss_id] = chunk_id_int
                        # Lưu attrs cấp chunk (lấy theo hàng đầu tiên cho mỗi chunk_id)
                        base_attrs = {
                            k: v
                            for k, v in row.items()
                            if k not in {"faiss_id", "chunk_id"}
                        }
                        self._id_attrs.setdefault(chunk_id_int, base_attrs)
                        # Lưu attrs cấp segment cho mỗi faiss_id
                        self._faiss_attrs[faiss_id] = base_attrs | {
                            "chunk_id": chunk_id_int,
                            "faiss_id": faiss_id,
                        }

        # Khởi tạo encoder
        # NOTE: luôn load trên CPU trước để tránh lỗi meta tensor (PyTorch >= 2.5).
        encoder_source: Optional[Path] = None

        env_model_dir = os.getenv("LEGALADVISOR_EMBEDDING_MODEL_DIR")
        if env_model_dir:
            candidate = Path(env_model_dir).expanduser().resolve()
            if candidate.exists():
                encoder_source = candidate

        if encoder_source is None and self.index_info.get("model_path"):
            candidate = Path(self.index_info["model_path"]).expanduser().resolve()
            if candidate.exists():
                encoder_source = candidate

        # Try to load a local encoder nếu có; fallback sang model gốc trên HF khi cần.
        model_name = self.model_info.get("model_name") or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        encoder_errors: List[str] = []
        self.encoder: Optional[SentenceTransformer] = None

        # Luôn load trên CPU trước để tránh lỗi meta tensor khi chuyển thẳng lên GPU
        if encoder_source is not None:
            try:
                self.encoder = self._load_sentence_transformer_safe(str(encoder_source))
            except Exception as exc:
                encoder_errors.append(f"{encoder_source}: {exc}")
                self._logger.warning(
                    "Failed to load local encoder %s; falling back to %s (%s)",
                    encoder_source,
                    model_name,
                    exc,
                )

        if self.encoder is None:
            try:
                self.encoder = self._load_sentence_transformer_safe(model_name)
            except Exception as exc:
                details = "; ".join(encoder_errors)
                if details:
                    details = f" Previous attempts: {details}"
                raise RuntimeError(f"Failed to load fallback encoder {model_name}: {exc}.{details}") from exc

        # Ensure encoder luôn tồn tại tới đây
        if self.encoder is None:
            raise RuntimeError(f"Failed to initialize encoder for {model_name}")
        assert self.encoder is not None

        # Lưu tên base model để fallback khi gặp lỗi encode
        self._base_model_name: str = str(model_name)

        # Giới hạn độ dài mã hoá để tránh lỗi position_embeddings out-of-range
        try:
            max_len_env = int(os.getenv("LEGALADVISOR_ENCODER_MAX_SEQ_LENGTH", "256"))
        except Exception:
            max_len_env = 256
        try:
            if hasattr(self.encoder, "max_seq_length"):
                # đặt thấp hơn một chút để an toàn
                self.encoder.max_seq_length = max(8, int(max_len_env))
            # Áp trực tiếp vào module Transformer đầu tiên (cách Sentence-Transformers khuyến nghị)
            try:
                first_mod = None
                try:
                    first_mod = self.encoder[0]
                except Exception:
                    first_mod = None
                if first_mod is not None and hasattr(first_mod, "max_seq_length"):
                    setattr(first_mod, "max_seq_length", max(8, int(max_len_env)))
            except Exception:
                pass
        except Exception:
            pass

        # Đảm bảo encoder đang ở đúng thiết bị mục tiêu
        # Chuyển encoder sang GPU sau khi đã materialize toàn bộ tham số nếu yêu cầu
        if self._device == "cuda":
            try:
                # Dùng to_empty nếu khả dụng để tránh sao chép không cần thiết (PyTorch >= 2.5)
                if hasattr(self.encoder, "to"):
                    self.encoder.to(torch.device("cuda"))
            except Exception as exc:
                self._logger.warning("Không thể chuyển encoder sang GPU (%s), fallback dùng CPU", exc)
                self._device = "cpu"

        # Đường dẫn JSONL content store (schema mới)
        self.processed_dir: Path = get_processed_data_dir()
        self.jsonl_path: Path = self.processed_dir / "zalo-legal" / "chunks_schema.jsonl"
        self._content_cache: Dict[int, str] = {}
        self._chunk_cache: Dict[int, Dict[str, Any]] = {}
        # Debug retrieval flag
        self.debug: bool = os.getenv("LEGALADVISOR_DEBUG_RETRIEVAL", "0") == "1"

        # Cache fallback nội dung Điều: (act_code_norm, article)-> content
        self._article_fallback_cache: Dict[tuple, str] = {}

        # Chỉ mục đảo đơn giản: (act_code_norm, suffix)-> [chunk_id]
        self._rev_index_ready: bool = False
        self._rev_index: Dict[tuple, List[int]] = {}
        
        # ===== OPTIMIZATION: Indexed JSONL cache =====
        # Load toàn bộ JSONL vào memory on-startup để tránh quét từ đầu cho mỗi chunk_id
        # Memory trade-off: ~60KB JSONL file = ~6MB RAM (acceptable)
        # Performance gain: O(n) scan → O(1) lookup per chunk_id
        self._all_records_cached: bool = False
        self._load_indexed_records_on_init()

    @staticmethod
    def _normalize_lookup_code(value: str) -> str:
        """Chuẩn hoá mã văn bản phục vụ tra cứu (bỏ dấu + casefold)."""
        text = (value or "").strip()
        if not text:
            return ""
        text = text.replace("Đ", "D").replace("đ", "d")
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return text.casefold()

    def encode_query(self, text: str) -> np.ndarray:
        """Mã hóa truy vấn thành numpy float32 contiguous và normalized."""
        assert self.encoder is not None
        # Cắt truy vấn theo token cấp tokenizer để tránh vượt quá max_position_embeddings
        try:
            tok = None
            max_len = None
            try:
                first_mod = self.encoder[0]
            except Exception:
                first_mod = None
            if first_mod is not None:
                tok = getattr(first_mod, "tokenizer", None)
                max_len = getattr(first_mod, "max_seq_length", None)
            if max_len is None:
                try:
                    max_len = int(os.getenv("LEGALADVISOR_ENCODER_MAX_SEQ_LENGTH", "512"))
                except Exception:
                    max_len = 256
            if tok is not None and isinstance(max_len, int) and max_len > 0:
                # tokenize & truncate rồi decode lại để bảo đảm chiều dài hợp lệ
                try:
                    ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=int(max_len))
                    text = tok.decode(ids, skip_special_tokens=True)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            query_embedding = self.encoder.encode([text], convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:
            msg = str(exc).lower()
            # Thử giảm max_seq_length và encode lại nếu lỗi do position/vocab
            if "index out of range" in msg or "position" in msg:
                try:
                    if hasattr(self.encoder, "max_seq_length"):
                        cur = int(getattr(self.encoder, "max_seq_length", 256) or 256)
                        new_len = max(16, min(cur, 128))
                        setattr(self.encoder, "max_seq_length", new_len)
                    # Áp luôn xuống module đầu tiên
                    try:
                        first_mod = self.encoder[0]
                        if hasattr(first_mod, "max_seq_length"):
                            setattr(first_mod, "max_seq_length", new_len)
                    except Exception:
                        pass
                except Exception:
                    pass
                try:
                    # Thử truncate lại bằng tokenizer nếu có
                    try:
                        first_mod = None
                        try:
                            first_mod = self.encoder[0]
                        except Exception:
                            first_mod = None
                        tok = getattr(first_mod, "tokenizer", None) if first_mod is not None else None
                        if tok is not None:
                            ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=int(new_len))
                            text = tok.decode(ids, skip_special_tokens=True)
                    except Exception:
                        pass
                    query_embedding = self.encoder.encode([text], convert_to_numpy=True, show_progress_bar=False)
                except Exception as exc2:
                    # Fallback: reload encoder từ tên base model mặc định
                    try:
                        fallback_name = self._base_model_name or "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                        self._logger.warning("Reloading fallback encoder due to encode error: %s", exc2)
                        self.encoder = self._load_sentence_transformer_safe(fallback_name)
                        if hasattr(self.encoder, "max_seq_length"):
                            setattr(self.encoder, "max_seq_length", 128)
                        try:
                            first_mod = self.encoder[0]
                            if hasattr(first_mod, "max_seq_length"):
                                setattr(first_mod, "max_seq_length", 128)
                        except Exception:
                            pass
                        try:
                            if hasattr(self.encoder, "to"):
                                self.encoder.to(self._device)
                        except Exception:
                            pass
                        # truncate thêm một lần bằng tokenizer nếu có
                        try:
                            first_mod = None
                            try:
                                first_mod = self.encoder[0]
                            except Exception:
                                first_mod = None
                            tok = getattr(first_mod, "tokenizer", None) if first_mod is not None else None
                            if tok is not None:
                                ids = tok.encode(text, add_special_tokens=True, truncation=True, max_length=128)
                                text = tok.decode(ids, skip_special_tokens=True)
                        except Exception:
                            pass
                        query_embedding = self.encoder.encode([text], convert_to_numpy=True, show_progress_bar=False)
                    except Exception:
                        # Cuối cùng, ném lại lỗi để caller bắt và ghi log
                        raise
            else:
                raise

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
        faiss.normalize_L2(query_embedding)
        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        return query_embedding

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Truy hồi tài liệu liên quan cho truy vấn (semantic) ở cấp Điều.

        - FAISS search trả về các vector/segment (multi-vector per Điều).
        - Hàm này nhóm theo chunk_id (tương ứng 1 Điều) và tính điểm doc-level
          bằng chiến lược gộp (top-m mean hoặc max).
        - Kết quả cuối cùng vẫn là danh sách Điều (mỗi phần tử 1 chunk_id).
        """
        try:
            query_embedding = self.encode_query(query)
            if self.debug:
                try:
                    self._logger.info(
                        "[retrieval] query='%s' emb_dim=%d emb_norm=%.4f",
                        query[:120], query_embedding.shape[1], float((query_embedding**2).sum())
                    )
                except Exception:
                    pass
            # Cấu hình gộp Điều từ ENV
            strategy = os.getenv("LEGALADVISOR_ARTICLE_GROUP_STRATEGY", "topm_mean").strip().lower()
            try:
                top_m = int(os.getenv("LEGALADVISOR_ARTICLE_GROUP_TOPM", "2"))
            except Exception:
                top_m = 2
            try:
                oversample = int(os.getenv("LEGALADVISOR_ARTICLE_GROUP_OVERSAMPLE", "5"))
            except Exception:
                oversample = 5
            top_m = max(1, top_m)
            oversample = max(1, oversample)

            # top_k là số Điều cần trả về; oversample số vector để FAISS search
            try:
                requested = int(top_k)
            except (TypeError, ValueError):
                requested = 3
            doc_k = max(1, requested)
            if int(self.index.ntotal) <= 0:
                return []
            raw_k = min(int(self.index.ntotal), max(doc_k, doc_k * oversample))
            if raw_k <= 0:
                return []

            distances, indices = self.index.search(query_embedding, raw_k)
            if self.debug:
                try:
                    self._logger.info(
                        "[retrieval] raw_distances=%s raw_indices=%s",
                        [float(x) for x in distances[0][: min(5, len(distances[0]))]],
                        [int(x) for x in indices[0][: min(5, len(indices[0]))]],
                    )
                except Exception:
                    pass

            # Nhóm các vector theo Điều (chunk_id)
            doc_hits: Dict[int, Dict[str, Any]] = {}
            for distance, idx in zip(distances[0], indices[0]):
                idx_int = int(idx)
                if idx_int < 0:
                    continue
                faiss_id = idx_int
                chunk_id = self._id_map.get(faiss_id, faiss_id)
                sc = float(distance)
                entry = doc_hits.get(chunk_id)
                if entry is None:
                    entry = {
                        "chunk_id": chunk_id,
                        "scores": [],
                        "segments": [],
                        "best_score": None,
                    }
                    doc_hits[chunk_id] = entry
                entry["scores"].append(sc)
                # Lưu thông tin segment (faiss_id, part, span_token_*)
                attrs = self._faiss_attrs.get(faiss_id, {})
                part = attrs.get("part")
                try:
                    part_int = int(part) if part is not None else None
                except Exception:
                    part_int = None
                seg_info: Dict[str, Any] = {
                    "faiss_id": faiss_id,
                    "score": sc,
                }
                if part_int is not None:
                    seg_info["part"] = part_int
                if "span_token_start" in attrs:
                    seg_info["span_token_start"] = attrs.get("span_token_start")
                if "span_token_end" in attrs:
                    seg_info["span_token_end"] = attrs.get("span_token_end")
                entry["segments"].append(seg_info)
                if entry["best_score"] is None or sc > float(entry["best_score"]):
                    entry["best_score"] = sc

            if not doc_hits:
                try:
                    self._logger.warning(
                        "[retrieval] EMPTY_RESULTS query='%s' k=%d ntotal=%d",
                        query[:120],
                        doc_k,
                        int(self.index.ntotal),
                    )
                except Exception:
                    pass
                return []

            # Tính điểm doc-level cho từng Điều
            articles: List[Dict[str, Any]] = []
            for chunk_id, info in doc_hits.items():
                scores = info.get("scores") or []
                if not scores:
                    continue
                scores_sorted = sorted(scores, reverse=True)
                if strategy == "max":
                    doc_score = float(scores_sorted[0])
                elif strategy == "mean":
                    doc_score = float(sum(scores_sorted) / max(1, len(scores_sorted)))
                else:  # topm_mean (mặc định)
                    m = max(1, min(top_m, len(scores_sorted)))
                    topm = scores_sorted[:m]
                    doc_score = float(sum(topm) / max(1, len(topm)))

                meta = self._get_chunk_metadata(chunk_id)
                chunk_record = self._load_chunk_record(chunk_id)
                content_preview = meta.get("preview") or meta.get("content") or ""
                if not content_preview and chunk_record:
                    content_preview = chunk_record.get("preview") or chunk_record.get("content") or ""
                content_full = chunk_record.get("content") if chunk_record else None

                # Sắp xếp danh sách segment theo score giảm dần (nếu cần dùng cho RAG)
                segs = info.get("segments") or []
                try:
                    segs = sorted(segs, key=lambda s: float(s.get("score", 0.0)), reverse=True)
                except Exception:
                    pass

                articles.append(
                    {
                        "id": chunk_id,
                        "chunk_id": chunk_id,
                        "corpus_id": meta.get("corpus_id"),
                        "type": meta.get("type"),
                        "number": meta.get("number"),
                        "year": meta.get("year"),
                        "suffix": meta.get("suffix"),
                        "content": content_preview,
                        "preview": content_preview,
                        "content_full": content_full or content_preview,
                        # Điểm doc-level sau khi gộp các segment của Điều
                        "score": doc_score,
                        # Một số thông tin debug thêm
                        "score_max": float(scores_sorted[0]),
                        "score_mean": float(sum(scores_sorted) / max(1, len(scores_sorted))),
                        "segments": segs,
                    }
                )

            # Sắp xếp theo doc_score giảm dần và cắt top_k Điều
            articles.sort(key=lambda r: r.get("score", 0.0), reverse=True)
            sliced = articles[:doc_k]
            return sliced
        except Exception as e:
            try:
                import traceback
                self._logger.error(
                    "[retrieval] EXCEPTION query='%s' error='%s'\n%s",
                    query[:120], e, traceback.format_exc()
                )
            except Exception:
                pass
            return []

    def get_chunk_content(self, chunk_id: int) -> Optional[str]:
        """Trả về nội dung chunk từ JSONL schema mới (cache theo ID)."""
        try:
            if chunk_id in self._content_cache:
                return self._content_cache.get(chunk_id)
            record = self._load_chunk_record(chunk_id)
            if not record:
                return None
            content = record.get('content') or ''
            self._content_cache[chunk_id] = content
            return content
        except Exception:
            return None

    def get_article_segments_text(self, chunk_id: int, max_segments: Optional[int] = None) -> List[Dict[str, Any]]:
        """Sinh lại các đoạn biểu diễn (segments) của một Điều từ content gốc.

        Dùng cùng chính sách segment (L, overlap) như khi build index để:
        - Ưu tiên ghép các đoạn trúng embedding vào context RAG.
        - Không thay đổi đơn vị nghiệp vụ (vẫn 1 Điều = 1 chunk_id).
        """
        try:
            text = self.get_chunk_content(int(chunk_id)) or ""
            text = text.strip()
            if not text:
                return []
            if self.encoder is None:
                return [{"part": 1, "text": text}]

            tok = _get_tokenizer(self.encoder)
            seg_len, overlap, _ = _get_segment_config(self.encoder)
            segments = segment_text_for_representation(text, tok, seg_len, overlap)
            if not segments:
                return [{"part": 1, "text": text}]

            results: List[Dict[str, Any]] = []
            for idx, seg in enumerate(segments, start=1):
                seg_text = str(seg.get("text") or "").strip()
                if not seg_text:
                    continue
                item: Dict[str, Any] = {
                    "part": int(idx),
                    "text": seg_text,
                }
                if "token_start" in seg:
                    item["span_token_start"] = seg.get("token_start")
                if "token_end" in seg:
                    item["span_token_end"] = seg.get("token_end")
                results.append(item)

            if max_segments is not None and max_segments > 0:
                results = results[: int(max_segments)]
            return results
        except Exception:
            return []

    def _ensure_reverse_index(self) -> None:
        """Xây dựng chỉ mục đảo (act_code_norm, article) -> [chunk_id] từ id_map attrs.

        Sử dụng nhanh gọn thuộc tính đã có trong id_map (corpus_id, suffix) để tránh quét toàn bộ JSONL.
        """
        if self._rev_index_ready:
            return
        try:
            from ..utils.law_registry import normalize_act_code  # lazy import để tránh phụ thuộc vòng
        except Exception:
            def normalize_act_code(x: str) -> str:
                return (x or '').strip().lower()

        rev: Dict[tuple, List[int]] = {}
        for chunk_id, attrs in self._id_attrs.items():
            corpus_id = str(attrs.get('corpus_id') or '').strip()
            if not corpus_id:
                continue
            raw_code = corpus_id.split('+')[0].strip()
            code_norm = normalize_act_code(raw_code) if raw_code else ''
            if not code_norm:
                continue
            lookup_code = self._normalize_lookup_code(code_norm)
            suffix = attrs.get('suffix')
            art: Optional[int] = None
            # Ưu tiên: từ attrs.suffix nếu có
            if suffix is not None and str(suffix).isdigit():
                try:
                    art = int(suffix)
                except Exception:
                    art = None
            # Fallback 1: từ phần sau dấu '+' trong corpus_id
            if art is None:
                try:
                    parts = corpus_id.split('+', 1)
                    if len(parts) == 2 and parts[1].isdigit():
                        art = int(parts[1])
                except Exception:
                    art = None
            # Fallback 2: đọc metadata/record để lấy suffix
            if art is None:
                try:
                    meta = self._get_chunk_metadata(int(chunk_id))
                    suf = meta.get('suffix')
                    if suf is not None and str(suf).isdigit():
                        art = int(suf)
                except Exception:
                    art = None
            if art is None:
                continue
            key = (lookup_code, int(art))
            rev.setdefault(key, []).append(int(chunk_id))
        self._rev_index = rev
        self._rev_index_ready = True

    def find_chunks_by_code_and_article(self, act_code_norm: str, article: int) -> List[int]:
        """Tìm các chunk_id ứng với mã văn bản (chuẩn hoá) và Điều.

        Trả về danh sách chunk_id (có thể nhiều nếu Điều dài, bị chia nhỏ)."""
        try:
            self._ensure_reverse_index()
            lookup = self._normalize_lookup_code(act_code_norm)
            key = (lookup, int(article))
            return list(self._rev_index.get(key, []))
        except Exception:
            return []

    def get_article_contents(self, act_code_norm: str, article: int) -> List[Dict[str, Any]]:
        """Lấy nội dung các chunk thuộc một Điều của văn bản.

        Trả về danh sách dict: {chunk_id, corpus_id, suffix, content}."""
        results: List[Dict[str, Any]] = []
        try:
            chunk_ids = self.find_chunks_by_code_and_article(act_code_norm, article)
            for cid in chunk_ids:
                meta = self._get_chunk_metadata(int(cid))
                content = self.get_chunk_content(int(cid)) or ''
                results.append({
                    'chunk_id': int(cid),
                    'corpus_id': meta.get('corpus_id'),
                    'suffix': meta.get('suffix'),
                    'content': content or (meta.get('content') or ''),
                })
        except Exception:
            return results
        # Sắp xếp theo suffix và chunk_id để hiển thị có thứ tự
        try:
            results.sort(key=lambda x: (int(x.get('suffix') or 0), int(x.get('chunk_id') or 0)))
        except Exception:
            pass
        return results

    # ---------------- Fallback trích xuất Điều khi thiếu chunk trực tiếp ---------------
    def _scan_act_code_records(self, act_code_norm: str) -> List[str]:
        """Quét toàn bộ JSONL để gom content của văn bản có act_code tương ứng.

        Tối ưu: dừng sớm nếu đã lấy > 0 và JSONL rất lớn? (Hiện dataset ~60k dòng chấp nhận được)
        Cache theo act_code_norm.
        """
        lookup_key = self._normalize_lookup_code(act_code_norm)
        key = (lookup_key, '__ALL__')
        if key in self._article_fallback_cache:
            cached = self._article_fallback_cache[key]
            return cached.split('\n\u0000\n') if cached else []
        contents: List[str] = []
        if not self.jsonl_path.exists():
            return contents
        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    corpus_id = str(record.get('corpus_id') or '').strip()
                    if not corpus_id:
                        continue
                    raw_code = corpus_id.split('+')[0].strip()
                    try:
                        from ..utils.law_registry import normalize_act_code  # lazy
                        code_norm = normalize_act_code(raw_code)
                    except Exception:
                        code_norm = raw_code.upper()
                    if self._normalize_lookup_code(code_norm) != lookup_key:
                        continue
                    content = (record.get('content') or record.get('preview') or '').replace('_', ' ').strip()
                    if content:
                        contents.append(content)
            # Cache chuỗi nối bằng sentinel để tránh list lớn trong dict
            self._article_fallback_cache[key] = '\n\u0000\n'.join(contents)
        except Exception:
            return contents
        return contents

    def extract_article_text_fallback(self, act_code_norm: str, article: int) -> Optional[str]:
        """Cố gắng trích nội dung Điều bằng regex nếu không có chunk mapping trực tiếp.

        Gộp tất cả content của văn bản và tìm pattern 'Điều <article>' đến trước 'Điều <next>' hoặc hết file.
        """
        lookup_key = self._normalize_lookup_code(act_code_norm)
        cache_key = (lookup_key, int(article))
        if cache_key in self._article_fallback_cache:
            return self._article_fallback_cache[cache_key]
        parts = self._scan_act_code_records(act_code_norm)
        if not parts:
            return None
        # Gộp với ngắt dòng an toàn
        combined = '\n'.join(parts)
        # Regex: Điều <article> (có thể có khoảng trắng) bắt đầu đoạn
        # Hỗ trợ cả định dạng có số 0 ở trước (vd: "Điều 05")
        pattern = re.compile(
            rf"(Điều\s+0*{int(article)}\b.*?)(?=\n\s*Điều\s+0*\d+\b|\Z)",
            re.IGNORECASE | re.DOTALL,
        )
        match = pattern.search(combined)
        if not match:
            return None
        text = match.group(1).strip()
        # Cache
        self._article_fallback_cache[cache_key] = text
        return text

    def get_article_text(self, act_code_norm: str, article: int) -> Optional[str]:
        """API hợp nhất: ưu tiên chunk mapping; fallback regex nếu rỗng."""
        chunks = self.get_article_contents(act_code_norm, article)
        if chunks:
            # Gộp nội dung các chunk (đã theo thứ tự)
            merged = '\n\n'.join((c.get('content') or '').strip() for c in chunks if c.get('content'))
            if merged.strip():
                return merged.strip()
        # Fallback
        return self.extract_article_text_fallback(act_code_norm, article)

    def get_document_text_all(self, act_code_norm: str) -> Optional[str]:
        """Ghép toàn bộ nội dung của một văn bản (mọi Điều) thành chuỗi duy nhất.

        Dùng làm fallback hiển thị khi nhóm tài liệu không có danh sách Điều cụ thể.
        """
        parts = self._scan_act_code_records(act_code_norm)
        if not parts:
            return None
        combined = '\n\n'.join(p.strip() for p in parts if p and p.strip())
        return combined.strip() if combined.strip() else None
        return results

    def _get_chunk_metadata(self, chunk_id: int) -> Dict[str, Any]:
        meta: Dict[str, Any] = dict(self._meta_by_id.get(chunk_id, {}))

        id_attrs = self._id_attrs.get(chunk_id)
        if id_attrs:
            for key, value in id_attrs.items():
                meta.setdefault(key, value)

        if not meta.get('corpus_id') or not meta.get('preview') or not meta.get('number'):
            record = self._load_chunk_record(chunk_id)
            if record:
                for key in ('corpus_id', 'type', 'number', 'year', 'suffix', 'preview'):
                    if key in record and record[key] and not meta.get(key):
                        meta[key] = record[key]
                meta.setdefault('content', record.get('content'))

        return meta

    @classmethod
    def _load_sentence_transformer_safe(cls, model_ref: str) -> SentenceTransformer:
        """Load SentenceTransformer an toàn, tránh lỗi meta tensor."""
        last_exc: Optional[BaseException] = None
        for disable_cuda in (False, True):
            with cls._cuda_disabled(disable_cuda):
                try:
                    model_kwargs = cls._default_model_args()
                    return SentenceTransformer(model_ref, device="cpu", model_kwargs=model_kwargs)
                except Exception as exc:
                    last_exc = exc
                    if not cls._is_meta_tensor_error(exc):
                        continue
                    try:
                        return cls._load_sentence_transformer_without_meta(model_ref)
                    except Exception as fallback_exc:
                        last_exc = fallback_exc
        if last_exc is not None:
            raise last_exc
        raise RuntimeError(f"Unknown failure while loading encoder {model_ref}")

    @classmethod
    def _load_sentence_transformer_without_meta(cls, model_ref: str) -> SentenceTransformer:
        """Fallback: tự build pipeline để tránh meta tensor."""
        model_path = Path(model_ref)
        if model_path.exists() and (model_path / "modules.json").exists():
            modules = cls._build_modules_from_local_config(model_path)
            model_kwargs = cls._default_model_args()
            model = SentenceTransformer(modules=modules, device="cpu", model_kwargs=model_kwargs)
            state_dict = cls._load_local_state_dict(model_path)
            if state_dict:
                state_dict = cls._adjust_local_state_dict(state_dict)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger = get_logger("legaladvisor.retrieval")
                    if missing:
                        logger.warning("Missing keys when loading encoder state dict: %s", missing)
                    if unexpected:
                        logger.warning("Unexpected keys when loading encoder state dict: %s", unexpected)
            return model

        # Remote model hoặc không có modules.json -> build mặc định Transformer + Pooling
        base_model_args = cls._default_model_args()
        try:
            transformer = models.Transformer(model_ref, model_args=base_model_args)
        except TypeError:
            transformer = models.Transformer(model_ref)
        cls._ensure_cpu_parameters(transformer.auto_model)
        pooling = models.Pooling(transformer.get_word_embedding_dimension())
        return SentenceTransformer(modules=[transformer, pooling], device="cpu")

    @classmethod
    def _build_modules_from_local_config(cls, root: Path) -> List[Any]:
        with open(root / "modules.json", "r", encoding="utf-8") as handle:
            modules_spec: List[Dict[str, Any]] = json.load(handle)

        modules_spec.sort(key=lambda item: int(item.get("idx", 0)))
        built_modules: List[Any] = []
        transformer_module: Optional[Any] = None

        for spec in modules_spec:
            module_type = spec.get("type")
            module_rel_path = spec.get("path", "")
            module_dir = root / module_rel_path if module_rel_path else root
            module = cls._instantiate_module(module_type, module_dir, transformer_module)
            if isinstance(module, models.Transformer):
                transformer_module = module
            built_modules.append(module)

        return built_modules

    @classmethod
    def _instantiate_module(
        cls,
        module_type: Optional[str],
        module_dir: Path,
        transformer_module: Optional[Any],
    ) -> Any:
        if not module_type:
            raise RuntimeError(f"Missing module type for {module_dir}")

        module_cls = cls._resolve_module_class(module_type)

        if module_type.endswith("Transformer"):
            transformer_kwargs_candidates = [
                {"model_args": cls._default_model_args()},
                {"model_args": {"low_cpu_mem_usage": False}},
                {},
            ]
            last_exc: Optional[BaseException] = None
            for kwargs in transformer_kwargs_candidates:
                try:
                    module = module_cls(str(module_dir), **kwargs)
                    cls._ensure_cpu_parameters(module.auto_model)
                    return module
                except TypeError as exc:
                    last_exc = exc
                    continue
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"Unable to instantiate transformer module at {module_dir}")
            return module_cls(str(module_dir))

        config_path = module_dir / "config.json"
        module_kwargs: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as cfg:
                module_kwargs = json.load(cfg)

        if module_type.endswith("Pooling") and transformer_module is not None:
            module_kwargs.setdefault(
                "word_embedding_dimension",
                transformer_module.get_word_embedding_dimension(),
            )

        return module_cls(**module_kwargs)

    @staticmethod
    def _resolve_module_class(class_path: str) -> Any:
        module_name, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)

    @staticmethod
    def _ensure_cpu_parameters(model: torch.nn.Module) -> None:
        """Đảm bảo toàn bộ tham số nằm trên CPU, dùng to_empty nếu cần."""
        if model is None:
            return

        try:
            first_param = next(model.parameters())
        except StopIteration:
            first_param = None

        if first_param is None:
            return

        if first_param.device.type == "cpu":
            return

        try:
            model.to(torch.device("cpu"), use_to_empty=True)
        except TypeError:
            if hasattr(model, "to_empty"):
                model.to_empty(device=torch.device("cpu"), recurse=True)
            else:
                model.to(torch.device("cpu"))

    @staticmethod
    def _load_local_state_dict(root: Path) -> Optional[Dict[str, torch.Tensor]]:
        safetensors_path = root / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors.torch import load_file  # type: ignore

                return load_file(str(safetensors_path))
            except Exception:
                pass

        bin_path = root / "pytorch_model.bin"
        if bin_path.exists():
            try:
                return torch.load(bin_path, map_location="cpu")
            except Exception:
                pass
        return None

    def _load_indexed_records_on_init(self) -> None:
        """Load toàn bộ JSONL file vào indexed cache on-startup.
        
        Tối ưu hoá: Thay vì quét JSONL từ đầu cho mỗi chunk_id (O(n) mỗi lần),
        load tất cả vào dict (O(1) lookup).
        
        Memory: ~60KB file = ~6MB RAM. Performance: -30s per request.
        """
        if self._all_records_cached:
            return
        
        if not self.jsonl_path.exists():
            self._all_records_cached = True
            return
        
        try:
            self._logger.info("[cache_init] Loading all records from JSONL into indexed cache...")
            import time
            t0 = time.time()
            
            with open(self.jsonl_path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    chunk_id = record.get('chunk_id')
                    if chunk_id is None:
                        continue
                    try:
                        chunk_id_int = int(chunk_id)
                    except (TypeError, ValueError):
                        continue
                    
                    # Lưu vào cache (không lưu vào _chunk_cache để tránh trùng)
                    self._chunk_cache[chunk_id_int] = record
            
            elapsed = time.time() - t0
            total_records = len(self._chunk_cache)
            self._logger.info(f"[cache_init] Loaded {total_records} records in {elapsed:.2f}s")
            self._all_records_cached = True
        except Exception as e:
            self._logger.warning(f"[cache_init] Failed to load indexed records: {e}")
            self._all_records_cached = True  # Mark as attempted to avoid retry
    
    def _load_chunk_record(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Load chunk record từ indexed cache (O(1) lookup).
        
        Nếu indexed cache không sẵn sàng, fallback quét JSONL (legacy).
        """
        if chunk_id in self._chunk_cache:
            return self._chunk_cache.get(chunk_id)

        # Nếu indexed cache đã load xong mà không tìm thấy → không có record
        if self._all_records_cached:
            self._chunk_cache[chunk_id] = {}
            return None

        # Fallback: quét JSONL (legacy, chỉ khi indexed cache chưa sẵn)
        if not self.jsonl_path.exists():
            return None

        try:
            with open(self.jsonl_path, 'r', encoding='utf-8') as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if int(record.get('chunk_id', -1)) == int(chunk_id):
                        self._chunk_cache[chunk_id] = record
                        return record
        except Exception:
            return None

        self._chunk_cache[chunk_id] = {}
        return None

    @staticmethod
    def _is_meta_tensor_error(exc: BaseException) -> bool:
        message = str(exc)
        return "Cannot copy out of meta tensor" in message or "to_empty" in message

    @staticmethod
    def _default_model_args() -> Dict[str, Any]:
        return {
            "low_cpu_mem_usage": False,
        }

    @staticmethod
    def _adjust_local_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        adjusted: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith(("0.", "1.")):
                adjusted[key] = value
                continue
            if key.startswith(("embeddings.", "encoder.", "pooler.")):
                adjusted[f"0.auto_model.{key}"] = value
                continue
            adjusted[key] = value
        return adjusted

    @staticmethod
    @contextmanager
    def _cuda_disabled(disable: bool):
        if not disable:
            yield
            return
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        try:
            yield
        finally:
            if old_cuda is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda


