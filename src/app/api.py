#!/usr/bin/env python3
"""
FastAPI backend cho LegalAdvisor
"""

import sys
import os
import signal
sys.path.append('../..')

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conint, constr
from typing import List, Dict, Any, Optional
import uvicorn
from pathlib import Path
import json
import time
import threading
from collections import defaultdict, deque
from datetime import datetime

# Import logger
try:
    from ..utils.logger import get_logger, log_performance, log_error
    logger = get_logger("legaladvisor.api")
except ImportError:
    # Fallback if utils not available
    import logging
    logger = logging.getLogger("legaladvisor.api")
    logger.setLevel(logging.INFO)

    def log_performance(operation, duration, metadata=None):
        logger.info(f"Performance: {operation} took {duration:.2f}s")

    def log_error(message):
        logger.error(message)

# Import RAG pipeline - sử dụng Groq-backed RAG class
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    # Ưu tiên import tuyệt đối để tránh lỗi 'attempted relative import with no known parent package'
    from src.rag.groq_rag import GroqRAG  # type: ignore
except ImportError:
    # Fallback if absolute import fails
    from ..rag.groq_rag import GroqRAG  # type: ignore

# Parse command line arguments
parser = argparse.ArgumentParser(description='LegalAdvisor API Server')
parser.add_argument('--host', default='0.0.0.0', help='Host để chạy server')
parser.add_argument('--port', type=int, default=8000, help='Port để chạy server')
parser.add_argument('--use-gpu', action='store_true', help='Sử dụng GPU nếu có sẵn')
args, unknown = parser.parse_known_args()

# Initialize RAG system: dùng Groq-backed RAG (lazy init để tăng tốc khởi động)
rag_system = None
rag_last_error: Optional[str] = None
rag_init_lock = threading.Lock()
rag_retry_info: Dict[str, Any] = {
    "attempts": 0,
    "last_attempt_at": None,
    "last_success_at": None,
    "last_error": None,
}

# Rate limiting (cấu hình qua ENV)
RATE_LIMIT_WINDOW = int(os.getenv("LEGALADVISOR_RATE_LIMIT_WINDOW", "60"))
RATE_LIMIT_MAX = int(os.getenv("LEGALADVISOR_RATE_LIMIT_MAX", "30"))
_rate_limit_store: Dict[str, deque] = defaultdict(deque)
_rate_limit_lock = threading.Lock()

# Toggle log nội dung câu hỏi (0/1)
LOG_QUESTIONS = os.getenv("LEGALADVISOR_LOG_QUESTIONS", "0") == "1"

# Sweep dọn dẹp rate-limit store định kỳ (0/1)
ENABLE_RATE_LIMIT_CLEANUP = os.getenv("LEGALADVISOR_RATE_LIMIT_CLEANUP", "1") == "1"
RATE_LIMIT_SWEEP_INTERVAL = int(os.getenv("LEGALADVISOR_RATE_LIMIT_SWEEP_INTERVAL", "60"))

# Cho phép bỏ qua khởi tạo RAG (phục vụ test)
SKIP_RAG_INIT = os.getenv("LEGALADVISOR_SKIP_RAG_INIT", "0") == "1"


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _init_rag(force: bool = False) -> bool:
    """Khởi tạo RAG và cập nhật trạng thái toàn cục."""
    global rag_system, rag_last_error

    with rag_init_lock:
        if rag_system is not None and not force:
            return True

        rag_retry_info["attempts"] = int(rag_retry_info.get("attempts", 0)) + 1
        rag_retry_info["last_attempt_at"] = _now_iso()

        try:
            from ..rag.groq_rag import GroqRAG
            rag_system = GroqRAG(use_gpu=args.use_gpu)
            rag_last_error = None
            rag_retry_info["last_success_at"] = rag_retry_info["last_attempt_at"]
            rag_retry_info["last_error"] = None
            # Only log, don't print to reduce console noise
            logger.info("Groq RAG initialized successfully")
            return True
        except Exception as e:
            error_msg = f"Failed to initialize Groq RAG: {e}"
            rag_system = None
            rag_last_error = str(e)
            rag_retry_info["last_error"] = str(e)
            log_error(error_msg)
            # Only log, don't print to reduce console noise
            return False


def _init_rag_background():
    """Khởi tạo RAG với retry nền."""
    max_retries = 3
    delay_seconds = 5
    for attempt in range(1, max_retries + 1):
        if _init_rag():
            return
        if attempt < max_retries:
            logger.warning(
                f"Retrying Groq RAG initialization in {delay_seconds}s (attempt {attempt}/{max_retries})"
            )
            time.sleep(delay_seconds)
    logger.error("Groq RAG failed to initialize after retries")


if not SKIP_RAG_INIT:
    threading.Thread(target=_init_rag_background, daemon=True).start()
else:
    logger.info("Skipping Groq RAG background init (LEGALADVISOR_SKIP_RAG_INIT=1)")


# Pydantic models
class QuestionRequest(BaseModel):
    # Giới hạn chiều dài câu hỏi để tránh lạm dụng (mặc định 1024)
    question: constr(min_length=1, max_length=1024)  # type: ignore
    # Mức chi tiết: brief | moderate | comprehensive (mặc định: moderate)
    detail_level: str = "brief"  # type: ignore
    # Legacy: giữ để backward compatible (không dùng)
    top_k: conint(ge=1, le=50) = 3  # type: ignore


class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    num_sources: int  # Số tài liệu (laws)
    num_segments_retrieved: int = 0  # Số segments retrieve để scoring (oversample)
    num_segments: int = 0  # Số segments total (toàn bộ của articles được chọn)
    num_articles: int = 0  # Số articles được chọn (qua adaptive threshold)
    sources_grouped: List[Dict[str, Any]] = []
    num_docs: int = 0  # Alias cho num_sources
    threshold_used: float = 0.0  # Threshold value áp dụng
    detail_level: str = "brief"  # Mức chi tiết được sử dụng
    status: str = "success"
    citations: List[Dict[str, Any]] = []


# Khởi tạo FastAPI
app = FastAPI(
    title="LegalAdvisor API",
    description="API cho hệ thống hỏi đáp pháp luật tiếng Việt",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    message: str
    rag_loaded: bool
    rag_error: Optional[str] = None

class HealthDiagnostics(BaseModel):
    rag_loaded: bool
    rag_error: Optional[str]
    retry_attempts: int
    last_attempt_at: Optional[str]
    last_success_at: Optional[str]

class WarmupResponse(BaseModel):
    status: str
    rag_loaded: bool
    warmed_retrieval: bool
    warmed_llm: bool
    message: Optional[str] = None

class CitationContentResponse(BaseModel):
    act_code: str
    article: int
    num_chunks: int
    items: List[Dict[str, Any]]
    merged_content: Optional[str] = None

class CitationContentBulkResponse(BaseModel):
    act_code: str
    articles: List[int] = []
    by_article: Dict[str, Optional[str]] = {}
    merged_combined: Optional[str] = None
    merged_all: Optional[str] = None


@app.get("/", tags=["General"])
async def root():
    """Trang chủ API"""
    return {
        "message": "Welcome to LegalAdvisor API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Kiểm tra trạng thái hệ thống"""
    status = "healthy" if rag_system else "degraded"
    return HealthResponse(
        status=status,
        message="LegalAdvisor API is running",
        rag_loaded=rag_system is not None,
        rag_error=rag_last_error
    )

@app.get("/health/details", response_model=HealthDiagnostics, tags=["Health"], summary="Chi tiết trạng thái RAG")
async def health_details():
    """Trả về thông tin chi tiết về trạng thái RAG."""
    return HealthDiagnostics(
        rag_loaded=rag_system is not None,
        rag_error=rag_last_error,
        retry_attempts=int(rag_retry_info.get("attempts", 0)),
        last_attempt_at=rag_retry_info.get("last_attempt_at"),
        last_success_at=rag_retry_info.get("last_success_at")
    )

@app.post("/warmup", response_model=WarmupResponse, tags=["Health"], summary="Khởi tạo và làm ấm hệ thống")
async def warmup(llm: bool = False):
    """Khởi tạo RAG và chạy một lượt retrieval (và tùy chọn LL.M) để giảm độ trễ lần đầu.

    - llm=False: chỉ warm retrieval (không tốn token Groq), mặc định.
    - llm=True: gọi một lượt generate_response với prompt ngắn để warm model (sẽ tốn token).
    """
    loaded = _init_rag(force=False)
    if not loaded or rag_system is None:
        return WarmupResponse(status="degraded", rag_loaded=False, warmed_retrieval=False, warmed_llm=False, message=rag_last_error)

    warmed_retrieval = False
    warmed_llm = False
    try:
        # Warm retrieval: một lượt search ngắn
        _ = rag_system.retrieve_documents("khởi tạo hệ thống", top_k=1)
        warmed_retrieval = True
    except Exception as e:
        log_error(f"Warmup retrieval failed: {e}")
    if llm:
        try:
            _ = rag_system.generate_response("Ping", context=None)
            warmed_llm = True
        except Exception as e:
            log_error(f"Warmup LLM failed: {e}")
    return WarmupResponse(status="ok" if warmed_retrieval or warmed_llm else "degraded",
                          rag_loaded=True,
                          warmed_retrieval=warmed_retrieval,
                          warmed_llm=warmed_llm)

@app.post("/debug/reinit", response_model=HealthResponse, tags=["Health"], summary="Force reinitialize RAG")
async def force_reinitialize_rag():
    """Ép khởi tạo lại hệ thống RAG."""
    success = _init_rag(force=True)
    status = "healthy" if success else "degraded"
    return HealthResponse(
        status=status,
        message="RAG reinitialized" if success else "Failed to reinitialize RAG",
        rag_loaded=rag_system is not None,
        rag_error=rag_last_error
    )


def _rate_limit_cleanup(timestamps: deque, now: float) -> None:
    while timestamps and now - timestamps[0] > RATE_LIMIT_WINDOW:
        timestamps.popleft()


def _sanitize_sensitive(text: Optional[str]) -> Optional[str]:
    """Mask thông tin nhạy cảm (ví dụ API key) nếu có xuất hiện trong chuỗi."""
    if not text:
        return text
    masked = str(text)
    for api_key_name in ("GROQ_API_KEY",):
        api_key = os.getenv(api_key_name)
        if api_key:
            masked = masked.replace(api_key, "***")
    return masked


def _rate_limit_sweeper():
    """Thread nền: dọn dẹp các key IP không còn timestamp hợp lệ để tránh leak bộ nhớ."""
    while True:
        try:
            now = time.time()
            with _rate_limit_lock:
                to_delete = []
                for ip, timestamps in _rate_limit_store.items():
                    _rate_limit_cleanup(timestamps, now)
                    if not timestamps:
                        to_delete.append(ip)
                for ip in to_delete:
                    del _rate_limit_store[ip]
        except Exception as e:
            logger.warning(f"Rate limit sweeper error: {e}")
        time.sleep(RATE_LIMIT_SWEEP_INTERVAL)


async def rate_limit_dependency(request: Request) -> None:
    if RATE_LIMIT_MAX <= 0:
        return

    client_host = request.client.host if request.client else "unknown"
    now = time.time()

    with _rate_limit_lock:
        timestamps = _rate_limit_store[client_host]
        _rate_limit_cleanup(timestamps, now)

        if len(timestamps) >= RATE_LIMIT_MAX:
            retry_after = max(0.0, RATE_LIMIT_WINDOW - (now - timestamps[0]))
            raise HTTPException(
                status_code=429,
                detail={
                    "message": "Rate limit exceeded",
                    "retry_after": round(retry_after, 2),
                    "limit_window": RATE_LIMIT_WINDOW,
                    "limit_max": RATE_LIMIT_MAX
                }
            )

        timestamps.append(now)


@app.post("/ask", response_model=AnswerResponse, tags=["QA"])
async def ask_question(request: QuestionRequest, _: None = Depends(rate_limit_dependency)):
    """Trả lời câu hỏi pháp luật"""

    if LOG_QUESTIONS:
        logger.info(f"Received question: {request.question}")
    else:
        logger.info("Received question: [masked]")
    start_time = time.time()

    if rag_system is None:
        if _init_rag():
            logger.info("Groq RAG reinitialized on-demand")
        else:
            log_error("RAG system not available")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "RAG system is not available",
                    "error": _sanitize_sensitive(rag_last_error),
                    "hint": "Verify GROQ_API_KEY and retrieval models are configured."
                }
            )

    try:
        # Xử lý câu hỏi
        # Sử dụng RAG.ask với detail_level (brief/moderate/comprehensive)
        result = rag_system.ask(request.question, detail_level=request.detail_level)

        response_time = time.time() - start_time
        log_performance("api_request", response_time, {
            "question": request.question if LOG_QUESTIONS else "[masked]",
            "confidence": float(result.get('confidence', 0.0)),
            "num_sources": int(result.get('num_sources', 0)),  # Số tài liệu (laws)
            "num_segments": int(result.get('num_segments', 0))  # Số segments tìm được
        })

        # Trích xuất tài liệu trích dẫn (citations) chỉ từ các nguồn thuộc TOP-K tài liệu
        citations: List[Dict[str, Any]] = []
        try:
            # Import chậm để tránh chi phí khởi động
            from ..retrieval.citation.extract import extract_citations  # type: ignore

            from ..utils.law_registry import get_registry  # type: ignore
            reg = None
            try:
                reg = get_registry()
            except Exception:
                reg = None

            sources = result.get('sources', []) or []
            # Chỉ cho phép trích dẫn từ các chunk thuộc tài liệu top-k (sources_grouped)
            allowed_act_codes: set = set()
            try:
                for g in (result.get('sources_grouped') or []):
                    ac = g.get('act_code')
                    if ac:
                        allowed_act_codes.add(str(ac))
            except Exception:
                allowed_act_codes = set()
            code_to_articles: Dict[str, set] = {}
            # Bản đồ: mã văn bản được trích dẫn -> {ref_act_code: {ref_articles: set[int], cited_articles: set[int]}}
            cited_to_refs: Dict[str, Dict[str, Dict[str, set]]] = {}
            for s in sources:
                content = (s.get('content_full') or s.get('content') or None)
                if not content:
                    try:
                        cid = s.get('chunk_id')
                        if cid is not None and rag_system is not None:
                            content = rag_system.get_chunk_content(int(cid))
                    except Exception:
                        content = None
                if not content:
                    continue
                # Xác định mã văn bản tham khảo (nguồn) tương ứng với chunk hiện tại
                ref_code_norm: Optional[str] = None
                ref_article_num: Optional[int] = None
                try:
                    corpus_id = str(s.get('corpus_id') or '').strip()
                    if corpus_id:
                        raw_code = corpus_id.split('+')[0].strip()
                        if raw_code:
                            try:
                                from ..utils.law_registry import normalize_act_code as _norm  # type: ignore
                                ref_code_norm = _norm(raw_code)
                            except Exception:
                                ref_code_norm = raw_code.upper()
                        # Lấy Điều (suffix) của văn bản tham chiếu hiện tại nếu có
                        try:
                            parts = corpus_id.split('+', 1)
                            if len(parts) == 2 and parts[1].isdigit():
                                ref_article_num = int(parts[1])
                        except Exception:
                            ref_article_num = None
                    # Ưu tiên suffix trong field riêng nếu có
                    if ref_article_num is None:
                        suf = s.get('suffix')
                        if suf is not None and str(suf).isdigit():
                            try:
                                ref_article_num = int(suf)
                            except Exception:
                                ref_article_num = None
                except Exception:
                    ref_code_norm = None
                # Bỏ qua nếu chunk không thuộc các tài liệu top-k
                if allowed_act_codes and (not ref_code_norm or ref_code_norm not in allowed_act_codes):
                    continue
                try:
                    hits = extract_citations(content, registry=reg, article_only=True)
                except Exception:
                    hits = []
                for h in hits:
                    if not h.act_code_norm or h.article is None:
                        continue
                    code = h.act_code_norm
                    if code not in code_to_articles:
                        code_to_articles[code] = set()
                    code_to_articles[code].add(int(h.article))
                    if ref_code_norm:
                        entry = cited_to_refs.setdefault(code, {}).setdefault(ref_code_norm, {"ref_articles": set(), "cited_articles": set()})
                        # Bổ sung Điều của văn bản THAM CHIẾU (nơi chứa trích dẫn)
                        if ref_article_num is not None:
                            entry["ref_articles"].add(int(ref_article_num))
                        # Ghi nhận Điều của văn bản BỊ TRÍCH DẪN (đã dùng ở phần tiêu đề bên ngoài)
                        entry["cited_articles"].add(int(h.article))

            for code, arts in code_to_articles.items():
                # Build supplemented_by list with act_code + articles
                supplemented_list: List[Dict[str, Any]] = []
                refs = cited_to_refs.get(code, {}) or {}
                for ref_code, ref_info in refs.items():
                    # Trả về các Điều của văn bản tham chiếu (nơi chứa tham chiếu)
                    ref_arts = sorted([int(a) for a in (ref_info.get("ref_articles") or set())])
                    supplemented_list.append({
                        "act_code": ref_code,
                        "articles": ref_arts,
                    })
                citations.append({
                    "act_code": code,
                    "articles": sorted(list(arts)),
                    "supplemented_by": supplemented_list,
                })
        except Exception:
            citations = []

        # Tạo phản hồi chuẩn hóa
        return AnswerResponse(
            question=result.get('question', request.question),
            answer=result.get('answer', ''),
            confidence=float(result.get('confidence', 0.0)),
            sources=result.get('sources', []),
            num_sources=int(result.get('num_sources', 0)),
            num_segments_retrieved=int(result.get('num_segments_retrieved', 0)),
            num_segments=int(result.get('num_segments', 0)),
            num_articles=int(result.get('num_articles', 0)),
            sources_grouped=result.get('sources_grouped', []),
            num_docs=int(result.get('num_docs', 0)),
            threshold_used=float(result.get('threshold_used', 0.0)),
            detail_level=result.get('detail_level', 'moderate'),
            status=result.get('status', 'success'),
            citations=citations,
        )

    except Exception as e:
        safe_err = _sanitize_sensitive(str(e))
        log_error(f"Error processing question: {safe_err}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Error processing question",
                "error": safe_err
            }
        )

@app.get("/stats", tags=["Statistics"])
async def get_stats():
    """Thống kê hệ thống"""

    if not rag_system:
        return {"error": "RAG system not loaded"}

    try:
        # Load metadata để lấy thống kê (hỗ trợ cấu trúc mới và cũ)
        env_models_dir = os.getenv("LEGALADVISOR_MODELS_DIR")
        if env_models_dir:
            model_dir = Path(env_models_dir)
        else:
            current_dir = Path(__file__).resolve().parent  # src/app
            root_dir = current_dir.parent.parent  # -> root
            model_dir = root_dir / "models" / "retrieval"
        metadata_path_old = model_dir / "metadata.json"
        metadata_path_new = model_dir / "index" / "metadata.json"

        metadata = None
        if metadata_path_old.exists():
            with open(metadata_path_old, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        elif metadata_path_new.exists():
            with open(metadata_path_new, 'r', encoding='utf-8') as f:
                # metadata mới là dict (tổng quan). Nếu cần chi tiết per-chunk, giữ nguyên lỗi “not found”.
                try:
                    metadata = json.load(f)
                except Exception:
                    metadata = None

        if isinstance(metadata, list):
            total_chunks = len(metadata)
            total_words = sum(item.get('word_count', 0) for item in metadata)
            model_name = getattr(rag_system, 'model_info', {}).get("model_name", "unknown")
            return {
                "total_chunks": total_chunks,
                "total_words": total_words,
                "avg_words_per_chunk": total_words / total_chunks if total_chunks > 0 else 0,
                "model_name": model_name,
            }
        elif isinstance(metadata, dict):
            # metadata mới (tổng quan) có thể chứa total_chunks
            total_chunks = int(metadata.get("total_chunks", 0))
            model_name = getattr(rag_system, 'model_info', {}).get("model_name", "unknown")
            return {
                "total_chunks": total_chunks,
                "total_words": None,
                "avg_words_per_chunk": None,
                "model_name": model_name,
            }
        else:
            return {"error": "Metadata not found"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/sources/{chunk_id}", tags=["Sources"])
async def get_source_content(chunk_id: int):
    """Lấy nội dung của một chunk"""

    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")

    try:
        content = rag_system.get_chunk_content(chunk_id)
        if content:
            return {"chunk_id": chunk_id, "content": content}
        else:
            raise HTTPException(status_code=404, detail="Chunk not found")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/citations/content", response_model=CitationContentResponse, tags=["Citations"], summary="Lấy nội dung trích dẫn theo mã văn bản và Điều")
async def get_citation_content(act_code: str, article: int):
    """Lấy nội dung các chunk thuộc một Điều trong văn bản được trích dẫn.

    - act_code: mã văn bản (dạng tự nhiên), sẽ được chuẩn hoá về `act_code_norm`.
    - article: số Điều (int).
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    try:
        try:
            from ..utils.law_registry import normalize_act_code  # type: ignore
        except Exception:
            def normalize_act_code(x: str) -> str:
                return (x or '').strip().lower()

        code_norm = normalize_act_code(act_code)
        if not code_norm:
            raise HTTPException(status_code=400, detail="act_code không hợp lệ")

        retr = getattr(rag_system, 'retriever', None)
        if retr is None:
            raise HTTPException(status_code=503, detail="Retriever not available")
        items = retr.get_article_contents(code_norm, int(article))
        merged = retr.get_article_text(code_norm, int(article))
        return {
            "act_code": code_norm,
            "article": int(article),
            "num_chunks": len(items),
            "items": items,
            "merged_content": merged,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/citations/content/bulk", response_model=CitationContentBulkResponse, tags=["Citations"], summary="Lấy nội dung trích dẫn hàng loạt theo danh sách Điều")
async def get_citation_content_bulk(act_code: str, articles: Optional[str] = None):
    """Trả về nội dung theo nhiều Điều trong một lần gọi. Nếu không truyền `articles`, trả về toàn văn bản.

    - act_code: mã văn bản (chuẩn hóa nội bộ).
    - articles: chuỗi số cách nhau bằng dấu phẩy, ví dụ: "5,6,7".
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not available")
    try:
        try:
            from ..utils.law_registry import normalize_act_code  # type: ignore
        except Exception:
            def normalize_act_code(x: str) -> str:
                return (x or '').strip().lower()

        code_norm = normalize_act_code(act_code)
        if not code_norm:
            raise HTTPException(status_code=400, detail="act_code không hợp lệ")
        retr = getattr(rag_system, 'retriever', None)
        if retr is None:
            raise HTTPException(status_code=503, detail="Retriever not available")

        if not articles:
            merged_all = retr.get_document_text_all(code_norm)
            return {
                "act_code": code_norm,
                "articles": [],
                "by_article": {},
                "merged_combined": merged_all,
                "merged_all": merged_all,
            }

        # Parse danh sách Điều
        try:
            arts_list = [int(a.strip()) for a in str(articles).split(',') if a.strip().isdigit()]
        except Exception:
            arts_list = []
        if not arts_list:
            merged_all = retr.get_document_text_all(code_norm)
            return {
                "act_code": code_norm,
                "articles": [],
                "by_article": {},
                "merged_combined": merged_all,
                "merged_all": merged_all,
            }

        by_article: Dict[str, Optional[str]] = {}
        sections: List[str] = []
        for art in arts_list:
            text = retr.get_article_text(code_norm, int(art))
            by_article[str(int(art))] = text
            if text:
                sections.append(f"Điều {int(art)}\n{text}")
        merged_combined = "\n\n".join(sections).strip() if sections else None
        return {
            "act_code": code_norm,
            "articles": [int(a) for a in arts_list],
            "by_article": by_article,
            "merged_combined": merged_combined,
            "merged_all": retr.get_document_text_all(code_norm),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Global variables for signal handling
server = None

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global server
    print(f"\n🛑 Nhận tín hiệu {signum}, đang dừng server...")
    if server:
        server.should_exit = True
    print("✅ Server đã dừng!")
    sys.exit(0)

def run_server(host="0.0.0.0", port=8000, reload=False):
    """Chạy server trực tiếp với signal handling tốt"""
    global server

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"🚀 Khởi động LegalAdvisor API server...")
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print("🛑 Nhấn Ctrl+C để dừng server")
    print("=" * 50)

    try:
        # Tạo uvicorn config
        config = uvicorn.Config(
            app=app,  # Sử dụng app instance trực tiếp thay vì string
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )

        # Tạo server instance
        server = uvicorn.Server(config)

        print(f"🎯 Server config: {host}:{port} (reload: {reload})")

        # Chạy server
        server.run()

    except Exception as e:
        print(f"❌ Lỗi khi khởi động server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Chạy server với cấu hình đã parse ở đầu file
    run_server(host=args.host, port=args.port, reload=False)

# Khởi động thread dọn dẹp rate-limit (đặt ở cuối để đảm bảo hàm đã được định nghĩa)
if ENABLE_RATE_LIMIT_CLEANUP:
    try:
        threading.Thread(target=_rate_limit_sweeper, daemon=True).start()
        logger.info("Rate limit sweeper thread started")
    except Exception as e:
        logger.warning(f"Cannot start rate limit sweeper: {e}")
