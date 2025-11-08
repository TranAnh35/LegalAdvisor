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

# Import RAG pipeline - Sử dụng GeminiRAG
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Parse command line arguments
parser = argparse.ArgumentParser(description='LegalAdvisor API Server')
parser.add_argument('--host', default='0.0.0.0', help='Host để chạy server')
parser.add_argument('--port', type=int, default=8000, help='Port để chạy server')
parser.add_argument('--use-gpu', action='store_true', help='Sử dụng GPU nếu có sẵn')
args, unknown = parser.parse_known_args()

# Initialize RAG system: luôn dùng GeminiRAG (lazy init để tăng tốc khởi động)
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
            from ..rag.gemini_rag import GeminiRAG
            rag_system = GeminiRAG(use_gpu=args.use_gpu)
            rag_last_error = None
            rag_retry_info["last_success_at"] = rag_retry_info["last_attempt_at"]
            rag_retry_info["last_error"] = None
            print("🤖 Đã khởi tạo GeminiRAG thành công!")
            logger.info("GeminiRAG initialized successfully")
            return True
        except Exception as e:
            error_msg = f"Failed to initialize GeminiRAG: {e}"
            rag_system = None
            rag_last_error = str(e)
            rag_retry_info["last_error"] = str(e)
            log_error(error_msg)
            print(f"❌ {error_msg}")
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
                f"Retrying GeminiRAG initialization in {delay_seconds}s (attempt {attempt}/{max_retries})"
            )
            time.sleep(delay_seconds)
    logger.error("GeminiRAG failed to initialize after retries")


if not SKIP_RAG_INIT:
    threading.Thread(target=_init_rag_background, daemon=True).start()
else:
    logger.info("Skipping GeminiRAG background init (LEGALADVISOR_SKIP_RAG_INIT=1)")

# Khởi động thread dọn dẹp rate-limit nếu bật
if False:
    pass

# Pydantic models
class QuestionRequest(BaseModel):
    # Giới hạn chiều dài câu hỏi để tránh lạm dụng (mặc định 1024)
    question: constr(min_length=1, max_length=1024)  # type: ignore
    # Ràng buộc top_k trong khoảng 1..50
    top_k: conint(ge=1, le=50) = 3  # type: ignore


class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    num_sources: int
    status: str


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
    api_key = os.getenv("GOOGLE_API_KEY")
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
            logger.info("GeminiRAG reinitialized on-demand")
        else:
            log_error("RAG system not available")
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "RAG system is not available",
                    "error": _sanitize_sensitive(rag_last_error),
                    "hint": "Verify GOOGLE_API_KEY and retrieval models are configured."
                }
            )

    try:
        # Xử lý câu hỏi
        # Sử dụng GeminiRAG.ask để lấy câu trả lời và nguồn
        result = rag_system.ask(request.question, top_k=request.top_k or 3)

        response_time = time.time() - start_time
        log_performance("api_request", response_time, {
            "question": request.question if LOG_QUESTIONS else "[masked]",
            "confidence": float(result.get('confidence', 0.0)),
            "num_sources": int(result.get('num_sources', 0))
        })

        # Tạo phản hồi chuẩn hóa, GeminiRAG hiện chưa trả về confidence -> mặc định 0.0
        return AnswerResponse(
            question=result.get('question', request.question),
            answer=result.get('answer', ''),
            confidence=float(result.get('confidence', 0.0)),
            sources=result.get('sources', []),
            num_sources=int(result.get('num_sources', 0)),
            status=result.get('status', 'success')
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
        # Load metadata để lấy thống kê
        # Xác định đường dẫn models/retrieval từ root dự án hoặc từ biến môi trường
        env_models_dir = os.getenv("LEGALADVISOR_MODELS_DIR")
        if env_models_dir:
            model_dir = Path(env_models_dir)
        else:
            current_dir = Path(__file__).resolve().parent  # src/app
            root_dir = current_dir.parent.parent  # -> root
            model_dir = root_dir / "models" / "retrieval"
        metadata_path = model_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            total_chunks = len(metadata)
            total_words = sum(item.get('word_count', 0) for item in metadata)

            return {
                "total_chunks": total_chunks,
                "total_words": total_words,
                "avg_words_per_chunk": total_words / total_chunks if total_chunks > 0 else 0,
                "model_name": rag_system.model_info.get("model_name", "unknown")
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
