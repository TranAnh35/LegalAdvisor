#!/usr/bin/env python3
"""
FastAPI backend cho LegalAdvisor
"""

import sys
import os
import signal
sys.path.append('../..')

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from pathlib import Path
import json
import time
import torch

# Import logger
try:
    from utils.logger import get_logger, log_performance, log_error
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
import sys
import os
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

def _init_rag_background():
    global rag_system
    try:
        from rag.gemini_rag import GeminiRAG
        rag_system = GeminiRAG(use_gpu=args.use_gpu)
        print("🤖 Đã khởi tạo GeminiRAG thành công (lazy)!")
    except Exception as e:
        print(f"❌ Lỗi khi khởi tạo GeminiRAG (lazy): {str(e)}")

import threading
threading.Thread(target=_init_rag_background, daemon=True).start()

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

# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class AnswerResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    num_sources: int
    status: str

class HealthResponse(BaseModel):
    status: str
    message: str
    rag_loaded: bool

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
    return HealthResponse(
        status="healthy" if rag_system else "degraded",
        message="LegalAdvisor API is running",
        rag_loaded=rag_system is not None
    )

@app.post("/ask", response_model=AnswerResponse, tags=["QA"])
async def ask_question(request: QuestionRequest):
    """Trả lời câu hỏi pháp luật"""

    logger.info(f"Received question: {request.question}")
    start_time = time.time()

    if not rag_system:
        log_error("RAG system not available")
        raise HTTPException(
            status_code=503,
            detail="RAG system is not available. Please check the system logs."
        )

    try:
        # Xử lý câu hỏi
        # Sử dụng GeminiRAG.ask để lấy câu trả lời và nguồn
        result = rag_system.ask(request.question, top_k=request.top_k or 3)

        response_time = time.time() - start_time
        log_performance("api_request", response_time, {
            "question": request.question,
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
        log_error(f"Error processing question '{request.question}': {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
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
