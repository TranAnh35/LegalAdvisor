"""
Configuration Management Module
Centralizes all environment variable lookups and defaults.
"""

import os
from dotenv import load_dotenv

# Load explicitly to ensure .env is parsed if running locally
load_dotenv()

class Config:
    @classmethod
    def get_bool(cls, key: str, default: bool = False) -> bool:
        val = str(os.getenv(key, str(default))).strip().lower()
        return val in ("1", "true", "yes", "on")

    @classmethod
    def get_int(cls, key: str, default: int = 0) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    @classmethod
    def get_float(cls, key: str, default: float = 0.0) -> float:
        try:
            return float(os.getenv(key, str(default)))
        except (ValueError, TypeError):
            return default

    @classmethod
    def get_str(cls, key: str, default: str = "") -> str:
        return os.getenv(key, default)

# ================= RAG CONFIG =================
RAG_GEMINI_MODEL = Config.get_str("LEGALADVISOR_GEMINI_MODEL", "gemini-2.5-flash-lite")
RAG_CONTEXT_TOTAL_CHARS = Config.get_int("LEGALADVISOR_CONTEXT_TOTAL_CHARS", 200000)
RAG_CONTEXT_DOC_CHARS = Config.get_int("LEGALADVISOR_CONTEXT_DOC_CHARS", 40000)
RAG_CONTEXT_CITATION_CHARS = Config.get_int("LEGALADVISOR_CONTEXT_CITATION_CHARS", 20000)
RAG_MAX_SEGMENTS_PER_ARTICLE = Config.get_int("LEGALADVISOR_CONTEXT_MAX_SEGMENTS_PER_ARTICLE", 10)

# ================= RETRIEVAL CONFIG =================
RETRIEVAL_USE_HYPERBOLIC = Config.get_bool("LEGALADVISOR_USE_HYPERBOLIC", False)
RETRIEVAL_GROUP_STRATEGY = Config.get_str("LEGALADVISOR_GROUP_STRATEGY", "mean")
RETRIEVAL_GROUP_TOPM = Config.get_int("LEGALADVISOR_GROUP_TOPM", 3)

# ================= API CONFIG =================
API_RATE_LIMIT_WINDOW = Config.get_int("LEGALADVISOR_RATE_LIMIT_WINDOW", 60)
API_RATE_LIMIT_MAX = Config.get_int("LEGALADVISOR_RATE_LIMIT_MAX", 30)
API_LOG_QUESTIONS = Config.get_bool("LEGALADVISOR_LOG_QUESTIONS", False)
API_SKIP_RAG_INIT = Config.get_bool("LEGALADVISOR_SKIP_RAG_INIT", False)

# ================= HIERARCHY RERANKER =================
RERANKER_ENABLED = Config.get_bool("LEGALADVISOR_HIERARCHY_RERANK", True)
RERANKER_ALPHA = Config.get_float("LEGALADVISOR_HIERARCHY_ALPHA", 0.15)
RERANKER_MAX_PER_DOC = Config.get_int("LEGALADVISOR_HIERARCHY_MAX_PER_DOC", 3)
RERANKER_AUTHORITY = Config.get_bool("LEGALADVISOR_HIERARCHY_AUTHORITY", True)

