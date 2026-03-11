"""
Orchestrator for Retrieval Services

Provides a unified interface (RetrievalFacade) that transparently selects
between Euclidean Search and Dual-Space Hyperbolic Search based on environment configuration.
"""

import os
from typing import List, Dict, Any

from .service import RetrievalService as EuclideanRetrievalService
from .hyperbolic_service import HyperbolicRetrievalService
from ..utils.logger import get_logger

class RetrievalOrchestrator:
    """
    Facade class that handles routing between different retrieval strategies.
    Should be used as a drop-in replacement for RetrievalService in RAG pipelines.
    """
    def __init__(self, use_gpu: bool = False):
        self._logger = get_logger("legaladvisor.retrieval_orchestrator")
        self.use_gpu = use_gpu
        
        self.use_hyperbolic = os.getenv("LEGALADVISOR_USE_HYPERBOLIC", "0").strip() in ("1", "true", "yes", "on")
        
        if self.use_hyperbolic:
            self._logger.info("Initializing Dual-Space Hyperbolic Retrieval")
            self._service = HyperbolicRetrievalService(use_gpu=use_gpu)
            # Expose encoder for compatibility with API endpoints expecting `model_info`
            self.encoder = getattr(self._service.euclidean_service, "encoder", None)
            self.metadata = getattr(self._service.euclidean_service, "metadata", {})
        else:
            self._logger.info("Initializing Standard Euclidean Retrieval")
            self._service = EuclideanRetrievalService(use_gpu=use_gpu)
            self.encoder = getattr(self._service, "encoder", None)
            self.metadata = getattr(self._service, "metadata", {})
            
        # Mock model_info based on underlying service
        self.model_info = getattr(self._service, "model_info", {"model_name": "unknown"})
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self._service.retrieve(query, top_k=top_k)
        
    def encode_query(self, query: str) -> Any:
        # Pass through to underlying euclidean service to get text embeddings if needed
        svc = getattr(self._service, "euclidean_service", self._service)
        return svc.encode_query(query)
        
    def get_chunk_content(self, chunk_id: int) -> str:
        svc = getattr(self._service, "euclidean_service", self._service)
        return svc.get_chunk_content(chunk_id)
        
    def get_article_text(self, code: str, article_num: int) -> str:
        svc = getattr(self._service, "euclidean_service", self._service)
        return svc.get_article_text(code, article_num)
        
    def get_article_contents(self, code: str, article_num: int) -> List[Dict[str, Any]]:
        svc = getattr(self._service, "euclidean_service", self._service)
        return svc.get_article_contents(code, article_num)
        
    def get_article_segments_text(self, chunk_id: int = None, max_segments: int = 10) -> List[Dict[str, Any]]:
        svc = getattr(self._service, "euclidean_service", self._service)
        if hasattr(svc, 'get_article_segments_text'):
            return svc.get_article_segments_text(chunk_id, max_segments)
        return []
        
    def get_document_text_all(self, code: str) -> str:
        svc = getattr(self._service, "euclidean_service", self._service)
        return svc.get_document_text_all(code)
