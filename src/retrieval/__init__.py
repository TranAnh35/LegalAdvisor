"""
Retrieval module for LegalAdvisor

Implements FAISS-based semantic search for legal documents.

Components:
- service: RetrievalService for document retrieval
- build_index: Building FAISS index from preprocessed data
- search: Search utilities
- hierarchy_reranker: Hierarchy-aware reranking for improved retrieval
- hyperbolic_service: Dual-Space Retrieval (Euclidean + Hyperbolic) Service
"""

from .service import RetrievalService
from .hierarchy_reranker import HierarchyReranker
from .hyperbolic_service import HyperbolicRetrievalService

__all__ = ['RetrievalService', 'HierarchyReranker', 'HyperbolicRetrievalService']
