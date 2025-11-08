"""
Retrieval module for LegalAdvisor

Implements FAISS-based semantic search for legal documents.

Components:
- service: RetrievalService for document retrieval
- build_index: Building FAISS index from preprocessed data
- search: Search utilities
"""

from .service import RetrievalService

__all__ = ['RetrievalService']
