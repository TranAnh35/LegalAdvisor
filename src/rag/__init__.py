"""
RAG (Retrieval-Augmented Generation) module for LegalAdvisor

Combines document retrieval with Gemini LLM for legal QA.

Components:
- gemini_rag: GeminiRAG for question answering
"""

from .gemini_rag import GeminiRAG

__all__ = ['GeminiRAG']
