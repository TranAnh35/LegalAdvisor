"""
RAG (Retrieval-Augmented Generation) module for LegalAdvisor

Combines document retrieval with a Groq LLM for legal QA.

Components:
- groq_rag: Groq-backed RAG class for question answering
"""

from .groq_rag import GroqRAG

__all__ = ["GroqRAG"]