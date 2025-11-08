"""
LegalAdvisor Source Module

Main source code for the LegalAdvisor NLP legal QA system.

Packages:
- app: FastAPI backend and Streamlit UI
- rag: Retrieval-Augmented Generation using Gemini
- retrieval: Document retrieval using FAISS
- data_preprocessing: Data preprocessing pipelines
- utils: Utility functions and helpers
- tools: Data processing tools
"""

__version__ = "1.0.0"
__author__ = "LegalAdvisor Team"
__all__ = [
    "app",
    "rag",
    "retrieval",
    "data_preprocessing",
    "utils",
    "tools"
]
