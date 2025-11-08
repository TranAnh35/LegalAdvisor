"""
Application layer for LegalAdvisor

Provides FastAPI backend and Streamlit UI.

Components:
- api: FastAPI application with endpoints
- ui: Streamlit web interface
"""

from .api import app

__all__ = ['app']
