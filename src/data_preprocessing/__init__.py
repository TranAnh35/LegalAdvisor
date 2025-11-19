"""
Data Preprocessing Module - LegalAdvisor

This module handles preprocessing of legal document data for the LegalAdvisor system.

Currently supports:
- Zalo-AI-Legal corpus preprocessing
"""

from .zalo_legal import (
    parse_corpus_id,
    preprocess_corpus,
    load_and_parse_corpus,
    save_schema_jsonl
)

__all__ = [
    'parse_corpus_id',
    'preprocess_corpus',
    'load_and_parse_corpus',
    'save_schema_jsonl'
]
