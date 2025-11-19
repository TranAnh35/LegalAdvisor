"""Citation extraction utilities for LegalAdvisor.

This package provides tools to extract legal citations (Điều/Khoản/Điểm + văn bản) from text.
"""

from .extract import (
    ActLocation,
    CitationHit,
    extract_citations,
)

__all__ = [
    "ActLocation",
    "CitationHit",
    "extract_citations",
]
