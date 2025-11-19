"""
Utility modules for LegalAdvisor

Includes:
- logger: Logging configuration
- paths: Path utilities
"""

from .logger import get_logger
from .paths import get_project_root, get_models_retrieval_dir, get_processed_data_dir

__all__ = [
    'get_logger',
    'get_project_root',
    'get_models_retrieval_dir',
    'get_processed_data_dir'
]
