#!/usr/bin/env python3
"""
Tiện ích chuẩn hóa đường dẫn cho LegalAdvisor.

Ưu tiên biến môi trường khi có:
- LEGALADVISOR_ROOT
- LEGALADVISOR_MODELS_DIR
- LEGALADVISOR_DATA_DIR
"""

from pathlib import Path
import os
from typing import Optional


def get_project_root() -> Path:
    """Xác định thư mục gốc dự án.

    Thứ tự:
    - ENV LEGALADVISOR_ROOT
    - Dò ngược từ file utils này đến khi thấy thư mục tên LegalAdvisor hoặc chứa thư mục con data/ và models/
    - Fallback: 3 cấp trên file hiện tại
    """
    env_root = os.getenv("LEGALADVISOR_ROOT")
    if env_root:
        p = Path(env_root).resolve()
        if p.exists():
            return p

    probe = Path(__file__).resolve()
    for parent in [probe.parent, *probe.parents]:
        # Ưu tiên tên thư mục LegalAdvisor
        if parent.name.lower() == "legaladvisor":
            return parent
        # Hoặc có cả data và models
        if (parent / "data").exists() and (parent / "models").exists():
            return parent

    # Fallback 3 cấp trên
    return Path(__file__).resolve().parents[3]


def get_models_retrieval_dir() -> Path:
    """Thư mục models/retrieval (ưu tiên ENV LEGALADVISOR_MODELS_DIR)."""
    env_models = os.getenv("LEGALADVISOR_MODELS_DIR")
    if env_models:
        p = Path(env_models).resolve()
        if p.exists():
            return p
    return get_project_root() / "models" / "retrieval"


def get_processed_data_dir() -> Path:
    """Thư mục data/processed (ưu tiên ENV LEGALADVISOR_DATA_DIR)."""
    env_data = os.getenv("LEGALADVISOR_DATA_DIR")
    if env_data:
        p = Path(env_data).resolve()
        if p.exists():
            return p
    return get_project_root() / "data" / "processed"


